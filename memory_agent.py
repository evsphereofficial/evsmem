"""
evsmem/memory_agent.py — Autonomous 2B LLM memory curator.

The agent owns **all** memory mutations.  It receives batches of raw
conversations, reasons about them, and calls tools to update the
memory store.  It never directly writes to SQLite — only through
the ``MemoryTools`` interface.

Architecture
------------
1. Deriver ingests raw messages from the ev-agent DB and groups them
   into batches (see :mod:`evsmem.deriver`).
2. Scheduler calls ``MemoryAgent.process_batch()`` once a batch
   reaches the minimum threshold (default 10 messages).
3. The agent retrieves relevant context via RAG (hybrid dense + FTS5).
4. It sends the batch + context to a local 2B LLM for reasoning.
5. The LLM emits JSON-structured tool calls.
6. The agent executes those calls through ``MemoryTools``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .schemas import BatchProcessingResult, MemoryAgentState, RawMessageBatch
from .memory_store import MemoryStore
from .memory_tools import MemoryTools
from .retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

# ── Default LLM endpoint (OpenAI-compatible) ──────────────────────────────

DEFAULT_LLM_ENDPOINT = "http://localhost:8080/v1/chat/completions"
DEFAULT_LLM_MODEL = "Qwen2.5-1.5B-Instruct"

# ── System prompt for the autonomous memory curator ───────────────────────

SYSTEM_PROMPT = """You are the Memory Agent — the autonomous memory curator for an AI coding assistant.

Your sole responsibility is maintaining the memory database. You never answer user questions directly.

## Your Role
You receive batches of raw conversations. For each batch, you must:
1. Understand the conversation
2. Extract important information
3. Update memory accordingly

## Your Tools
You have access to these tools — use them to update memory:
- create_memory(category, content, confidence, importance, metadata)
- update_memory(memory_id, category, updates)
- delete_memory(memory_id, category)
- search_memory(query, category, top_k, threshold)

## Memory Categories
- hot_memory: Current context, expires quickly
- user_preference: Persistent user preferences
- behavior_pattern: Detected behavior patterns
- conclusion: High-confidence inferences
- long_term_memory: Important facts (projects, skills, devices, goals)
- relationship: Connections between entities

## Processing Rules
1. ALWAYS search before creating — avoid duplicates
2. Extract ALL preferences, behaviors, and conclusions you can find
3. Update confidence based on repetition — repeated info = higher confidence
4. Hot memories should reference current conversation context
5. Long-term memories are for permanent facts
6. Relationships connect entities (tools, projects, concepts)

## Output Format
You MUST respond with a JSON object containing:
{
    "summary": "Brief conversation summary",
    "memory_ops": [
        {
            "tool": "create_memory" | "update_memory" | "delete_memory",
            "params": { ... tool parameters ... },
            "reasoning": "Why this operation"
        }
    ],
    "confidence": 0.95,
    "requires_human_review": false
}
"""


# ── MemoryAgent ────────────────────────────────────────────────────────────


class MemoryAgent:
    """Autonomous memory curator powered by a 2B local LLM.

    Processes batches of raw conversations, reasons about them,
    and updates memory via tool calls.

    Parameters
    ----------
    llm_endpoint:
        HTTP endpoint for an OpenAI-compatible LLM API
        (llama.cpp, Ollama, vLLM, etc.).
    llm_model:
        Model name sent in the API request body.
    store:
        ``MemoryStore`` instance for direct batch-queue queries.
    retrieval:
        ``RetrievalEngine`` instance for RAG context retrieval.
    tools:
        ``MemoryTools`` instance for executing agent tool calls.
    """

    def __init__(
        self,
        llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
        llm_model: str = DEFAULT_LLM_MODEL,
        store: MemoryStore | None = None,
        retrieval: RetrievalEngine | None = None,
        tools: MemoryTools | None = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.store = store
        self.retrieval = retrieval
        self.tools = tools
        self.state = MemoryAgentState()

    # ── Public entry point ─────────────────────────────────────────────

    def process_batch(self, batch: RawMessageBatch) -> BatchProcessingResult:
        """Process a single batch of raw conversations.

        Flow
        ----
        1. Build context with relevant memories (RAG).
        2. Format conversation text for the LLM.
        3. Call the local LLM with system prompt + context + conversation.
        4. Parse the JSON tool-call response.
        5. Execute each tool call through ``MemoryTools``.
        6. Return a ``BatchProcessingResult`` with created/updated IDs.
        """
        self.state.status = "processing"
        self.state.current_batch_id = batch.id

        context = self._build_context(batch)
        conversation_text = self._format_conversation(batch)
        llm_response = self._call_llm(context, conversation_text)
        parsed = self._parse_response(llm_response)

        result = self._execute_operations(
            parsed.get("memory_ops", []),
            batch_id=batch.id,
        )
        result.summary = parsed.get("summary", "")
        result.batch_id = batch.id

        self.state.status = "idle"
        self.state.current_batch_id = None
        self.state.last_processed_at = datetime.utcnow()
        self.state.cycle_count += 1

        return result

    # ── RAG context builder ────────────────────────────────────────────

    def _build_context(self, batch: RawMessageBatch) -> str:
        """Retrieve relevant memories from the store as context for the LLM.

        Uses the last 5 messages of the batch as the query text, searches
        across all categories, and formats the top results as a string.
        """
        if not self.retrieval:
            return "No context available."

        texts = [m.content for m in batch.messages]
        query = " ".join(texts[-5:]) if len(texts) >= 5 else " ".join(texts)

        # Hybrid retrieval across all categories
        results = self.retrieval.retrieve_relevant_context(query, top_k=15)

        context_parts: list[str] = []
        for r in results[:10]:
            cat = r.get("_category", "unknown")
            content = r.get("content", "")
            score = r.get("_rrf_score", r.get("_similarity", 0.0))
            context_parts.append(
                f"[{cat}](score:{score:.3f}) {content[:200]}"
            )

        return (
            "\n".join(context_parts)
            if context_parts
            else "No relevant context found."
        )

    # ── Conversation formatter ─────────────────────────────────────────

    @staticmethod
    def _format_conversation(batch: RawMessageBatch) -> str:
        """Format a batch of messages into a human-readable conversation log."""
        lines: list[str] = []
        for msg in batch.messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)

    # ── LLM call ───────────────────────────────────────────────────────

    def _call_llm(self, context: str, conversation: str) -> str:
        """Call the local LLM via HTTP (OpenAI-compatible API).

        Falls back to an empty JSON response on any connection or
        parsing failure so the pipeline does not stall.
        """
        import requests

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\n## Relevant Context\n"
                    + context
                ),
            },
            {
                "role": "user",
                "content": (
                    "Process this conversation batch.\n\n"
                    + conversation
                    + "\n\nRespond with the JSON object only."
                ),
            },
        ]

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(
                self.llm_endpoint, json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return json.dumps({
                "summary": f"LLM call failed: {exc}",
                "memory_ops": [],
                "confidence": 0.0,
                "requires_human_review": True,
            })

    # ── Response parser ────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        """Parse the JSON tool-call response from the LLM.

        Handles markdown code-block fences (`` ```json … ``` ``) that
        smaller models sometimes emit even when instructed to output
        raw JSON.
        """
        try:
            text = response.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                # Remove opening fence
                first_newline = text.find("\n")
                if first_newline != -1:
                    text = text[first_newline + 1:]
                # Remove trailing fence
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)

            # Ensure required keys exist
            parsed.setdefault("memory_ops", [])
            parsed.setdefault("summary", "")
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("requires_human_review", False)

            return parsed

        except json.JSONDecodeError:
            logger.error(
                "Failed to parse LLM response: %s", response[:200]
            )
            return {
                "summary": "Failed to parse LLM response",
                "memory_ops": [],
                "confidence": 0.0,
                "requires_human_review": True,
            }

    # ── Tool-call executor ─────────────────────────────────────────────

    def _execute_operations(
        self,
        ops: list[dict[str, Any]],
        batch_id: int,
    ) -> BatchProcessingResult:
        """Execute each parsed tool operation in sequence.

        For ``create_memory`` operations, a pre-write duplicate check
        is performed via ``RetrievalEngine.find_duplicate``.  If a
        near-duplicate exists, the operation is converted into a
        confidence-boosting update instead.
        """
        result = BatchProcessingResult(batch_id=batch_id)

        for op in ops:
            tool = op.get("tool")
            params = op.get("params", {})
            reasoning = op.get("reasoning", "")

            try:
                if tool == "create_memory" and self.tools:
                    cat = params.get("category", "long_term_memory")
                    content = params.get("content", "")

                    # Pre-write duplicate check
                    dup = (
                        self.retrieval.find_duplicate(content, cat)
                        if self.retrieval
                        else None
                    )
                    if dup is not None:
                        existing_conf = dup.get("confidence", 0.5)
                        new_conf = min(1.0, existing_conf + 0.1)
                        self.tools.update_memory(
                            dup["id"],
                            cat,
                            {
                                "confidence": new_conf,
                                "updated_at": datetime.utcnow().isoformat(),
                            },
                        )
                        result.memories_updated.append(dup["id"])
                        logger.debug(
                            "Merged duplicate (batch %d): %s -> %s "
                            "(conf %.2f->%.2f) — %s",
                            batch_id,
                            cat,
                            dup["id"][:8],
                            existing_conf,
                            new_conf,
                            reasoning[:80],
                        )
                    else:
                        create_result = self.tools.create_memory(**params)
                        if create_result.success:
                            result.memories_created.append(create_result.memory_id)
                            logger.debug(
                                "Created %s (batch %d): %s — %s",
                                cat,
                                batch_id,
                                create_result.memory_id[:8],
                                reasoning[:80],
                            )

                elif tool == "update_memory" and self.tools:
                    memory_id = params.get("memory_id", "")
                    category = params.get("category", "long_term_memory")
                    updates = params.get("updates", {})
                    self.tools.update_memory(memory_id, category, updates)
                    result.memories_updated.append(memory_id)
                    logger.debug(
                        "Updated %s (batch %d): %s — %s",
                        category,
                        batch_id,
                        memory_id[:8],
                        reasoning[:80],
                    )

                elif tool == "delete_memory" and self.tools:
                    self.tools.delete_memory(**params)
                    logger.debug(
                        "Deleted (batch %d): %s — %s",
                        batch_id,
                        params.get("memory_id", "?")[:8],
                        reasoning[:80],
                    )

                result.tool_calls_executed += 1

            except Exception as exc:
                logger.error(
                    "Tool call failed: %s(%s) — %s",
                    tool,
                    params,
                    exc,
                )

        # Update agent state counters
        self.state.total_memories_created += len(result.memories_created)
        self.state.total_memories_updated += len(result.memories_updated)

        logger.info(
            "Batch %d done: %d created, %d updated, %d tool calls — %s",
            batch_id,
            len(result.memories_created),
            len(result.memories_updated),
            result.tool_calls_executed,
            result.summary[:100] if result.summary else "(no summary)",
        )

        return result

    # ── Query helper (used by ``ask_memory_agent`` tool) ───────────────

    def ask(self, question: str, context: str | None = None) -> str:
        """Answer a question by searching memory and reasoning over results.

        This is the implementation backing the ``ask_memory_agent``
        EvAgent tool.  It performs RAG over the memory store, then
        asks the LLM to synthesise an answer.
        """
        if not self.retrieval:
            return "Retrieval engine not available."

        results = self.retrieval.retrieve_relevant_context(
            question, top_k=20
        )

        memory_context = "\n".join(
            [
                (
                    f"[{r.get('_category', 'unknown')}]"
                    f"(score:{r.get('_rrf_score', 0.0):.2f}): "
                    f"{r.get('content', '')[:300]}"
                )
                for r in results[:15]
            ]
        )

        ask_prompt = (
            "Based on the stored memories, answer this question concisely.\n\n"
            f"Question: {question}\n\n"
            f"Relevant memories:\n{memory_context}\n"
            + (f"\nAdditional context: {context}\n" if context else "")
            + "\nProvide a concise answer synthesising the relevant information."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory retrieval assistant. "
                    "Answer based on stored memories only."
                ),
            },
            {"role": "user", "content": ask_prompt},
        ]

        try:
            import requests

            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024,
            }
            resp = requests.post(
                self.llm_endpoint, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("ask() LLM call failed: %s", exc)
            return f"Error querying memory agent: {exc}"
