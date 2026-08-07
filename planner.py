"""
evsmem/planner.py — EvAgent-facing tool interfaces.

Exposes memory_search and ask_memory_agent as callable tools
for the main EvAgent process (TypeScript/Node.js tool dispatch).

Uses MemoryPipeline as the singleton orchestrator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from evsmem.memory_pipeline import MemoryPipeline

logger = logging.getLogger(__name__)

# ── Singleton pipeline instance ───────────────────────────────────────────

_pipeline: Optional[MemoryPipeline] = None


def get_pipeline() -> MemoryPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MemoryPipeline()
        _pipeline.initialize()
    return _pipeline


def set_pipeline(pipeline: MemoryPipeline) -> None:
    """Override the global pipeline (for testing or custom config)."""
    global _pipeline
    _pipeline = pipeline


# ============================================================
# Tool: memory_search
# ============================================================

MEMORY_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "memory_search",
    "description": "Search across all memory categories using semantic search. "
    "Returns relevant memories with similarity scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (natural language)",
            },
            "category": {
                "type": "string",
                "enum": [
                    "hot_memory",
                    "user_preference",
                    "behavior_pattern",
                    "conclusion",
                    "long_term_memory",
                    "relationship",
                    None,
                ],
                "description": "Optional category filter",
            },
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of results to return",
            },
            "threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Minimum similarity threshold (0-1)",
            },
        },
        "required": ["query"],
    },
}


def memory_search(
    query: str,
    category: Optional[str] = None,
    top_k: int = 10,
    threshold: float = 0.5,
) -> str:
    """
    EvAgent tool: semantic memory search.

    Performs hybrid search (dense vector + FTS5 keyword) across all memories.
    Returns a JSON string for the EvAgent tool dispatch system.

    Args:
        query: Natural language search query.
        category: Optional category filter (e.g. ``"user_preference"``).
        top_k: Maximum number of results (default 10).
        threshold: Minimum similarity / RRF score (default 0.5).

    Returns:
        JSON string with ``success``, ``results``, and ``total`` fields.
    """
    try:
        pipeline = get_pipeline()

        # If a specific category is requested, pass it directly to the
        # retrieval engine for a more efficient search.
        if category is not None:
            results = pipeline.retrieval.retrieve_relevant_context(
                query=query, category=category, top_k=top_k * 2
            )
        else:
            results = pipeline.search(query, top_k=top_k * 2)

        # Normalise and filter results
        filtered: list[dict[str, Any]] = []
        for r in results:
            r_cat: str = r.get("_category") or r.get("category", "unknown")
            r_score: float = (
                r.get("_rrf_score")
                or r.get("_similarity")
                or r.get("score", 0.0)
            )

            # Post-filter in case the retrieval engine returned
            # more categories than requested (edge case safety).
            if category and r_cat != category:
                continue
            if r_score < threshold:
                continue

            filtered.append({
                "id": r.get("id", ""),
                "content": str(r.get("content", ""))[:500],
                "category": r_cat,
                "score": round(float(r_score), 4),
                "confidence": r.get("confidence", 0.5),
                "importance": r.get("importance", 5),
                "created_at": str(r.get("created_at", "")),
                "metadata": r.get("metadata", {}),
            })

        return json.dumps({
            "success": True,
            "results": filtered[:top_k],
            "total": len(filtered),
        })

    except Exception as e:
        logger.error("memory_search failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e),
            "results": [],
        })


# ============================================================
# Tool: ask_memory_agent
# ============================================================

ASK_MEMORY_AGENT_SCHEMA: dict[str, Any] = {
    "name": "ask_memory_agent",
    "description": "Ask the autonomous memory agent a question. The agent "
    "searches all memories and synthesises an answer. Use this "
    "for complex questions that require reasoning across memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to answer based on stored memories",
            },
            "context": {
                "type": "string",
                "description": "Optional additional context to help the agent",
            },
        },
        "required": ["question"],
    },
}


def ask_memory_agent(question: str, context: Optional[str] = None) -> str:
    """
    EvAgent tool: delegate a reasoning query to the memory agent.

    The memory agent performs RAG: it searches for relevant memories,
    then uses the 2B LLM to synthesise an answer.

    Args:
        question: The question to answer.
        context: Optional additional context for the prompt.

    Returns:
        JSON string with ``success``, ``answer``, and optional ``error``.
    """
    try:
        pipeline = get_pipeline()
        answer = pipeline.query(question, context)

        return json.dumps({
            "success": True,
            "answer": answer,
        })

    except Exception as e:
        logger.error("ask_memory_agent failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e),
            "answer": "",
        })


# ============================================================
# Tool registry for EvAgent integration
# ============================================================

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "memory_search": {
        "handler": memory_search,
        "schema": MEMORY_SEARCH_SCHEMA,
    },
    "ask_memory_agent": {
        "handler": ask_memory_agent,
        "schema": ASK_MEMORY_AGENT_SCHEMA,
    },
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return JSON schemas for all tools (for EvAgent registration)."""
    return [info["schema"] for info in TOOL_REGISTRY.values()]


def execute_tool(name: str, params: dict[str, Any]) -> str:
    """
    Execute a tool by name with parameters.

    This is the entry point called by the EvAgent tool dispatch system
    (TypeScript/Node.js).

    Args:
        name: Tool name (``"memory_search"`` or ``"ask_memory_agent"``).
        params: Tool parameters dict.

    Returns:
        Tool result as a JSON string.
    """
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    try:
        return tool["handler"](**params)
    except Exception as e:
        logger.error("execute_tool(%s) failed: %s", name, e)
        return json.dumps({"success": False, "error": str(e)})
