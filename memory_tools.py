"""
evsmem/memory_tools.py — Tool wrappers for the Memory Agent.

All memory mutations go through these tools. The agent never
directly accesses SQLite.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .memory_store import MemoryStore
from .retrieval import RetrievalEngine
from .schemas import (
    CreateMemoryOutput,
    DeleteMemoryOutput,
    MemoryResult,
    MEMORY_CATEGORIES,
    SearchMemoryOutput,
    UpdateMemoryOutput,
)

logger = logging.getLogger(__name__)


# ============================================================
# LLM Tool Definitions (JSON Schema for function-calling LLMs)
# ============================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "create_memory",
        "description": "Create a new memory. Automatically checks for duplicates before creating.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "hot_memory",
                        "user_preference",
                        "behavior_pattern",
                        "conclusion",
                        "long_term_memory",
                        "relationship",
                    ],
                    "description": "Memory category",
                },
                "content": {
                    "type": "string",
                    "description": "The memory content text",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence in this memory (0-1)",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance level (1-10)",
                },
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "update_memory",
        "description": "Update an existing memory by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory UUID"},
                "category": {"type": "string", "description": "Memory category"},
                "updates": {
                    "type": "object",
                    "description": "Fields to update",
                    "properties": {
                        "content": {"type": "string"},
                        "confidence": {"type": "number"},
                        "importance": {"type": "integer"},
                        "metadata": {"type": "object"},
                    },
                },
            },
            "required": ["memory_id", "category", "updates"],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a memory by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["memory_id", "category"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search memories semantically.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max results",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum similarity threshold",
                },
            },
            "required": ["query"],
        },
    },
]


# ============================================================
# MemoryTools
# ============================================================


class MemoryTools:
    """
    Tool wrappers for the autonomous memory agent.

    Every create/update operation includes a pre-write RAG check
    to avoid duplicates. The retrieval engine is consulted first,
    and if a semantically similar memory exists, the tool updates
    it instead of creating a new one.
    """

    def __init__(
        self,
        store: MemoryStore,
        retrieval: RetrievalEngine | None = None,
    ) -> None:
        self.store = store
        self.retrieval = retrieval

    # ============================================================
    # CREATE
    # ============================================================

    def create_memory(
        self,
        category: str,
        content: str,
        confidence: float = 0.5,
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
        source_batch_id: int | None = None,
        **extra_fields: Any,
    ) -> CreateMemoryOutput:
        """
        Create a new memory.

        Pre-write RAG: if a semantically similar memory exists
        (threshold > 0.85), returns the existing one as duplicate.
        """
        # Pre-write dedup check
        if self.retrieval is not None:
            duplicate = self.retrieval.find_duplicate(
                content, category, threshold=0.85
            )
            if duplicate is not None:
                # Merge: update confidence and importance on the duplicate
                merge_updates: dict[str, Any] = {
                    "confidence": min(
                        1.0, duplicate.get("confidence", 0.5) + 0.1
                    ),
                    "importance": max(
                        duplicate.get("importance", 5), importance
                    ),
                    "updated_at": datetime.utcnow().isoformat(),
                }
                self.store.update_memory(
                    duplicate["id"],
                    category,
                    **merge_updates,
                )
                logger.info("Duplicate found, merged into %s", duplicate["id"])
                return CreateMemoryOutput(
                    memory_id=duplicate["id"],
                    category=category,
                    success=True,
                    duplicate=True,
                    merged_with=duplicate["id"],
                )

        # Create new memory — MemoryStore.create_memory takes **fields
        memory_id = self.store.create_memory(
            category,
            content=content,
            confidence=confidence,
            importance=importance,
            metadata=metadata or {},
            source_batch_id=source_batch_id,
            **extra_fields,
        )

        # Generate and store embedding
        if memory_id:
            self._embed_and_store(memory_id, category, content)

        return CreateMemoryOutput(
            memory_id=memory_id,
            category=category,
            success=bool(memory_id),
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update_memory(
        self,
        memory_id: str,
        category: str,
        updates: dict[str, Any],
    ) -> UpdateMemoryOutput:
        """Update an existing memory. Re-embeds if content changes."""
        # MemoryStore.update_memory takes (memory_id, category, **updates)
        success = self.store.update_memory(
            memory_id,
            category,
            **updates,
        )

        # Re-embed if content changed
        if success and "content" in updates:
            self._embed_and_store(
                memory_id, category, str(updates["content"])
            )

        return UpdateMemoryOutput(memory_id=memory_id, success=success)

    # ============================================================
    # DELETE
    # ============================================================

    def delete_memory(
        self,
        memory_id: str,
        category: str,
    ) -> DeleteMemoryOutput:
        """Delete a memory."""
        success = self.store.delete_memory(
            memory_id,
            category,
        )
        return DeleteMemoryOutput(memory_id=memory_id, success=success)

    # ============================================================
    # SEARCH
    # ============================================================

    def search_memory(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> SearchMemoryOutput:
        """
        Search memories using hybrid retrieval.

        If retrieval engine is available, uses hybrid search.
        Otherwise falls back to text LIKE search.
        """
        if self.retrieval is not None:
            results = self.retrieval.retrieve_relevant_context(
                query=query,
                category=category,
                top_k=top_k,
            )
        else:
            raw = self.store.search_by_text(
                category, query, top_k
            )
            results = [
                {
                    "content": r["content"],
                    "category": category or "unknown",
                    "score": 0.5,
                }
                for r in raw
            ]

        # Convert to MemoryResult objects
        memory_results = self._dicts_to_memory_results(results, threshold)

        return SearchMemoryOutput(results=memory_results)

    # ============================================================
    # HELPERS
    # ============================================================

    def _embed_and_store(
        self,
        memory_id: str,
        category: str,
        content: str,
    ) -> None:
        """Generate embedding and store it."""
        try:
            from .embeddings import generate_embedding

            embedding = generate_embedding(content)
            if embedding:
                self.store.store_embedding(
                    memory_id,
                    category,
                    embedding,
                )
        except Exception as e:
            logger.warning(
                "Embedding generation failed for %s: %s", memory_id, e
            )

    def _dicts_to_memory_results(
        self,
        dicts: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> list[MemoryResult]:
        """
        Convert raw dicts (from retrieval or text search) into
        ``MemoryResult`` objects.  Results whose score falls below
        *threshold* are omitted.
        """
        results: list[MemoryResult] = []

        for d in dicts:
            # Extract score: prefer _rrf_score, then _similarity,
            # then embedded score dict key
            score: float = 0.5
            for key in ("_rrf_score", "_similarity", "score"):
                val = d.get(key)
                if val is not None:
                    score = float(val)
                    break

            if score < threshold:
                continue

            cat: str = (
                d.get("_category")
                or d.get("category")
                or "long_term_memory"
            )
            model_cls = MEMORY_CATEGORIES.get(cat)

            # Strip internal keys before model construction
            clean: dict[str, Any] = {
                k: v
                for k, v in d.items()
                if not k.startswith("_") and k != "category"
            }

            # Expand any JSON-string fields that may not have been decoded
            for field in (
                "metadata",
                "evidence",
                "tags",
                "properties",
                "supporting_evidence",
            ):
                if field in clean and isinstance(clean[field], str):
                    try:
                        clean[field] = json.loads(clean[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

            if model_cls is not None:
                try:
                    memory_obj = model_cls(**clean)
                except Exception as exc:
                    logger.warning(
                        "Failed to construct %s model: %s", cat, exc
                    )
                    continue
            else:
                # Fallback: use LongTermMemory
                from .schemas import LongTermMemory

                memory_obj = LongTermMemory(
                    content=str(clean.get("content", "")),
                    memory_type=cat,
                )

            results.append(
                MemoryResult(
                    memory=memory_obj,
                    score=round(score, 4),
                    category=cat,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ============================================================
    # Batch helpers
    # ============================================================

    def get_memory(
        self,
        memory_id: str,
        category: str,
    ) -> dict[str, Any] | None:
        """Fetch a single memory as a raw dict."""
        return self.store.get_memory(
            memory_id,
            category,
        )
