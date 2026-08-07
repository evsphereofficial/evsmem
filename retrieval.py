"""Hybrid retrieval engine — dense vector + FTS5 keyword search with RRF fusion.

Provides ``RetrievalEngine`` which layers FTS5 full-text search on top of the
existing ``MemoryStore`` vector similarity search and fuses results via
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

__all__ = ["RetrievalEngine"]

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .embeddings import generate_embedding, embedding_dimension
from .memory_store import MemoryStore
from .schemas import (
    CATEGORY_TABLE_MAP,
    MEMORY_CATEGORIES,
    MemoryResult,
)

logger = logging.getLogger("evsmem.retrieval")

# ── FTS5 column configuration per memory table ────────────────────────────
# Only text columns that are meaningful for keyword search are included.

_FTS_COLUMNS: dict[str, list[str]] = {
    "hot_memories": ["content"],
    "user_preferences": ["content", "preference_key", "preference_value"],
    "behavior_patterns": ["content", "pattern_type"],
    "conclusions_mem": ["content", "conclusion_type"],
    "long_term_memories": ["content", "memory_type", "tags"],
    "relationships": ["content", "source_entity", "target_entity", "relationship_type"],
}

# Reverse map: DB table name → category key
_TABLE_TO_CATEGORY: dict[str, str] = {v: k for k, v in CATEGORY_TABLE_MAP.items()}

# Fields stored as JSON TEXT in DB (mirrors memory_store._JSON_FIELDS)
_JSON_FIELDS: set[str] = {"metadata", "evidence", "tags", "properties", "supporting_evidence"}

# Fields injected by MemoryStore.similarity_search that should be stripped
# when constructing Pydantic models.
_INTERNAL_KEYS: set[str] = {"_category", "_similarity", "_keyword_score", "_fts_rank"}


# ── Helpers ───────────────────────────────────────────────────────────────


def _expand_json_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Deserialise JSON string fields in-place (mirrors memory_store._row_to_dict)."""
    for field in _JSON_FIELDS:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _strip_internal_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Remove internal metadata keys before constructing Pydantic models."""
    for key in _INTERNAL_KEYS:
        d.pop(key, None)
    return d


def _safe_fts_query(raw_query: str) -> str:
    """Convert a free-text query into a safe FTS5 OR query.

    Splits on whitespace, discards very short tokens, and joins with ``OR``
    so that any single term can produce a match (lenient keyword search).
    """
    tokens = raw_query.split()
    safe: list[str] = []
    for token in tokens:
        token = token.strip().strip('"').strip("'")
        if len(token) > 1:  # skip single chars and punctuation-only tokens
            # Escape any remaining double-quote characters
            token = token.replace('"', "")
            if token and len(token) > 1:
                safe.append(token)
    if not safe:
        # Fallback: use the original query as a phrase (may fail)
        return f'"{raw_query}"'
    return " OR ".join(safe)


# ── RetrievalEngine ───────────────────────────────────────────────────────


class RetrievalEngine:
    """Hybrid search engine combining dense BGE-M3 embeddings with FTS5 keyword search.

    Usage::

        store = MemoryStore(db_path="evsmem.db")
        engine = RetrievalEngine(store)

        # Hybrid search
        results = engine.retrieve_relevant_context("user likes hiking", top_k=5)

        # Pre-write duplicate check
        dup = engine.find_duplicate("user prefers dark mode", "user_preference")
    """

    def __init__(
        self,
        store: MemoryStore,
        db_path: str | Path | None = None,
    ) -> None:
        self.store = store
        # Access the same database file that MemoryStore uses
        self._db_path: Path = Path(db_path) if db_path else store._db_path  # type: ignore[attr-defined]
        self._local = threading.local()
        self._init_fts()

    # ── Connection management ─────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection to the same DB file as MemoryStore."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Explicitly close the thread-local connection if open."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ── FTS5 initialisation ───────────────────────────────────────────────

    def _init_fts(self) -> None:
        """Create FTS5 virtual tables with content-sync for each memory category.

        Uses ``content=`` so that INSERT / UPDATE / DELETE on the source table
        automatically propagates to the FTS index via built-in triggers.
        """
        conn = self._conn()
        for table_name, columns in _FTS_COLUMNS.items():
            col_def = ", ".join(columns)
            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_fts
                    USING fts5(
                        {col_def},
                        content='{table_name}',
                        content_rowid='rowid'
                    )
                    """
                )
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 init for %s failed: %s", table_name, exc)
        conn.commit()

    def rebuild_fts_index(self) -> None:
        """Rebuild all FTS indexes from the current source table contents.

        Uses the standard FTS5 rebuild command (``INSERT INTO … VALUES('rebuild')``)
        which re-indexes all rows from the content table without manual DELETE/INSERT.
        Call this once after ``RetrievalEngine`` is first created if the source
        tables already contained data before the FTS virtual tables were created.
        """
        conn = self._conn()
        for table_name in _FTS_COLUMNS:
            fts_table = f"{table_name}_fts"
            try:
                conn.execute(
                    f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')"
                )
                n = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM {fts_table}"
                ).fetchone()
                count = n["cnt"] if n else 0
                logger.info("Rebuilt FTS index for %s (%d rows)", table_name, count)
            except sqlite3.OperationalError as exc:
                logger.warning("FTS rebuild for %s failed: %s", table_name, exc)
        conn.commit()

    # ── Core hybrid search ────────────────────────────────────────────────

    def retrieve_relevant_context(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining dense vector similarity + FTS5 keyword score.

        Steps:
        1. Generate query embedding via BGE-M3
        2. Search FTS5 for keyword matches
        3. Search MemoryStore for dense vector matches
        4. Fuse scores with Reciprocal Rank Fusion (RRF)
        5. Return *top_k* results with ``_rrf_score`` key

        Each result dict contains all columns from the source memory table plus
        ``_category``, ``_similarity``, and ``_rrf_score``.
        """
        if not query or not query.strip():
            return []

        # 1. Embedding
        query_emb = generate_embedding(query)

        fetch_k = top_k * 3  # over-fetch for fusion quality

        # 2. Vector search via MemoryStore
        vector_results = self.store.similarity_search(
            query_embedding=query_emb,
            memory_category=category,
            top_k=fetch_k,
        )

        # 3. Keyword search via FTS5
        keyword_results = self._keyword_search(
            query=query,
            category=category,
            top_k=fetch_k,
        )

        # 4. RRF fusion
        fused = self._rrf_fusion(vector_results, keyword_results, k=60)

        return fused[:top_k]

    def _keyword_search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """FTS5 keyword search across memory tables.

        Returns dicts with all source columns plus ``_category`` and
        ``_keyword_score`` (higher = better match, derived from BM25).
        """
        if not query or not query.strip():
            return []

        conn = self._conn()
        results: list[dict[str, Any]] = []

        tables: list[str]
        if category is not None:
            tbl = CATEGORY_TABLE_MAP.get(category)
            if tbl is None:
                raise ValueError(f"Unknown memory category: {category}")
            tables = [tbl]
        else:
            tables = list(CATEGORY_TABLE_MAP.values())

        for table_name in tables:
            if table_name is None:
                continue
            cat = _TABLE_TO_CATEGORY.get(table_name, "unknown")
            fts_table = f"{table_name}_fts"

            rows: list[sqlite3.Row] = []

            # Try the raw query first — FTS5 may handle it natively
            try:
                rows = conn.execute(
                    f"""
                    SELECT t.*, bm25({fts_table}) AS _fts_rank
                    FROM {fts_table}
                    JOIN {table_name} t ON t.rowid = {fts_table}.rowid
                    WHERE {fts_table} MATCH ?
                    ORDER BY _fts_rank
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                pass

            # Fallback: safe OR'ed terms
            if not rows:
                safe_query = _safe_fts_query(query)
                try:
                    rows = conn.execute(
                        f"""
                        SELECT t.*, bm25({fts_table}) AS _fts_rank
                        FROM {fts_table}
                        JOIN {table_name} t ON t.rowid = {fts_table}.rowid
                        WHERE {fts_table} MATCH ?
                        ORDER BY _fts_rank
                        LIMIT ?
                        """,
                        (safe_query, top_k),
                    ).fetchall()
                except sqlite3.OperationalError:
                    logger.debug("FTS5 MATCH failed for %s with query: %s", table_name, safe_query)
                    continue

            for row in rows:
                d: dict[str, Any] = dict(row)
                _expand_json_fields(d)
                # The BM25 rank from FTS5 is negative; lower (more negative) = better match.
                # Convert to a positive score where higher = better.
                rank_val = d.pop("_fts_rank", None)
                kw_score: float = 0.0
                if rank_val is not None:
                    kw_score = 1.0 / (1.0 + abs(rank_val))
                d["_category"] = cat
                d["_keyword_score"] = kw_score
                results.append(d)

        # Sort by keyword score descending
        results.sort(key=lambda x: x.get("_keyword_score", 0.0), reverse=True)
        return results[:top_k]

    # ── RRF Fusion ────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion.

        Fuses two ranked lists by combining reciprocal ranks:

            score(d) = sum( 1 / (k + rank(d)) )

        where *rank(d)* is the 0-based position of document *d* in each list.
        Results are returned sorted by descending fused score with an
        ``_rrf_score`` key added.
        """
        rrf_scores: dict[str, float] = {}
        results_map: dict[str, dict[str, Any]] = {}

        # Helper: add a document's contribution
        def _add_list(docs: list[dict[str, Any]]) -> None:
            for rank, doc in enumerate(docs):
                mem_id = doc.get("id")
                if not mem_id:
                    continue
                rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + 1.0 / (k + rank + 1)
                if mem_id not in results_map:
                    results_map[mem_id] = doc

        _add_list(vector_results)
        _add_list(keyword_results)

        # Sort by descending RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        fused: list[dict[str, Any]] = []
        for mem_id in sorted_ids:
            doc = dict(results_map[mem_id])  # shallow copy
            doc["_rrf_score"] = round(rrf_scores[mem_id], 6)
            fused.append(doc)

        return fused

    # ── Semantic similarity search (for pre-write dedup, etc.) ──────────

    def search_similar_memories(
        self,
        query: str,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> list[MemoryResult]:
        """Search for semantically similar memories across all categories.

        Returns ``MemoryResult`` objects (Pydantic models) sorted by
        descending relevance.  Only results with a similarity / RRF score
        above *threshold* are included.
        """
        if not query or not query.strip():
            return []

        # Use hybrid retrieval
        fused = self.retrieve_relevant_context(query, category=None, top_k=top_k * 2)

        memory_results: list[MemoryResult] = []
        for doc in fused:
            sim = doc.get("_similarity") or doc.get("_rrf_score", 0.0)
            if sim < threshold:
                continue

            cat: str = doc.get("_category", "long_term_memory")
            model_cls = MEMORY_CATEGORIES.get(cat)
            if model_cls is None:
                continue

            # Strip internal keys before constructing the Pydantic model
            clean = _strip_internal_keys(dict(doc))
            try:
                memory_obj = model_cls(**clean)
            except Exception as exc:
                logger.warning("Failed to construct %s model: %s", cat, exc)
                continue

            score = float(
                max(doc.get("_rrf_score", 0.0), doc.get("_similarity", 0.0))
            )
            memory_results.append(
                MemoryResult(
                    memory=memory_obj,
                    score=round(score, 4),
                    category=cat,
                )
            )

        memory_results.sort(key=lambda x: x.score, reverse=True)
        return memory_results[:top_k]

    # ── Pre-write duplicate detection ─────────────────────────────────────

    def find_duplicate(
        self,
        content: str,
        category: str,
        threshold: float = 0.85,
    ) -> dict[str, Any] | None:
        """Check if a semantically similar memory already exists.

        Before creating a new memory, call this to detect duplicates.
        Returns the existing memory dict (without internal keys) if
        similarity > *threshold*, otherwise ``None``.

        Only matches within the same *category* are considered duplicates.
        """
        if not content or not content.strip():
            return None

        # Quick vector-only search (more precise for duplicate detection)
        query_emb = generate_embedding(content)
        results = self.store.similarity_search(
            query_embedding=query_emb,
            memory_category=category,
            top_k=5,
        )

        for doc in results:
            sim = doc.get("_similarity", 0.0)
            if sim >= threshold:
                # Return a clean copy without internal keys
                clean = _strip_internal_keys(dict(doc))
                _expand_json_fields(clean)
                return clean

        return None
