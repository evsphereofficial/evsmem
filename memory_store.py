"""SQLite persistence layer for the autonomous evsmem memory agent.

Provides connection management, raw-conversation queuing, category-aware CRUD
for all memory types, text-based search, and BLOB embedding storage.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterator

from evsmem.config import (
    EVSMEM_DB_PATH,
    CATEGORY_TABLE_MAP,
    get_table_name,
    get_all_new_table_names,
    RAW_CONVERSATIONS_TABLE,
    RAW_BATCHES_TABLE,
    CURSOR_STATE_TABLE,
    HOT_MEMORIES_TABLE,
    USER_PREFERENCES_TABLE,
    BEHAVIOR_PATTERNS_TABLE,
    CONCLUSIONS_TABLE,
    LONG_TERM_MEMORIES_TABLE,
    RELATIONSHIPS_TABLE,
)
from .schemas import (
    RawMessageBatch,
)


# ── Helpers ──

def _utcnow() -> str:
    return datetime.utcnow().isoformat()


def _serialise_json(value: Any) -> str:
    """Serialise a Python value to a JSON string for TEXT columns."""
    return json.dumps(value, default=str)


def _deserialise_json(value: str | None) -> Any:
    """Deserialise a JSON string from a TEXT column back to a Python value."""
    if value is None:
        return {} if value is None else None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a list of floats into a BLOB using struct (little-endian f32)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    """Unpack a BLOB back into a list of floats."""
    if blob is None:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ── Base column definitions shared by all memory tables ──

_BASE_COLUMNS = [
    ("id", "TEXT PRIMARY KEY"),
    ("content", "TEXT NOT NULL"),
    ("embedding", "BLOB"),
    ("confidence", "REAL DEFAULT 0.5"),
    ("importance", "INTEGER DEFAULT 5"),
    ("recency", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("source_batch_id", "INTEGER"),
    ("metadata", "TEXT DEFAULT '{}'"),
]

# Category-specific extra columns beyond the base
# Keys match the evsmem_ prefixed table names from config
_CATEGORY_EXTRA_COLUMNS: dict[str, list[tuple[str, str]]] = {
    HOT_MEMORIES_TABLE: [
        ("expires_at", "TIMESTAMP"),
        ("context_id", "TEXT NOT NULL"),
    ],
    USER_PREFERENCES_TABLE: [
        ("preference_key", "TEXT NOT NULL UNIQUE"),
        ("preference_value", "TEXT NOT NULL"),
        ("category", "TEXT NOT NULL"),
    ],
    BEHAVIOR_PATTERNS_TABLE: [
        ("pattern_type", "TEXT NOT NULL"),
        ("frequency", "REAL DEFAULT 0.0"),
        ("evidence", "TEXT DEFAULT '[]'"),
    ],
    CONCLUSIONS_TABLE: [
        ("conclusion_type", "TEXT NOT NULL"),
        ("supporting_evidence", "TEXT DEFAULT '[]'"),
        # confidence is inherited from _BASE_COLUMNS
    ],
    LONG_TERM_MEMORIES_TABLE: [
        ("memory_type", "TEXT NOT NULL"),
        ("tags", "TEXT DEFAULT '[]'"),
    ],
    RELATIONSHIPS_TABLE: [
        ("source_entity", "TEXT NOT NULL"),
        ("target_entity", "TEXT NOT NULL"),
        ("relationship_type", "TEXT NOT NULL"),
        ("properties", "TEXT DEFAULT '{}'"),
    ],
}

# Map from DB table names (values) back to category keys (keys)
_TABLE_TO_CATEGORY = {v: k for k, v in CATEGORY_TABLE_MAP.items()}

# Fields that are stored as JSON TEXT in the DB
_JSON_FIELDS = {"metadata", "evidence", "tags", "properties", "supporting_evidence"}


class MemoryStore:
    """SQLite persistence layer for the memory agent.

    Thread-safe via ``threading.local`` connections.  All CRUD uses
    parameterised queries.  Embeddings are stored as packed ``f32`` BLOBs.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else Path(EVSMEM_DB_PATH)
        self._local = threading.local()
        self._init_schema()

    # ── Connection management ──────────────────────────────────────────────

    def _raw_conn(self) -> sqlite3.Connection:
        """Obtain the thread-local connection (not yet wrapped as context manager)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a connection with autocommit on exit.

        Rolls back on exception; commits on success.
        """
        conn = self._raw_conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the thread-local connection and release the database file.

        Call this when the store is no longer needed, especially on Windows
        where the SQLite WAL journal keeps the file locked.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ── Schema initialisation ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create all tables if they do not already exist."""
        conn = self._raw_conn()

        # ── Raw conversation queue ──
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {RAW_CONVERSATIONS_TABLE} (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                batch_id        INTEGER,
                processed       INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS {RAW_BATCHES_TABLE} (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_ids     TEXT NOT NULL,
                message_count   INTEGER NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed       INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS {CURSOR_STATE_TABLE} (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # ── Per-category memory tables ──
        for table_name, extra_cols in _CATEGORY_EXTRA_COLUMNS.items():
            all_cols = _BASE_COLUMNS + extra_cols
            col_defs = ",\n    ".join(f"{name} {dtype}" for name, dtype in all_cols)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {col_defs}
                )
            """)

        conn.commit()

    # ── Raw conversation queue ─────────────────────────────────────────────

    def append_raw_message(
        self, conversation_id: str, role: str, content: str
    ) -> int:
        """Insert a single raw message.  Returns the new row id."""
        with self.get_connection() as conn:
            cur = conn.execute(
                f"INSERT INTO {RAW_CONVERSATIONS_TABLE} (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, _utcnow()),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def create_batch(self, conversation_id: str, message_ids: list[int]) -> int:
        """Create a batch from a list of raw message ids.  Returns batch id."""
        with self.get_connection() as conn:
            cur = conn.execute(
                f"INSERT INTO {RAW_BATCHES_TABLE} (conversation_id, message_ids, message_count, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, _serialise_json(message_ids), len(message_ids), _utcnow()),
            )
            batch_id = cur.lastrowid

            # Mark messages as belonging to this batch
            for mid in message_ids:
                conn.execute(
                    f"UPDATE {RAW_CONVERSATIONS_TABLE} SET batch_id = ?, processed = 1 WHERE id = ?",
                    (batch_id, mid),
                )
            return batch_id  # type: ignore[return-value]

    def get_unprocessed_batches(self, min_size: int = 10) -> list[RawMessageBatch]:
        """Return batches whose message count >= *min_size* and not yet processed."""
        results: list[RawMessageBatch] = []
        with self.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM {RAW_BATCHES_TABLE} WHERE processed = 0 AND message_count >= ? "
                "ORDER BY created_at ASC",
                (min_size,),
            ).fetchall()
            for row in rows:
                msg_ids = _deserialise_json(row["message_ids"])
                msg_rows = conn.execute(
                    f"SELECT * FROM {RAW_CONVERSATIONS_TABLE} WHERE id IN ({','.join('?' * len(msg_ids))})",
                    msg_ids,
                ).fetchall()
                messages = [
                    {
                        "id": m["id"],
                        "conversation_id": m["conversation_id"],
                        "role": m["role"],
                        "content": m["content"],
                        "created_at": m["created_at"],
                    }
                    for m in msg_rows
                ]
                from .schemas import RawMessage
                results.append(
                    RawMessageBatch(
                        id=row["id"],
                        conversation_id=row["conversation_id"],
                        messages=[RawMessage(**m) for m in messages],
                        processed=bool(row["processed"]),
                    )
                )
        return results

    def mark_batch_processed(self, batch_id: int) -> None:
        """Mark a batch as processed (consumed)."""
        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE {RAW_BATCHES_TABLE} SET processed = 1 WHERE id = ?",
                (batch_id,),
            )

    def get_cursor(self) -> int:
        """Return the last-processed raw message id (0 if none)."""
        with self.get_connection() as conn:
            row = conn.execute(
                f"SELECT value FROM {CURSOR_STATE_TABLE} WHERE key = 'last_raw_message_id'"
            ).fetchone()
            return int(row["value"]) if row else 0

    def update_cursor(self, message_id: int) -> None:
        """Persist the last-processed raw message id."""
        with self.get_connection() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {CURSOR_STATE_TABLE} (key, value) VALUES ('last_raw_message_id', ?)",
                (str(message_id),),
            )

    # ── Generic CRUD (category-based dispatch) ────────────────────────────

    def create_memory(self, memory_category: str, **fields: Any) -> str:
        """Insert a memory into the appropriate category table.

        Returns the new memory id.
        """
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        # Separate base columns from extra columns
        extra_cols = _CATEGORY_EXTRA_COLUMNS.get(table, [])
        all_cols = _BASE_COLUMNS + extra_cols
        col_names = [c[0] for c in all_cols]

        # Build row values – apply defaults / JSON-serialise where needed
        row: dict[str, Any] = {
            "id": fields.get("id", str(uuid.uuid4())),
            "content": fields.get("content", ""),
            "embedding": fields.get("embedding"),
            "confidence": fields.get("confidence", 0.5),
            "importance": fields.get("importance", 5),
            "recency": _utcnow(),
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "source_batch_id": fields.get("source_batch_id"),
            "metadata": _serialise_json(fields.get("metadata", {})),
        }

        for col_name, _dtype in extra_cols:
            default: Any = None
            if col_name in ("evidence", "tags", "supporting_evidence", "message_ids"):
                default = "[]"
            elif col_name in ("properties",):
                default = "{}"
            elif col_name in ("frequency",):
                default = 0.0
            elif col_name in ("expires_at",):
                default = _utcnow()

            val = fields.get(col_name, default)
            # JSON-serialise list / dict fields
            if col_name in _JSON_FIELDS and not isinstance(val, str):
                val = _serialise_json(val)
            row[col_name] = val

        # Embedding BLOB encoding
        emb = row.get("embedding")
        if emb is not None and not isinstance(emb, bytes):
            row["embedding"] = _pack_embedding(emb)

        # INSERT
        placeholders = ", ".join(f":{c}" for c in col_names)
        quoted_cols = ", ".join(col_names)

        with self.get_connection() as conn:
            conn.execute(
                f"INSERT INTO {table} ({quoted_cols}) VALUES ({placeholders})",
                row,
            )
        return row["id"]

    def update_memory(self, memory_id: str, memory_category: str, **updates: Any) -> bool:
        """Update fields on an existing memory.  Returns True if a row changed."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        if not updates:
            return False

        # JSON-serialise relevant fields
        for k in list(updates):
            if k in _JSON_FIELDS and not isinstance(updates[k], str):
                updates[k] = _serialise_json(updates[k])

        # Embedding encoding
        if "embedding" in updates and updates["embedding"] is not None and not isinstance(updates["embedding"], bytes):
            updates["embedding"] = _pack_embedding(updates["embedding"])

        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = memory_id

        with self.get_connection() as conn:
            cur = conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE id = :id",
                updates,
            )
            return cur.rowcount > 0

    def delete_memory(self, memory_id: str, memory_category: str) -> bool:
        """Delete a memory by id.  Returns True if a row was removed."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        with self.get_connection() as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def get_memory(self, memory_id: str, memory_category: str) -> dict[str, Any] | None:
        """Fetch a single memory as a dict.  Returns None if not found."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        with self.get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row, table)

    # ── Query ──────────────────────────────────────────────────────────────

    def search_by_text(
        self,
        memory_category: str | None,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Basic LIKE-based text search across one (or all) category tables.

        This is a fallback text search; FTS5 will be layered in ``retrieval.py``.
        """
        like = f"%{query}%"
        results: list[dict[str, Any]] = []

        if memory_category is not None:
            tables = [CATEGORY_TABLE_MAP.get(memory_category)]
            if tables[0] is None:
                raise ValueError(f"Unknown memory category: {memory_category}")
        else:
            tables = list(CATEGORY_TABLE_MAP.values())

        with self.get_connection() as conn:
            for table in tables:
                if table is None:
                    continue
                cat = _TABLE_TO_CATEGORY.get(table, "unknown")
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (like, top_k),
                ).fetchall()
                for row in rows:
                    d = self._row_to_dict(row, table)
                    d["_category"] = cat
                    results.append(d)
                if len(results) >= top_k:
                    break

        return results[:top_k]

    def get_all_memories_by_category(self, memory_category: str) -> list[dict[str, Any]]:
        """Return every row from a single category table."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        with self.get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY updated_at DESC").fetchall()
            return [self._row_to_dict(r, table) for r in rows]

    # ── Embedding storage & similarity search ──────────────────────────────

    def store_embedding(
        self, memory_id: str, memory_category: str, embedding: list[float]
    ) -> None:
        """Update the embedding BLOB for an existing memory."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        blob = _pack_embedding(embedding)
        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE {table} SET embedding = ?, updated_at = ? WHERE id = ?",
                (blob, _utcnow(), memory_id),
            )

    def get_embedding(
        self, memory_id: str, memory_category: str
    ) -> list[float] | None:
        """Retrieve the embedding BLOB as a list of floats."""
        table = CATEGORY_TABLE_MAP.get(memory_category)
        if table is None:
            raise ValueError(f"Unknown memory category: {memory_category}")

        with self.get_connection() as conn:
            row = conn.execute(
                f"SELECT embedding FROM {table} WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            return _unpack_embedding(row["embedding"])

    def similarity_search(
        self,
        query_embedding: list[float],
        memory_category: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Exhaustive cosine-similarity search across one or all tables.

        .. note::
            This performs a full table scan and computes cosine similarity in
            Python.  For large datasets, consider an ANN index (e.g. usearch).
        """
        import numpy as np

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm

        tables: list[str]
        if memory_category is not None:
            tbl = CATEGORY_TABLE_MAP.get(memory_category)
            if tbl is None:
                raise ValueError(f"Unknown memory category: {memory_category}")
            tables = [tbl]
        else:
            tables = list(CATEGORY_TABLE_MAP.values())

        scored: list[tuple[float, dict[str, Any]]] = []

        with self.get_connection() as conn:
            for table in tables:
                if table is None:
                    continue
                cat = _TABLE_TO_CATEGORY.get(table, "unknown")
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE embedding IS NOT NULL"
                ).fetchall()
                for row in rows:
                    blob = row["embedding"]
                    if blob is None:
                        continue
                    emb = _unpack_embedding(blob)
                    if emb is None or len(emb) != len(q):
                        continue
                    v = np.array(emb, dtype=np.float32)
                    v_norm = np.linalg.norm(v)
                    if v_norm < 1e-10:
                        continue
                    sim = float(np.dot(q, v / v_norm))
                    d = self._row_to_dict(row, table)
                    d["_category"] = cat
                    d["_similarity"] = sim
                    scored.append((sim, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row, table: str) -> dict[str, Any]:
        """Convert a ``sqlite3.Row`` to a plain dict with JSON fields expanded
        and embeddings decoded."""
        d = dict(row)

        # Expand JSON TEXT fields
        for field in _JSON_FIELDS:
            if field in d and isinstance(d[field], str):
                d[field] = _deserialise_json(d[field])

        # Decode embedding BLOB
        if "embedding" in d:
            d["embedding"] = _unpack_embedding(d["embedding"])

        # Convert boolean-like ints
        for field in ("processed",):
            if field in d and isinstance(d[field], int):
                d[field] = bool(d[field])

        return d
