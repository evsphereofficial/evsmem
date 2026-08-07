"""
evsmem/ddl.py — Canonical DDL statements for all evsmem-prefixed tables.

This module is the single source of truth for the evsmem database schema.
All 10 new tables use the ``evsmem_`` prefix to avoid collisions with the
existing 17 Honcho tables (workspaces, agents, sessions, peers, messages,
memories, conclusions, skills, reputation, recommendations, notifications,
deriver_state).

Helping functions
-----------------
* ``get_ddl_statements()`` — returns a list of all CREATE TABLE / DDL strings.
* ``get_table_names()`` — returns the list of table names (without the FTS5 virtual).
* ``get_fts_table_name()`` — returns the unified FTS5 virtual table name.

Usage
-----
.. code-block:: python

    from evsmem.ddl import get_ddl_statements, get_table_names

    for stmt in get_ddl_statements():
        conn.execute(stmt)
"""

from __future__ import annotations

from typing import List

from evsmem.config import (
    RAW_CONVERSATIONS_TABLE,
    RAW_BATCHES_TABLE,
    CURSOR_STATE_TABLE,
    HOT_MEMORIES_TABLE,
    USER_PREFERENCES_TABLE,
    BEHAVIOR_PATTERNS_TABLE,
    CONCLUSIONS_TABLE,
    LONG_TERM_MEMORIES_TABLE,
    RELATIONSHIPS_TABLE,
    SCHEMA_VERSION_TABLE,
)

# ---------------------------------------------------------------------------
# FTS5 virtual table name (unified cross-category full-text search)
# ---------------------------------------------------------------------------
FTS_TABLE: str = "evsmem_fts"

# ---------------------------------------------------------------------------
# DDL collection — ordered so that referenced tables exist first
# ---------------------------------------------------------------------------

_ALL_DDL_STATEMENTS: List[str] = [
    # ── 1. Raw conversation queue ──────────────────────────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {RAW_CONVERSATIONS_TABLE} (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        batch_id        INTEGER,
        processed       INTEGER DEFAULT 0
    )
    """,
    # ── 2. Raw message batches ────────────────────────────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {RAW_BATCHES_TABLE} (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        message_ids     TEXT NOT NULL,
        message_count   INTEGER NOT NULL,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed       INTEGER DEFAULT 0
    )
    """,
    # ── 3. Cursor state (high-water mark for ingestion) ────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {CURSOR_STATE_TABLE} (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # ── 4. Hot memories (transient, expiring) ──────────────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {HOT_MEMORIES_TABLE} (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        embedding       BLOB,
        confidence      REAL DEFAULT 0.5,
        importance      INTEGER DEFAULT 5,
        recency         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id INTEGER,
        metadata        TEXT DEFAULT '{{}}',
        expires_at      TIMESTAMP,
        context_id      TEXT NOT NULL
    )
    """,
    # ── 5. User preferences (learnt facts about the user) ──────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {USER_PREFERENCES_TABLE} (
        id               TEXT PRIMARY KEY,
        content          TEXT NOT NULL,
        embedding        BLOB,
        confidence       REAL DEFAULT 0.5,
        importance       INTEGER DEFAULT 5,
        recency          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id  INTEGER,
        metadata         TEXT DEFAULT '{{}}',
        preference_key   TEXT NOT NULL UNIQUE,
        preference_value TEXT NOT NULL,
        category         TEXT NOT NULL
    )
    """,
    # ── 6. Behaviour patterns (frequency-tracked habits) ───────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {BEHAVIOR_PATTERNS_TABLE} (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        embedding       BLOB,
        confidence      REAL DEFAULT 0.5,
        importance      INTEGER DEFAULT 5,
        recency         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id INTEGER,
        metadata        TEXT DEFAULT '{{}}',
        pattern_type    TEXT NOT NULL,
        frequency       REAL DEFAULT 0.0,
        evidence        TEXT DEFAULT '[]'
    )
    """,
    # ── 7. Conclusions (derived insights with supporting evidence) ─────────
    f"""
    CREATE TABLE IF NOT EXISTS {CONCLUSIONS_TABLE} (
        id                 TEXT PRIMARY KEY,
        content            TEXT NOT NULL,
        embedding          BLOB,
        confidence         REAL DEFAULT 0.5,
        importance         INTEGER DEFAULT 5,
        recency            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id    INTEGER,
        metadata           TEXT DEFAULT '{{}}',
        conclusion_type    TEXT NOT NULL,
        supporting_evidence TEXT DEFAULT '[]'
    )
    """,
    # ── 8. Long-term memories (persistent knowledge with tags) ─────────────
    f"""
    CREATE TABLE IF NOT EXISTS {LONG_TERM_MEMORIES_TABLE} (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        embedding       BLOB,
        confidence      REAL DEFAULT 0.5,
        importance      INTEGER DEFAULT 5,
        recency         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id INTEGER,
        metadata        TEXT DEFAULT '{{}}',
        memory_type     TEXT NOT NULL,
        tags            TEXT DEFAULT '[]'
    )
    """,
    # ── 9. Relationships (entity-entity connections) ───────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {RELATIONSHIPS_TABLE} (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        embedding         BLOB,
        confidence        REAL DEFAULT 0.5,
        importance        INTEGER DEFAULT 5,
        recency           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_batch_id   INTEGER,
        metadata          TEXT DEFAULT '{{}}',
        source_entity     TEXT NOT NULL,
        target_entity     TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        properties        TEXT DEFAULT '{{}}'
    )
    """,
    # ── 10. Schema version tracking ────────────────────────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA_VERSION_TABLE} (
        version     INTEGER PRIMARY KEY,
        applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT
    )
    """,
    # ── 11. Unified FTS5 virtual table ─────────────────────────────────────
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE}
    USING fts5(
        content,
        memory_type,
        tags,
        preference_key,
        pattern_type,
        tokenize='porter unicode61'
    )
    """,
]


def get_ddl_statements() -> List[str]:
    """Return all CREATE TABLE / DDL statements for schema creation.

    Returns:
        A list of SQL strings, each a single ``CREATE TABLE IF NOT EXISTS``
        or ``CREATE VIRTUAL TABLE IF NOT EXISTS`` statement.
    """
    return list(_ALL_DDL_STATEMENTS)


def get_table_names() -> List[str]:
    """Return all evsmem-prefixed table names (excluding the FTS5 virtual table).

    Returns:
        List of 10 table name strings.
    """
    return [
        RAW_CONVERSATIONS_TABLE,
        RAW_BATCHES_TABLE,
        CURSOR_STATE_TABLE,
        HOT_MEMORIES_TABLE,
        USER_PREFERENCES_TABLE,
        BEHAVIOR_PATTERNS_TABLE,
        CONCLUSIONS_TABLE,
        LONG_TERM_MEMORIES_TABLE,
        RELATIONSHIPS_TABLE,
        SCHEMA_VERSION_TABLE,
    ]


def get_fts_table_name() -> str:
    """Return the unified FTS5 virtual table name.

    Returns:
        ``"evsmem_fts"``
    """
    return FTS_TABLE


# Re-export for convenience
ALL_DDL_STATEMENTS = _ALL_DDL_STATEMENTS

__all__ = [
    "ALL_DDL_STATEMENTS",
    "FTS_TABLE",
    "get_ddl_statements",
    "get_table_names",
    "get_fts_table_name",
]
