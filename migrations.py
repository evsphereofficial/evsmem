"""
evsmem/migrations.py — Idempotent SQLite schema migration runner.

Uses the ``evsmem_schema_version`` table to track which version of the
schema has been applied.  All DDL statements come from ``evsmem.ddl``.

Usage
-----
.. code-block:: python

    from evsmem.migrations import run_migration

    applied = run_migration("path/to/evsmem.db")
    if applied:
        print("Schema migrated successfully")
    else:
        print("Schema already up to date")
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from evsmem.ddl import get_ddl_statements, get_table_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

#: Current schema version. Increment this when the DDL changes.
SCHEMA_VERSION: int = 1

#: Human-readable description of each version.
_VERSION_DESCRIPTIONS: dict[int, str] = {
    1: (
        "Initial evsmem schema: 10 evsmem-prefixed tables (raw queue, cursor, "
        "6 memory categories, schema version) + unified FTS5 virtual table"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the highest applied schema version from the database.

    Returns ``0`` if the ``evsmem_schema_version`` table does not exist yet
    (fresh database) or is empty.

    Args:
        conn: An open SQLite connection.

    Returns:
        The highest ``version`` integer in ``evsmem_schema_version``, or ``0``.
    """
    try:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM evsmem_schema_version"
        ).fetchone()
        return row["v"] if row and row["v"] is not None else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet — not an error
        return 0


def _schema_version_has_table(conn: sqlite3.Connection) -> bool:
    """Return ``True`` if the ``evsmem_schema_version`` table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='evsmem_schema_version'"
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_migration(db_path: str) -> bool:
    """Run pending schema migrations on the given database file.

    The migration is **idempotent** — calling it multiple times is safe.
    If the schema is already at the latest version, this function is a no-op
    and returns ``False``.

    Args:
        db_path: Absolute or relative path to the SQLite database file.

    Returns:
        ``True`` if the schema was migrated (new tables / columns created).
        ``False`` if it was already up to date.

    Raises:
        sqlite3.Error: If any SQL statement fails (transaction is rolled back).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    try:
        current_version = get_schema_version(conn)

        if current_version >= SCHEMA_VERSION:
            logger.info(
                "Schema at v%s (current=%s) — nothing to do",
                current_version,
                SCHEMA_VERSION,
            )
            return False

        logger.info(
            "Migrating schema from v%s to v%s",
            current_version,
            SCHEMA_VERSION,
        )

        # ── Execute all DDL statements ─────────────────────────────────
        for ddl in get_ddl_statements():
            logger.debug("Executing DDL (first 80 chars): %s ...", ddl.strip()[:80])
            conn.execute(ddl)

        # ── Record the migration ───────────────────────────────────────
        description = _VERSION_DESCRIPTIONS.get(
            SCHEMA_VERSION,
            f"Schema version {SCHEMA_VERSION}",
        )
        conn.execute(
            "INSERT INTO evsmem_schema_version (version, description) VALUES (?, ?)",
            (SCHEMA_VERSION, description),
        )

        conn.commit()
        logger.info("Migration to v%s complete — %s", SCHEMA_VERSION, description)

        # ── Rebuild FTS index if there's existing data ─────────────────
        try:
            conn.execute("INSERT INTO evsmem_fts(evsmem_fts) VALUES('rebuild')")
            logger.debug("FTS5 index rebuilt")
        except sqlite3.OperationalError:
            # FTS table may not exist or rebuild may not be needed
            pass

        return True

    except Exception:
        conn.rollback()
        logger.exception("Migration to v%s failed — rolled back", SCHEMA_VERSION)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def needs_migration(db_path: str) -> bool:
    """Quick check — does the database need a schema migration?

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ``True`` if the database is not yet at the latest schema version.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            v = get_schema_version(conn)
            return v < SCHEMA_VERSION
        finally:
            conn.close()
    except sqlite3.Error:
        return True


def get_migration_status(db_path: str) -> dict:
    """Return a detailed migration status dict for diagnostics.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dict with keys ``db_path``, ``current_version``, ``latest_version``,
        ``needs_migration``, ``tables_present``.
    """
    status: dict = {
        "db_path": db_path,
        "current_version": 0,
        "latest_version": SCHEMA_VERSION,
        "needs_migration": True,
        "tables_present": [],
    }
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            status["current_version"] = get_schema_version(conn)
            status["needs_migration"] = status["current_version"] < SCHEMA_VERSION

            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'evsmem_%' ORDER BY name"
            ).fetchall()
            status["tables_present"] = [r["name"] for r in rows]

            # Check FTS virtual table
            fts_row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='evsmem_fts'"
            ).fetchone()
            status["fts_present"] = fts_row is not None
        finally:
            conn.close()
    except sqlite3.Error as exc:
        status["error"] = str(exc)
    return status


__all__ = [
    "SCHEMA_VERSION",
    "run_migration",
    "needs_migration",
    "get_migration_status",
    "get_schema_version",
]
