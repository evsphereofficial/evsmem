"""Verify deriver_state reset mechanics (non-destructive).

Connects to the evsmem SQLite database and verifies that the
'last_processed_rowid' key can be updated (the same mechanics used by
cleanup_memories.py), prints all rows BEFORE and AFTER a temporary
update, then RESTORES the original value so this script is safe to run
standalone.  'last_synced_rowid' is only read, never modified.
The agents/skills tables are never touched.

Run with: D:\\Programming\\AiProjects\\EvAgent\\evsmem\\.venv\\Scripts\\python.exe
"""

import sqlite3
import sys

DB_PATH = r"C:\Users\Rehan\.evsmem\evsmem.db"
TARGET_KEY = "last_processed_rowid"
READONLY_KEY = "last_synced_rowid"
TABLE = "deriver_state"


def connect() -> sqlite3.Connection:
    """Open the evsmem database with a 10-second busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if the given table exists in the database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def dump_state(conn: sqlite3.Connection, label: str) -> None:
    """Print every row in deriver_state (key, value, updated_at)."""
    rows = conn.execute(
        "SELECT key, value, updated_at FROM deriver_state ORDER BY key"
    ).fetchall()
    print(f"--- deriver_state {label} ---")
    if not rows:
        print("  (no rows)")
    for row in rows:
        print(f"  key={row['key']!r}  value={row['value']!r}  updated_at={row['updated_at']!r}")


def get_key(conn: sqlite3.Connection, key: str):
    """Fetch a single deriver_state row by key, or None if missing."""
    return conn.execute(
        "SELECT key, value, updated_at FROM deriver_state WHERE key = ?",
        (key,),
    ).fetchone()


def run_update(conn: sqlite3.Connection, sql: str, params: tuple) -> None:
    """Execute a single UPDATE inside its own transaction (atomic)."""
    try:
        conn.execute("BEGIN")
        conn.execute(sql, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main() -> int:
    conn = connect()
    try:
        if not table_exists(conn, TABLE):
            print(f"ERROR: table {TABLE!r} does not exist in {DB_PATH}.", file=sys.stderr)
            return 1

        # 1. All rows BEFORE the temporary update.
        dump_state(conn, "BEFORE")

        # 2. Confirm the target key exists.
        row = get_key(conn, TARGET_KEY)
        if row is None:
            print(
                f"ERROR: key {TARGET_KEY!r} not found in {TABLE}.", file=sys.stderr
            )
            return 1
        original = (row["value"], row["updated_at"])
        print(
            f"OK: {TARGET_KEY!r} exists -> "
            f"value={original[0]!r}, updated_at={original[1]!r}"
        )

        # 3. Temporary update (same mechanics as cleanup_memories.py).
        run_update(
            conn,
            "UPDATE deriver_state SET value = '0', updated_at = datetime('now') "
            "WHERE key = ?",
            (TARGET_KEY,),
        )

        # 4. All rows AFTER the temporary update, plus the updated row itself.
        dump_state(conn, "AFTER temporary update")
        updated = get_key(conn, TARGET_KEY)
        print(
            f"OK: {TARGET_KEY!r} updated -> "
            f"value={updated['value']!r}, updated_at={updated['updated_at']!r}"
        )

        # 5. RESTORE the original value so the script is non-destructive.
        run_update(
            conn,
            "UPDATE deriver_state SET value = ?, updated_at = ? WHERE key = ?",
            (original[0], original[1], TARGET_KEY),
        )
        print(
            f"OK: {TARGET_KEY!r} restored -> "
            f"value={original[0]!r}, updated_at={original[1]!r}"
        )

        # 6. Read-only check: 'last_synced_rowid' must be untouched.
        synced = get_key(conn, READONLY_KEY)
        if synced is None:
            print(f"OK: {READONLY_KEY!r} not present in deriver_state (untouched).")
        else:
            print(
                f"OK: {READONLY_KEY!r} untouched -> "
                f"value={synced['value']!r}, updated_at={synced['updated_at']!r}"
            )

        # 7. Final state confirms the restore.
        dump_state(conn, "AFTER restore (final)")
        print("Verification complete (non-destructive).")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
