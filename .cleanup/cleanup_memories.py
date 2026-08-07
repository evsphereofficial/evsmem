#!/usr/bin/env python3
"""
evsmem cleanup: delete memories whose content matches '%something to remember%',
delete their FTS5 index rows, and reset deriver_state.last_processed_rowid to '0'.

WRITE CONSTRAINT: the only tables this script may write to are:
    - memories
    - memories_fts
    - deriver_state
The `agents` and `skills` tables are touched READ-ONLY (counted before/after
and asserted unchanged). No DDL is executed on any table.

Run with:  D:\\Programming\\AiProjects\\EvAgent\\evsmem\\.venv\\Scripts\\python.exe  cleanup_memories.py
Uses only the stdlib `sqlite3` module.
"""

import sqlite3

DB_PATH = r"C:\Users\Rehan\.evsmem\evsmem.db"

# Verbatim match predicate from the task spec.
MATCH_SQL = (
    "SELECT rowid, id, content FROM memories "
    "WHERE lower(trim(content)) LIKE '%something to remember%'"
)
MATCH_DELETE_SQL = (
    "DELETE FROM memories WHERE lower(trim(content)) LIKE ?"
)
MATCH_PATTERN = "%something to remember%"

# The FTS table must be the expected external-content FTS5 table keyed by the
# memories rowid; refuse to run against anything else.
FTS_MARKER_CONTENT = "content='memories'"
FTS_MARKER_ROWID = "content_rowid='rowid'"


def _truncate(content, width=80):
    """Flatten to one line and truncate to `width` chars."""
    flat = (content or "").replace("\r", " ").replace("\n", " ")
    if len(flat) <= width:
        return flat
    return flat[: width - 3] + "..."


def main():
    print("=" * 78)
    print("evsmem CLEANUP - delete memories matching '%something to remember%'")
    print("DB: %s" % DB_PATH)
    print("=" * 78)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        # Autocommit mode; the transaction is managed explicitly with
        # BEGIN IMMEDIATE ... COMMIT / ROLLBACK below.
        conn.isolation_level = None

        conn.execute("PRAGMA busy_timeout = 10000")
        # Enable WAL mode (must run outside a transaction; not a table write).
        conn.execute("PRAGMA journal_mode = WAL")
        # NOTE: PRAGMA foreign_keys is intentionally left at its default;
        # it is not needed for this cleanup.

        # --- Defensive schema sanity checks (read-only) ---------------------
        fts_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if fts_row is None:
            raise RuntimeError("memories_fts table not found; refusing to run")
        fts_def = fts_row[0]
        if FTS_MARKER_CONTENT not in fts_def or FTS_MARKER_ROWID not in fts_def:
            raise RuntimeError(
                "memories_fts is not the expected external-content FTS5 table "
                "(missing %r / %r); refusing to run"
                % (FTS_MARKER_CONTENT, FTS_MARKER_ROWID)
            )

        # --- Step 0: pre-transaction counts (read-only) ---------------------
        agents_before = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        skills_before = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        print("Pre-flight counts: agents=%d skills=%d" % (agents_before, skills_before))

        # --- Step 1-5: one transaction --------------------------------------
        try:
            conn.execute("BEGIN IMMEDIATE")

            # (a) Collect rowids of matching memories.
            rows = conn.execute(MATCH_SQL).fetchall()
            rowids = [r["rowid"] for r in rows]

            # (b) Print each match (id + content truncated to 80 chars).
            if rows:
                print("\nDeleting %d matching memory row(s):" % len(rows))
                for r in rows:
                    print("  id=%s | %s" % (r["id"], _truncate(r["content"])))
            else:
                print("\nNo memories match the pattern; nothing to delete.")

            # (c) Delete the corresponding FTS index rows (external content,
            #     keyed by the memories rowid).
            fts_cleaned = 0
            for rowid in rowids:
                cur = conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rowid,))
                fts_cleaned += cur.rowcount

            # (d) Delete the memories themselves.
            cur = conn.execute(MATCH_DELETE_SQL, (MATCH_PATTERN,))
            deleted_count = cur.rowcount

            # (e) Reset the deriver cursor.
            cur = conn.execute(
                "UPDATE deriver_state SET value='0', updated_at=datetime('now') "
                "WHERE key='last_processed_rowid'"
            )
            deriver_updated = cur.rowcount

            # (f) Post-transaction counts + assertion: agents/skills must be
            #     identical to the pre-flight values (this script never writes
            #     to them). Assert BEFORE committing so a violation rolls back.
            agents_after = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            skills_after = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            if agents_after != agents_before or skills_after != skills_before:
                raise RuntimeError(
                    "agents/skills counts changed during cleanup "
                    "(agents %d->%d, skills %d->%d); rolling back"
                    % (agents_before, agents_after, skills_before, skills_after)
                )

            conn.execute("COMMIT")
        except Exception:
            conn.rollback()  # nothing partial is committed
            raise

        # --- Final summary (post-commit reads) ------------------------------
        last_rowid_row = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_processed_rowid'"
        ).fetchone()
        agents_after = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        skills_after = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

        print("\n" + "=" * 78)
        print("CLEANUP SUMMARY")
        print("=" * 78)
        print("deleted_count:              %d" % deleted_count)
        print("fts_cleaned_count:          %d" % fts_cleaned)
        print("deriver_state rows updated: %d" % deriver_updated)
        print(
            "last_processed_rowid after: %s"
            % (last_rowid_row["value"] if last_rowid_row else "<key missing>")
        )
        print("agents_count before/after:  %d / %d" % (agents_before, agents_after))
        print("skills_count before/after:  %d / %d" % (skills_before, skills_after))

        if deriver_updated == 0:
            print(
                "WARNING: deriver_state has no 'last_processed_rowid' key; "
                "no row was updated."
            )
        if fts_cleaned != deleted_count:
            print(
                "WARNING: FTS rows cleaned (%d) differ from memories deleted (%d); "
                "manual verification recommended."
                % (fts_cleaned, deleted_count)
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
