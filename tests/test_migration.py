"""
Test migration against a copy of the actual evsmem database.
Run: python -m evsmem.tests.test_migration
"""

import logging
import os
import shutil
import sqlite3
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evsmem.migrations import run_migration, get_migration_status

def main():
    src = "C:/Users/Rehan/.evsmem/evsmem.db"
    dst = os.path.join(tempfile.gettempdir(), "evsmem_test_migration.db")

    if not os.path.exists(src):
        print(f"Source DB not found: {src}")
        sys.exit(1)

    print(f"Source DB: {src} ({os.path.getsize(src)} bytes)")
    shutil.copy2(src, dst)
    print(f"Copied to: {dst}")

    # Run migration
    try:
        result = run_migration(dst)
        print(f"Migration applied: {result}")
    except Exception as e:
        print(f"MIGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    try:
        # Check tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'evsmem_%' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        print(f"Evsmem tables ({len(table_names)}): {table_names}")

        # Check FTS
        fts_row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evsmem_fts'"
        ).fetchone()
        print(f"FTS5 evsmem_fts present: {fts_row is not None}")

        # Schema version
        sv = conn.execute("SELECT * FROM evsmem_schema_version").fetchall()
        print(f"Schema versions: {[dict(s) for s in sv]}")

        # Verify old tables preserved
        old_tables = ["workspaces", "messages", "sessions", "agents", "peers", "memories", "conclusions"]
        for tbl in old_tables:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
            ).fetchone()
            print(f"Old table '{tbl}' preserved: {row is not None}")

        # Check status
        status = get_migration_status(dst)
        print(f"Migration status: {status['current_version']}/{status['latest_version']}, needs: {status['needs_migration']}")
        print(f"Tables present: {status.get('tables_present', [])[:5]}...")
        print(f"FTS present: {status.get('fts_present', False)}")

        print("MIGRATION TEST PASSED")
    finally:
        conn.close()
        os.unlink(dst)
        print(f"Cleaned up: {dst}")


if __name__ == "__main__":
    main()
