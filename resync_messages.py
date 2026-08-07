"""One-time repair: rebuild the evsmem `messages` mirror from the ev-agent DB.

The old sync had a streaming race (assistant replies captured mid-stream or
skipped entirely). This script deletes the auto-save message mirror, resets
the sync cursor, and re-syncs EVERY message using the fixed deriver logic
(settle delay + stable source-id upsert). Messages younger than 60s will be
picked up by the running deriver shortly after.

Usage (STOP the evsmem server first):
  .venv\\Scripts\\python.exe resync_messages.py
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Skip per-message embeddings during the rebuild (the raw mirror is rebuilt
# fast; memory/conclusion embeddings come from the LLM pass anyway).
os.environ["DERIVER_SYNC_NO_EMBED"] = "1"

from deriver import Deriver

DB_PATH = Path.home() / ".evsmem" / "evsmem.db"


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("BEGIN")
    cur = conn.execute(
        """DELETE FROM messages WHERE session_id IN (
               SELECT s.id FROM sessions s JOIN workspaces w ON s.workspace_id = w.id
               WHERE s.name = 'auto-save' AND w.name = 'ev-agent')"""
    )
    deleted = cur.rowcount
    conn.execute(
        "INSERT OR REPLACE INTO deriver_state (key, value, updated_at) "
        "VALUES ('last_synced_rowid', '0', datetime('now'))"
    )
    conn.commit()
    conn.close()
    print(f"Deleted {deleted} auto-save messages; sync cursor reset to 0.")

    d = Deriver()
    total = 0
    while True:
        n = d._sync_ev_sessions()
        if n == 0:
            break
        total += n
        print(f"  synced {total} so far...")
    print(f"Resync complete: {total} messages rebuilt.")


if __name__ == "__main__":
    main()
