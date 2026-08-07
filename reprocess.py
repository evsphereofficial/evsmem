"""One-time migration: reprocess ALL raw user messages through gemma4 to
populate hot_memory / cold_memory / conclusions / peer metadata from the
full conversation history.

Resumable: each message is marked is_processed = 1 after analysis, so an
interrupted run resumes where it left off (run the same command again).

Usage (from the evsmem dir):
  .venv\\Scripts\\python.exe reprocess.py                # process everything
  .venv\\Scripts\\python.exe reprocess.py --limit 500    # process only 500 messages
  .venv\\Scripts\\python.exe reprocess.py --sleep 1.0    # pace yourself (GPU/heat)

IMPORTANT: stop the evsmem server while running this so the two processes
don't both load the 12B model into GPU memory at once.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient
from deriver import Deriver

DB_PATH = Path.home() / ".evsmem" / "evsmem.db"


def ensure_is_processed_column():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN is_processed INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()


def load_unprocessed(limit=0):
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    sql = """SELECT m.rowid, m.id, m.content, m.role, m.session_id, s.workspace_id
             FROM messages m JOIN sessions s ON m.session_id = s.id
             WHERE m.is_processed = 0 AND m.role = 'user' AND m.content != ''
             ORDER BY m.rowid ASC"""
    if limit > 0:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max messages to process (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between messages")
    args = ap.parse_args()

    ensure_is_processed_column()

    d = Deriver()
    llm = LLMClient()
    if not llm.is_available():
        print("gemma4 model not available; exiting")
        return

    msgs = load_unprocessed(args.limit)
    total = len(msgs)
    print(f"Reprocessing {total} unprocessed user messages...")
    if total == 0:
        print("Nothing to do (all messages already processed).")
        return

    t0 = time.time()
    done = 0
    for i, msg in enumerate(msgs, 1):
        try:
            parsed = d._analyze_message_with_llm(llm, msg)
            if parsed:
                d._store_llm_results(parsed, msg)
            done += 1
        except Exception as e:
            print(f"[err] rowid={msg['rowid']}: {e}")
        finally:
            d._mark_message_processed(msg["rowid"])

        if i % 10 == 0 or i == total:
            elapsed = max(time.time() - t0, 1e-9)
            rate = i / elapsed
            eta_min = (total - i) / max(rate, 1e-9) / 60
            print(f"  {i}/{total} done ({rate:.2f} msg/s, ETA {eta_min:.1f} min)")

        if args.sleep:
            time.sleep(args.sleep)

    print(f"Done. Processed {done} messages in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
