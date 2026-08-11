"""Tests for the curation Deriver: hour-batch window, rolling analysis cursor,
and idempotent success/failure marking (messages stay unprocessed after an LLM
failure so the batch is retried on the next run).

Run from the evsmem directory:
    .venv\\Scripts\\python.exe -m unittest test_deriver -v
"""

import gc
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import deriver


class FakeLLM:
    """Plain generate-only LLM (local GGUF fallback contract)."""

    def __init__(self, payload):
        self._payload = payload

    def is_available(self):
        return True

    def generate(self, messages, max_tokens=512, temperature=0.1):
        return self._payload


class DeriverBatchWindowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "evsmem.db"
        self._patchers = [
            mock.patch.object(deriver, "DB_PATH", self.db_path),
            mock.patch.object(deriver, "EV_SESSION_DB", Path(self._tmp.name) / "no-session.db"),
        ]
        for p in self._patchers:
            p.start()

        import crud
        self._crud = crud
        crud.DB_PATH = self.db_path
        crud._SCHEMA_READY = False
        crud._local = threading.local()
        crud.get_db()  # init full schema on the temp DB

        self.now = datetime.now(timezone.utc)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO workspaces (id, name, metadata, created_at) VALUES ('ws1','ev-agent','{}', ?)",
            (self.now.isoformat(),),
        )
        conn.execute(
            "INSERT INTO sessions (id, workspace_id, name, metadata, created_at) VALUES ('s1','ws1','auto-save','{}', ?)",
            (self.now.isoformat(),),
        )
        conn.execute(
            "INSERT INTO peers (id, workspace_id, name, metadata, created_at) VALUES ('p1','ws1','user','{}', ?)",
            (self.now.isoformat(),),
        )
        conn.commit()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        try:
            self._crud.close_db()
        finally:
            self._crud.DB_PATH = Path.home() / ".evsmem" / "evsmem.db"
            gc.collect()
            try:
                self._tmp.cleanup()
            except Exception:
                pass

    def _add_message(self, mid, minutes_ago, content, processed=0, peer="p1"):
        ts = (self.now - timedelta(minutes=minutes_ago)).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO messages
               (id, session_id, peer_id, content, role, message_type, metadata,
                embedding, created_at, is_processed)
               VALUES (?, ?, ?, ?, 'user', 'message', '{}', NULL, ?, ?)""",
            (mid, "s1", peer, content, ts, processed),
        )
        conn.commit()
        conn.close()

    def test_hour_batch_collects_all_unprocessed_in_window_ascending(self):
        # 25 messages all inside the hour window (no LIMIT hard-coded anywhere).
        for i in range(25):
            self._add_message(f"m{i}", minutes_ago=5 + i, content=f"message {i}")
        # Outside the window / already processed / empty content.
        self._add_message("old", minutes_ago=90, content="too old")
        self._add_message("done", minutes_ago=10, content="already done", processed=1)
        self._add_message("empty", minutes_ago=10, content="")

        d = deriver.Deriver()
        batch = d._get_hour_batch()
        self.assertEqual(len(batch), 25)
        ids = [m["id"] for m in batch]
        self.assertEqual(ids, sorted(ids, key=lambda s: int(s[1:])))  # ordered by rowid ASC
        self.assertNotIn("old", ids)
        self.assertNotIn("done", ids)
        self.assertNotIn("empty", ids)

    def test_hour_batch_rolling_cursor_excludes_messages_before_last_pass(self):
        self._add_message("before_cursor", minutes_ago=40, content="covered by last pass")
        self._add_message("after_cursor", minutes_ago=20, content="new since last pass")
        conn = deriver.get_db()
        conn.execute(
            "INSERT OR REPLACE INTO deriver_state (key, value, updated_at) VALUES ('last_analysis_at', ?, ?)",
            ((self.now - timedelta(minutes=30)).isoformat(), self.now.isoformat()),
        )
        conn.commit()
        conn.close()

        d = deriver.Deriver()
        ids = [m["id"] for m in d._get_hour_batch()]
        self.assertNotIn("before_cursor", ids)
        self.assertIn("after_cursor", ids)

    def test_successful_pass_marks_processed_and_advances_cursor(self):
        self._add_message("m1", minutes_ago=10, content="I prefer dark mode and short answers.")
        payload = json.dumps({
            "user": {"name": "Rehan"}, "hot_memories": [], "cold_memories": [],
            "behaviours": [], "preferences": [], "rules": [], "conclusions": [],
            "agent_assessment": {},
        })
        d = deriver.Deriver()
        d._get_llm = lambda: ("local", FakeLLM(payload))
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        ok = d._process_new_messages_with_llm(stats)
        self.assertTrue(ok)
        self.assertEqual(stats["messages_processed"], 1)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT is_processed FROM messages WHERE id='m1'").fetchone()
        self.assertEqual(row["is_processed"], 1)
        cursor = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_analysis_at'"
        ).fetchone()
        self.assertIsNotNone(cursor)
        conn.close()

    def test_failed_pass_leaves_messages_unprocessed_and_cursor_untouched(self):
        self._add_message("m1", minutes_ago=10, content="message that will fail analysis")
        d = deriver.Deriver()
        d._get_llm = lambda: ("local", FakeLLM("```json\n{this is not json\n```"))
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        with self.assertRaises(RuntimeError):
            d._process_new_messages_with_llm(stats)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT is_processed FROM messages WHERE id='m1'").fetchone()
        self.assertEqual(row["is_processed"], 0)
        cursor = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_analysis_at'"
        ).fetchone()
        self.assertIsNone(cursor)
        conn.close()


if __name__ == "__main__":
    unittest.main()