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
import time
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

    def _add_message_ts(self, mid, ts, content, peer="p1"):
        """Insert a message with an explicit created_at string (any format)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO messages
               (id, session_id, peer_id, content, role, message_type, metadata,
                embedding, created_at, is_processed)
               VALUES (?, ?, ?, ?, 'user', 'message', '{}', NULL, ?, 0)""",
            (mid, "s1", peer, content, ts),
        )
        conn.commit()
        conn.close()

    def _run_pass(self, d, payload):
        d._get_llm = lambda: ("local", FakeLLM(payload))
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        ok = d._process_new_messages_with_llm(stats)
        return ok, stats

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

    def test_hour_batch_handles_sqlite_space_format_timestamps(self):
        """Legacy rows with SQLite 'YYYY-MM-DD HH:MM:SS' timestamps must be
        placed in the correct hour window. The old string-comparison SQL
        silently excluded them (' ' < 'T' in ASCII); parsing created_at with
        datetime.fromisoformat fixes that."""
        inside = (self.now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        outside = (self.now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        self._add_message_ts("space_new", inside, "space-format inside window")
        self._add_message_ts("space_old", outside, "space-format outside window")
        self._add_message("iso_new", minutes_ago=10, content="iso-format inside window")

        d = deriver.Deriver()
        ids = [m["id"] for m in d._get_hour_batch()]
        self.assertIn("space_new", ids)
        self.assertIn("iso_new", ids)
        self.assertNotIn("space_old", ids)

    def test_cursor_advances_to_newest_processed_message_not_now(self):
        """The analysis cursor lands on the newest message's created_at, not
        wall-clock now, so a message that arrives WHILE the LLM call is running
        stays ahead of the cursor and is curated on the next pass (previously
        it was skipped forever)."""
        self._add_message("m1", minutes_ago=10, content="I prefer dark mode and short answers.")
        payload = json.dumps({
            "user": {"name": "Rehan"}, "hot_memories": [], "cold_memories": [],
            "behaviours": [], "preferences": [], "rules": [], "conclusions": [],
            "agent_assessment": {},
        })
        d = deriver.Deriver()
        ok, _ = self._run_pass(d, payload)
        self.assertTrue(ok)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_analysis_at'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(cursor)
        # Cursor == the newest processed message's created_at (not now).
        self.assertEqual(cursor["value"], (self.now - timedelta(minutes=10)).isoformat())
        self.assertLess(cursor["value"], datetime.now(timezone.utc).isoformat())

    def test_batch_cap_chunks_catch_up_without_loss(self):
        """With a 5-message cap and 7 unprocessed in-window messages, the first
        pass processes the oldest 5 (rowid ASC) and advances the cursor to the
        5th's created_at; the next pass picks up the remaining 2 — nothing is
        lost, and a single LLM invocation stays bounded."""
        minutes = [30, 25, 20, 15, 10, 8, 5]  # m0 oldest ... m6 newest
        for i, m in enumerate(minutes):
            self._add_message(f"m{i}", minutes_ago=m, content=f"message {i}")
        payload = json.dumps({
            "user": {}, "hot_memories": [], "cold_memories": [],
            "behaviours": [], "preferences": [], "rules": [], "conclusions": [],
            "agent_assessment": {},
        })
        d = deriver.Deriver()

        with mock.patch.object(deriver, "BATCH_MAX_MSGS", 5):
            ok1, stats1 = self._run_pass(d, payload)
        self.assertTrue(ok1)
        self.assertEqual(stats1["messages_processed"], 5)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        remaining = [r["id"] for r in conn.execute(
            "SELECT id FROM messages WHERE is_processed=0 ORDER BY rowid ASC"
        ).fetchall()]
        cursor = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_analysis_at'"
        ).fetchone()
        conn.close()
        self.assertEqual(remaining, ["m5", "m6"])
        self.assertEqual(cursor["value"], (self.now - timedelta(minutes=10)).isoformat())

        # Second pass catches up the remaining two (no cap pressure now).
        ok2, stats2 = self._run_pass(d, payload)
        self.assertTrue(ok2)
        self.assertEqual(stats2["messages_processed"], 2)


class SessionStartIdempotencyTest(unittest.TestCase):
    """The session-start trigger must not re-fire the LLM analysis when a
    successful pass already ran inside the current hour window (app restart
    mid-hour) — the job skips and waits for the next hour boundary."""

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

    def _set_cursor(self, minutes_ago: float):
        conn = deriver.get_db()
        now = datetime.now(timezone.utc)
        conn.execute(
            "INSERT OR REPLACE INTO deriver_state (key, value, updated_at) VALUES ('last_analysis_at', ?, ?)",
            ((now - timedelta(minutes=minutes_ago)).isoformat(), now.isoformat()),
        )
        conn.commit()
        conn.close()

    def test_no_cursor_means_window_due_immediately(self):
        d = deriver.Deriver()
        with mock.patch.object(deriver, "DERIVE_INTERVAL", 3600.0):
            self.assertEqual(d._seconds_until_next_window(), 0.0)

    def test_fresh_cursor_returns_remaining_window(self):
        self._set_cursor(minutes_ago=10)
        d = deriver.Deriver()
        with mock.patch.object(deriver, "DERIVE_INTERVAL", 3600.0):
            remaining = d._seconds_until_next_window()
        # 50 minutes left in the hour window (allow small clock skew).
        self.assertGreater(remaining, 2900.0)
        self.assertLessEqual(remaining, 3600.0)

    def test_stale_cursor_means_window_due(self):
        self._set_cursor(minutes_ago=90)  # older than the 60-min window
        d = deriver.Deriver()
        with mock.patch.object(deriver, "DERIVE_INTERVAL", 3600.0):
            self.assertEqual(d._seconds_until_next_window(), 0.0)

    def test_run_forever_skips_immediate_pass_when_window_not_due(self):
        """With a fresh last_analysis_at inside the window, run_forever must NOT
        call run_once immediately; it syncs and waits for the hour boundary."""
        self._set_cursor(minutes_ago=10)
        d = deriver.Deriver()
        calls = []

        def fake_run_once():
            calls.append("run_once")
            return 0

        def fake_sync():
            calls.append("sync")
            return 0

        d.run_once = fake_run_once
        d._sync_ev_sessions = fake_sync
        d._running = True

        import deriver as _deriver
        with mock.patch.object(_deriver, "DERIVE_INTERVAL", 3600.0), \
             mock.patch.object(_deriver, "SYNC_INTERVAL", 0.05), \
             mock.patch.object(_deriver.time, "sleep", return_value=None):
            thread = threading.Thread(target=d.run_forever, daemon=True)
            thread.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "sync" in calls:
                    break
            d.stop()
            thread.join(timeout=2.0)

        self.assertNotIn("run_once", calls)  # skipped — window not due
        self.assertIn("sync", calls)          # raw capture still runs

    def test_run_forever_fires_immediate_pass_when_window_due(self):
        """With no cursor (first run), run_forever fires run_once immediately —
        the session-start catch-up pass."""
        d = deriver.Deriver()
        calls = []

        def fake_run_once():
            calls.append("run_once")
            return 0

        def fake_sync():
            calls.append("sync")
            return 0

        d.run_once = fake_run_once
        d._sync_ev_sessions = fake_sync
        d._running = True

        import deriver as _deriver
        with mock.patch.object(_deriver, "DERIVE_INTERVAL", 3600.0), \
             mock.patch.object(_deriver, "SYNC_INTERVAL", 0.05), \
             mock.patch.object(_deriver.time, "sleep", return_value=None):
            thread = threading.Thread(target=d.run_forever, daemon=True)
            thread.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "run_once" in calls:
                    break
            d.stop()
            thread.join(timeout=2.0)

        self.assertIn("run_once", calls)  # first run always fires


if __name__ == "__main__":
    unittest.main()