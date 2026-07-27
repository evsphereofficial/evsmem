import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deriver


class DeriverCursorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "evsmem.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL);
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL
            );
            INSERT INTO sessions VALUES ('session-1', 'workspace-1');
            INSERT INTO messages VALUES ('message-1', 'session-1', 'remember this', 'user');
            """
        )
        conn.commit()
        conn.close()
        self.db_patch = patch.object(deriver, "DB_PATH", self.db_path)
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    def test_failed_extraction_does_not_advance_cursor(self):
        worker = deriver.Deriver()
        worker._last_message_id = 0

        with (
            patch.object(worker, "_sync_ev_sessions", return_value=0),
            patch.object(worker, "_analyze_session_reputation"),
            patch.object(worker, "_check_auto_generate_trigger"),
            patch.object(worker, "_extract_facts", return_value=None),
        ):
            worker.run_once()

        self.assertEqual(worker._last_message_id, 0)
        self.assertEqual(worker._get_unprocessed_messages()[0]["id"], "message-1")

    def test_successful_extraction_updates_profile_then_advances_cursor(self):
        worker = deriver.Deriver()
        worker._last_message_id = 0
        facts = [
            {
                "category": "USER_INFO",
                "description": "User prefers concise answers",
                "session_id": "session-1",
                "workspace_id": "workspace-1",
            }
        ]

        with (
            patch.object(worker, "_sync_ev_sessions", return_value=0),
            patch.object(worker, "_analyze_session_reputation"),
            patch.object(worker, "_check_auto_generate_trigger"),
            patch.object(worker, "_extract_facts", return_value=facts),
            patch.object(worker, "_store_conclusions") as store,
            patch.object(worker, "_update_peer_representation") as update,
        ):
            worker.run_once()

        store.assert_called_once_with(facts)
        update.assert_called_once_with(facts)
        self.assertEqual(worker._last_message_id, 1)
        conn = sqlite3.connect(self.db_path)
        value = conn.execute(
            "SELECT value FROM deriver_state WHERE key='last_message_rowid'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(value, "1")


if __name__ == "__main__":
    unittest.main()
