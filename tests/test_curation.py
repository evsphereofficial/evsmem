"""
Tests for the memory-curation layer: crud update/delete/move functions,
curation audit trail, atomic tool-call application, and the deriver's
function-calling integration.

Run from the evsmem directory:
    .venv\\Scripts\\python.exe -m unittest tests.test_curation -v
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crud
import deriver


def _cleanup(tmp):
    gc.collect()
    try:
        tmp.cleanup()
    except Exception:
        pass


class CuratedDBTestCase(unittest.TestCase):
    """Set up a fresh temp evsmem DB shared by crud + deriver."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "evsmem.db"
        old_crud_db, old_deriver_db = crud.DB_PATH, deriver.DB_PATH
        old_schema, old_local = crud._SCHEMA_READY, crud._local
        old_ev = deriver.EV_SESSION_DB

        crud.DB_PATH = self.db_path
        deriver.DB_PATH = self.db_path
        deriver.EV_SESSION_DB = Path(self._tmp.name) / "no-session.db"
        crud._SCHEMA_READY = False
        crud._local = threading.local()
        self._restore = (old_crud_db, old_deriver_db, old_schema, old_local, old_ev)

        self.conn = crud.get_db()
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO workspaces (id, name, metadata, created_at) VALUES ('ws1','ev-agent','{}', ?)",
            (now,),
        )
        self.conn.execute(
            "INSERT INTO sessions (id, workspace_id, name, metadata, created_at) VALUES ('s1','ws1','auto-save','{}', ?)",
            (now,),
        )
        self.conn.execute(
            "INSERT INTO peers (id, workspace_id, name, metadata, created_at) VALUES ('p1','ws1','user','{}', ?)",
            (now,),
        )
        self.conn.commit()

    def tearDown(self):
        try:
            crud.close_db()
        finally:
            crud.DB_PATH, deriver.DB_PATH, crud._SCHEMA_READY, crud._local, deriver.EV_SESSION_DB = self._restore
            _cleanup(self._tmp)

    def _add_message(self, mid, minutes_ago, content, processed=0, session="s1"):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
        self.conn.execute(
            """INSERT INTO messages
               (id, session_id, peer_id, content, role, message_type, metadata,
                embedding, created_at, is_processed)
               VALUES (?, ?, 'p1', ?, 'user', 'message', '{}', NULL, ?, ?)""",
            (mid, session, content, ts, processed),
        )
        self.conn.commit()


class CrudCurationTest(CuratedDBTestCase):
    def test_add_memory_row_across_tables(self):
        crud.add_memory_row("memories", "ws1", {"content": "building a rust CLI", "type": "project"})
        crud.add_memory_row("behaviour", "ws1", {"content": "user gets frustrated when code fails"})
        crud.add_memory_row("preferences", "ws1", {"content": "user prefers python"})
        crud.add_memory_row("rules", "ws1", {"content": "never force delete"})
        crud.add_memory_row("agent_written_memory", "ws1", {"content": "refactored xyz"})
        crud.add_memory_row("users", "ws1", {"name": "Rehan", "location": "Mumbai"})

        self.assertEqual(len(crud.get_behaviours("ws1")), 1)
        self.assertEqual(len(crud.get_preferences("ws1")), 1)
        self.assertEqual(len(crud.get_rules("ws1")), 1)
        self.assertEqual(len(crud.list_agent_written_memories("ws1")), 1)
        user = crud.get_user("ws1")
        self.assertEqual(user["name"], "Rehan")
        self.assertEqual(user["location"], "Mumbai")

        mems = [r for r in self.conn.execute(
            "SELECT * FROM memories WHERE workspace_id='ws1'").fetchall()]
        self.assertEqual(len(mems), 1)
        self.assertEqual(mems[0]["type"], "cold_memory")
        self.assertEqual(mems[0]["memory_type"], "project")

    def test_add_memory_row_validates_table(self):
        with self.assertRaises(ValueError):
            crud.add_memory_row("agents", "ws1", {"content": "nope"})

    def test_update_memory_row_in_place(self):
        crud.add_memory_row("memories", "ws1", {"content": "dark mode", "importance": 0.5})
        mem = dict(self.conn.execute(
            "SELECT * FROM memories WHERE workspace_id='ws1'").fetchone())
        res = crud.update_memory_row("memories", mem["id"], {"importance": 0.98, "type": "hot_memory"},
                                     reason="high value")
        self.assertTrue(res["ok"])
        self.assertEqual(res["action"], "updated")
        row = dict(self.conn.execute("SELECT * FROM memories WHERE id=?", (mem["id"],)).fetchone())
        self.assertEqual(row["importance"], 0.98)
        self.assertEqual(row["type"], "hot_memory")

    def test_update_memory_row_missing(self):
        res = crud.update_memory_row("memories", "does-not-exist", {"importance": 0.9})
        self.assertFalse(res["ok"])
        self.assertEqual(res["action"], "missing")

    def test_move_row_between_tables(self):
        crud.add_memory_row("behaviour", "ws1", {"content": "user paces while coding"})
        beh = crud.get_behaviours("ws1")[0]
        res = crud.update_memory_row("behaviour", beh["id"], move_to_table="memories",
                                     reason="behavioural fact belongs in memories")
        self.assertTrue(res["ok"])
        self.assertEqual(res["action"], "moved")
        # source deleted, target inserted
        self.assertIsNone(self.conn.execute(
            "SELECT id FROM behaviour WHERE id=?", (beh["id"],)).fetchone())
        moved = dict(self.conn.execute(
            "SELECT * FROM memories WHERE id=?", (res["new_row_id"],)).fetchone())
        self.assertEqual(moved["type"], "cold_memory")
        self.assertEqual(json.loads(moved["metadata"])["curation"]["moved_from"], "behaviour")

    def test_move_users_to_memories(self):
        crud.add_memory_row("users", "ws1", {"name": "Rehan", "occupation": "engineer"})
        user = crud.get_user("ws1")
        res = crud.update_memory_row("users", user["id"], move_to_table="memories", reason="profile snapshot")
        self.assertTrue(res["ok"])
        moved = dict(self.conn.execute("SELECT * FROM memories WHERE id=?", (res["new_row_id"],)).fetchone())
        self.assertIn("engineer", moved["content"])
        self.assertIsNone(crud.get_user("ws1"))

    def test_move_into_users_rejected(self):
        crud.add_memory_row("memories", "ws1", {"content": "some fact"})
        mem = dict(self.conn.execute("SELECT * FROM memories WHERE workspace_id='ws1'").fetchone())
        with self.assertRaises(ValueError):
            crud.update_memory_row("memories", mem["id"], move_to_table="users")

    def test_delete_memory_row_hard_delete_with_audit(self):
        crud.add_memory_row("rules", "ws1", {"content": "old rule"})
        rule = crud.get_rules("ws1")[0]
        res = crud.delete_memory_row("rules", rule["id"], reason="superseded")
        self.assertTrue(res["ok"])
        self.assertIsNone(self.conn.execute("SELECT id FROM rules WHERE id=?", (rule["id"],)).fetchone())

        audit = crud.list_curation_audit()
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["action"], "delete")
        self.assertEqual(audit[0]["table_name"], "rules")
        self.assertEqual(audit[0]["reason"], "superseded")
        self.assertEqual(audit[0]["content"], "old rule")

    def test_update_audits_reason(self):
        crud.add_memory_row("preferences", "ws1", {"content": "dark mode"})
        pref = crud.get_preferences("ws1")[0]
        crud.update_memory_row("preferences", pref["id"], {"importance": 0.4}, reason="less important now")
        audits = crud.list_curation_audit()
        self.assertEqual([a["action"] for a in audits], ["update"])
        self.assertEqual(audits[0]["reason"], "less important now")

    def test_atomic_tool_call_transaction_rolls_back(self):
        """A failing call rolls back the entire tool round."""
        conn = crud.get_db()
        conn.execute("BEGIN")
        crud.add_memory_row("preferences", "ws1", {"content": "will be rolled back"}, conn=conn)
        with self.assertRaises(ValueError):
            crud.add_memory_row("bogus-table", "ws1", {"content": "boom"})
        try:
            conn.rollback()
        except Exception:
            pass
        self.assertEqual(crud.get_preferences("ws1"), [])


class FakeLLM:
    """Plain generate-only LLM (local fallback)."""

    def __init__(self, payload):
        self._payload = payload

    def is_available(self):
        return True

    def generate(self, messages, max_tokens=512, temperature=0.1):
        return self._payload


class FakeToolLLM:
    """LLM with generate_with_tools returning queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.rounds = 0

    def is_available(self):
        return True

    def generate_with_tools(self, messages, tools=None, tool_choice="auto", max_tokens=512, temperature=0.1):
        self.rounds += 1
        if not self._responses:
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": []}}]}
        return self._responses.pop(0)


def _tool_response(tool_calls, content=None):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        }]
    }


def _tool_call(call_id, name, arguments, content=None):
    if content:
        return {"role": "assistant", "content": content, "tool_calls": []}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class DeriverToolIntegrationTest(CuratedDBTestCase):
    def _make_deriver(self, llm):
        d = deriver.Deriver()
        d._get_llm = lambda: ("remote" if getattr(llm, "generate_with_tools", None) else "local", llm)
        return d

    def test_plain_generate_fallback_parses_json(self):
        self._add_message("m1", 10, "I prefer dark mode and short answers.")
        payload = json.dumps({
            "user": {"name": "Rehan"},
            "hot_memories": [{"content": "user prefers dark mode", "importance": 0.9,
                              "confidence": 0.9, "durability": 0.9, "type": "preference"}],
            "cold_memories": [], "behaviours": [], "preferences": [], "rules": [],
            "conclusions": [], "agent_assessment": {},
        })
        d = self._make_deriver(FakeLLM(payload))
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        ok = d._process_new_messages_with_llm(stats)
        self.assertTrue(ok)
        self.assertEqual(stats["messages_processed"], 1)
        self.assertEqual(stats["json_rows"].get("memories"), 1)
        self.assertEqual(stats["json_rows"].get("users"), 1)
        # message marked processed only after success
        n = self.conn.execute("SELECT is_processed FROM messages WHERE id='m1'").fetchone()[0]
        self.assertEqual(n, 1)

    def test_tool_calls_applied_and_results_fed_back(self):
        self._add_message("m1", 10, "user prefers dark mode")
        # Round 1: add_memory_rows tool call; Round 2: final JSON content.
        tool_llm = FakeToolLLM([
            _tool_response([
                {"id": "c1", "type": "function",
                 "function": {"name": "add_memory_rows",
                              "arguments": json.dumps({"rows": [
                                  {"table": "preferences", "content": "user prefers dark mode",
                                   "importance": 0.9},
                              ]})}},
            ]),
            _tool_response([], content=json.dumps({
                "user": {}, "hot_memories": [{"content": "user prefers dark mode", "importance": 0.9,
                                              "confidence": 0.9, "durability": 0.9, "type": "preference"}],
                "cold_memories": [], "behaviours": [], "preferences": [], "rules": [],
                "conclusions": [], "agent_assessment": {},
            })),
        ])
        d = self._make_deriver(tool_llm)
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        ok = d._process_new_messages_with_llm(stats)
        self.assertTrue(ok)
        self.assertEqual(tool_llm.rounds, 2)
        self.assertEqual(stats["tool_rounds"], 1)
        self.assertEqual(stats["tool_inserted"].get("preferences"), 1)
        # tool-call insert is deduped by the JSON path (same meaning) — no dup row
        prefs = crud.get_preferences("ws1")
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["content"], "user prefers dark mode")

    def test_update_and_delete_tools(self):
        # Seed rows the model will fix.
        crud.add_memory_row("behaviour", "ws1", {"content": "user likes python", "importance": 0.9})
        crud.add_memory_row("memories", "ws1", {"content": "dup dark mode fact", "importance": 0.4})
        beh = crud.get_behaviours("ws1")[0]
        mem = dict(self.conn.execute("SELECT * FROM memories WHERE workspace_id='ws1'").fetchone())

        self._add_message("m1", 10, "an actual new message")
        tool_llm = FakeToolLLM([
            _tool_response([
                {"id": "c1", "type": "function",
                 "function": {"name": "update_memory_row",
                              "arguments": json.dumps({"table": "behaviour", "row_id": beh["id"],
                                                       "move_to_table": "memories",
                                                       "reason": "is a preference not behaviour"})}},
                {"id": "c2", "type": "function",
                 "function": {"name": "delete_memory_row",
                              "arguments": json.dumps({"table": "memories", "row_id": mem["id"],
                                                       "reason": "duplicate"})}},
            ]),
            _tool_response([], content=json.dumps({
                "user": {}, "hot_memories": [], "cold_memories": [], "behaviours": [],
                "preferences": [], "rules": [], "conclusions": [], "agent_assessment": {},
            })),
        ])
        d = self._make_deriver(tool_llm)
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        ok = d._process_new_messages_with_llm(stats)
        self.assertTrue(ok)
        self.assertEqual(stats["tool_moved"], 1)
        self.assertEqual(stats["tool_deleted"], 1)
        # behaviour row moved to memories
        self.assertIsNone(self.conn.execute(
            "SELECT id FROM behaviour WHERE id=?", (beh["id"],)).fetchone())
        moved = dict(self.conn.execute(
            "SELECT * FROM memories WHERE content='user likes python'").fetchone())
        self.assertEqual(moved["type"], "cold_memory")
        # dup memory hard-deleted
        self.assertIsNone(self.conn.execute(
            "SELECT id FROM memories WHERE id=?", (mem["id"],)).fetchone())

    def test_tool_transaction_failure_rolls_back(self):
        self._add_message("m1", 10, "hello")
        tool_llm = FakeToolLLM([
            _tool_response([
                {"id": "c1", "type": "function",
                 "function": {"name": "add_memory_rows",
                              "arguments": json.dumps({"rows": [
                                  {"table": "preferences", "content": "kept preference"},
                                  {"table": "not-a-table", "content": "boom"},
                              ]})}},
            ]),
        ])
        d = self._make_deriver(tool_llm)
        stats = {"messages_processed": 0, "engine": "none", "tool_rounds": 0,
                 "json_rows": {}, "tool_inserted": {}, "tool_updated": 0,
                 "tool_moved": 0, "tool_deleted": 0}
        with self.assertRaises(ValueError):
            d._process_new_messages_with_llm(stats)
        # entire round rolled back; message stays unprocessed
        self.assertEqual(crud.get_preferences("ws1"), [])
        n = self.conn.execute("SELECT is_processed FROM messages WHERE id='m1'").fetchone()[0]
        self.assertEqual(n, 0)


class SchedulerTest(unittest.TestCase):
    def test_run_forever_immediate_pass_then_interval(self):
        d = deriver.Deriver()
        calls = []
        orig_run_once = d.run_once
        orig_sync = d._sync_ev_sessions

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
        with mock.patch.object(_deriver, "DERIVE_INTERVAL", 5.0), \
             mock.patch.object(_deriver, "SYNC_INTERVAL", 5.0), \
             mock.patch.object(_deriver.time, "sleep", return_value=None):
            thread = threading.Thread(target=d.run_forever, daemon=True)
            thread.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if "run_once" in calls:
                    break
            d.stop()
            thread.join(timeout=2.0)

        self.assertIn("run_once", calls)   # immediate first pass
        self.assertIn("sync", calls)       # sync ran too


if __name__ == "__main__":
    unittest.main()