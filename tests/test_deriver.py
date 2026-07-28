"""
Tests for the ingestion-only Deriver.

Covers:
- Checking for new messages from mock ev-agent DB
- Honcho-style schema (message + part tables)
- Simple 'messages' table schema
- 'conversations' table schema
- Ingestion pipeline (raw queue → batches)
- Cursor advancement
- Polling cycle
- Edge cases: empty messages, missing tables, DB not found
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from evsmem.memory_store import MemoryStore
from evsmem.deriver import Deriver


# ============================================================
# Fixtures: mock ev-agent databases
# ============================================================

@pytest.fixture
def honcho_evagent_db():
    """Create a mock ev-agent database with Honcho-style schema (message + part)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            data TEXT,
            time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE part (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            data TEXT,
            time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Insert messages with JSON data
    conn.execute(
        "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
        ("msg-1", "session-1", json.dumps({"role": "user"})),
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
        ("msg-2", "session-1", json.dumps({"role": "assistant"})),
    )
    # Insert text parts
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-1", "msg-1", json.dumps({"type": "text", "text": "Hello!"})),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
        ("part-2", "msg-2", json.dumps({"type": "text", "text": "Hi there!"})),
    )
    conn.commit()
    conn.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture
def simple_messages_db():
    """Create a mock DB with a simple 'messages' table."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        ("conv1", "user", "Hello from simple schema"),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        ("conv1", "assistant", "Response from simple schema"),
    )
    conn.commit()
    conn.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture
def conversations_db():
    """Create a mock DB with a 'conversations' table."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "INSERT INTO conversations (content, role) VALUES (?, ?)",
        ("Hello from conversations table", "user"),
    )
    conn.execute(
        "INSERT INTO conversations (content, role) VALUES (?, ?)",
        ("Response", "assistant"),
    )
    conn.commit()
    conn.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture
def evsmem_db():
    """Create a temporary path for the evsmem database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    try:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    except PermissionError:
        pass


# ============================================================
# Honcho-style schema tests
# ============================================================

class TestDeriverHonchoSchema:
    def test_check_for_new_messages(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        assert len(messages) >= 1
        assert "content" in messages[0]
        assert "rowid" in messages[0]
        assert "session_id" in messages[0]
        assert "role" in messages[0]

    def test_check_returns_message_content(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        contents = [m["content"] for m in messages]
        assert "Hello!" in contents
        assert "Hi there!" in contents

    def test_check_returns_correct_roles(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        roles = set(m["role"] for m in messages)
        assert "user" in roles
        assert "assistant" in roles

    def test_poll_once_ingests(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        count = deriver.poll_once()
        assert count >= 1

    def test_poll_once_idempotent(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        count1 = deriver.poll_once()
        assert count1 >= 1
        count2 = deriver.poll_once()
        assert count2 == 0  # No new messages

    def test_cursor_advances(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        deriver.poll_once()
        cursor = deriver.get_cursor()
        assert cursor >= 2  # Two messages ingested


# ============================================================
# Simple 'messages' table tests
# ============================================================

class TestDeriverSimpleMessages:
    def test_check_simple_messages(self, simple_messages_db, evsmem_db):
        deriver = Deriver(simple_messages_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        assert len(messages) >= 1
        assert "Hello from simple schema" in messages[0]["content"]

    def test_poll_once_simple(self, simple_messages_db, evsmem_db):
        deriver = Deriver(simple_messages_db, evsmem_db, poll_interval=0.1)
        count = deriver.poll_once()
        assert count >= 1

    def test_ingest_simple_then_idempotent(self, simple_messages_db, evsmem_db):
        deriver = Deriver(simple_messages_db, evsmem_db, poll_interval=0.1)
        count1 = deriver.poll_once()
        assert count1 >= 1
        count2 = deriver.poll_once()
        assert count2 == 0


# ============================================================
# 'conversations' table tests
# ============================================================

class TestDeriverConversations:
    def test_check_conversations(self, conversations_db, evsmem_db):
        deriver = Deriver(conversations_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        assert len(messages) >= 1
        # Content should be present
        contents = [m.get("content", "") for m in messages]
        assert any("Hello from conversations table" in c for c in contents)

    def test_poll_once_conversations(self, conversations_db, evsmem_db):
        deriver = Deriver(conversations_db, evsmem_db, poll_interval=0.1)
        count = deriver.poll_once()
        assert count >= 1


# ============================================================
# Raw ingestion pipeline tests
# ============================================================

class TestDeriverIngestion:
    def test_ingest_creates_batches(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        assert len(messages) >= 1
        ingested = deriver.ingest(messages)
        assert ingested >= 1
        # Check that batches were created
        store = deriver._get_store()
        batches = store.get_unprocessed_batches(min_size=1)
        assert len(batches) >= 1

    def test_ingest_empty_messages(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        ingested = deriver.ingest([])
        assert ingested == 0

    def test_ingest_cursor_advances(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        initial_cursor = deriver.get_cursor()
        deriver.ingest(messages)
        final_cursor = deriver.get_cursor()
        assert final_cursor > initial_cursor

    def test_full_poll_cycle(self, honcho_evagent_db, evsmem_db):
        """Test the complete check→ingest→batch flow."""
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        ingested = deriver.poll_once()
        assert ingested >= 1

        # Verify messages are in the evsmem DB
        store = deriver._get_store()
        # The cursor should have advanced
        cursor = store.get_cursor()
        assert cursor >= 1

    def test_run_once_alias(self, honcho_evagent_db, evsmem_db):
        """run_once is an alias for poll_once."""
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        count = deriver.run_once()
        assert count >= 1


# ============================================================
# Edge Cases
# ============================================================

class TestDeriverEdgeCases:
    def test_db_not_found(self, evsmem_db):
        """Non-existent ev-agent DB should not raise."""
        deriver = Deriver("/nonexistent/path/db.sqlite", evsmem_db,
                          poll_interval=0.1)
        messages = deriver.check_for_new_messages()
        assert messages == []

    def test_poll_once_no_db(self, evsmem_db):
        deriver = Deriver("/nonexistent/path/db.sqlite", evsmem_db,
                          poll_interval=0.1)
        count = deriver.poll_once()
        assert count == 0  # Graceful handling

    def test_empty_messages_table(self, evsmem_db):
        """A valid ev-agent DB with no rows should return empty."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        try:
            deriver = Deriver(tmp.name, evsmem_db, poll_interval=0.1)
            messages = deriver.check_for_new_messages()
            assert messages == []
        finally:
            os.unlink(tmp.name)

    def test_empty_content_skipped(self, honcho_evagent_db, evsmem_db):
        """Messages with empty content should not stall ingestion."""
        # Add a message with empty content
        conn = sqlite3.connect(honcho_evagent_db)
        conn.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
            ("msg-empty", "session-1", json.dumps({"role": "user"})),
        )
        conn.execute(
            "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
            ("part-empty", "msg-empty",
             json.dumps({"type": "text", "text": ""})),
        )
        conn.commit()
        conn.close()

        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        # Should not raise
        count = deriver.poll_once()
        assert count >= 1  # Should still process non-empty messages

    def test_extract_text_from_parts_empty(self):
        """_extract_text_from_parts should handle missing part data."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row  # _extract_text_from_parts expects Row objects
        conn.execute("""
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                data TEXT,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        result = Deriver._extract_text_from_parts(conn, "nonexistent-msg")
        assert result == ""
        conn.close()
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    def test_extract_text_from_reasoning_parts(self):
        """Should extract text from 'reasoning' type parts too."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row  # _extract_text_from_parts expects Row objects
        # Must include time_created column for ORDER BY in _extract_text_from_parts
        conn.execute("""
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                data TEXT,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO part (id, message_id, data) VALUES (?, ?, ?)",
            ("p1", "msg-1",
             json.dumps({"type": "reasoning", "text": "step by step"})),
        )
        conn.commit()

        result = Deriver._extract_text_from_parts(conn, "msg-1")
        assert "step by step" in result
        conn.close()
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    def test_get_cursor_initial(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        assert deriver.get_cursor() == 0

    def test_update_cursor_persists(self, honcho_evagent_db, evsmem_db):
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        deriver.update_cursor(42)
        assert deriver.get_cursor() == 42

    def test_empty_poll_returns_zero(self, honcho_evagent_db, evsmem_db):
        """Poll with no new messages should return 0."""
        deriver = Deriver(honcho_evagent_db, evsmem_db, poll_interval=0.1)
        # First poll should find messages
        count1 = deriver.poll_once()
        assert count1 >= 1
        # Second poll should find nothing
        count2 = deriver.poll_once()
        assert count2 == 0
