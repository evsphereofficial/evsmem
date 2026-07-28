"""
evsmem/deriver.py — Pure ingestion watcher.

Monitors the ev-agent SQLite database for new conversation messages,
appends them to the evsmem raw conversation queue, and maintains a cursor.
No interpretation, summarization, or classification is performed.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)

from evsmem.config import (
    EVAGENT_DB_PATH,
    EVSMEM_DB_PATH,
    DERIVER_POLL_INTERVAL,
)


class Deriver:
    """
    Lightweight ingestion-only component.

    Watches the ev-agent database for new messages and feeds
    them into the evsmem memory system's raw conversation queue
    (the ``raw_conversations`` table). A high-water cursor is
    persisted in ``cursor_state`` to ensure exactly-once delivery.
    """

    def __init__(
        self,
        evagent_db_path: str | Path | None = None,
        evsmem_db_path: str | Path | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self.evagent_db_path = Path(evagent_db_path) if evagent_db_path else Path(EVAGENT_DB_PATH)
        self.evsmem_db_path = Path(evsmem_db_path) if evsmem_db_path else Path(EVSMEM_DB_PATH)
        self.poll_interval = poll_interval if poll_interval is not None else DERIVER_POLL_INTERVAL
        self._running = False
        self._evsmem_store: Optional["MemoryStore"] = None  # lazy import

    # ── evsmem store access ─────────────────────────────────────────────

    def _get_store(self) -> "MemoryStore":
        """Lazy-load MemoryStore to avoid circular imports at module level."""
        if self._evsmem_store is None:
            from evsmem.memory_store import MemoryStore  # type: ignore[import-untyped]

            self._evsmem_store = MemoryStore(self.evsmem_db_path)
        return self._evsmem_store

    # ── Cursor management ───────────────────────────────────────────────

    def get_cursor(self) -> int:
        """Return the last-processed ev-agent message rowid (0 if none)."""
        return self._get_store().get_cursor()

    def update_cursor(self, message_rowid: int) -> None:
        """Persist the high-water mark."""
        self._get_store().update_cursor(message_rowid)

    # ── Ev-agent DB access ──────────────────────────────────────────────

    def _connect_evagent(self) -> sqlite3.Connection:
        """Open a read-only connection to the ev-agent SQLite database."""
        conn = sqlite3.connect(f"file:{self.evagent_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Message detection ───────────────────────────────────────────────

    def check_for_new_messages(self) -> list[dict]:
        """
        Query the ev-agent DB for messages newer than the current cursor.

        Handles the Honcho-style schema used by ev-agent (``message`` table
        with a JSON ``data`` column, plus a ``part`` table for text content).
        Falls back to simple table discovery for other schemas.

        Returns a list of dicts with keys::

            rowid         — raw rowid from ev-agent DB
            session_id    — conversation / session identifier
            role          — 'user' or 'assistant'
            content       — plain-text body of the message
        """
        cursor_rowid = self.get_cursor()

        if not self.evagent_db_path.exists():
            logger.debug("ev-agent DB not found at %s", self.evagent_db_path)
            return []

        try:
            conn = self._connect_evagent()
        except sqlite3.Error as exc:
            logger.warning("Cannot open ev-agent DB: %s", exc)
            return []

        try:
            # ── Probe the available tables ──
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            messages: list[dict] = []

            # ── Strategy A: Honcho-style schema (message + part) ──
            if "message" in tables and "part" in tables:
                rows = conn.execute(
                    """
                    SELECT m.rowid,
                           m.id          AS message_id,
                           m.session_id,
                           m.data        AS message_data,
                           m.time_created
                    FROM message m
                    WHERE m.rowid > ?
                    ORDER BY m.rowid ASC
                    """,
                    (cursor_rowid,),
                ).fetchall()

                for row in rows:
                    # Parse the JSON data blob
                    try:
                        data = (
                            json.loads(row["message_data"])
                            if isinstance(row["message_data"], str)
                            else (row["message_data"] or {})
                        )
                    except (json.JSONDecodeError, TypeError):
                        data = {}

                    role = data.get("role", "unknown")

                    # Retrieve text parts from the part table
                    content = self._extract_text_from_parts(conn, row["message_id"])

                    messages.append(
                        {
                            "rowid": row["rowid"],
                            "session_id": row["session_id"],
                            "role": role,
                            "content": content,
                        }
                    )

            # ── Strategy B: Simple 'messages' table ──
            elif "messages" in tables:
                cols = {
                    col[1]
                    for col in conn.execute(
                        "PRAGMA table_info(messages)"
                    ).fetchall()
                }
                if "rowid" in cols:
                    id_col = "rowid"
                elif "id" in cols:
                    id_col = "id"
                else:
                    id_col = "rowid"

                select_cols = [id_col, "session_id", "role", "content"]
                select_list = ", ".join(
                    f"m.{c}" if c != id_col else f"m.{id_col} AS rowid"
                    for c in select_cols
                )

                try:
                    rows = conn.execute(
                        f"SELECT {select_list} FROM messages m "
                        f"WHERE m.{id_col} > ? ORDER BY m.{id_col} ASC",
                        (cursor_rowid,),
                    ).fetchall()
                    for r in rows:
                        messages.append(dict(r))
                except sqlite3.OperationalError:
                    pass

            # ── Strategy C: 'conversations' table ──
            elif "conversations" in tables:
                cols = {
                    col[1]
                    for col in conn.execute(
                        "PRAGMA table_info(conversations)"
                    ).fetchall()
                }
                if "id" in cols and "content" in cols:
                    id_col = "rowid"
                    # Always select rowid (implicit in SQLite, not in PRAGMA)
                    select_cols = ["c.rowid AS rowid"]
                    for col_name in ("content", "role"):
                        if col_name in cols:
                            select_cols.append(f"c.{col_name}")
                    select_list = ", ".join(select_cols)
                    if not select_list:
                        select_list = "rowid, content"
                    try:
                        rows = conn.execute(
                            f"SELECT {select_list} FROM conversations c "
                            f"WHERE c.rowid > ? ORDER BY c.rowid ASC",
                            (cursor_rowid,),
                        ).fetchall()
                        for r in rows:
                            d = dict(r)
                            d.setdefault("role", "user")
                            d.setdefault("session_id", "default")
                            messages.append(d)
                    except sqlite3.OperationalError:
                        pass

            return messages

        except sqlite3.Error as exc:
            logger.warning("Error querying ev-agent DB: %s", exc)
            return []
        finally:
            conn.close()

    @staticmethod
    def _extract_text_from_parts(
        conn: sqlite3.Connection, message_id: str
    ) -> str:
        """Assemble plain text from the ``part`` table for a given message."""
        try:
            part_rows = conn.execute(
                "SELECT data FROM part WHERE message_id=? ORDER BY time_created ASC",
                (message_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return ""

        text_parts: list[str] = []
        for p in part_rows:
            try:
                pd = (
                    json.loads(p["data"])
                    if isinstance(p["data"], str)
                    else (p["data"] or {})
                )
            except (json.JSONDecodeError, TypeError):
                continue
            if pd.get("type") in ("text", "reasoning"):
                text = (pd.get("text") or "").strip()
                if text:
                    text_parts.append(text[:4000])
        return " | ".join(text_parts[:10]) if text_parts else ""

    # ── Ingestion pipeline ──────────────────────────────────────────────

    def ingest(self, messages: list[dict]) -> int:
        """
        Append raw messages to the evsmem raw queue and create batches.

        Args:
            messages: List of message dicts from :meth:`check_for_new_messages`.

        Returns:
            Number of messages ingested.
        """
        if not messages:
            return 0

        store = self._get_store()
        batch_message_ids: list[int] = []
        last_conversation_id: str | None = None
        ingested = 0

        for msg in messages:
            rowid = msg["rowid"]
            conversation_id = msg.get("session_id", "default")
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if not content:
                # Still advance cursor for empty messages so we don't stall
                self.update_cursor(rowid)
                continue

            # Insert into raw queue
            msg_db_id = store.append_raw_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
            )
            ingested += 1

            # Group consecutive messages from the same conversation
            if last_conversation_id is not None and conversation_id != last_conversation_id:
                # Flush previous batch
                if batch_message_ids:
                    store.create_batch(last_conversation_id, batch_message_ids)
                    logger.debug(
                        "Created batch for conversation %s (%d messages)",
                        last_conversation_id,
                        len(batch_message_ids),
                    )
                batch_message_ids = []

            batch_message_ids.append(msg_db_id)
            last_conversation_id = conversation_id

            # Advance cursor after every message
            self.update_cursor(rowid)

        # Flush the final batch
        if batch_message_ids and last_conversation_id is not None:
            store.create_batch(last_conversation_id, batch_message_ids)
            logger.debug(
                "Created batch for conversation %s (%d messages)",
                last_conversation_id,
                len(batch_message_ids),
            )

        logger.info("Ingested %d raw message(s)", ingested)
        return ingested

    # ── Polling ─────────────────────────────────────────────────────────

    def poll_once(self) -> int:
        """
        Single poll cycle: check for new messages and ingest them.

        Returns:
            Number of messages ingested (0 if none or on error).
        """
        try:
            messages = self.check_for_new_messages()
            if messages:
                return self.ingest(messages)
            return 0
        except Exception as exc:
            logger.error("Deriver poll error: %s", exc, exc_info=True)
            return 0

    def run_once(self) -> int:
        """Alias for :meth:`poll_once` — used by external schedulers."""
        return self.poll_once()

    def start(self) -> None:
        """Start the blocking polling loop."""
        self._running = True
        logger.info(
            "Deriver started (poll interval: %.1fs, ev-agent DB: %s)",
            self.poll_interval,
            self.evagent_db_path,
        )
        while self._running:
            count = self.poll_once()
            if count:
                logger.debug("Deriver ingested %d message(s) in this cycle", count)
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to exit."""
        self._running = False
        logger.info("Deriver stopped")


# ── Background thread helper ────────────────────────────────────────────


def start_deriver_thread(
    evagent_db_path: str | Path | None = None,
    evsmem_db_path: str | Path | None = None,
    poll_interval: float | None = None,
) -> Deriver:
    """Start the deriver in a daemon background thread. Returns the instance."""
    d = Deriver(
        evagent_db_path=evagent_db_path,
        evsmem_db_path=evsmem_db_path,
        poll_interval=poll_interval,
    )
    t = Thread(target=d.start, daemon=True, name="evsmem-deriver")
    t.start()
    logger.info("Deriver thread started (daemon)")
    return d
