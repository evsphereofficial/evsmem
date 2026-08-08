"""
evsmem Deriver — background session sync.

Reads new messages from ev-agent sessions and syncs them
into the evsmem database for memory storage.

Usage:
  from deriver import Deriver
  d = Deriver()
  d.run_once()          # process all new messages
  d.run_forever()       # poll every N seconds
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

DB_PATH = Path.home() / ".evsmem" / "evsmem.db"

# Ev-agent session DB path
EV_SESSION_DB = Path.home() / ".local" / "share" / "ev-agent" / "ev-agent-local.db"

POLL_INTERVAL = float(os.getenv("DERIVER_POLL_INTERVAL", "5.0"))
MAX_FACTS_PER_BATCH = int(os.getenv("DERIVER_MAX_FACTS", "10"))


# ── LLM Analysis Prompt ──

LLM_ANALYSIS_PROMPT = """\
You are a meticulous memory-extraction engine for a conversation between a user and an AI coding assistant. Your ONLY job is to extract durable facts. Analyze the message carefully and output ONE valid JSON object.

Message: "{content}"

Required JSON schema:
{{
  "user_name": "exact user name if stated, else null",
  "user_mood": "detected mood (happy/frustrated/neutral/urgent/etc) or null",
  "user_preferences": ["concrete stated preference 1", "preference 2"],
  "hot_memories": [
    {{"content": "a CRITICAL fact that must always be remembered", "importance": 0.9, "confidence": 0.9, "durability": 0.9, "type": "architecture_decision"}},
    {{"content": "another critical always-relevant fact", "importance": 0.95, "confidence": 0.85, "durability": 0.9, "type": "user"}}
  ],
  "cold_memories": [
    {{"content": "a DETAILED, specific fact: include names, projects, versions, decisions, code paths, and enough context (2-3 sentences) to stand alone", "importance": 0.5, "confidence": 0.8, "durability": 0.5, "type": "project"}},
    {{"content": "another detailed fact retrievable on request", "importance": 0.4, "confidence": 0.7, "durability": 0.3, "type": "event"}}
  ],
  "conclusions": ["a new insight about the user or project, phrased uniquely"],
  "agent_assessment": {{"agent_name": "mentioned agent or ''", "verdict": "positive|negative|neutral", "detail": "one-line assessment"}}
}}

Detailed rules:
1. HOT_MEMORIES = EVERY user info and behavior fact: identity (name, age, location), preferences, habits, moods, work style, constraints, key relationships, recurring behaviors, decisions the user makes about themselves. These are ALWAYS injected into every session. Include EVERY user info/behavior fact from this message; keep each short, precise, importance 0.8-1.0. Max 8 per message.
2. COLD_MEMORIES = everything else — the EXTRAS, injected on demand: project/technical details, what the user is building, the task, the topic, the code/architecture/approach, technical decisions, tooling, versions, bugs, design rationale, people/projects mentioned. MOST IMPORTANTLY capture WHAT THE USER IS TALKING ABOUT in detail. Be DETAILED — project names, exact terms, versions, file paths, and 2-3 sentences of context so the fact stands alone. importance 0.3-0.7.
3. Every content value MUST be derived strictly from the message. Never invent facts, never output generic placeholders, never copy these instructions.
4. Per-memory attributes:
   - importance: 0.0 (trivial) to 1.0 (must-never-forget). Hot 0.8-1.0, cold 0.3-0.7.
   - confidence: how sure are we this fact is true, 0.0-1.0 (default 0.8).
   - durability: how long this fact is likely to stay true, 0.0 (transient/debugging detail) to 1.0 (permanent fact). Architecture decisions ~0.9, a failed command ~0.1.
   - type: one of: user | preference | project | decision | architecture_decision | tooling | environment | event | debugging_event | conversation_insight.
5. If the message has no memorable facts, output empty arrays: "hot_memories": [], "cold_memories": [], "conclusions": [].
6. Return ONLY the JSON object. No markdown fences, no commentary."""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS deriver_state (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL,
               updated_at TEXT NOT NULL
           )"""
    )
    # Ensure the is_processed column exists on messages (added by crud schema,
    # but the deriver connects directly and must not depend on crud.get_db()).
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN is_processed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    return conn


class Deriver:
    """Background fact extractor for evsmem."""

    def __init__(self):
        self._last_processed_rowid: Optional[int] = None
        self._running = False

    # ── Cursor-based message tracking ──

    def _get_unprocessed_messages(self) -> list[dict]:
        """Get unprocessed messages using the per-row is_processed flag.

        Only messages created within a short lookback window (default 24h) AND
        older than 30 seconds are considered — so the deriver handles LIVE
        messages and never re-processes old backlog/history.
        """
        lookback_hours = int(os.getenv("EVSMEM_PROCESS_LOOKBACK_HOURS", "24"))
        conn = get_db()
        try:
            rows = conn.execute(
                f"""SELECT m.rowid, m.id, m.content, m.role, m.session_id,
                           s.workspace_id, m.metadata
                    FROM messages m
                    JOIN sessions s ON m.session_id = s.id
                    WHERE m.is_processed = 0 AND m.content != ''
                      AND m.created_at >= strftime('%Y-%m-%dT%H:%M:%f', 'now', '-{lookback_hours} hours')
                      AND m.created_at <= strftime('%Y-%m-%dT%H:%M:%f', 'now', '-30 seconds')
                    ORDER BY m.rowid DESC
                    LIMIT 20""",
            ).fetchall()

            out = []
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r["metadata"] or "{}")
                except Exception:
                    pass
                out.append({
                    "rowid": r["rowid"],
                    "id": r["id"],
                    "content": r["content"],
                    "role": r["role"],
                    "session_id": r["session_id"],
                    "workspace_id": r["workspace_id"],
                    "is_subagent": bool(meta.get("is_subagent", False)),
                })
            return out
        finally:
            conn.close()

    def _mark_message_processed(self, rowid: int):
        """Mark a single message as processed (is_processed = 1)."""
        conn = get_db()
        try:
            conn.execute("UPDATE messages SET is_processed = 1 WHERE rowid = ?", (rowid,))
            conn.commit()
        finally:
            conn.close()

    # ── LLM Processing Pipeline ──

    def _process_new_messages_with_llm(self) -> int:
        """Process unprocessed messages through local LLM to extract facts.

        Returns the number of messages successfully processed.
        """
        from llm_client import LLMClient

        llm = LLMClient()
        if not llm.is_available():
            logger.debug("LLM not available, skipping message processing")
            return 0

        messages = self._get_unprocessed_messages()
        if not messages:
            return 0

        processed = 0
        for msg in messages:
            rowid = msg["rowid"]
            try:
                parsed = self._analyze_message_with_llm(llm, msg)
                if parsed:
                    self._store_llm_results(parsed, msg)
                    processed += 1
            except Exception as e:
                logger.warning(f"LLM processing error for msg {rowid}: {e}")
            finally:
                self._mark_message_processed(rowid)

        if processed > 0:
            logger.info(f"Deriver: processed {processed} messages")
        return processed

    def _analyze_message_with_llm(self, llm, msg: dict) -> Optional[dict]:
        """Build prompt, call LLM, and parse the JSON response.

        Args:
            llm: An LLMClient instance.
            msg: Message dict with 'content', 'role', etc.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        content = (msg.get("content") or "").strip()
        if not content:
            return None

        if content.startswith("[auto:"):
            idx = content.find("] ", 6)
            if idx != -1:
                content = content[idx+2:].strip()

        role = msg.get("role", "user")
        if msg.get("is_subagent"):
            content = (
                "[NOTE: This message is from an INTERNAL SUBAGENT session (an AI subagent working a task), "
                "NOT a direct statement from the human user. Do NOT infer the human user's name, preferences, "
                "identity, or mood from it. Only extract concrete technical/project facts if clearly present.]\n"
                + content
            )
        elif role == "assistant":
            content = (
                "[NOTE: This is the AI ASSISTANT's response in the conversation. It contains the DETAILED "
                "technical/project context of what the user is working on. Extract the facts about what is being "
                "built and discussed IN DETAIL. Do NOT attribute these statements as the human user's own "
                "words or preferences.]\n"
                + content
            )

        if len(content) < 10:
            return None

        prompt = LLM_ANALYSIS_PROMPT.format(content=content[:6000])

        raw = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1,
        )

        if not raw:
            logger.debug(f"LLM returned empty for message {msg.get('rowid')}")
            return None

        return self._parse_llm_output(raw)

    def _parse_llm_output(self, raw: str) -> Optional[dict]:
        """Parse JSON from the LLM's raw text output.

        Handles markdown code fences and stray text before/after JSON.
        """
        text = raw.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            # Find the first { or [ after the opening fence
            start = text.find("{")
            if start == -1:
                start = text.find("[")
            if start != -1:
                text = text[start:]
            # Remove trailing fence
            end = text.rfind("}")
            if end == -1:
                end = text.rfind("]")
            if end != -1:
                text = text[: end + 1]

        # Try to find JSON object boundaries
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            logger.debug(f"LLM output was not a dict: {type(result)}")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse LLM JSON: {e}")
            return None

    def _store_llm_results(self, parsed: dict, msg: dict):
        """Dispatch LLM results to the appropriate storage targets.

        Populates conclusions, memories, peer metadata, and reputation.
        """
        session_id = msg.get("session_id")
        workspace_id = msg.get("workspace_id")
        if not session_id or not workspace_id:
            return

        self._store_llm_conclusions(parsed, session_id)
        self._store_llm_memories(parsed, workspace_id, session_id)
        self._update_peer_from_llm(parsed, workspace_id)
        self._store_agent_assessment(parsed, workspace_id)

    def _store_llm_conclusions(self, parsed: dict, session_id: str):
        """Store derived facts/insights as conclusions via crud."""
        conclusions = parsed.get("conclusions")
        if not conclusions or not isinstance(conclusions, list):
            return

        from embeddings import EmbeddingClient

        _ec = EmbeddingClient()
        import crud as _crud

        seen = set()
        try:
            conn = sqlite3.connect(str(Path.home() / ".evsmem" / "evsmem.db"))
            seen = set(row[0] for row in conn.execute(
                "SELECT content FROM conclusions"
            ).fetchall())
            conn.close()
        except Exception:
            pass

        for conc_text in conclusions:
            if not conc_text or not isinstance(conc_text, str):
                continue
            conc_text = conc_text.strip()
            if not conc_text or conc_text in seen:
                continue
            if len(conc_text) < 8 or conc_text.lower().startswith("something to remember"):
                continue
            seen.add(conc_text)
            try:
                emb = None
                if _ec.is_available():
                    try:
                        emb = _ec.embed(conc_text[:2000])
                    except Exception:
                        pass

                _crud.create_conclusion(
                    session_id=session_id,
                    content=conc_text,
                    metadata={
                        "category": "derived_insight",
                        "source": "deriver_llm",
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                    embedding=emb,
                )
            except Exception as e:
                logger.debug(f"Failed to store conclusion: {e}")

    def _store_llm_memories(self, parsed: dict, workspace_id: str, session_id: str):
        """Store long-term memory items with embeddings via crud.

        Supports three tiers from the LLM output:
          - hot_memories   -> type='hot_memory'   (critical, always injected)
          - cold_memories  -> type='cold_memory'  (detailed, on-demand)
          - memories       -> type='conversation_insight' (legacy generic)
        """
        from embeddings import EmbeddingClient

        _ec = EmbeddingClient()
        import crud as _crud

        tiers = (
            ("hot_memories", "hot_memory", 0.9),
            ("cold_memories", "cold_memory", 0.5),
            ("memories", "conversation_insight", 0.5),
        )

        for key, mem_type, default_imp in tiers:
            items = parsed.get(key)
            if not items or not isinstance(items, list):
                continue
            for mem in items:
                content = mem.get("content") if isinstance(mem, dict) else None
                if not content or not isinstance(content, str):
                    continue
                content = content.strip()
                if len(content) < 8:
                    continue
                low = content.lower()
                if low.startswith("something to remember") or low.startswith("user switched to a rust cli tool") or low.startswith("a concrete fact or preference") or low in (
                    "remember something", "a memory", "a memory to remember",
                ):
                    continue

                # Exact-content dedup: skip if this memory already exists for the workspace
                try:
                    dconn = sqlite3.connect(str(Path.home() / ".evsmem" / "evsmem.db"))
                    dup = dconn.execute(
                        "SELECT 1 FROM memories WHERE workspace_id=? AND lower(trim(content))=lower(trim(?)) LIMIT 1",
                        (workspace_id, content),
                    ).fetchone()
                    dconn.close()
                    if dup:
                        logger.info(f"Duplicate memory skipped: {len(content)} chars")
                        continue
                except Exception:
                    pass

                importance = float(mem.get("importance", default_imp))
                confidence = float(mem.get("confidence", 0.8))
                durability = float(mem.get("durability", 0.5))
                mem_semantic_type = str(mem.get("type", "conversation_insight"))
                try:
                    emb = None
                    if _ec.is_available():
                        try:
                            emb = _ec.embed(content[:2000])
                        except Exception:
                            pass

                    logger.info(f"Storing {mem_type}: {len(content)} chars")
                    _crud.create_memory(
                        workspace_id=workspace_id,
                        type=mem_type,
                        content=content,
                        importance=importance,
                        confidence=confidence,
                        durability=durability,
                        memory_type=mem_semantic_type,
                        source="deriver_llm",
                        metadata={"session_id": session_id, "tier": mem_type},
                        embedding=emb,
                    )
                except Exception as e:
                    logger.debug(f"Failed to store memory: {e}")

    def _update_peer_from_llm(self, parsed: dict, workspace_id: str):
        """Update user peer metadata with extracted user info."""
        import crud as _crud

        # Check if we have any user info to update
        user_name = parsed.get("user_name")
        user_mood = parsed.get("user_mood")
        user_preferences = parsed.get("user_preferences")
        if not user_name and not user_mood and not user_preferences:
            return

        # Find or create the user peer for this workspace
        try:
            peer = _crud.get_or_create_peer(workspace_id, "user")
            if not peer:
                return

            pid = peer["id"]
            meta_update = {}

            if user_name and isinstance(user_name, str):
                meta_update["user_name"] = user_name

            if user_mood and isinstance(user_mood, str):
                meta_update["user_mood"] = user_mood.lower()
                meta_update["mood_updated_at"] = datetime.now(timezone.utc).isoformat()

            if user_preferences and isinstance(user_preferences, list):
                meta_update["preferences"] = json.dumps(user_preferences)

            if meta_update:
                _crud.update_peer(pid, metadata=meta_update)
        except Exception as e:
            logger.debug(f"Failed to update peer metadata: {e}")

    def _store_agent_assessment(self, parsed: dict, workspace_id: str):
        """Store agent performance signals into reputation table."""
        assessment = parsed.get("agent_assessment")
        if not assessment or not isinstance(assessment, dict):
            return

        agent_name = assessment.get("agent_name")
        verdict = assessment.get("verdict")
        if not agent_name or not verdict:
            return

        import crud as _crud

        try:
            if verdict == "positive":
                _crud.record_success(
                    entity_type="agent",
                    entity_name=agent_name,
                    task_type="general",
                )
            elif verdict == "negative":
                _crud.record_failure(
                    entity_type="agent",
                    entity_name=agent_name,
                    task_type="general",
                )
            elif verdict == "neutral":
                _crud.record_dispatch(
                    entity_type="agent",
                    entity_name=agent_name,
                    task_type="general",
                )
        except Exception as e:
            logger.debug(f"Failed to record agent assessment: {e}")

    # ── DB operations (kept) ──

    # ── Public API ──

    def _sync_ev_sessions(self) -> int:
        """Pull new messages from ev-agent session DB into evsmem.
        Replaces the TypeScript auto-save hook that isn't working.
        Uses deterministic hash-based IDs and cursor-based pagination
        via deriver_state to prevent message duplication."""
        if not EV_SESSION_DB.exists():
            return 0

        # Read sync cursor from deriver_state
        last_rowid = 0
        try:
            state_conn = get_db()
            row = state_conn.execute(
                "SELECT value FROM deriver_state WHERE key='last_synced_rowid'"
            ).fetchone()
            if row:
                last_rowid = int(row["value"])
            state_conn.close()
        except Exception:
            last_rowid = 0

        try:
            src = sqlite3.connect(str(EV_SESSION_DB))
            src.row_factory = sqlite3.Row

            # Get all messages newer than the sync cursor, ordered by rowid ASC.
            # Every message is stored INSTANTLY (upsert by source id) — partial
            # streams get replaced by their final text on later polls.
            rows = src.execute("""
                SELECT m.rowid, m.id, m.session_id, m.data, m.time_created,
                       s.title AS session_title
                FROM message m
                JOIN session s ON m.session_id = s.id
                WHERE m.rowid > ?
                ORDER BY m.rowid ASC
                LIMIT 50
            """, (last_rowid,)).fetchall()

            if not rows:
                src.close()
                return 0

            # Phase 1: Parse messages and compute embeddings (BEFORE evsmem connection)
            # This prevents DB lock contention — the embedding call is a slow network
            # request that should not hold the evsmem DB connection open.
            prepared = []
            max_rowid = last_rowid
            settle_ms = 60000
            for row in rows:
                try:
                    data = json.loads(row["data"]) if isinstance(row["data"], str) else (row["data"] or {})
                except (json.JSONDecodeError, TypeError):
                    data = {}
                role = data.get("role", "user")
                content = ""

                # Get text from parts
                parts = src.execute(
                    "SELECT data FROM part WHERE message_id=? ORDER BY time_created ASC",
                    (row["id"],),
                ).fetchall()
                text_parts = []
                for p in parts:
                    try:
                        pd = json.loads(p["data"]) if isinstance(p["data"], str) else (p["data"] or {})
                    except (json.JSONDecodeError, TypeError):
                        pd = {}
                    if pd.get("type") in ("text", "reasoning"):
                        text = (pd.get("text") or "").strip()
                        if text and len(text) > 2:
                            text_parts.append(text[:2000])
                content = " | ".join(text_parts[:5]) if text_parts else str(data.get("summary", ""))[:500]

                # A message is "settled" once it is old enough that streaming is
                # guaranteed done. Everything is stored INSTANTLY, but the sync
                # cursor only advances past settled messages so in-flight
                # (partial) messages are re-fetched and upserted to their final
                # text on later polls.
                now_ms = int(time.time() * 1000)
                settled = (not row["time_created"]) or ((now_ms - row["time_created"]) >= settle_ms)

                if not content:
                    if settled:
                        max_rowid = max(max_rowid, row["rowid"])
                    else:
                        break  # still streaming; hold the cursor
                    continue

                # Embed from full content (BEFORE evsmem connection opens)
                msg_emb = None
                if os.getenv("DERIVER_SYNC_NO_EMBED") != "1":
                    try:
                        from embeddings import EmbeddingClient
                        _ec = EmbeddingClient()
                        if _ec.is_available():
                            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
                            with ThreadPoolExecutor(max_workers=1) as pool:
                                future = pool.submit(_ec.embed, content[:8000])
                                msg_emb = future.result(timeout=15)
                    except FutureTimeout:
                        logger.warning("Embedding timed out (15s) - skipping")
                    except Exception:
                        pass
                msg_emb_json = json.dumps(msg_emb) if msg_emb else None

                # Stable ID derived from the SOURCE message id — survives content
                # updates (reasoning-only -> final text) and prevents duplicates.
                deterministic_id = hashlib.sha256(row["id"].encode()).hexdigest()[:16]

                prepared.append({
                    "row": row,
                    "role": role,
                    "content": content,
                    "msg_emb_json": msg_emb_json,
                    "deterministic_id": deterministic_id,
                })

                if settled:
                    max_rowid = max(max_rowid, row["rowid"])
                else:
                    break  # stored now, but hold cursor until streaming completes

            src.close()

            if not prepared:
                # Persist cursor even when no messages were prepared
                try:
                    cursor_conn = get_db()
                    cursor_conn.execute(
                        """INSERT OR REPLACE INTO deriver_state (key, value, updated_at)
                           VALUES ('last_synced_rowid', ?, ?)""",
                        (str(max_rowid), datetime.now(timezone.utc).isoformat()),
                    )
                    cursor_conn.commit()
                    cursor_conn.close()
                except Exception:
                    pass
                return 0

            # Phase 2: Write each message with its own evsmem connection.
            # Each iteration opens a fresh connection, writes one message, commits,
            # and closes — preventing DB lock contention with other processes.
            synced_count = 0
            for prep in prepared:
                row = prep["row"]
                role = prep["role"]
                content = prep["content"]
                msg_emb_json = prep["msg_emb_json"]
                deterministic_id = prep["deterministic_id"]

                hcon = get_db()
                try:
                    # Get evsmem workspace
                    ws = hcon.execute(
                        "SELECT id FROM workspaces WHERE name='ev-agent' ORDER BY created_at ASC LIMIT 1"
                    ).fetchone()
                    if not ws:
                        continue
                    wid = ws["id"]

                    # Get or create a session for auto-saved messages
                    evsmem_sid = None
                    hs = hcon.execute(
                        "SELECT id FROM sessions WHERE workspace_id=? AND name='auto-save'",
                        (wid,),
                    ).fetchone()
                    if hs:
                        evsmem_sid = hs["id"]
                    else:
                        hcon.execute(
                            "INSERT INTO sessions (id, workspace_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                            (str(uuid4()), wid, "auto-save", "{}", datetime.now(timezone.utc).isoformat()),
                        )
                        hcon.commit()
                        hs = hcon.execute(
                            "SELECT id FROM sessions WHERE workspace_id=? AND name='auto-save'",
                            (wid,),
                        ).fetchone()
                        if hs:
                            evsmem_sid = hs["id"]

                    if not evsmem_sid:
                        continue

                    # Get or create peers
                    user_peer = hcon.execute(
                        "SELECT id FROM peers WHERE workspace_id=? AND name='user'",
                        (wid,),
                    ).fetchone()
                    if not user_peer:
                        hcon.execute(
                            "INSERT INTO peers (id, workspace_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                            (str(uuid4()), wid, "user", "{}", datetime.now(timezone.utc).isoformat()),
                        )
                        hcon.commit()
                        user_peer = hcon.execute(
                            "SELECT id FROM peers WHERE workspace_id=? AND name='user'",
                            (wid,),
                        ).fetchone()
                    user_pid = user_peer["id"] if user_peer else None

                    agent_peer = hcon.execute(
                        "SELECT id FROM peers WHERE workspace_id=? AND name='agent'",
                        (wid,),
                    ).fetchone()
                    if not agent_peer:
                        hcon.execute(
                            "INSERT INTO peers (id, workspace_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                            (str(uuid4()), wid, "agent", "{}", datetime.now(timezone.utc).isoformat()),
                        )
                        hcon.commit()
                        agent_peer = hcon.execute(
                            "SELECT id FROM peers WHERE workspace_id=? AND name='agent'",
                            (wid,),
                        ).fetchone()
                    agent_pid = agent_peer["id"] if agent_peer else None

                    peer_id = user_pid if role == "user" else agent_pid
                    if not peer_id:
                        continue

                    hcon.execute(
                        """INSERT INTO messages
                           (id, session_id, peer_id, content, role, message_type, metadata, embedding, created_at, is_processed)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                           ON CONFLICT(id) DO UPDATE SET
                             session_id=excluded.session_id,
                             peer_id=excluded.peer_id,
                             content=excluded.content,
                             role=excluded.role,
                             metadata=excluded.metadata,
                             embedding=excluded.embedding,
                             created_at=excluded.created_at,
                             is_processed=CASE WHEN messages.content = excluded.content THEN messages.is_processed ELSE 0 END""",
                        (
                            deterministic_id,
                            evsmem_sid,
                            peer_id,
                            f"[auto:{row['session_id'][:16]}] [{row['session_title'] or '?'}] {content[:12000]}",
                            role,
                            "message",
                            json.dumps({
                                "scope": "session",
                                "type": "auto-save",
                                "ev_session_id": row["session_id"],
                                "is_subagent": bool(row["session_title"] and "subagent" in str(row["session_title"]).lower()),
                            }),
                            msg_emb_json,
                            datetime.fromtimestamp(row["time_created"] / 1000,
                                                    tz=timezone.utc).isoformat() if row["time_created"] else datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    hcon.commit()
                    synced_count += 1
                finally:
                    hcon.close()

            # Persist sync cursor to deriver_state
            try:
                cursor_conn = get_db()
                cursor_conn.execute(
                    """INSERT OR REPLACE INTO deriver_state (key, value, updated_at)
                       VALUES ('last_synced_rowid', ?, ?)""",
                    (str(max_rowid), datetime.now(timezone.utc).isoformat()),
                )
                cursor_conn.commit()
                cursor_conn.close()
            except Exception as e:
                logger.warning(f"Failed to persist sync cursor: {e}")

            if synced_count > 0:
                logger.info(f"Deriver: synced {synced_count} ev-agent messages to evsmem")
            return synced_count
        except Exception as e:
            logger.warning(f"Deriver ev-session sync error: {e}")
            return 0

    def _analyze_session_reputation(self):
        """Analyze sessions for reputation updates - corrections, doom loops, and dispatch counting."""
        conn = get_db()
        try:
            # ── Dispatch counting ──
            # Count dispatches from evsmem sessions (each ended session with an agent is a dispatch)
            ended_sessions = conn.execute("""
                SELECT p.name as agent_name, s.id, COUNT(m.id) as msg_count
                FROM sessions s
                JOIN messages m ON m.session_id = s.id
                JOIN peers p ON m.peer_id = p.id
                WHERE s.created_at > strftime('%Y-%m-%dT%H:%M:%f', 'now', '-30 days') AND p.name != 'user'
                GROUP BY s.id, p.name
            """).fetchall()

            for s in ended_sessions:
                agent_name = s["agent_name"] or "unknown"
                if not agent_name:
                    continue

                # Record dispatch
                existing = conn.execute(
                    "SELECT id, dispatch_count FROM reputation WHERE entity_type='agent' AND entity_name=? AND task_type=''",
                    (agent_name,)
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE reputation SET dispatch_count = dispatch_count + 1, last_used_at = datetime('now') WHERE id=?",
                        (existing["id"],)
                    )
                else:
                    conn.execute(
                        "INSERT INTO reputation (id, entity_type, entity_name, task_type, dispatch_count, created_at) VALUES (?, 'agent', ?, '', 1, datetime('now'))",
                        (str(uuid4()), agent_name)
                    )

            conn.commit()

            # ── Existing correction/doom_loop detection ──
            sessions = conn.execute("""
                SELECT s.id, s.workspace_id
                FROM sessions s
                WHERE s.ended_at IS NOT NULL
                ORDER BY s.ended_at DESC
                LIMIT 5
            """).fetchall()

            correction_pattern = re.compile(
                r"\b(no|actually|wrong|don't|stop|that's not|incorrect|"
                r"try again|fix it|instead|not what|mistake|error|revert|undo)\b",
                re.IGNORECASE,
            )

            for session in sessions:
                sid = session["id"]
                wid = session["workspace_id"]

                # Get all tool-call type messages
                tool_rows = conn.execute(
                    """SELECT content, message_type, metadata, role
                       FROM messages WHERE session_id=? AND message_type='tool_call'""",
                    (sid,),
                ).fetchall()

                # Get user messages with adjacent context
                user_rows = conn.execute(
                    """SELECT content, role, created_at
                       FROM messages WHERE session_id=? AND role='user'
                       ORDER BY created_at ASC""",
                    (sid,),
                ).fetchall()

                # Detect doom loop: >10 consecutive tool calls without user interjection
                if len(tool_rows) > 10:
                    # Find any agents/skills referenced in this session
                    agent_names = conn.execute(
                        """SELECT DISTINCT p.name FROM messages m
                           JOIN peers p ON m.peer_id = p.id
                           WHERE m.session_id=? AND p.name='agent'""",
                        (sid,),
                    ).fetchall()
                    for agent_row in agent_names:
                        try:
                            import crud as _crud
                            _crud.record_doom_loop("agent", agent_row["name"], "general")
                        except Exception:
                            pass

                # Detect corrections in user messages
                for msg in user_rows:
                    if correction_pattern.search(msg["content"]):
                        # Find the agent peer for this session
                        agent_rows = conn.execute(
                            """SELECT DISTINCT p.name FROM messages m
                               JOIN peers p ON m.peer_id = p.id
                               WHERE m.session_id=? AND p.name='agent'""",
                            (sid,),
                        ).fetchall()
                        for agent_row in agent_rows:
                            try:
                                import crud as _crud
                                _crud.record_correction("agent", agent_row["name"], "general")
                            except Exception:
                                pass
                        break  # One correction event per session is enough
        except Exception as e:
            logger.warning(f"[reputation] Analysis error: {e}")
        finally:
            conn.close()

    def _write_recommendations_json(self):
        """Write pending recommendations to JSON file for TypeScript system prompt to read."""
        try:
            conn = get_db()
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE status='pending' ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            conn.close()
            recs = [dict(r) for r in rows]
            output = {
                "recommendations": recs,
                "instruction": "Call the generate_agent tool NOW for each pending recommendation above to create the specialist agent. Do not skip this step.",
            }
            (Path.home() / ".evsmem" / "recommendations.json").write_text(json.dumps(output, indent=2))
        except Exception as e:
            logger.warning(f"[auto-gen] Failed to write recommendations.json: {e}")

    def _check_auto_generate_trigger(self):
        """Auto-generate specialist agents when reputation shows repeated poor performance.
        Instead of calling generate_agent.py directly (which is invisible), writes
        recommendations that the main agent can see in the system prompt and act on
        via the generate_agent tool call (visible in TUI)."""
        conn = get_db()
        try:
            rows = conn.execute("""
                SELECT entity_name, task_type, dispatch_count, success_count, failure_count, correction_count
                FROM reputation
                WHERE entity_type='agent'
                  AND dispatch_count >= 3
                  AND task_type != ''
                  AND (CAST(success_count AS REAL) / CAST(MAX(dispatch_count, 1) AS REAL)) < 0.5
                ORDER BY dispatch_count DESC
                LIMIT 3
            """).fetchall()

            for r in rows:
                task_type = r["task_type"]
                agent_name = task_type.lower().replace(" ", "-").replace("_", "-") + "-specialist"

                # Check if agent already exists
                exists = conn.execute("SELECT 1 FROM agents WHERE name=?", (agent_name,)).fetchone()
                if exists:
                    continue

                # Check if recommendation already pending
                already = conn.execute(
                    "SELECT 1 FROM recommendations WHERE entity_name=? AND status='pending'",
                    (agent_name,),
                ).fetchone()
                if already:
                    continue

                logger.info(f"[auto-gen] Recommending specialist for {task_type} "
                            f"(dispatched {r['dispatch_count']}x, "
                            f"success rate {r['success_count']}/{r['dispatch_count']})")

                # Write recommendation for the main agent to act on (visible in TUI)
                rid = str(uuid4())
                conn.execute(
                    """INSERT INTO recommendations
                       (id, entity_type, entity_name, task_type, dispatch_count, failure_count, reason, status, created_at)
                       VALUES (?, 'agent', ?, ?, ?, ?, ?, 'pending', datetime('now'))""",
                    (rid, agent_name, task_type, r["dispatch_count"], r["failure_count"],
                     f"Agent '{agent_name}' is recommended for {task_type} tasks "
                     f"(dispatched {r['dispatch_count']}x, "
                     f"failed {r['failure_count']}x, "
                     f"success rate {r['success_count']}/{r['dispatch_count']}). "
                     f"Use the generate_agent tool to create this specialist."),
                )
                conn.commit()

                # Write notification for visibility
                conn.execute(
                    "INSERT INTO notifications (id, type, title, message, data) VALUES (?, 'agent_recommended', ?, ?, ?)",
                    (str(uuid4()),
                     f"Recommendation: {agent_name}",
                     f"Recommended specialist agent for {task_type} tasks",
                     json.dumps({"agent_name": agent_name, "task_type": task_type})),
                )
                conn.commit()

            # Update JSON file for TypeScript system prompt
            self._write_recommendations_json()

        except Exception as e:
            logger.warning(f"[auto-gen] Error: {e}")
        finally:
            conn.close()

    def _prune_hot_memories(self):
        """Cap the hot-memory pool so the always-injected system prompt stays small.
        If more than EVSMEM_HOT_CAP hot memories exist, demote the lowest-scored
        ones (importance x durability) to cold_memory."""
        cap = int(os.getenv("EVSMEM_HOT_CAP", "100"))
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT id FROM memories WHERE type='hot_memory' "
                "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC"
            ).fetchall()
            if len(rows) <= cap:
                return
            excess = [r["id"] for r in rows[cap:]]
            for mid in excess:
                conn.execute(
                    "UPDATE memories SET type='cold_memory', "
                    "metadata=json_set(COALESCE(metadata,'{}'), '$.tier', 'cold_memory', '$.demoted_from_hot', 'true') "
                    "WHERE id=?",
                    (mid,),
                )
            conn.commit()
            logger.info(f"[hot] Demoted {len(excess)} low-value hot memories to cold")
        except Exception as e:
            logger.warning(f"[hot] Prune error: {e}")
        finally:
            conn.close()

    def _write_hot_memories_json(self):
        """Write the top hot memories to ~/.evsmem/hot_memories.json so the
        ev-agent system prompt can inject them (same pattern as
        recommendations.json). Capped to 8, ranked by importance x durability."""
        try:
            conn = get_db()
            rows = conn.execute(
                "SELECT content, importance FROM memories WHERE type='hot_memory' "
                "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC LIMIT 40"
            ).fetchall()
            conn.close()
            items = [{"content": r["content"], "importance": r["importance"]} for r in rows]
            payload = {
                "memories": items,
                "instruction": "These are hot memories — always-relevant facts about the user, injected every session.",
            }
            (Path.home() / ".evsmem" / "hot_memories.json").write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning(f"[hot] Failed to write hot_memories.json: {e}")

    def run_once(self) -> int:
        """Sync ev-agent sessions to evsmem, process with LLM, then analyze."""
        synced = self._sync_ev_sessions()
        try:
            self._process_new_messages_with_llm()
        except Exception as e:
            logger.warning(f"LLM processing error: {e}")
        try:
            self._analyze_session_reputation()
        except Exception as e:
            logger.warning(f"Reputation analysis error: {e}")
        try:
            self._check_auto_generate_trigger()
        except Exception as e:
            logger.warning(f"Auto-generate trigger error: {e}")
        try:
            self._prune_hot_memories()
        except Exception as e:
            logger.warning(f"Hot-memory prune error: {e}")
        try:
            self._write_hot_memories_json()
        except Exception as e:
            logger.warning(f"Hot-memory JSON write error: {e}")
        return synced

    def run_forever(self):
        """Poll for new messages indefinitely."""
        self._running = True
        logger.info(f"Deriver started (poll every {POLL_INTERVAL}s)")
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Deriver error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)

    def stop(self):
        self._running = False


# ── Standalone entry point ──
def start_deriver_thread() -> Deriver:
    """Start the deriver in a background thread. Returns the deriver instance."""
    d = Deriver()
    t = Thread(target=d.run_forever, daemon=True)
    t.start()
    logger.info("Deriver thread started")
    return d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting evsmem Deriver (standalone)...")
    d = Deriver()
    d.run_forever()