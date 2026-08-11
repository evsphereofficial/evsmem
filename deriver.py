"""
evsmem Deriver — background session sync + batched memory curation.

Reads new messages from ev-agent sessions and syncs them into the evsmem
database for memory storage. A scheduler runs one LLM analysis pass at start
and then once per 60-minute window (EVSMEM_DERIVE_INTERVAL); each pass batches
ALL unprocessed messages from the past hour into a single LLM request and
stores the extracted rows across the memory tables. Message sync (upsert by
source id) keeps running between passes.

Usage:
  from deriver import Deriver
  d = Deriver()
  d.run_once()          # one full pass (sync + analysis + hot JSON)
  d.run_forever()       # scheduler: immediate pass, then hourly
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, Thread
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

# Legacy 5s poll interval — kept for backward compatibility, no longer used by
# the scheduler (see EVSMEM_DERIVE_INTERVAL below).
POLL_INTERVAL = float(os.getenv("DERIVER_POLL_INTERVAL", "5.0"))
MAX_FACTS_PER_BATCH = int(os.getenv("DERIVER_MAX_FACTS", "10"))

# Scheduler cadence: one LLM analysis pass immediately at start, then once per
# EVSMEM_DERIVE_INTERVAL seconds (default 60 minutes). Message sync keeps
# running every EVSMEM_SYNC_INTERVAL seconds between analysis passes.
DERIVE_INTERVAL = max(float(os.getenv("EVSMEM_DERIVE_INTERVAL", "3600")), 5.0)
SYNC_INTERVAL = max(float(os.getenv("EVSMEM_SYNC_INTERVAL", "30")), 5.0)

# Batch window (hours) used on the FIRST run before any analysis cursor exists.
BATCH_WINDOW_HOURS = float(os.getenv("EVSMEM_BATCH_HOURS", "1"))

# Remote LLM (DeepSeek) is preferred when available; the local GGUF LLMClient
# is the fallback. DeepSeekClient is owned by llm_client.py (parallel work) —
# import defensively so this module works with or without it.
try:
    from llm_client import DeepSeekClient
except ImportError:  # pragma: no cover - depends on parallel work in llm_client.py
    DeepSeekClient = None


# ── LLM Analysis Prompt ──

LLM_ANALYSIS_PROMPT = """\
You are a meticulous memory-extraction engine for a conversation between a user and an AI coding assistant. Your ONLY job is to extract durable facts. Analyze the message carefully and output ONE valid JSON object.

Message: "{content}"

Required JSON schema:
{{
  "user": {{"name": "extracted name or null", "age": "extracted age or null", "location": "extracted location or null", "username": "extracted username or null", "email": "extracted email or null", "occupation": "extracted occupation or null", "education": "extracted education or null", "interests": "extracted interests or null", "mood": "extracted mood or null", "github": "extracted github handle or null"}},
  "user_name": "extracted name or null",
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
  "behaviours": [
    {{"content": "a recurring behavior pattern of the user", "importance": 0.9}},
    {{"content": "another recurring behavior pattern", "importance": 0.85}}
  ],
  "preferences": [
    {{"content": "a stated preference of the user", "importance": 0.9}}
  ],
  "rules": [
    {{"content": "an explicit instruction/rule the user wants followed ALWAYS (e.g. 'make small git commits')", "importance": 1.0}}
  ],
  "conclusions": ["a new insight about the user or project, phrased uniquely"],
  "agent_assessment": {{"agent_name": "mentioned agent or ''", "verdict": "positive|negative|neutral", "detail": "one-line assessment"}}
}}

Detailed rules:
1. BEHAVIOURS = ONLY the user's recurring EMOTIONAL / BEHAVIORAL patterns: how they react and behave (frustration, anger, impatience, excitement, temper, moods, how they communicate under stress). Example: "user gets frustrated easily and uses profane language when code fails". NEVER put work-style preferences, project facts, or tool choices here — those belong in PREFERENCES. Max 4 per message.
2. PREFERENCES = ONLY stated likes/dislikes/choices for things: technology (Python vs TypeScript), UI (dark mode, no emojis), interaction style (direct/no-fluff responses), how they like to work (small commits, basedev branch). Example: "user prefers Python over TypeScript". NEVER store ev-agent architecture/config/test instructions here. Max 8 per message.
3. RULES = ONLY hard, explicit instructions the user gives about how the agent should behave toward them — usually phrased with always/never/do not/must. Examples: "never use force delete", "make small git commits to inflate commits", "do not create files without permission", "no emojis". NEVER store ev-agent's own workflow/architecture/config/test/token/UI instructions here. Max 5 per message.
4. HOT_MEMORIES = every user info fact NOT already covered by behaviours/preferences/rules: identity, mood, work style, constraints, relationships, decisions. Always injected. Max 8 per message.
5. COLD_MEMORIES = everything else — the EXTRAS, injected on demand: project/technical details, what the user is building, the task, the topic, code/architecture/approach, technical decisions, tooling, versions, bugs, design rationale. MOST IMPORTANTLY capture WHAT THE USER IS TALKING ABOUT in detail. Be DETAILED — project names, exact terms, versions, file paths, 2-3 sentences of context. importance 0.3-0.7.
6. Every content value MUST be derived strictly from the message. Never invent facts, never output generic placeholders, never copy these instructions.
7. CRITICAL: NEVER store ev-agent's OWN internal workflow/architecture/config/test instructions as rules, preferences, or behaviours — e.g. anything mentioning MANDATORY RULE / MANDATORY WORKFLOW, plan-architect, router, routing token, dispatch, subagent, task(), verify(), evaluate(), system prompt, Domain Boundary, the orchestrator workflow, rule numbering ("rule 8"), test loops, integration-test designs, token-expiry design, difficulty-gated planning, or UI/config rendering requests ("render as a compact single line"). Only store instructions/preferences the HUMAN USER gave about how THEY want to work with the agent.
8. Per-memory attributes:
   - importance: 0.0 (trivial) to 1.0 (must-never-forget).
   - confidence: how sure we are this fact is true, 0.0-1.0 (default 0.8).
   - durability: how long this fact is likely to stay true, 0.0 (transient) to 1.0 (permanent).
   - type: user | preference | project | decision | architecture_decision | tooling | environment | event | debugging_event | conversation_insight.
9. If the message has no memorable facts, output empty arrays for all fields.
10. Return ONLY the JSON object. No markdown fences, no commentary."""


# ── Batch-mode prompt (one analysis request per hour-window) ──

LLM_BATCH_PREAMBLE = """\
You are analyzing a BATCH of {count} messages exchanged during the last hour between a human user and an AI coding assistant. Produce ONE valid JSON object covering the WHOLE batch: array items (hot_memories, cold_memories, behaviours, preferences, rules, conclusions) may draw facts from ANY of the messages. The per-message maximums scale with the batch size (e.g. up to 4 behaviours per message → up to {count}*4 behaviours total). Never invent facts, never output generic placeholders, and NEVER store ev-agent's own internal workflow/architecture/config/test instructions."""


# Additional system context appended to every batch analysis. It gives the
# model the last hour of existing memory state so it can detect duplicates and
# fix wrong-table placements instead of blindly re-adding rows.
LLM_CONTEXT_SECTION = """\
You will also receive the existing memory state below. Use it to detect duplicates and correct wrong-table placements:
- If an extracted fact already exists (same meaning), do NOT re-add it.
- If a fact belongs in another table, place it in the correct table, or use the update_memory_row tool to move/update the existing row (row ids are shown below).
- If an existing row is a duplicate of another or is obsolete, use delete_memory_row (with a reason).
- NEVER store ev-agent internal workflow details in any table.

EXISTING MEMORY STATE:
--- Last 60 minutes of existing memories (id | content | importance | type) ---
{recent_memories}
--- Latest memory (id | content | importance | type) ---
{latest_memory}
--- Latest behaviours (id | content | importance) ---
{behaviours}
--- User profile ---
{user}
--- Latest preferences (id | content | importance) ---
{preferences}
--- Latest rules (id | content | importance) ---
{rules}"""


# ── OpenAI function-calling tools for memory curation ──

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_memory_rows",
            "description": (
                "Insert one or more NEW memory rows across the memory tables. "
                "Use only for facts extracted from the batch that do not already "
                "exist in the provided memory state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "table": {
                                    "type": "string",
                                    "enum": ["memories", "behaviour", "preferences", "rules", "users", "agent_written_memory"],
                                },
                                "content": {"type": "string", "description": "The fact to store (for table=users this is a summary line; structured user fields are set via the name/age/... fields)."},
                                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "durability": {"type": "number", "minimum": 0, "maximum": 1},
                                "type": {"type": "string", "description": "For memories: 'hot_memory'/'cold_memory' tier or a semantic type (project, user, decision...). For others: optional."},
                                "memory_type": {"type": "string"},
                            },
                            "required": ["table", "content"],
                        },
                    }
                },
                "required": ["rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory_row",
            "description": (
                "Update an existing memory row (content, importance, type, ...) "
                "or MOVE it to a different table when it was placed wrong. "
                "Always provide a short reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["memories", "behaviour", "preferences", "rules", "users", "agent_written_memory"],
                    },
                    "row_id": {"type": "string", "description": "The id of the existing row (shown in the memory state context)."},
                    "fields": {"type": "object", "description": "Columns to change, e.g. {\"content\": \"...\", \"importance\": 0.9}."},
                    "move_to_table": {
                        "type": ["string", "null"],
                        "enum": [None, "memories", "behaviour", "preferences", "rules", "agent_written_memory"],
                        "description": "If set, move the row to this table instead of updating in place.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["table", "row_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory_row",
            "description": (
                "Permanently delete an existing memory row (duplicate of another "
                "row, wrong-table placeholder, or obsolete/contradicted fact). "
                "Always provide a reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["memories", "behaviour", "preferences", "rules", "users", "agent_written_memory"],
                    },
                    "row_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["table", "row_id", "reason"],
            },
        },
    },
]


def _as_dict(v):
    """Coerce a dict / pydantic model / arbitrary object into a plain dict."""
    if isinstance(v, dict):
        return v
    for attr in ("model_dump", "dict"):
        fn = getattr(v, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    if hasattr(v, "__dict__"):
        return dict(v.__dict__)
    return {}


def _empty_tool_stats() -> dict:
    return {"inserted": {}, "updated": 0, "moved": 0, "deleted": 0,
            "applied": False, "rounds": 0}


def _is_workflow_noise(text):
    """True if the content looks like ev-agent's internal workflow/architecture
    instructions (system-prompt or subagent boilerplate) that must NEVER be
    stored as user rules/preferences/behaviours."""
    low = (text or "").lower()
    patterns = (
        "mandatory rule", "mandatory workflow", "plan-architect", "plan_architect",
        "routing token", "selection token", "plan token", "phase system",
        "domain boundary", "you are a specialist agent", "dispatch progress",
        "dispatched x of", "evaluator required", "call the router", "call evaluate",
        "call generate_agent", "do not skip this step", "bash tool is not available",
        "skipping the chain", "workflow is enforced", "token enforcement",
        "rule 8", "rule 9", "rule 10", "rule numbering", "difficulty-gated planning",
        "loop test", "integration test design", "test design in the final message",
        "compact single line", "collapsed ui", "token-expiry", "token expiration",
        "linked to the completion of to-do", "to-do items", "system prompt must be strict",
        "the agent must use search memory", "use memory tools as much as possible",
        "render as", "instead of the generic tool", "bypass complex orchestration",
    )
    return any(p in low for p in patterns)


# The full evsmem schema is ensured ONCE per process via crud.ensure_schema()
# (thread-safe, shared with crud.get_db) — _init_schema performs writes that
# would otherwise cause "database is locked" contention on every connection.


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    # busy_timeout MUST be set before any statement that can take a write lock
    # (journal_mode / CREATE TABLE / ALTER), or those fail instantly with
    # "database is locked" under concurrent writers.
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
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
    # Ensure the full evsmem schema exists ONCE per process (shared guard).
    try:
        import crud as _crud
        _crud.ensure_schema(conn)
    except Exception:
        pass
    return conn


class Deriver:
    """Background fact extractor for evsmem."""

    def __init__(self):
        self._last_processed_rowid: Optional[int] = None
        self._running = False
        # Guards analysis passes so two passes never run concurrently (e.g. a
        # slow LLM batch overlapping the next scheduled window).
        self._analysis_lock = Lock()

    # ── Batch message tracking ──

    def _get_hour_batch(self) -> list[dict]:
        """All unprocessed messages accumulated since the previous successful
        analysis pass — a rolling 1-hour window (no LIMIT), oldest first.

        On the first run the window defaults to the last `BATCH_WINDOW_HOURS`
        (default 1). Messages younger than 30 seconds are excluded so streamed
        messages settle first. The window cursor (`deriver_state.last_analysis_at`)
        only advances on a successful pass, so after an LLM failure the same
        batch is left unprocessed and retried on the next run.
        """
        window_hours = float(os.getenv("EVSMEM_BATCH_HOURS", str(BATCH_WINDOW_HOURS)))
        conn = get_db()
        try:
            cursor = conn.execute(
                "SELECT value FROM deriver_state WHERE key='last_analysis_at'"
            ).fetchone()

            sql = """SELECT m.rowid, m.id, m.content, m.role, m.session_id,
                            s.workspace_id, m.metadata
                     FROM messages m
                     JOIN sessions s ON m.session_id = s.id
                     WHERE m.is_processed = 0 AND m.content != ''
                       AND m.created_at <= ?"""
            params: list = [(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()]
            if cursor and cursor["value"]:
                # Rolling window: messages created after the last successful pass.
                sql += " AND m.created_at > ?"
                params.append(cursor["value"])
            else:
                # First run: the last hour (or EVSMEM_BATCH_HOURS if set).
                sql += " AND m.created_at >= ?"
                params.append((datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat())
            sql += " ORDER BY m.rowid ASC"

            rows = conn.execute(sql, params).fetchall()

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

    def _get_workspace_batches(self) -> dict[str, list[dict]]:
        """Group the hour batch by workspace (each workspace is analyzed
        separately). In practice all synced messages share the 'ev-agent'
        workspace, but this keeps the pipeline correct if that ever changes."""
        batches: dict[str, list[dict]] = {}
        for msg in self._get_hour_batch():
            batches.setdefault(msg["workspace_id"], []).append(msg)
        return batches

    def _mark_message_processed(self, rowid: int):
        """Mark a single message as processed (is_processed = 1)."""
        conn = get_db()
        try:
            conn.execute("UPDATE messages SET is_processed = 1 WHERE rowid = ?", (rowid,))
            conn.commit()
        finally:
            conn.close()

    def _mark_messages_processed(self, rowids: list[int]):
        """Mark a whole batch processed in one UPDATE (post-pass success path)."""
        if not rowids:
            return
        conn = get_db()
        try:
            for i in range(0, len(rowids), 500):
                chunk = rowids[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                conn.execute(
                    f"UPDATE messages SET is_processed = 1 WHERE rowid IN ({placeholders})",
                    chunk,
                )
            conn.commit()
        finally:
            conn.close()

    def _advance_analysis_cursor(self):
        """Record the successful analysis time so the next pass only picks up
        messages written after it (idempotent rolling window)."""
        now = datetime.now(timezone.utc).isoformat()
        conn = get_db()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO deriver_state (key, value, updated_at)
                   VALUES ('last_analysis_at', ?, ?)""",
                (now, now),
            )
            conn.commit()
        finally:
            conn.close()

    # ── LLM Processing Pipeline ──

    def _get_llm(self):
        """Return (engine, llm): remote DeepSeek when available (with tool
        calling), otherwise the local GGUF LLMClient fallback."""
        from llm_client import LLMClient

        if DeepSeekClient is not None:
            try:
                llm = DeepSeekClient()
                available = getattr(llm, "is_available", None)
                if available is None or available():
                    logger.info("Using remote DeepSeek client for memory analysis")
                    return "remote", llm
            except Exception as e:
                logger.warning(f"DeepSeek client failed to initialize: {e}")

        llm = LLMClient()
        if llm.is_available():
            return "local", llm
        return None, None

    def _process_new_messages_with_llm(self, stats: dict) -> bool:
        """Analyze the hour-batch through the LLM and store the extracted rows.

        `stats` is filled in-place (engine, messages_processed, json_rows,
        tool_inserted/updated/moved/deleted, tool_rounds) so the caller can
        emit one structured log line per run. Returns True on success.

        Idempotency contract: messages are marked is_processed=1 and the
        analysis cursor is advanced ONLY after the whole pass succeeds; on LLM
        failure (exception, empty reply, unparseable JSON) the batch is left
        unprocessed and retried on the next run.
        """
        engine, llm = self._get_llm()
        stats["engine"] = engine or "none"
        if llm is None:
            logger.debug("LLM not available, skipping message processing")
            return False

        batches = self._get_workspace_batches()
        total = sum(len(v) for v in batches.values())
        stats["messages_processed"] = total
        if total == 0:
            # Nothing new; roll the window forward so it doesn't grow unbounded.
            self._advance_analysis_cursor()
            return True

        try:
            for workspace_id, msgs in batches.items():
                parsed, tool_stats = self._analyze_batch_with_llm(llm, msgs, workspace_id, engine)
                self._accumulate_tool_stats(stats, tool_stats)
                if parsed:
                    json_rows = self._count_json_rows(parsed)
                    for table, n in json_rows.items():
                        stats["json_rows"][table] = stats["json_rows"].get(table, 0) + n
                    self._store_batch_llm_results(parsed, msgs)
                elif not tool_stats["applied"]:
                    # LLM produced neither parseable JSON nor tool actions.
                    raise RuntimeError("LLM returned no parseable analysis and no tool calls")

            # Post-pass success path: only now mark the whole batch processed.
            all_rowids = [m["rowid"] for batch in batches.values() for m in batch]
            self._mark_messages_processed(all_rowids)
            self._advance_analysis_cursor()
            return True
        except Exception:
            # Leave messages unprocessed and the cursor unadvanced → retried next run.
            raise

    def _accumulate_tool_stats(self, stats: dict, tool_stats: dict):
        for table, n in (tool_stats.get("inserted") or {}).items():
            stats["tool_inserted"][table] = stats["tool_inserted"].get(table, 0) + n
        stats["tool_updated"] += tool_stats.get("updated", 0)
        stats["tool_moved"] += tool_stats.get("moved", 0)
        stats["tool_deleted"] += tool_stats.get("deleted", 0)
        stats["tool_rounds"] = max(stats.get("tool_rounds", 0), tool_stats.get("rounds", 0))

    def _format_batch_for_prompt(self, msgs: list[dict]) -> str:
        """Render the batch as one prompt block (strips [auto:..] prefixes and
        adds role/subagent notes, mirroring the single-message path)."""
        parts = []
        for i, m in enumerate(msgs, 1):
            content = (m.get("content") or "").strip()
            if content.startswith("[auto:"):
                idx = content.find("] ", 6)
                if idx != -1:
                    content = content[idx + 2:].strip()
            role = m.get("role", "user")
            if m.get("is_subagent"):
                content = (
                    "[NOTE: This message is from an INTERNAL SUBAGENT session, NOT a "
                    "direct statement from the human user. Do NOT infer the user's name, "
                    "preferences, identity, or mood from it. Only extract concrete "
                    "technical/project facts if clearly present.]\n" + content
                )
            elif role == "assistant":
                content = (
                    "[NOTE: This is the AI ASSISTANT's response. It contains the DETAILED "
                    "technical/project context of what is being built. Do NOT attribute "
                    "these statements as the human user's own words or preferences.]\n" + content
                )
            parts.append(f"--- Message {i} (role={role}) ---\n{content}")
        return "\n\n".join(parts)

    def _build_memory_context(self, workspace_id: str) -> str:
        """Snapshot the existing memory state (last hour of memories, latest
        memory, behaviours, user profile, preferences, rules) so the model can
        detect duplicates and fix wrong-table placements."""
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn = get_db()
        try:
            recent = conn.execute(
                """SELECT id, content, importance, type FROM memories
                   WHERE workspace_id=? AND created_at >= ?
                   ORDER BY created_at DESC LIMIT 15""",
                (workspace_id, one_hour_ago),
            ).fetchall()
            latest = conn.execute(
                """SELECT id, content, importance, type FROM memories
                   WHERE workspace_id=? ORDER BY created_at DESC LIMIT 1""",
                (workspace_id,),
            ).fetchone()
            behaviours = conn.execute(
                """SELECT id, content, importance FROM behaviour
                   WHERE workspace_id=? ORDER BY created_at DESC LIMIT 5""",
                (workspace_id,),
            ).fetchall()
            prefs = conn.execute(
                """SELECT id, content, importance FROM preferences
                   WHERE workspace_id=? ORDER BY created_at DESC LIMIT 5""",
                (workspace_id,),
            ).fetchall()
            rules = conn.execute(
                """SELECT id, content, importance FROM rules
                   WHERE workspace_id=? ORDER BY created_at DESC LIMIT 5""",
                (workspace_id,),
            ).fetchall()
            user = conn.execute(
                """SELECT name, age, location, username, email, occupation,
                          education, interests, mood, github
                   FROM users WHERE workspace_id=?""",
                (workspace_id,),
            ).fetchone()
        finally:
            conn.close()

        def _fmt(rows, cols):
            out = []
            for r in rows:
                out.append(" | ".join(str(r[c] or "") for c in cols))
            return "\n".join(out) if out else "(none)"

        return LLM_CONTEXT_SECTION.format(
            recent_memories=_fmt(recent, ("id", "content", "importance", "type")),
            latest_memory=(
                f"{latest['id']} | {latest['content']} | {latest['importance']} | {latest['type']}"
                if latest else "(none)"
            ),
            behaviours=_fmt(behaviours, ("id", "content", "importance")),
            user=json.dumps(dict(user), default=str) if user else "(unknown)",
            preferences=_fmt(prefs, ("id", "content", "importance")),
            rules=_fmt(rules, ("id", "content", "importance")),
        )

    def _analyze_batch_with_llm(self, llm, msgs: list[dict], workspace_id: str, engine: str):
        """Build the batch prompt, call the LLM (with tool calling when the
        client supports it), and return (parsed_json_or_None, tool_stats)."""
        content = self._format_batch_for_prompt(msgs)
        if len(content) < 10:
            return None, _empty_tool_stats()

        context = self._build_memory_context(workspace_id)
        batch_cap = int(os.getenv("EVSMEM_BATCH_MAX_CHARS", "24000"))
        truncated = content if len(content) <= batch_cap else content[:batch_cap] + "\n...[truncated]"
        user_prompt = (
            LLM_BATCH_PREAMBLE.format(count=len(msgs))
            + "\n\n"
            + LLM_ANALYSIS_PROMPT.format(content=truncated)
        )
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_llm_with_tools(llm, messages, workspace_id, engine)

    def _call_llm_with_tools(self, llm, messages: list[dict], workspace_id: str,
                             engine: str, max_rounds: int = 3):
        """Run the batch through generate_with_tools when available (applying
        tool calls atomically and feeding results back for up to `max_rounds`
        rounds), else fall back to plain generate + JSON parse.

        Returns (parsed_json_or_None, tool_stats).
        """
        tool_stats = _empty_tool_stats()
        gen = getattr(llm, "generate_with_tools", None)
        max_tokens = 4096 if engine == "remote" else 2048

        if gen is None:
            # Plain chat completion (local GGUF fallback): ask for the JSON
            # schema output directly; no tool loop.
            raw = llm.generate(messages=messages, max_tokens=max_tokens, temperature=0.1)
            if not raw:
                return None, tool_stats
            return self._parse_llm_output(raw), tool_stats

        conversation = list(messages)
        final_content = ""
        for rnd in range(max_rounds + 1):
            try:
                try:
                    resp = gen(messages=conversation, tools=MEMORY_TOOLS,
                               tool_choice="auto", max_tokens=max_tokens, temperature=0.1)
                except TypeError:
                    resp = gen(messages=conversation, tools=MEMORY_TOOLS,
                               max_tokens=max_tokens, temperature=0.1)
            except Exception as e:
                logger.warning(f"generate_with_tools round {rnd} failed: {e}")
                break

            content, tool_calls = self._normalize_llm_response(resp)
            final_content = content or final_content

            if not tool_calls:
                break

            tool_stats["rounds"] += 1
            stats, results = self._apply_tool_calls(tool_calls, workspace_id)
            tool_stats["applied"] = tool_stats["applied"] or bool(stats["inserted"] or stats["updated"] or stats["moved"] or stats["deleted"])
            for table, n in stats["inserted"].items():
                tool_stats["inserted"][table] = tool_stats["inserted"].get(table, 0) + n
            tool_stats["updated"] += stats["updated"]
            tool_stats["moved"] += stats["moved"]
            tool_stats["deleted"] += stats["deleted"]

            # Feed the tool results back so the model can finish/continue.
            formatted_calls = []
            for tc in tool_calls:
                tcd = _as_dict(tc)
                fn = tcd.get("function") or tcd
                cid = tcd.get("id") or f"call_{uuid4().hex[:8]}"
                name = fn.get("name") or tcd.get("name") or ""
                formatted_calls.append({
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": fn.get("arguments") or tcd.get("arguments") or "{}",
                    },
                })
            conversation.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": formatted_calls,
            })
            for tc in formatted_calls:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": json.dumps(results.get(tc["id"], {"ok": False, "error": "unknown tool call"}),
                                         default=str),
                })

            if rnd >= max_rounds - 1:
                break

        if not final_content.strip():
            return None, tool_stats
        return self._parse_llm_output(final_content), tool_stats

    def _normalize_llm_response(self, resp):
        """Extract (content, tool_calls) from common generate_with_tools return
        shapes: OpenAI chat-completion dict, {content, tool_calls} dict, plain
        string, or a model object."""
        if resp is None:
            return "", []
        if isinstance(resp, str):
            return resp, []
        resp = _as_dict(resp)
        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            msg = resp["choices"][0].get("message") or {}
            return msg.get("content") or "", msg.get("tool_calls") or []
        if resp.get("message") and isinstance(resp["message"], dict):
            return resp["message"].get("content") or "", resp["message"].get("tool_calls") or []
        return resp.get("content") or "", resp.get("tool_calls") or []

    def _apply_tool_calls(self, tool_calls, workspace_id: str):
        """Execute the model's tool_calls against crud atomically (single
        SQLite transaction: all-or-nothing). Embeddings are computed BEFORE the
        transaction opens so no slow network/GPU call holds the DB lock.

        Returns (aggregate_stats, per_call_results)."""
        import crud as _crud
        from embeddings import EmbeddingClient

        stats = {"inserted": {}, "updated": 0, "moved": 0, "deleted": 0}
        calls = []
        for tc in tool_calls:
            tcd = _as_dict(tc)
            fn = tcd.get("function") or tcd
            name = fn.get("name") or tcd.get("name")
            if not name:
                continue
            try:
                args = fn.get("arguments") or tcd.get("arguments") or "{}"
                args = json.loads(args) if isinstance(args, str) else (args or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append((name, args, tcd.get("id") or f"call_{uuid4().hex[:8]}"))

        if not calls:
            return stats, {}

        # Embed all new/updated contents before opening the transaction.
        _ec = EmbeddingClient()
        def _embed(text):
            try:
                if _ec.is_available():
                    return _ec.embed((text or "")[:2000])
            except Exception:
                pass
            return None
        for name, args, _ in calls:
            if name == "add_memory_rows":
                for row in args.get("rows") or []:
                    if isinstance(row, dict) and row.get("content") and "embedding" not in row:
                        row["embedding"] = _embed(row["content"])
            elif name == "update_memory_row":
                fields = args.get("fields")
                if isinstance(fields, dict) and fields.get("content") and "embedding" not in fields:
                    fields["embedding"] = _embed(fields["content"])

        results = {}
        conn = _crud.get_db()
        try:
            conn.execute("BEGIN")
            for name, args, call_id in calls:
                if name == "add_memory_rows":
                    inserted_here = 0
                    for row in args.get("rows") or []:
                        if not isinstance(row, dict) or not row.get("table"):
                            continue
                        _crud.add_memory_row(row["table"], workspace_id, row, conn=conn)
                        inserted_here += 1
                        stats["inserted"][row["table"]] = stats["inserted"].get(row["table"], 0) + 1
                    results[call_id] = {"ok": True, "inserted": inserted_here}
                elif name == "update_memory_row":
                    res = _crud.update_memory_row(
                        args.get("table"), args.get("row_id"),
                        args.get("fields") or {}, args.get("move_to_table"),
                        args.get("reason") or "", conn=conn,
                    )
                    if res.get("ok"):
                        if res.get("action") == "moved":
                            stats["moved"] += 1
                        elif res.get("action") == "updated":
                            stats["updated"] += 1
                    results[call_id] = res
                elif name == "delete_memory_row":
                    res = _crud.delete_memory_row(
                        args.get("table"), args.get("row_id"),
                        args.get("reason") or "", conn=conn,
                    )
                    if res.get("ok"):
                        stats["deleted"] += 1
                    results[call_id] = res
                else:
                    results[call_id] = {"ok": False, "error": f"unknown tool '{name}'"}
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning(f"Tool-call transaction failed, rolled back: {e}")
            raise

        return stats, results

    def _store_batch_llm_results(self, parsed: dict, msgs: list[dict]):
        """Dispatch a batch-level analysis result to the storage targets using
        the first session id for metadata (the store functions already dedup)."""
        if not msgs:
            return
        first = msgs[0]
        session_ids = []
        for m in msgs:
            sid = m.get("session_id")
            if sid and sid not in session_ids:
                session_ids.append(sid)
        msg = {
            "session_id": session_ids[0] if session_ids else first.get("session_id"),
            "workspace_id": first.get("workspace_id"),
            "_batch_size": len(msgs),
            "_batch_session_ids": session_ids,
        }
        self._store_llm_results(parsed, msg)

    def _count_json_rows(self, parsed: dict) -> dict:
        """Count rows the JSON analysis asked us to insert, per table."""
        counts = {}
        mapping = (
            ("hot_memories", "memories"), ("cold_memories", "memories"),
            ("memories", "memories"), ("behaviours", "behaviour"),
            ("preferences", "preferences"), ("rules", "rules"),
            ("conclusions", "conclusions"),
        )
        for key, table in mapping:
            items = parsed.get(key)
            if isinstance(items, list):
                counts[table] = counts.get(table, 0) + len(items)
        user = parsed.get("user")
        if isinstance(user, dict) and any(user.values()):
            counts["users"] = counts.get("users", 0) + 1
        return counts

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
            max_tokens=2048,
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
        self._store_llm_classified(parsed, workspace_id, session_id)
        self._store_user_info(parsed, workspace_id)
        self._update_peer_from_llm(parsed, workspace_id)
        self._store_agent_assessment(parsed, workspace_id)

    def _store_user_info(self, parsed: dict, workspace_id: str):
        """Upsert structured user info into the users table."""
        user = parsed.get("user")
        if not user or not isinstance(user, dict):
            return
        def _g(k):
            return str(user.get(k) or "").strip() or None
        vals = {
            "name": _g("name"), "age": _g("age"), "location": _g("location"),
            "username": _g("username"), "email": _g("email"),
            "occupation": _g("occupation"), "education": _g("education"),
            "interests": _g("interests"), "mood": _g("mood"), "github": _g("github"),
        }
        if not any(vals.values()):
            return
        import crud as _crud
        try:
            _crud.upsert_user(workspace_id, **vals)
        except Exception as e:
            logger.debug(f"Failed to store user info: {e}")

    def _store_llm_classified(self, parsed: dict, workspace_id: str, session_id: str):
        """Store behaviours / preferences / rules into their dedicated tables
        (small, capped, always-injected — the most important user-adaptation
        memory)."""
        from embeddings import EmbeddingClient
        _ec = EmbeddingClient()
        import crud as _crud

        for key, store_fn, mem_type, default_imp in (
            ("behaviours", _crud.create_behaviour, "behaviour", 0.9),
            ("preferences", _crud.create_preference, "preference", 0.9),
            ("rules", _crud.create_rule, "rule", 1.0),
        ):
            items = parsed.get(key)
            if not items or not isinstance(items, list):
                continue
            for mem in items:
                content = mem.get("content") if isinstance(mem, dict) else None
                if not content or not isinstance(content, str):
                    continue
                content = content.strip()
                if len(content) < 5:
                    continue
                if _is_workflow_noise(content):
                    logger.info(f"Skipping workflow noise ({mem_type}): {content[:60]}")
                    continue
                importance = float(mem.get("importance", default_imp))
                try:
                    emb = None
                    if _ec.is_available():
                        try:
                            emb = _ec.embed(content[:2000])
                        except Exception:
                            pass
                    logger.info(f"Storing {mem_type}: {len(content)} chars")
                    store_fn(
                        workspace_id=workspace_id,
                        content=content,
                        importance=importance,
                        confidence=float(mem.get("confidence", 0.9)),
                        durability=float(mem.get("durability", 0.9)),
                        metadata={"session_id": session_id},
                        embedding=emb,
                    )
                except Exception as e:
                    logger.debug(f"Failed to store {mem_type}: {e}")

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

        # Load existing memory contents once (recent 300) for normalized + fuzzy dedup.
        def _norm(t):
            import re as _re
            s = _re.sub(r"[.!?]+$", "", (t or "").strip().lower())
            return _re.sub(r"\s+", " ", s)

        existing_contents = []
        try:
            dconn = sqlite3.connect(str(Path.home() / ".evsmem" / "evsmem.db"))
            existing_contents = [r[0] for r in dconn.execute(
                "SELECT content FROM memories WHERE workspace_id=? "
                "ORDER BY created_at DESC LIMIT 300",
                (workspace_id,),
            ).fetchall()]
            dconn.close()
        except Exception:
            pass

        import difflib

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
                if _is_workflow_noise(content):
                    logger.info(f"Skipping workflow noise ({mem_type}): {content[:60]}")
                    continue

                # Normalized + fuzzy dedup: skip near-duplicate memories
                norm = _norm(content)
                dup = False
                for ec in existing_contents:
                    ec_norm = _norm(ec)
                    if ec_norm == norm or difflib.SequenceMatcher(None, ec_norm, norm).ratio() >= 0.85:
                        dup = True
                        break
                if dup:
                    logger.info(f"Duplicate memory skipped: {len(content)} chars")
                    continue
                existing_contents.append(content)

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
        """Write behaviours / preferences / rules / hot memories to
        ~/.evsmem/hot_memories.json so the ev-agent system prompt can inject
        them (same pattern as recommendations.json). Behavioural memory is the
        MOST IMPORTANT and always comes first."""
        try:
            conn = get_db()
            ws = conn.execute(
                "SELECT id FROM workspaces WHERE name='ev-agent' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            wid = ws["id"] if ws else None

            if wid:
                # Sweep near-duplicates first so the injected set stays clean.
                try:
                    import crud as _crud
                    _crud.dedup_classified("behaviour", wid)
                    _crud.dedup_classified("preferences", wid)
                    _crud.dedup_classified("rules", wid)
                except Exception:
                    pass
                # Purge ev-agent workflow/architecture instructions from the
                # classified tables (demoted to memories, never deleted).
                for t in ("behaviour", "preferences", "rules"):
                    rows = conn.execute(
                        f"SELECT id, content, importance, confidence, durability, metadata FROM {t} WHERE workspace_id=?",
                        (wid,),
                    ).fetchall()
                    for r in rows:
                        if _is_workflow_noise(r["content"]):
                            now = datetime.now(timezone.utc).isoformat()
                            conn.execute(
                                """INSERT INTO memories
                                   (id, workspace_id, user_id, agent_name, type, memory_type, content,
                                    embedding, importance, confidence, durability, observed_count, source,
                                    metadata, created_at, updated_at)
                                   VALUES (?, ?, '', '', 'cold_memory', 'workflow_noise', ?, NULL, ?, ?, ?, 1, 'demoted_workflow_noise', ?, ?, ?)""",
                                (str(uuid4()), wid, r["content"], r["importance"], r["confidence"], r["durability"],
                                 r["metadata"] or '{}', now, now),
                            )
                            conn.execute(f"DELETE FROM {t} WHERE id=?", (r["id"],))
                    conn.commit()

            beh, pref, rules, hot = [], [], [], []
            user = {}
            if wid:
                beh = [{"content": r["content"], "importance": r["importance"]} for r in conn.execute(
                    "SELECT content, importance FROM behaviour WHERE workspace_id=? "
                    "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC", (wid,)).fetchall()]
                pref = [{"content": r["content"], "importance": r["importance"]} for r in conn.execute(
                    "SELECT content, importance FROM preferences WHERE workspace_id=? "
                    "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC", (wid,)).fetchall()]
                rules = [{"content": r["content"], "importance": r["importance"]} for r in conn.execute(
                    "SELECT content, importance FROM rules WHERE workspace_id=? "
                    "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC", (wid,)).fetchall()]
                hot = [{"content": r["content"], "importance": r["importance"]} for r in conn.execute(
                    """SELECT content, importance, memory_type FROM memories
                       WHERE workspace_id=? AND type='hot_memory'
                       ORDER BY
                         CASE WHEN memory_type IN ('user','preference','mood','relationship','behavior') THEN 0 ELSE 1 END,
                         (importance * COALESCE(durability, 0.5)) DESC,
                         created_at DESC
                       LIMIT 100""",
                    (wid,)).fetchall()]
                u = conn.execute(
                    "SELECT name, age, location, username, occupation, education, interests, mood, github "
                    "FROM users WHERE workspace_id=?", (wid,)).fetchone()
                if u:
                    user = {
                        "name": u["name"] or "",
                        "age": u["age"] or "",
                        "location": u["location"] or "",
                        "username": u["username"] or "",
                        "occupation": u["occupation"] or "",
                        "education": u["education"] or "",
                        "interests": u["interests"] or "",
                        "mood": u["mood"] or "",
                        "github": u["github"] or "",
                    }
            conn.close()

            payload = {
                "user": user,
                "behaviours": beh,
                "preferences": pref,
                "rules": rules,
                "memories": hot,
                "instruction": "Behavioural memory is the MOST IMPORTANT — adapt to the user's behaviour, preferences, and rules above all else.",
            }
            (Path.home() / ".evsmem" / "hot_memories.json").write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning(f"[hot] Failed to write hot_memories.json: {e}")

    def run_once(self) -> int:
        """Sync ev-agent sessions to evsmem, run one batched LLM analysis pass,
        then analyze reputation / auto-gen triggers / hot-memory JSON.

        Emits one structured per-run log line (JSON) with the number of messages
        processed, the engine used (remote/local), rows inserted per table, and
        update/move/delete counts.
        """
        synced = self._sync_ev_sessions()
        stats = {
            "messages_processed": 0,
            "engine": "none",
            "tool_rounds": 0,
            "json_rows": {},
            "tool_inserted": {},
            "tool_updated": 0,
            "tool_moved": 0,
            "tool_deleted": 0,
            "hot_json_written": False,
            "error": None,
        }
        t0 = time.monotonic()
        try:
            self._process_new_messages_with_llm(stats)
        except Exception as e:
            stats["error"] = f"llm_pass: {e}"
            logger.warning(f"LLM processing error: {e}")
        try:
            self._analyze_session_reputation()
        except Exception as e:
            stats["error"] = f"reputation: {e}"
            logger.warning(f"Reputation analysis error: {e}")
        try:
            self._check_auto_generate_trigger()
        except Exception as e:
            logger.warning(f"Auto-generate trigger error: {e}")
        try:
            self._prune_hot_memories()
            self._write_hot_memories_json()
            stats["hot_json_written"] = True
        except Exception as e:
            logger.warning(f"Hot-memory JSON write error: {e}")

        stats["duration_s"] = round(time.monotonic() - t0, 3)
        logger.info("deriver_analysis_pass " + json.dumps(stats, default=str))
        return synced

    def run_forever(self):
        """Idempotent scheduler: one analysis pass immediately at start, then
        once per EVSMEM_DERIVE_INTERVAL (default 60 minutes). A threading.Lock
        prevents overlapping passes. Message sync (upsert by source id) keeps
        running every EVSMEM_SYNC_INTERVAL between analysis passes so new
        messages flow into evsmem without triggering the LLM."""
        self._running = True
        logger.info(
            f"Deriver started (analysis pass now, then every {DERIVE_INTERVAL:.0f}s; "
            f"message sync every {SYNC_INTERVAL:.0f}s)"
        )
        next_analysis = 0.0
        while self._running:
            if time.monotonic() >= next_analysis:
                if self._analysis_lock.acquire(blocking=False):
                    try:
                        self.run_once()
                    except Exception as e:
                        logger.error(f"Deriver pass error: {e}", exc_info=True)
                    finally:
                        self._analysis_lock.release()
                else:
                    logger.warning("Deriver: analysis pass skipped — previous pass still running")
                next_analysis = time.monotonic() + DERIVE_INTERVAL
            else:
                try:
                    self._sync_ev_sessions()
                except Exception as e:
                    logger.warning(f"Deriver sync error: {e}")
            time.sleep(max(1.0, min(SYNC_INTERVAL, DERIVE_INTERVAL)))

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