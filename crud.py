"""SQLite CRUD operations for evsmem."""

import json, re, sqlite3, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

APP_DIR = Path.home() / ".evsmem"
DB_PATH = APP_DIR / "evsmem.db"

_local = threading.local()


def get_db():
    if not hasattr(_local, "conn") or _local.conn is None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(conn)
        _local.conn = conn
    return _local.conn


def close_db():
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
        _local.conn = None


def parse_row(row):
    d = dict(row)
    for key in ("metadata", "embedding", "task_embedding", "domain_embedding"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT 'default',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            name TEXT NOT NULL DEFAULT 'default',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            ended_at TEXT,
            summary TEXT
        );
        CREATE TABLE IF NOT EXISTS peers (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            name TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(workspace_id, name)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            peer_id TEXT NOT NULL REFERENCES peers(id),
            content TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'user',
            message_type TEXT NOT NULL DEFAULT 'message',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            embedding TEXT,
            is_processed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS conclusions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            content TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            embedding TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_peers_workspace ON peers(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_conclusions_session ON conclusions(session_id);
        DELETE FROM conclusions
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM conclusions
                GROUP BY session_id, content
            );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_conclusions_session_content
            ON conclusions(session_id, content);
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            skills TEXT NOT NULL DEFAULT '',
            system_prompt TEXT DEFAULT '',
            permissions TEXT DEFAULT '{}',
            model TEXT DEFAULT '',
            color TEXT DEFAULT '',
            mode TEXT DEFAULT 'subagent',
            embedding TEXT,
            domain TEXT DEFAULT '',
            domain_embedding TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            content TEXT DEFAULT '',
            agent_name TEXT DEFAULT '',
            embedding TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_agents_workspace ON agents(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_skills_workspace ON skills(workspace_id);
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            user_id TEXT DEFAULT '',
            agent_name TEXT DEFAULT '',
            type TEXT NOT NULL DEFAULT 'personal',
            memory_type TEXT DEFAULT '',
            content TEXT NOT NULL,
            content_ts TEXT DEFAULT '',
            embedding TEXT,
            importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.8,
            durability REAL DEFAULT 0.5,
            observed_count INTEGER DEFAULT 1,
            source TEXT DEFAULT 'conversation',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_memories_workspace_type ON memories(workspace_id, type);
        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(workspace_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(workspace_id, agent_name);
        CREATE TABLE IF NOT EXISTS behaviour (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            content TEXT NOT NULL,
            type TEXT DEFAULT 'behaviour',
            importance REAL DEFAULT 0.9,
            confidence REAL DEFAULT 0.8,
            durability REAL DEFAULT 0.9,
            source TEXT DEFAULT 'deriver_llm',
            metadata TEXT DEFAULT '{}',
            embedding TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS preferences (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            content TEXT NOT NULL,
            type TEXT DEFAULT 'preference',
            importance REAL DEFAULT 0.9,
            confidence REAL DEFAULT 0.8,
            durability REAL DEFAULT 0.9,
            source TEXT DEFAULT 'deriver_llm',
            metadata TEXT DEFAULT '{}',
            embedding TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            content TEXT NOT NULL,
            type TEXT DEFAULT 'rule',
            importance REAL DEFAULT 1.0,
            confidence REAL DEFAULT 0.9,
            durability REAL DEFAULT 0.95,
            source TEXT DEFAULT 'deriver_llm',
            metadata TEXT DEFAULT '{}',
            embedding TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            name TEXT DEFAULT '',
            age TEXT DEFAULT '',
            location TEXT DEFAULT '',
            username TEXT DEFAULT '',
            email TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(workspace_id)
        );
        CREATE TABLE IF NOT EXISTS reputation (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            task_type TEXT NOT NULL DEFAULT '',
            task_embedding TEXT,
            embedding TEXT,
            domain TEXT DEFAULT '',
            domain_embedding TEXT,
            dispatch_count           INTEGER DEFAULT 0,
            success_count            INTEGER DEFAULT 0,
            failure_count            INTEGER DEFAULT 0,
            correction_count         INTEGER DEFAULT 0,
            doom_loop_count          INTEGER DEFAULT 0,
            avg_tokens_consumed      REAL DEFAULT 0,
            avg_time_seconds         REAL DEFAULT 0,
            total_tokens             INTEGER DEFAULT 0,
            use_count                INTEGER DEFAULT 0,
            avg_semantic_score       REAL DEFAULT 0,
            avg_accuracy             REAL DEFAULT 0,
            avg_completeness         REAL DEFAULT 0,
            avg_quality_score        REAL DEFAULT 0,
            avg_efficiency_score     REAL DEFAULT 0,
            avg_tool_usage_score     REAL DEFAULT 0,
            total_evaluations        INTEGER DEFAULT 0,
            overthinking_count       INTEGER DEFAULT 0,
            last_used_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entity_type, entity_name, task_type)
        );
        CREATE INDEX IF NOT EXISTS idx_reputation_type ON reputation(entity_type, entity_name);
        CREATE INDEX IF NOT EXISTS idx_reputation_task ON reputation(task_type);
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            data TEXT DEFAULT '{}',
            read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL DEFAULT 'agent',
            entity_name TEXT NOT NULL,
            task_type TEXT NOT NULL DEFAULT '',
            dispatch_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            reason TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    # Migration: add domain columns to agents (safe for existing DBs)
    for col in ("domain TEXT DEFAULT ''", "domain_embedding TEXT"):
        try:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # Migration: add domain columns to reputation (safe for existing DBs)
    for col in ("domain TEXT DEFAULT ''", "domain_embedding TEXT"):
        try:
            conn.execute(f"ALTER TABLE reputation ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # Migration: add evaluator columns to reputation (safe for existing DBs)
    for col in (
        "avg_accuracy REAL DEFAULT 0",
        "avg_completeness REAL DEFAULT 0",
        "avg_quality_score REAL DEFAULT 0",
        "avg_efficiency_score REAL DEFAULT 0",
        "avg_tool_usage_score REAL DEFAULT 0",
        "total_evaluations INTEGER DEFAULT 0",
        "overthinking_count INTEGER DEFAULT 0",
        "better_agent TEXT DEFAULT ''",
    ):
        try:
            conn.execute(f"ALTER TABLE reputation ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Migration: add is_processed column to messages (safe for existing DBs)
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN is_processed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add durability column to memories (safe for existing DBs)
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN durability REAL DEFAULT 0.5")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _uuid():
    import uuid
    return str(uuid.uuid4())


# ── Workspace ──
def create_workspace(name="default", metadata=None):
    conn = get_db()
    wid = _uuid()
    conn.execute(
        "INSERT INTO workspaces (id, name, metadata, created_at) VALUES (?, ?, ?, ?)",
        (wid, name, json.dumps(metadata or {}), _ts()),
    )
    conn.commit()
    return get_workspace(wid)


def get_workspace(wid):
    conn = get_db()
    row = conn.execute("SELECT * FROM workspaces WHERE id=?", (wid,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def list_workspaces():
    conn = get_db()
    return [parse_row(r) for r in conn.execute("SELECT * FROM workspaces ORDER BY created_at DESC").fetchall()]


def get_workspace_by_name(name):
    """Return the OLDEST workspace with the given name (deterministic)."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM workspaces WHERE name=? ORDER BY created_at ASC LIMIT 1",
        (name,),
    ).fetchone()
    if not row:
        return None
    return parse_row(row)


def list_workspaces_by_name(name):
    """Return ALL workspaces with the given name."""
    conn = get_db()
    return [
        parse_row(r)
        for r in conn.execute(
            "SELECT * FROM workspaces WHERE name=? ORDER BY created_at DESC",
            (name,),
        ).fetchall()
    ]


def get_or_create_workspace(name, metadata=None):
    """Get existing workspace by name or create one."""
    w = get_workspace_by_name(name)
    if w:
        return w
    return create_workspace(name, metadata)


# ── Agents ──

def list_agents(workspace_id):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM agents WHERE workspace_id=? ORDER BY name", (workspace_id,)
    ).fetchall()]


def upsert_agent(workspace_id, name, description="", skills="", system_prompt="",
                 permissions="{}", model="", color="", mode="subagent", embedding=None,
                 domain="", domain_embedding=None):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM agents WHERE workspace_id=? AND name=?", (workspace_id, name)
    ).fetchone()
    aid = existing["id"] if existing else _uuid()
    emb_json = json.dumps(embedding) if embedding else None
    domain_emb_json = json.dumps(domain_embedding) if domain_embedding else None
    conn.execute(
        """INSERT OR REPLACE INTO agents
           (id, workspace_id, name, description, skills, system_prompt, permissions,
            model, color, mode, embedding, domain, domain_embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (aid, workspace_id, name, description, skills, system_prompt, permissions,
         model, color, mode, emb_json, domain, domain_emb_json, _ts()),
    )
    conn.commit()
    return conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()


def delete_agent(workspace_id, name):
    conn = get_db()
    conn.execute("DELETE FROM agents WHERE workspace_id=? AND name=?", (workspace_id, name))
    conn.commit()


def search_agents(workspace_id, q_emb, top_n=5, query="", domain="", domain_emb=None):
    """Search agents by embedding similarity with name boost and optional domain filtering.
    
    Domain filters agents (exact match on domain field) rather than blending
    domain similarity into the ranking score.
    """
    import numpy as np
    import re
    conn = get_db()
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10: return []
    q = q / q_norm

    # Build query — filter by domain string if provided (exact match)
    sql = "SELECT * FROM agents WHERE workspace_id=? AND embedding IS NOT NULL"
    params = [workspace_id]
    if domain:
        sql += " AND domain=?"
        params.append(domain)

    rows = conn.execute(sql, params).fetchall()
    # Extract meaningful keywords from query for name boosting
    stopwords = {"the","a","an","for","and","with","from","to","of","in","on","at","by","is","are","was","were","be","this","that","it","its","app","desktop","web","ui"}
    keywords = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', query) if w.lower() not in stopwords)
    scored = []
    for r in rows:
        try:
            emb = json.loads(r["embedding"])
        except: continue
        if len(emb) != len(q): continue
        v = np.array(emb, dtype=np.float32)
        sim = float(np.dot(q, v) / (np.linalg.norm(v) + 1e-10))
        # Boost name + skills matches heavily — role names and skills should take priority
        name_lower = r["name"].lower()
        skills_text = (r["skills"] or "").lower()
        boost = 0.5 if any(
            kw in name_lower or name_lower in kw or kw in skills_text
            for kw in keywords
        ) else 0
        base_score = sim + boost

        scored.append((base_score, parse_row(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in scored[:top_n]]


# ── Skills ──

def list_skills(workspace_id):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM skills WHERE workspace_id=? ORDER BY name", (workspace_id,)
    ).fetchall()]


def upsert_skill(workspace_id, name, description="", content="", embedding=None):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM skills WHERE workspace_id=? AND name=?", (workspace_id, name)
    ).fetchone()
    sid = existing["id"] if existing else _uuid()
    emb_json = json.dumps(embedding) if embedding else None
    conn.execute(
        """INSERT OR REPLACE INTO skills
           (id, workspace_id, name, description, content, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (sid, workspace_id, name, description, content, emb_json, _ts()),
    )
    conn.commit()
    return conn.execute("SELECT * FROM skills WHERE id=?", (sid,)).fetchone()


def delete_skill(workspace_id, name):
    conn = get_db()
    conn.execute("DELETE FROM skills WHERE workspace_id=? AND name=?", (workspace_id, name))
    conn.commit()


def search_skills(workspace_id, q_emb, top_n=5, query="", agent_name=""):
    """Search skills by embedding similarity with name boost."""
    import numpy as np
    import re
    conn = get_db()
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10: return []
    q = q / q_norm
    if agent_name:
        rows = conn.execute(
            "SELECT * FROM skills WHERE workspace_id=? AND agent_name=? AND embedding IS NOT NULL",
            (workspace_id, agent_name)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM skills WHERE workspace_id=? AND embedding IS NOT NULL", (workspace_id,)
        ).fetchall()
    stopwords = {"the","a","an","for","and","with","from","to","of","in","on","at","by","is","are","was","were","be","this","that","it","its","app","desktop","web","ui"}
    keywords = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', query) if w.lower() not in stopwords)
    scored = []
    for r in rows:
        try:
            emb = json.loads(r["embedding"])
        except: continue
        if len(emb) != len(q): continue
        v = np.array(emb, dtype=np.float32)
        sim = float(np.dot(q, v) / (np.linalg.norm(v) + 1e-10))
        name_lower = r["name"].lower()
        boost = 0
        for kw in keywords:
            if kw in name_lower or name_lower in kw:
                boost += 0.15
        scored.append((sim + boost, parse_row(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in scored[:top_n]]


# ── Session ──
def create_session(workspace_id, name="default", metadata=None):
    conn = get_db()
    sid = _uuid()
    conn.execute(
        "INSERT INTO sessions (id, workspace_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (sid, workspace_id, name, json.dumps(metadata or {}), _ts()),
    )
    conn.commit()
    return get_session(sid)


def get_session(sid):
    conn = get_db()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def list_sessions(workspace_id):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM sessions WHERE workspace_id=? ORDER BY created_at DESC", (workspace_id,)
    ).fetchall()]


def update_session(sid, **kwargs):
    conn = get_db()
    sets = []
    vals = []
    for k, v in kwargs.items():
        if v is not None:
            sets.append(f"{k}=?")
            vals.append(json.dumps(v) if isinstance(v, dict) else v)
    if sets:
        vals.append(sid)
        conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
    return get_session(sid)


def end_session(sid):
    return update_session(sid, ended_at=_ts())


# ── Peer ──
def create_peer(workspace_id, name, metadata=None):
    conn = get_db()
    pid = _uuid()
    try:
        conn.execute(
            "INSERT INTO peers (id, workspace_id, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (pid, workspace_id, name, json.dumps(metadata or {}), _ts()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        row = conn.execute("SELECT * FROM peers WHERE workspace_id=? AND name=?", (workspace_id, name)).fetchone()
        return parse_row(row) if row else None
    return get_peer(pid)


def get_peer(pid):
    conn = get_db()
    row = conn.execute("SELECT * FROM peers WHERE id=?", (pid,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def get_or_create_peer(workspace_id, name, metadata=None):
    conn = get_db()
    row = conn.execute("SELECT * FROM peers WHERE workspace_id=? AND name=?", (workspace_id, name)).fetchone()
    if row:
        return parse_row(row)
    return create_peer(workspace_id, name, metadata)


def list_peers(workspace_id):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM peers WHERE workspace_id=? ORDER BY created_at", (workspace_id,)
    ).fetchall()]


def update_peer(pid, metadata=None, name=None):
    conn = get_db()
    fields = []
    params = []
    if metadata is not None:
        # Merge — don't replace the entire metadata dict
        existing = get_peer(pid)
        if existing:
            merged = {**(existing.get("metadata") or {}), **metadata}
            fields.append("metadata=?")
            params.append(json.dumps(merged))
        else:
            fields.append("metadata=?")
            params.append(json.dumps(metadata))
    if name is not None:
        fields.append("name=?")
        params.append(name)
    if not fields:
        return get_peer(pid)
    params.append(pid)
    conn.execute(f"UPDATE peers SET {', '.join(fields)} WHERE id=?", params)
    conn.commit()
    return get_peer(pid)


# ── Message ──
def create_message(session_id, peer_id, content="", role="user", message_type="message", metadata=None, embedding=None):
    conn = get_db()
    mid = _uuid()
    conn.execute(
        "INSERT INTO messages (id, session_id, peer_id, content, role, message_type, metadata, created_at, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, session_id, peer_id, content, role, message_type, json.dumps(metadata or {}), _ts(),
         json.dumps(embedding) if embedding else None),
    )
    conn.commit()
    return get_message(mid)


def get_message(mid):
    conn = get_db()
    row = conn.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def list_messages(session_id, limit=100, offset=0):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (session_id, limit, offset),
    ).fetchall()]


def search_messages_by_text(session_id, query, limit=5):
    """Keyword search on message content (fallback when embeddings unavailable)."""
    if not query:
        return list_messages(session_id, limit=limit)
    conn = get_db()
    like = f"%{query}%"
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM messages WHERE session_id=? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
        (session_id, like, limit),
    ).fetchall()]


def search_messages(session_id, q_emb, top_k=5, max_distance=1.5):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id=? AND embedding IS NOT NULL", (session_id,)
    ).fetchall()

    import numpy as np
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return []
    q = q / q_norm
    q_dim = len(q)

    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            continue
        # Skip embeddings with wrong dimensions (e.g. old 768-dim vs new 1024-dim)
        if len(emb) != q_dim:
            continue
        v = np.array(emb, dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        sim = float(np.dot(q, v / v_norm))
        dist = 1.0 - sim
        if dist <= max_distance:
            scored.append((dist, parse_row(row)))

    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored[:top_k]]


# ── Conclusion ──
def create_conclusion(session_id, content, metadata=None, embedding=None):
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM conclusions WHERE session_id=? AND content=?",
        (session_id, content),
    ).fetchone()
    if existing:
        return parse_row(existing)

    cid = _uuid()
    conn.execute(
        "INSERT INTO conclusions (id, session_id, content, metadata, created_at, embedding) VALUES (?,?,?,?,?,?)",
        (cid, session_id, content, json.dumps(metadata or {}), _ts(),
         json.dumps(embedding) if embedding else None),
    )
    conn.commit()
    return get_conclusion(cid)


def get_conclusion(cid):
    conn = get_db()
    row = conn.execute("SELECT * FROM conclusions WHERE id=?", (cid,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def list_conclusions(session_id):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM conclusions WHERE session_id=? ORDER BY created_at", (session_id,)
    ).fetchall()]


# ── Stats ──
def session_stats(session_id):
    conn = get_db()
    msg_count = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]
    peer_count = conn.execute("SELECT COUNT(DISTINCT peer_id) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]
    conc_count = conn.execute("SELECT COUNT(*) FROM conclusions WHERE session_id=?", (session_id,)).fetchone()[0]
    return {
        "message_count": msg_count,
        "peer_count": peer_count,
        "conclusion_count": conc_count,
    }


# ── Context ──
def get_context(session_id, q_emb, top_k=5, max_distance=1.5):
    msgs = search_messages(session_id, q_emb, top_k, max_distance)
    conn = get_db()
    conclusions = [parse_row(r) for r in conn.execute(
        "SELECT * FROM conclusions WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, top_k),
    ).fetchall()]
    return {"messages": msgs, "conclusions": conclusions}


# ── Memories ──

def create_memory(workspace_id, type, content, user_id=None, agent_name=None,
                  importance=0.5, confidence=0.8, durability=0.5, memory_type=None,
                  source='conversation', metadata=None, embedding=None):
    """Store a memory with type (tier), memory_type (semantic), importance,
    confidence, and durability."""
    conn = get_db()
    mid = _uuid()
    metadata_json = json.dumps(metadata or {})
    emb_json = json.dumps(embedding) if embedding else None
    mem_type = memory_type or type
    conn.execute(
        """INSERT INTO memories
           (id, workspace_id, user_id, agent_name, type, memory_type, content,
            embedding, importance, confidence, durability, observed_count, source,
            metadata, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mid, workspace_id, user_id or '', agent_name or '', type, mem_type, content,
         emb_json, importance, confidence, durability, 1, source,
         metadata_json, _ts(), _ts()),
    )
    conn.commit()
    return get_memory(mid)


def get_memory(memory_id):
    """Get a single memory by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    if not row:
        return None
    return parse_row(row)


def _store_classified(table, workspace_id, content, mem_type, importance, confidence,
                     durability, source='deriver_llm', metadata=None, embedding=None, cap=100):
    """Dedup + capped insert into behaviour / preferences / rules tables.
    Rows beyond the cap are DEMOTED to the memories table (as cold_memory)
    instead of being deleted, so nothing is ever lost."""
    conn = get_db()
    existing = conn.execute(
        f"SELECT id FROM {table} WHERE workspace_id=? AND lower(trim(content))=lower(trim(?))",
        (workspace_id, content),
    ).fetchone()
    if existing:
        return conn.execute(f"SELECT * FROM {table} WHERE id=?", (existing["id"],)).fetchone()
    mid = _uuid()
    emb_json = json.dumps(embedding) if embedding else None
    conn.execute(
        f"""INSERT INTO {table}
           (id, workspace_id, content, type, importance, confidence, durability, source, metadata, embedding, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mid, workspace_id, content, mem_type, importance, confidence, durability, source,
         json.dumps(metadata or {}), emb_json, _ts(), _ts()),
    )
    # Demote excess rows (over cap) into memories as cold_memory instead of deleting.
    excess = conn.execute(
        f"""SELECT id, content, importance, confidence, durability, metadata FROM {table}
            WHERE workspace_id=?
            ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC
            LIMIT -1 OFFSET ?""",
        (workspace_id, cap),
    ).fetchall()
    for row in excess:
        try:
            conn.execute(
                """INSERT INTO memories
                   (id, workspace_id, user_id, agent_name, type, memory_type, content,
                    embedding, importance, confidence, durability, observed_count, source,
                    metadata, created_at, updated_at)
                   VALUES (?, ?, '', '', 'cold_memory', ?, ?, NULL, ?, ?, ?, 1, 'demoted', ?, ?, ?)""",
                (_uuid(), workspace_id, mem_type, row["content"],
                 row["importance"], row["confidence"], row["durability"],
                 row["metadata"] or '{}', _ts(), _ts()),
            )
            conn.execute(f"DELETE FROM {table} WHERE id=?", (row["id"],))
        except Exception:
            pass
    conn.commit()
    return conn.execute(f"SELECT * FROM {table} WHERE id=?", (mid,)).fetchone()


def create_behaviour(workspace_id, content, importance=0.9, confidence=0.8, durability=0.9,
                     metadata=None, embedding=None, cap=100):
    return _store_classified("behaviour", workspace_id, content, "behaviour", importance,
                             confidence, durability, metadata=metadata, embedding=embedding, cap=cap)


def create_preference(workspace_id, content, importance=0.9, confidence=0.8, durability=0.9,
                      metadata=None, embedding=None, cap=100):
    return _store_classified("preferences", workspace_id, content, "preference", importance,
                             confidence, durability, metadata=metadata, embedding=embedding, cap=cap)


def create_rule(workspace_id, content, importance=1.0, confidence=0.9, durability=0.95,
                metadata=None, embedding=None, cap=100):
    return _store_classified("rules", workspace_id, content, "rule", importance,
                             confidence, durability, metadata=metadata, embedding=embedding, cap=cap)


def list_classified(table, workspace_id, limit=100):
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        f"SELECT * FROM {table} WHERE workspace_id=? "
        "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC LIMIT ?",
        (workspace_id, limit),
    ).fetchall()]


def get_behaviours(workspace_id, limit=100):
    return list_classified("behaviour", workspace_id, limit)


def get_preferences(workspace_id, limit=100):
    return list_classified("preferences", workspace_id, limit)


def get_rules(workspace_id, limit=100):
    return list_classified("rules", workspace_id, limit)


def upsert_user(workspace_id, name=None, age=None, location=None, username=None, metadata=None):
    """Create or update the single user row for a workspace. Existing non-empty
    fields are preserved when the new value is empty."""
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE workspace_id=?", (workspace_id,)).fetchone()
    uid = existing["id"] if existing else _uuid()
    meta_json = json.dumps(metadata or {})
    conn.execute(
        """INSERT INTO users (id, workspace_id, name, age, location, username, metadata, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(workspace_id) DO UPDATE SET
             name = CASE WHEN excluded.name != '' THEN excluded.name ELSE users.name END,
             age = CASE WHEN excluded.age != '' THEN excluded.age ELSE users.age END,
             location = CASE WHEN excluded.location != '' THEN excluded.location ELSE users.location END,
             username = CASE WHEN excluded.username != '' THEN excluded.username ELSE users.username END,
             metadata = excluded.metadata,
             updated_at = excluded.updated_at""",
        (uid, workspace_id, name or '', age or '', location or '', username or '',
         meta_json, _ts(), _ts()),
    )
    conn.commit()
    return get_user(workspace_id)


def get_user(workspace_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE workspace_id=?", (workspace_id,)).fetchone()
    return parse_row(row) if row else None


def get_hot_memories(workspace_id, limit=40):
    """Get hot memories (type='hot_memory') — always-injected.
    Ranked by importance x durability (newest tiebreak) and capped to `limit`
    so the system prompt doesn't balloon with tokens."""
    conn = get_db()
    return [parse_row(r) for r in conn.execute(
        "SELECT * FROM memories WHERE workspace_id=? AND type='hot_memory' "
        "ORDER BY (importance * COALESCE(durability, 0.5)) DESC, created_at DESC LIMIT ?",
        (workspace_id, limit),
    ).fetchall()]


def search_memories(workspace_id, query_text, query_emb, type="", user_id="",
                    agent_name="", top_n=10, include_importance=True):
    """Hybrid search: semantic cosine * 0.5 + keyword BM25-normalized * 0.3 + importance * 0.2."""
    conn = get_db()
    import numpy as np

    # 1. Build parameterized WHERE clause
    filters = [("m.workspace_id", "=", workspace_id)]
    if type:
        filters.append(("m.type", "=", type))
    if user_id:
        filters.append(("m.user_id", "=", user_id))
    if agent_name:
        filters.append(("m.agent_name", "=", agent_name))

    where_parts = []
    params = []
    for col, op, val in filters:
        where_parts.append(f"{col} {op} ?")
        params.append(val)

    where_sql = " AND ".join(where_parts)

    # 2. Load memories (keyword scoring computed in Python, no FTS dependency)
    sql = f"""SELECT m.rowid, m.*
              FROM memories m
              WHERE {where_sql}
              ORDER BY m.created_at DESC"""

    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    # 2b. Extract significant words from the text query for keyword scoring
    _stop = {"the","a","an","and","for","with","from","to","of","in","on","at","by","is","are","was","were","be","this","that","it","its"}
    query_words = [
        w for w in re.findall(r"[a-zA-Z0-9]{3,}", (query_text or "").lower())
        if w not in _stop
    ]

    # 3. Normalize query embedding (if available)
    q = None
    if query_emb is not None:
        q = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
        else:
            q = None

    # 4. Compute hybrid scores
    results = []
    for r in rows:
        # Semantic cosine similarity (0 if no query embedding or no stored embedding)
        sem_score = 0.0
        if q is not None:
            try:
                stored_emb = json.loads(r["embedding"])
            except (json.JSONDecodeError, TypeError):
                stored_emb = None
            if stored_emb is not None:
                v = np.array(stored_emb, dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    sem_score = float(np.dot(q, v / v_norm))

        # Keyword score — fraction of query words present in content
        kw_score = 0.0
        content_lower = (r["content"] or "").lower()
        if query_words and content_lower:
            hits = sum(1 for w in query_words if w in content_lower)
            kw_score = hits / len(query_words)

        # Importance (boost high-importance memories)
        imp_score = r["importance"] if include_importance else 0.0

        # Confidence factor (reliable memories weighted higher)
        conf_score = r["confidence"]

        # Durability (long-lived facts weigh more than transient events)
        dur_score = r["durability"] if r["durability"] is not None else 0.5

        # Recency decay (recently created memories get a small boost)
        recency = 0.0
        try:
            created_dt = datetime.fromisoformat(r["created_at"])
            age_days = (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400
            recency = max(0.0, 1.0 - age_days / 60.0)
        except Exception:
            pass

        final = (sem_score * 0.4 + kw_score * 0.2 + imp_score * 0.2
                 + conf_score * 0.05 + dur_score * 0.1 + recency * 0.05)

        results.append({
            "id": r["id"],
            "content": r["content"],
            "type": r["type"],
            "memory_type": r["memory_type"],
            "importance": r["importance"],
            "confidence": r["confidence"],
            "durability": round(float(dur_score), 3),
            "observed_count": r["observed_count"],
            "source": r["source"],
            "score": round(float(final), 4),
            "created_at": r["created_at"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def update_memory_confidence(memory_id, confidence):
    """Update confidence and increment observed_count for a memory."""
    conn = get_db()
    conn.execute(
        "UPDATE memories SET confidence=?, observed_count=observed_count+1, updated_at=? WHERE id=?",
        (confidence, _ts(), memory_id),
    )
    conn.commit()
    return get_memory(memory_id)


def delete_memory(memory_id):
    """Soft delete a memory by clearing content and setting a tombstone."""
    conn = get_db()
    row = conn.execute("SELECT rowid FROM memories WHERE id=?", (memory_id,)).fetchone()
    if row:
        # Soft delete: clear content and mark
        conn.execute(
            "UPDATE memories SET content='', content_ts='__deleted__', embedding=NULL, updated_at=? WHERE id=?",
            (_ts(), memory_id),
        )
        conn.commit()
    return {"ok": True}


def preflight_command(workspace_id, command, query_emb):
    """Check a command against stored command_policy memories. Strictest policy wins."""
    conn = get_db()
    command_lower = command.lower()
    rows = conn.execute(
        "SELECT * FROM memories WHERE workspace_id=? AND type='command_policy' AND embedding IS NOT NULL",
        (workspace_id,),
    ).fetchall()

    matches = []
    for r in rows:
        meta = json.loads(r["metadata"])
        pattern = meta.get("command_pattern", r["content"])
        match_type = meta.get("match_type", "substring")
        if match_type == "exact" and command_lower == pattern.lower():
            matches = [r]
            break
        elif match_type == "substring" and pattern.lower() in command_lower:
            matches.append(r)
        elif match_type == "regex" and re.search(pattern, command_lower):
            matches = [r]
            break

    if not matches:
        return {"decision": "allow", "reason": "No matching policy"}

    # Strictest wins (block > warn > allow)
    decisions = {"allow": 0, "warn": 1, "block": 2}
    matches.sort(key=lambda m: -decisions.get(json.loads(m["metadata"]).get("decision", "warn"), 0))
    m = matches[0]
    meta = json.loads(m["metadata"])
    return {
        "decision": meta.get("decision", "warn"),
        "reason": meta.get("reason", m["content"]),
        "matched_memory_id": m["id"],
        "matched_content": m["content"][:200],
    }


# ── Reputation ──

def record_dispatch(entity_type, entity_name, task_type="", semantic_score=0.0, task_embedding=None):
    """Record a dispatch. Same (entity_type, entity_name, task_type) gets upserted."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()

    now = _ts()
    if existing:
        row = conn.execute("SELECT * FROM reputation WHERE id=?", (existing["id"],)).fetchone()
        dc = row["dispatch_count"] or 0
        avg_ss = row["avg_semantic_score"] or 0
        new_avg_ss = ((avg_ss * dc) + semantic_score) / (dc + 1) if dc > 0 else semantic_score
        conn.execute(
            """UPDATE reputation
               SET dispatch_count = dispatch_count + 1,
                   avg_semantic_score = ?,
                   use_count = use_count + 1,
                   last_used_at = ?,
                   updated_at = ?
               WHERE id=?""",
            (new_avg_ss, now, now, existing["id"]),
        )
    else:
        rid = _uuid()
        task_emb_json = json.dumps(task_embedding) if task_embedding else None
        conn.execute(
            """INSERT INTO reputation
               (id, entity_type, entity_name, task_type, task_embedding,
                dispatch_count, avg_semantic_score, use_count,
                last_used_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, 1, ?, ?, ?)""",
            (rid, entity_type, entity_name, task_type, task_emb_json,
             semantic_score, now, now, now),
        )
    conn.commit()
    return get_reputation(entity_type, entity_name, task_type)


def record_success(entity_type, entity_name, task_type, tokens_used=0, time_seconds=0):
    """Record a successful completion. Increments success_count, updates averages."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()

    now = _ts()
    if not existing:
        rid = _uuid()
        conn.execute(
            """INSERT INTO reputation
               (id, entity_type, entity_name, task_type, dispatch_count, success_count,
                avg_tokens_consumed, avg_time_seconds, total_tokens,
                last_used_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, ?, ?, ?)""",
            (rid, entity_type, entity_name, task_type, now, now, now),
        )
        conn.commit()
        existing = conn.execute(
            "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
            (entity_type, entity_name, task_type),
        ).fetchone()

    row = conn.execute("SELECT * FROM reputation WHERE id=?", (existing["id"],)).fetchone()
    dc = row["dispatch_count"] or 0
    sc = row["success_count"] or 0
    avg_tokens = row["avg_tokens_consumed"] or 0
    avg_time = row["avg_time_seconds"] or 0
    total_tok = row["total_tokens"] or 0

    new_sc = sc + 1
    new_total_tokens = total_tok + tokens_used
    new_avg_tokens = new_total_tokens / max(new_sc, 1)
    new_avg_time = ((avg_time * sc) + time_seconds) / new_sc if sc > 0 else time_seconds

    conn.execute(
        """UPDATE reputation
           SET success_count = ?,
               avg_tokens_consumed = ?,
               avg_time_seconds = ?,
               total_tokens = ?,
               use_count = use_count + 1,
               last_used_at = ?,
               updated_at = ?
           WHERE id=?""",
        (new_sc, new_avg_tokens, new_avg_time, new_total_tokens, now, now, existing["id"]),
    )
    conn.commit()
    return get_reputation(entity_type, entity_name, task_type)


def record_failure(entity_type, entity_name, task_type, tokens_used=0, time_seconds=0):
    """Record a failure. Increments failure_count."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()

    now = _ts()
    if not existing:
        rid = _uuid()
        conn.execute(
            """INSERT INTO reputation
               (id, entity_type, entity_name, task_type, dispatch_count, failure_count,
                avg_tokens_consumed, avg_time_seconds, total_tokens,
                last_used_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, ?, ?, ?)""",
            (rid, entity_type, entity_name, task_type, now, now, now),
        )
        conn.commit()
        existing = conn.execute(
            "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
            (entity_type, entity_name, task_type),
        ).fetchone()

    row = conn.execute("SELECT * FROM reputation WHERE id=?", (existing["id"],)).fetchone()
    dc = row["dispatch_count"] or 0
    fc = row["failure_count"] or 0
    total_tok = row["total_tokens"] or 0
    avg_time = row["avg_time_seconds"] or 0

    new_total_tokens = total_tok + tokens_used
    new_avg_tokens = new_total_tokens / max(dc + 1, 1)
    new_avg_time = ((avg_time * dc) + time_seconds) / (dc + 1) if dc > 0 else time_seconds

    conn.execute(
        """UPDATE reputation
           SET failure_count = ?,
               avg_tokens_consumed = ?,
               avg_time_seconds = ?,
               total_tokens = ?,
               use_count = use_count + 1,
               last_used_at = ?,
               updated_at = ?
           WHERE id=?""",
        (fc + 1, new_avg_tokens, new_avg_time, new_total_tokens, now, now, existing["id"]),
    )
    conn.commit()
    return get_reputation(entity_type, entity_name, task_type)


def record_correction(entity_type, entity_name, task_type):
    """Record that user corrected the output. Increments correction_count."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()

    now = _ts()
    if not existing:
        rid = _uuid()
        conn.execute(
            """INSERT INTO reputation
               (id, entity_type, entity_name, task_type, dispatch_count, correction_count,
                last_used_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?)""",
            (rid, entity_type, entity_name, task_type, now, now, now),
        )
        conn.commit()
        existing = conn.execute(
            "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
            (entity_type, entity_name, task_type),
        ).fetchone()

    conn.execute(
        """UPDATE reputation
           SET correction_count = correction_count + 1,
               use_count = use_count + 1,
               last_used_at = ?,
               updated_at = ?
           WHERE id=?""",
        (now, now, existing["id"]),
    )
    conn.commit()
    return get_reputation(entity_type, entity_name, task_type)


def record_doom_loop(entity_type, entity_name, task_type):
    """Record that agent entered a doom loop. Increments doom_loop_count."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()

    now = _ts()
    if not existing:
        rid = _uuid()
        conn.execute(
            """INSERT INTO reputation
               (id, entity_type, entity_name, task_type, dispatch_count, doom_loop_count,
                last_used_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?)""",
            (rid, entity_type, entity_name, task_type, now, now, now),
        )
        conn.commit()
        existing = conn.execute(
            "SELECT id FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
            (entity_type, entity_name, task_type),
        ).fetchone()

    conn.execute(
        """UPDATE reputation
           SET doom_loop_count = doom_loop_count + 1,
               use_count = use_count + 1,
               last_used_at = ?,
               updated_at = ?
           WHERE id=?""",
        (now, now, existing["id"]),
    )
    conn.commit()
    return get_reputation(entity_type, entity_name, task_type)


def record_evaluation(entity_type, entity_name, task_type, evaluation):
    """Record evaluator results into reputation table.
    Updates rolling averages for quality scores."""
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type)
    ).fetchone()

    now = _ts()
    if existing:
        te = existing["total_evaluations"] or 0
        new_te = te + 1

        # Rolling averages
        aa = ((existing["avg_accuracy"] or 0) * te + evaluation.get("accuracy", 0)) / new_te
        ac = ((existing["avg_completeness"] or 0) * te + evaluation.get("completeness", 0)) / new_te
        aq = ((existing["avg_quality_score"] or 0) * te + evaluation.get("quality_score", 0)) / new_te
        ae = ((existing["avg_efficiency_score"] or 0) * te + evaluation.get("efficiency_score", 0)) / new_te
        at = ((existing["avg_tool_usage_score"] or 0) * te + evaluation.get("tool_usage_score", 0)) / new_te

        # Track better_agent — keep the latest non-empty suggestion
        ba = evaluation.get("better_agent", "") or ""
        if not ba:
            ba = existing["better_agent"] or ""

        conn.execute("""
            UPDATE reputation SET
                avg_accuracy=?,
                avg_completeness=?,
                avg_quality_score=?,
                avg_efficiency_score=?,
                avg_tool_usage_score=?,
                total_evaluations=?,
                doom_loop_count=doom_loop_count+?,
                overthinking_count=overthinking_count+?,
                better_agent=?,
                updated_at=?
            WHERE id=?
        """, (aa, ac, aq, ae, at, new_te,
              1 if evaluation.get("doom_loop") else 0,
              evaluation.get("overthinking_count", 0),
              ba,
              now,
              existing["id"]))
    else:
        rid = _uuid()
        ba = evaluation.get("better_agent", "") or ""
        # Insert a fresh row with initial evaluation values
        conn.execute("""
            INSERT INTO reputation
                (id, entity_type, entity_name, task_type,
                 avg_accuracy, avg_completeness, avg_quality_score,
                 avg_efficiency_score, avg_tool_usage_score,
                 total_evaluations, doom_loop_count, overthinking_count,
                 better_agent, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
        """, (rid, entity_type, entity_name, task_type,
              evaluation.get("accuracy", 0),
              evaluation.get("completeness", 0),
              evaluation.get("quality_score", 0),
              evaluation.get("efficiency_score", 0),
              evaluation.get("tool_usage_score", 0),
              1 if evaluation.get("doom_loop") else 0,
              evaluation.get("overthinking_count", 0),
              ba, now, now))
    conn.commit()


def get_reputation(entity_type, entity_name, task_type=""):
    """Get reputation for a specific (entity, task_type) combo."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM reputation WHERE entity_type=? AND entity_name=? AND task_type=?",
        (entity_type, entity_name, task_type),
    ).fetchone()
    if not row:
        return None
    return parse_row(row)


def search_reputation(query_emb=None, entity_type="agent", top_n=10):
    """Search reputations by semantic similarity to task_type.
    Returns reputation records with computed final_score.
    """
    import numpy as np
    conn = get_db()

    q = None
    if query_emb is not None:
        q = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm

    rows = conn.execute(
        "SELECT * FROM reputation WHERE entity_type=?",
        (entity_type,),
    ).fetchall()

    now = _ts()
    scored = []
    for r in rows:
        # Semantic similarity
        sim = 0.0
        if q is not None:
            try:
                emb = json.loads(r["task_embedding"])
            except (json.JSONDecodeError, TypeError):
                emb = None
            if emb is not None:
                v = np.array(emb, dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    sim = float(np.dot(q, v / v_norm))

        dc = r["dispatch_count"] or 0
        sc = r["success_count"] or 0
        cc = r["correction_count"] or 0
        dlc = r["doom_loop_count"] or 0

        success_rate = sc / max(dc, 1)
        correction_rate = cc / max(dc, 1)
        doom_loop_rate = dlc / max(dc, 1)

        avg_time = r["avg_time_seconds"] or 0
        efficiency_score = 1.0 - min(avg_time / 300.0, 1.0)

        # Recently used boost (within last 24 hours)
        recency_boost = 0.0
        last_used = r["last_used_at"] if "last_used_at" in r.keys() else None
        if last_used:
            try:
                last_dt = datetime.fromisoformat(last_used)
                now_dt = datetime.fromisoformat(now)
                hours_since = (now_dt - last_dt).total_seconds() / 3600
                if hours_since < 24:
                    recency_boost = 1.0
            except (ValueError, TypeError):
                pass

        final_score = (
            success_rate * 0.4
            + (1.0 - correction_rate) * 0.2
            + (1.0 - doom_loop_rate) * 0.2
            + efficiency_score * 0.1
            + recency_boost * 0.1
        )

        result = parse_row(r)
        result["final_score"] = round(float(final_score), 4)
        result["semantic_similarity"] = round(float(sim), 4)
        scored.append(result)

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_n]
