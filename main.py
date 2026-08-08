"""evsmem — FastAPI application."""

import json, logging, os, sys, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure package root is on path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import crud
from embeddings import EmbeddingClient
import models
from deriver import start_deriver_thread

logger = logging.getLogger(__name__)

embedder = EmbeddingClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"evsmem starting — DB: {crud.DB_PATH}")

    # Warm up embedding model in background thread so it stays in GPU memory
    def _warmup():
        try:
            logger.info("Warming up embedding model (background)...")
            embedder._ensure_model()
            logger.info("Embedding model ready on GPU")
        except Exception as e:
            logger.warning(f"Embedding model warmup failed: {e}")

    import threading
    warmup_thread = threading.Thread(target=_warmup, daemon=True)
    warmup_thread.start()

    embed_ok = embedder.is_available()
    logger.info(f"Embeddings: {'available' if embed_ok else 'UNAVAILABLE (LM Studio not running)'}")
    deriver = start_deriver_thread()
    logger.info("Deriver started in background thread")
    yield
    # Shutdown
    if deriver:
        deriver.stop()
    crud.close_db()


app = FastAPI(
    title="evsmem",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Embedding endpoint ──

class EmbedRequest(BaseModel):
    input: str | list[str]
    model: Optional[str] = None


@app.post("/v1/embeddings")
async def embed_text(body: EmbedRequest):
    """Generate embeddings using the loaded BGE-M3 model (GPU)."""
    texts = [body.input] if isinstance(body.input, str) else body.input
    result = embedder.embed_batch(texts)
    data = [
        {"object": "embedding", "index": i, "embedding": emb}
        for i, emb in enumerate(result)
    ]
    return {"object": "list", "data": data, "model": embedder.model_name()}


# ── Health ──
@app.get("/health")
async def health():
    return models.HealthResponse(
        embeddings="available" if embedder.is_available() else "unavailable"
    )


# ── Workspaces ──
@app.post("/workspaces", response_model=models.Workspace)
async def create_workspace(body: models.WorkspaceCreate):
    return crud.create_workspace(body.name, body.metadata)

@app.get("/workspaces", response_model=list[models.Workspace])
async def list_workspaces():
    return crud.list_workspaces()

# ── Workspace by name (MUST be before parameterized /workspaces/{id} routes) ──
@app.get("/workspace/by-name/{name}", response_model=models.Workspace)
async def get_workspace_by_name(name: str):
    """Return the OLDEST workspace with the given name (deterministic). Creates one if none exists."""
    w = crud.get_workspace_by_name(name)
    if not w:
        w = crud.create_workspace(name)
    return w


@app.post("/workspace/by-name/{name}/search")
async def search_by_name(name: str, body: models.ContextRequest):
    """Search across ALL workspaces with the given name."""
    workspaces = crud.list_workspaces_by_name(name)
    if not workspaces:
        return {"results": []}

    q_emb = None
    if body.last_message and embedder.is_available():
        try:
            q_emb = embedder.embed(body.last_message[:2000])
        except Exception:
            pass

    results = []
    for w in workspaces:
        sessions = crud.list_sessions(w["id"])
        for s in sessions[:10]:  # last 10 sessions per workspace
            if q_emb:
                msgs = crud.search_messages(s["id"], q_emb, body.search_top_k, body.search_max_distance)
                # Fallback to text search if vector search found nothing
                if not msgs:
                    msgs = crud.search_messages_by_text(s["id"], body.last_message or "", limit=5)
            else:
                # Fallback: keyword search when embeddings unavailable
                msgs = crud.search_messages_by_text(s["id"], body.last_message or "", limit=5)
            if msgs:
                results.append({"session_id": s["id"], "session_name": s["name"], "matches": msgs})

    return {"results": results}


# ── Agent CRUD ──

class AgentCreate(BaseModel):
    name: str
    description: str = ""
    skills: str = ""
    system_prompt: str = ""
    permissions: str = "{}"
    model: str = ""
    color: str = ""
    mode: str = "subagent"
    domain: str = ""


@app.get("/workspaces/{workspace_id}/agents")
async def list_agents(workspace_id: str):
    return crud.list_agents(workspace_id)


@app.post("/workspaces/{workspace_id}/agents")
async def upsert_agent(workspace_id: str, body: AgentCreate):
    # Generate embedding
    emb = None
    text = f"{body.name} {body.description}".strip()
    if text and embedder.is_available():
        try:
            emb = embedder.embed(text[:2000])
        except Exception:
            pass
    # Generate domain embedding
    domain_emb = None
    if body.domain and embedder.is_available():
        try:
            domain_emb = embedder.embed(body.domain[:2000])
        except Exception:
            pass
    a = crud.upsert_agent(
        workspace_id, body.name, body.description, body.skills,
        body.system_prompt, body.permissions, body.model, body.color, body.mode, emb,
        body.domain, domain_emb,
    )
    return a


@app.delete("/workspaces/{workspace_id}/agents/{name}")
async def delete_agent(workspace_id: str, name: str):
    crud.delete_agent(workspace_id, name)
    return {"ok": True}


# ── Skill CRUD ──

class SkillCreate(BaseModel):
    name: str
    description: str = ""
    content: str = ""


@app.get("/workspaces/{workspace_id}/skills")
async def list_skills(workspace_id: str):
    return crud.list_skills(workspace_id)


@app.post("/workspaces/{workspace_id}/skills")
async def upsert_skill(workspace_id: str, body: SkillCreate):
    emb = None
    text = f"{body.name} {body.description}".strip()
    if text and embedder.is_available():
        try:
            emb = embedder.embed(text[:2000])
        except Exception:
            pass
    s = crud.upsert_skill(workspace_id, body.name, body.description, body.content, emb)
    return s


@app.delete("/workspaces/{workspace_id}/skills/{name}")
async def delete_skill(workspace_id: str, name: str):
    crud.delete_skill(workspace_id, name)
    return {"ok": True}


# ── Semantic search across agents + skills + sessions ──

class SemanticSearchRequest(BaseModel):
    query: str
    top_n_agents: int = 5
    top_n_skills: int = 5
    top_n_sessions: int = 5
    agent_name: str = ""
    domain: str = ""


@app.post("/workspace/by-name/{name}/semantic-search")
async def semantic_search(name: str, body: SemanticSearchRequest):
    """Search agents, skills, and sessions in one call using GPU embeddings."""
    workspaces = crud.list_workspaces_by_name(name)
    if not workspaces:
        return {"agents": [], "skills": [], "sessions": []}
    wid = workspaces[-1]["id"]  # Oldest workspace (consistent with ensureWorkspace)

    q_emb = None
    if body.query and embedder.is_available():
        try:
            q_emb = embedder.embed(body.query[:2000])
        except Exception:
            pass

    # Compute domain embedding if domain is provided
    domain_emb = None
    if body.domain and embedder.is_available():
        try:
            domain_emb = embedder.embed(body.domain[:2000])
        except Exception:
            pass

    result = {"agents": [], "skills": [], "sessions": []}

    if q_emb:
        result["agents"] = crud.search_agents(wid, q_emb, body.top_n_agents, body.query,
                                               domain=body.domain, domain_emb=domain_emb)
        result["skills"] = crud.search_skills(wid, q_emb, body.top_n_skills, body.query, agent_name=body.agent_name)
    else:
        result["agents"] = crud.list_agents(wid)[:body.top_n_agents]
        result["skills"] = crud.list_skills(wid)[:body.top_n_skills]

    # Session search (text fallback always available)
    sessions = crud.list_sessions(wid)
    for s in sessions[:10]:
        if q_emb:
            msgs = crud.search_messages(s["id"], q_emb, body.top_n_sessions, 2.0)
            if not msgs:
                msgs = crud.search_messages_by_text(s["id"], body.query, limit=3)
        else:
            msgs = crud.search_messages_by_text(s["id"], body.query, limit=3)
        if msgs:
            result["sessions"].append({"session_id": s["id"], "session_name": s["name"], "matches": msgs})

    return result


@app.get("/workspaces/{workspace_id}", response_model=models.Workspace)
async def get_workspace(workspace_id: str):
    w = crud.get_workspace(workspace_id)
    if not w: raise HTTPException(404)
    return w


# ── Sessions ──
@app.post("/workspaces/{workspace_id}/sessions", response_model=models.Session)
async def create_session(workspace_id: str, body: models.SessionCreate):
    w = crud.get_workspace(workspace_id)
    if not w: raise HTTPException(404, "Workspace not found")
    return crud.create_session(workspace_id, body.name, body.metadata)

@app.get("/workspaces/{workspace_id}/sessions", response_model=list[models.Session])
async def list_sessions(workspace_id: str):
    return crud.list_sessions(workspace_id)

@app.get("/workspaces/{workspace_id}/sessions/{session_id}", response_model=models.Session)
async def get_session(workspace_id: str, session_id: str):
    s = crud.get_session(session_id)
    if not s: raise HTTPException(404)
    return s

@app.patch("/workspaces/{workspace_id}/sessions/{session_id}", response_model=models.Session)
async def update_session(workspace_id: str, session_id: str, body: models.SessionUpdate):
    s = crud.update_session(session_id, metadata=body.metadata, name=body.name)
    if not s: raise HTTPException(404)
    return s

@app.post("/workspaces/{workspace_id}/sessions/{session_id}/end", response_model=models.Session)
async def end_session(workspace_id: str, session_id: str):
    s = crud.end_session(session_id)
    if not s: raise HTTPException(404)
    return s

@app.get("/workspaces/{workspace_id}/sessions/{session_id}/summary", response_model=models.SummaryResponse)
async def session_summary(workspace_id: str, session_id: str):
    s = crud.get_session(session_id)
    if not s: raise HTTPException(404)
    stats = crud.session_stats(session_id)
    return models.SummaryResponse(
        summary=s.get("summary", ""),
        **stats,
    )


# ── Peers ──
@app.post("/workspaces/{workspace_id}/peers", response_model=models.Peer)
async def create_peer(workspace_id: str, body: models.PeerCreate):
    w = crud.get_workspace(workspace_id)
    if not w: raise HTTPException(404, "Workspace not found")
    return crud.create_peer(workspace_id, body.name, body.metadata)

@app.get("/workspaces/{workspace_id}/peers", response_model=list[models.Peer])
async def list_peers(workspace_id: str):
    return crud.list_peers(workspace_id)

@app.get("/workspaces/{workspace_id}/peers/{peer_id}", response_model=models.Peer)
async def get_peer(workspace_id: str, peer_id: str):
    p = crud.get_peer(peer_id)
    if not p: raise HTTPException(404)
    return p

@app.patch("/workspaces/{workspace_id}/peers/{peer_id}", response_model=models.Peer)
async def update_peer(workspace_id: str, peer_id: str, body: models.PeerUpdate):
    p = crud.get_peer(peer_id)
    if not p: raise HTTPException(404, "Peer not found")
    return crud.update_peer(peer_id, metadata=body.metadata)


# ── Messages ──
@app.post("/workspaces/{workspace_id}/sessions/{session_id}/messages", response_model=models.Message)
async def create_message(workspace_id: str, session_id: str, body: models.MessageCreate):
    s = crud.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    # Auto-create or resolve peers
    peer = crud.get_or_create_peer(workspace_id, body.role)
    # Generate embedding if content exists
    emb = None
    if body.content and embedder.is_available():
        try: emb = embedder.embed(body.content[:2000])
        except Exception: pass
    return crud.create_message(session_id, peer["id"], body.content, body.role,
                                body.message_type, body.metadata, emb)

@app.get("/workspaces/{workspace_id}/sessions/{session_id}/messages", response_model=list[models.Message])
async def list_messages(workspace_id: str, session_id: str,
                         limit: int = Query(100), offset: int = Query(0)):
    return crud.list_messages(session_id, limit, offset)


# ── Context / Search ──
@app.post("/workspaces/{workspace_id}/sessions/{session_id}/context")
async def get_context(workspace_id: str, session_id: str, body: models.ContextRequest):
    s = crud.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    
    q_emb = None
    if body.last_message and embedder.is_available():
        try: q_emb = embedder.embed(body.last_message[:2000])
        except Exception: pass
    
    if q_emb:
        result = crud.get_context(session_id, q_emb, body.search_top_k, body.search_max_distance)
    else:
        # Fallback: return recent messages
        msgs = crud.search_messages_by_text(session_id, body.last_message or "", limit=10)
        result = {"messages": msgs, "conclusions": []}

    return {
        "context": result["messages"] + result["conclusions"],
        "representation": {
            "observer": body.observer,
            "observed": body.observed,
            "message_count": len(result["messages"]),
            "conclusion_count": len(result["conclusions"]),
        }
    }


# ── Conclusions ──
@app.post("/workspaces/{workspace_id}/sessions/{session_id}/conclusions", response_model=models.Conclusion)
async def create_conclusion(workspace_id: str, session_id: str, body: models.ConclusionCreate):
    s = crud.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found")
    emb = None
    if body.content and embedder.is_available():
        try: emb = embedder.embed(body.content[:2000])
        except Exception: pass
    return crud.create_conclusion(session_id, body.content, body.metadata, emb)

@app.get("/workspaces/{workspace_id}/sessions/{session_id}/conclusions", response_model=list[models.Conclusion])
async def list_conclusions(workspace_id: str, session_id: str):
    return crud.list_conclusions(session_id)


# ── Search across sessions ──
@app.post("/workspaces/{workspace_id}/search")
async def search_across_sessions(workspace_id: str, body: models.ContextRequest):
    """Search across all sessions in a workspace."""
    sessions = crud.list_sessions(workspace_id)
    if not sessions:
        return {"results": []}
    
    q_emb = None
    if body.last_message and embedder.is_available():
        try: q_emb = embedder.embed(body.last_message[:2000])
        except Exception: pass
    
    results = []
    for s in sessions[:20]:  # Search last 20 sessions
        if q_emb:
            msgs = crud.search_messages(s["id"], q_emb, body.search_top_k, body.search_max_distance)
        else:
            msgs = crud.search_messages_by_text(s["id"], body.last_message or "", limit=5)
        if msgs:
            results.append({"session_id": s["id"], "session_name": s["name"], "matches": msgs})
    
    return {"results": results}


# ── Memories ──

@app.post("/workspace/by-name/{name}/memories")
async def create_memory(name: str, body: models.MemoryCreate):
    """Store a memory with type, importance, confidence."""
    ws = crud.get_or_create_workspace(name)
    emb = None
    if body.content and embedder.is_available():
        try:
            emb = embedder.embed(body.content[:2000])
        except Exception:
            pass
    return crud.create_memory(
        ws["id"], body.type, body.content, body.user_id, body.agent_name,
        body.importance, body.confidence, body.source, body.metadata,
        embedding=emb,
    )


@app.post("/workspace/by-name/{name}/memories/search")
async def search_memories(name: str, body: models.MemorySearchRequest):
    """Search memories by semantic similarity + keyword + importance boost."""
    ws = crud.get_or_create_workspace(name)
    q_emb = None
    if body.query and embedder.is_available():
        try:
            q_emb = embedder.embed(body.query[:2000])
        except Exception:
            pass
    return crud.search_memories(
        ws["id"], body.query, q_emb, body.type, body.user_id,
        body.agent_name, body.top_n,
    )


@app.post("/workspace/by-name/{name}/memories/preflight")
async def preflight_command(name: str, body: models.PreflightRequest):
    """Check a command against stored command_policy memories."""
    ws = crud.get_or_create_workspace(name)
    q_emb = None
    if body.command and embedder.is_available():
        try:
            q_emb = embedder.embed(body.command[:2000])
        except Exception:
            pass
    return crud.preflight_command(ws["id"], body.command, q_emb)


@app.get("/workspace/by-name/{name}/memories/hot")
async def get_hot_memories_endpoint(name: str, limit: int = Query(15)):
    """Get hot memories (always-injected) — highest importance first."""
    ws = crud.get_or_create_workspace(name)
    return {"memories": crud.get_hot_memories(ws["id"], limit)}


class MemoryTableSearchRequest(BaseModel):
    table: str = "memories"
    query: str = ""
    top_n: int = 10


@app.post("/workspace/by-name/{name}/memory-tables/search")
async def search_memory_table_endpoint(name: str, body: MemoryTableSearchRequest):
    """Semantic search across one memory table:
    memories | conclusions | preferences | behaviour | users | agent_written_memory."""
    ws = crud.get_or_create_workspace(name)
    q_emb = None
    if body.query and embedder.is_available():
        try:
            q_emb = embedder.embed(body.query[:2000])
        except Exception:
            pass
    results = crud.search_memory_table(ws["id"], body.table, body.query, q_emb, body.top_n)
    return {"table": body.table, "results": results}


class AgentMemoryWriteRequest(BaseModel):
    content: str
    memory_type: str = "agent_note"
    importance: float = 0.6
    confidence: float = 0.8
    durability: float = 0.5
    metadata: dict = {}


@app.post("/workspace/by-name/{name}/memory-tables/agent-write")
async def agent_write_memory(name: str, body: AgentMemoryWriteRequest):
    """Write a memory authored by the agent into agent_written_memory."""
    ws = crud.get_or_create_workspace(name)
    emb = None
    if body.content and embedder.is_available():
        try:
            emb = embedder.embed(body.content[:2000])
        except Exception:
            pass
    return crud.create_agent_written_memory(
        ws["id"], body.content, body.memory_type, body.importance,
        body.confidence, body.durability, body.metadata, emb,
    )


@app.get("/workspace/by-name/{name}/memory-tables/agent-write")
async def agent_list_memory(name: str, limit: int = Query(50)):
    """List agent-written memories (most recent first)."""
    ws = crud.get_or_create_workspace(name)
    return {"memories": crud.list_agent_written_memories(ws["id"], limit)}


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str):
    """Get a single memory by ID."""
    m = crud.get_memory(memory_id)
    if not m:
        raise HTTPException(404, "Memory not found")
    return m


@app.patch("/memories/{memory_id}/confidence")
async def update_memory_confidence(memory_id: str, body: dict):
    """Update confidence for a memory (increments observed_count)."""
    confidence = body.get("confidence", 0.8)
    m = crud.update_memory_confidence(memory_id, confidence)
    if not m:
        raise HTTPException(404, "Memory not found")
    return m


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Soft delete a memory."""
    m = crud.get_memory(memory_id)
    if not m:
        raise HTTPException(404, "Memory not found")
    return crud.delete_memory(memory_id)


# ── Reputation ──

@app.post("/workspace/by-name/{name}/reputation/search")
async def search_reputation_endpoint(name: str, body: models.ReputationSearchRequest):
    """Search reputation by semantic similarity to query."""
    ws = crud.get_or_create_workspace(name)
    q_emb = None
    if body.query and embedder.is_available():
        try:
            q_emb = embedder.embed(body.query[:2000])
        except Exception:
            pass
    return crud.search_reputation(q_emb, body.entity_type, body.top_n)


@app.post("/workspace/by-name/{name}/reputation/record")
async def record_reputation(name: str, body: models.ReputationAction):
    """Record a dispatch/success/failure/correction/doom_loop.
    Returns threshold_met for auto-gen detection.
    """
    ws = crud.get_or_create_workspace(name)
    # Generate task embedding for semantic matching when dispatching
    task_emb = None
    if body.task_type and embedder.is_available() and body.action == "dispatch":
        try:
            task_emb = embedder.embed(body.task_type[:2000])
        except Exception:
            pass

    if body.action == "dispatch":
        crud.record_dispatch(body.entity_type, body.entity_name, body.task_type,
                              body.semantic_score, task_emb)
    elif body.action == "success":
        crud.record_success(body.entity_type, body.entity_name, body.task_type,
                             body.tokens_used, body.time_seconds)
    elif body.action == "failure":
        crud.record_failure(body.entity_type, body.entity_name, body.task_type,
                             body.tokens_used, body.time_seconds)
    elif body.action == "correction":
        crud.record_correction(body.entity_type, body.entity_name, body.task_type)
    elif body.action == "doom_loop":
        crud.record_doom_loop(body.entity_type, body.entity_name, body.task_type)
    else:
        raise HTTPException(400, f"Unknown action: {body.action}")

    # Check if auto-gen threshold is met: dispatch_count >= 1 and success_rate < 0.5
    threshold_met = False
    if body.task_type:
        rep = crud.get_reputation(body.entity_type, body.entity_name, body.task_type)
        if rep:
            dc = rep.get("dispatch_count", 0) or 0
            sc = rep.get("success_count", 0) or 0
            rate = sc / max(dc, 1) if dc > 0 else 0
            if dc >= 1 and rate < 0.5:
                threshold_met = True

    return {"status": "ok", "threshold_met": threshold_met}


@app.post("/workspace/by-name/{name}/reputation/evaluate")
async def evaluate_reputation(name: str, body: models.EvaluationRequest):
    """Record evaluator results for an agent/task combination.
    Updates rolling averages for quality scores and tracks doom_loop/overthinking."""
    ws = crud.get_or_create_workspace(name)
    crud.record_evaluation(body.entity_type, body.entity_name, body.task_type, body.model_dump())
    return {"status": "ok"}


@app.get("/workspace/by-name/{name}/reputation/{entity_type}/{entity_name}")
async def get_reputation_endpoint(name: str, entity_type: str, entity_name: str,
                                    task_type: str = Query("")):
    """Get reputation for a specific entity."""
    ws = crud.get_or_create_workspace(name)
    r = crud.get_reputation(entity_type, entity_name, task_type)
    if not r:
        raise HTTPException(404, "Reputation record not found")
    return r


@app.get("/workspace/by-name/{name}/reputation/better-agents")
async def get_better_agent_recommendations(name: str, limit: int = Query(10)):
    """Get reputation records where a better_agent was recommended, sorted by recency."""
    ws = crud.get_or_create_workspace(name)
    conn = crud.get_db()
    rows = conn.execute(
        "SELECT * FROM reputation WHERE entity_type='agent' AND better_agent IS NOT NULL AND better_agent != '' ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [crud.parse_row(r) for r in rows]


# ── Notifications ──

@app.get("/workspace/by-name/{name}/notifications")
async def get_notifications(name: str):
    """Get unread notifications about auto-generated agents and other events."""
    ws = crud.get_or_create_workspace(name)
    conn = crud.get_db()
    rows = conn.execute("SELECT * FROM notifications WHERE read=0 ORDER BY created_at DESC LIMIT 10").fetchall()
    return [dict(r) for r in rows]


# ── Recommendations ──

@app.get("/workspace/by-name/{name}/recommendations")
async def get_recommendations(name: str):
    """Get pending auto-generation recommendations created by the deriver."""
    ws = crud.get_or_create_workspace(name)
    conn = crud.get_db()
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE status='pending' ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/workspace/by-name/{name}/recommendations")
async def create_recommendation(name: str, body: models.CreateRecommendationRequest):
    """Create an auto-generation recommendation live (bypasses deriver cycle)."""
    ws = crud.get_or_create_workspace(name)
    conn = crud.get_db()
    
    # Dedup: check if a pending recommendation already exists for this entity
    existing = conn.execute(
        "SELECT id FROM recommendations WHERE entity_name=? AND status='pending'",
        (body.entity_name,),
    ).fetchone()
    if existing:
        conn.close()
        return {"status": "already_exists", "id": existing["id"]}
    
    rid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO recommendations
           (id, entity_type, entity_name, task_type, dispatch_count, failure_count, reason, status, created_at)
           VALUES (?, 'agent', ?, ?, ?, ?, ?, 'pending', datetime('now'))""",
        (rid, body.entity_name, body.task_type, body.dispatch_count, body.failure_count, body.reason),
    )
    conn.commit()
    conn.close()
    return {"status": "created", "id": rid}


@app.post("/workspace/by-name/{name}/recommendations/{rec_id}/fulfill")
async def fulfill_recommendation(name: str, rec_id: str):
    """Mark a recommendation as fulfilled (agent was generated)."""
    ws = crud.get_or_create_workspace(name)
    conn = crud.get_db()
    conn.execute("UPDATE recommendations SET status='fulfilled' WHERE id=?", (rec_id,))
    conn.commit()
    return {"status": "ok"}
