"""Pydantic schemas for evsmem API — mirrors Honcho API types."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


def _now():
    return datetime.now(timezone.utc).isoformat()

def _id():
    return str(uuid.uuid4())


# ── Workspace ──
class WorkspaceCreate(BaseModel):
    name: str = "default"
    metadata: dict[str, Any] = {}

class Workspace(BaseModel):
    id: str
    name: str
    metadata: dict[str, Any] = {}
    created_at: str

class WorkspaceGet(BaseModel):
    id: str

# ── Session ──
class SessionCreate(BaseModel):
    name: str = "default"
    metadata: dict[str, Any] = {}
    session_configuration: dict[str, Any] = {}

class Session(BaseModel):
    id: str
    workspace_id: str
    name: str
    metadata: dict[str, Any] = {}
    created_at: str
    ended_at: Optional[str] = None
    summary: Optional[str] = None

class SessionGet(BaseModel):
    session_id: str
    workspace_id: str

class SessionUpdate(BaseModel):
    metadata: Optional[dict[str, Any]] = None
    name: Optional[str] = None

# ── Peer ──
class PeerCreate(BaseModel):
    name: str
    metadata: dict[str, Any] = {}

class Peer(BaseModel):
    id: str
    workspace_id: str
    name: str
    metadata: dict[str, Any] = {}
    created_at: str

class PeerGet(BaseModel):
    peer_id: str
    workspace_id: str

class PeerUpdate(BaseModel):
    metadata: Optional[dict[str, Any]] = None

# ── Message ──
class MessageCreate(BaseModel):
    content: str = ""
    role: str = "user"
    metadata: dict[str, Any] = {}
    message_type: str = "message"

class Message(BaseModel):
    id: str
    session_id: str
    peer_id: str
    content: str
    role: str
    message_type: str
    metadata: dict[str, Any] = {}
    created_at: str

class MessageGet(BaseModel):
    message_id: str
    session_id: str

# ── Context / Search ──
class ContextRequest(BaseModel):
    last_message: str = ""
    observer: str = "agent"
    observed: str = "user"
    search_top_k: int = 5
    search_max_distance: float = 1.5
    include_most_derived: bool = True

class ContextResponse(BaseModel):
    context: list[dict[str, Any]]
    representation: Optional[dict[str, Any]] = None

# ── Conclusion ──
class ConclusionCreate(BaseModel):
    content: str
    metadata: dict[str, Any] = {}

class Conclusion(BaseModel):
    id: str
    session_id: str
    content: str
    metadata: dict[str, Any] = {}
    created_at: str

# ── Summary ──
class SummaryResponse(BaseModel):
    summary: str
    message_count: int
    peer_count: int
    conclusion_count: int

# ── Health ──
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    embeddings: str = "available"

# ── Memory ──
class MemoryCreate(BaseModel):
    content: str
    type: str = "personal"
    user_id: str = ""
    agent_name: str = ""
    importance: float = 0.5
    confidence: float = 0.8
    source: str = "conversation"
    metadata: dict = {}

class MemorySearchRequest(BaseModel):
    query: str
    type: str = ""
    user_id: str = ""
    agent_name: str = ""
    top_n: int = 10
    include_importance: bool = True

class PreflightRequest(BaseModel):
    command: str
    cwd: str = ""

class CreateRecommendationRequest(BaseModel):
    entity_name: str
    task_type: str = ""
    dispatch_count: int = 0
    failure_count: int = 0
    reason: str = ""

# ── Reputation ──

class ReputationRecord(BaseModel):
    entity_type: str
    entity_name: str
    task_type: str = ""
    dispatch_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    correction_count: int = 0
    doom_loop_count: int = 0
    avg_tokens_consumed: float = 0
    avg_time_seconds: float = 0
    total_tokens: int = 0
    use_count: int = 0
    avg_semantic_score: float = 0
    final_score: float = 0

class ReputationSearchRequest(BaseModel):
    query: str
    entity_type: str = "agent"
    top_n: int = 10

class ReputationAction(BaseModel):
    action: str  # 'dispatch', 'success', 'failure', 'correction', 'doom_loop'
    entity_type: str = "agent"
    entity_name: str
    task_type: str = ""
    tokens_used: int = 0
    time_seconds: float = 0
    semantic_score: float = 0.0


class EvaluationRequest(BaseModel):
    entity_type: str = "agent"
    entity_name: str
    task_type: str = ""
    accuracy: float = 0
    completeness: float = 0
    quality_score: float = 0
    efficiency_score: float = 0
    tool_usage_score: float = 0
    doom_loop: bool = False
    overthinking_count: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    time_taken_seconds: float = 0
    verdict: str = "accept"
    better_agent: str = ""
