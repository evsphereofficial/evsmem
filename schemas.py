"""Pydantic v2 data models for the autonomous evsmem memory agent.

This module defines the schemas for the memory lifecycle: raw conversation
ingestion, all memory categories, tool I/O interfaces, agent state, and
batch processing results.
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Raw Conversation Queue ──

class RawMessage(BaseModel):
    """A single raw message from a conversation, queued for processing."""
    id: int
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


class RawMessageBatch(BaseModel):
    """A batch of raw messages ready for memory derivation."""
    id: int = 0  # auto-increment PK from DB
    messages: list[RawMessage]
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    processed: bool = False
    conversation_id: str


# ── Base Memory ──

class BaseMemory(BaseModel):
    """Root memory model — all memory categories inherit from this."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: list[float] | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: int = Field(default=5, ge=1, le=10)
    recency: datetime = Field(default_factory=lambda: datetime.utcnow())
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    source_batch_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Memory Categories ──

class HotMemory(BaseMemory):
    """Transient memory with an expiry and context binding."""
    expires_at: datetime
    context_id: str


class UserPreference(BaseMemory):
    """A learned preference about the user."""
    preference_key: str
    preference_value: str
    category: str


class BehaviorPattern(BaseMemory):
    """A detected behavioral pattern with frequency tracking."""
    pattern_type: str
    frequency: float = 0.0  # 0.0–1.0 scale
    evidence: list[str] = Field(default_factory=list)


class Conclusion(BaseMemory):
    """A derived conclusion or insight with supporting evidence."""
    conclusion_type: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # overrides base


class LongTermMemory(BaseMemory):
    """Persistent long-term knowledge with tags for categorisation."""
    memory_type: str
    tags: list[str] = Field(default_factory=list)


class Relationship(BaseMemory):
    """A connection between two observed entities."""
    source_entity: str
    target_entity: str
    relationship_type: str  # e.g. "trusts", "works_with", "mentions"
    properties: dict[str, Any] = Field(default_factory=dict)


# ── Category registry (for dispatch) ──

MEMORY_CATEGORIES: dict[str, type[BaseMemory]] = {
    "hot_memory": HotMemory,
    "user_preference": UserPreference,
    "behavior_pattern": BehaviorPattern,
    "conclusion": Conclusion,
    "long_term_memory": LongTermMemory,
    "relationship": Relationship,
}

# Map each category to its SQL table name
CATEGORY_TABLE_MAP: dict[str, str] = {
    "hot_memory": "hot_memories",
    "user_preference": "user_preferences",
    "behavior_pattern": "behavior_patterns",
    "conclusion": "conclusions_mem",       # distinguishes from Honcho conclusions
    "long_term_memory": "long_term_memories",
    "relationship": "relationships",
}


# ── Tool Input / Output Schemas ──

class CreateMemoryInput(BaseModel):
    """Input for creating a new memory."""
    category: str
    content: str
    confidence: float = 0.5
    importance: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_batch_id: int | None = None


class CreateMemoryOutput(BaseModel):
    """Result of a memory creation attempt."""
    memory_id: str
    category: str
    success: bool
    duplicate: bool = False
    merged_with: str | None = None


class UpdateMemoryInput(BaseModel):
    """Input for updating an existing memory."""
    memory_id: str
    updates: dict[str, Any]


class UpdateMemoryOutput(BaseModel):
    """Result of a memory update."""
    memory_id: str
    success: bool


class DeleteMemoryInput(BaseModel):
    """Input for deleting a memory."""
    memory_id: str


class DeleteMemoryOutput(BaseModel):
    """Result of a memory deletion."""
    memory_id: str
    success: bool


class SearchMemoryInput(BaseModel):
    """Input for searching memories."""
    query: str
    category: str | None = None
    top_k: int = 10
    threshold: float = 0.5


SearchMemoryOutputModel = HotMemory | UserPreference | BehaviorPattern | Conclusion | LongTermMemory | Relationship


class MemoryResult(BaseModel):
    """A single memory search result with its relevance score."""
    memory: SearchMemoryOutputModel
    score: float
    category: str


class SearchMemoryOutput(BaseModel):
    """Container for search results."""
    results: list[MemoryResult]


class AskMemoryAgentInput(BaseModel):
    """Input for querying the memory agent."""
    question: str
    context: str | None = None


class AskMemoryAgentOutput(BaseModel):
    """Response from the memory agent."""
    answer: str
    sources: list[str] = Field(default_factory=list)


# ── Agent / Batch State ──

class MemoryAgentState(BaseModel):
    """Persistent state of the autonomous memory agent."""
    status: Literal["idle", "processing", "sleeping"] = "idle"
    last_processed_at: datetime | None = None
    cycle_count: int = 0
    total_memories_created: int = 0
    total_memories_updated: int = 0
    current_batch_id: int | None = None


class BatchProcessingResult(BaseModel):
    """Outcome of processing a single batch of raw messages."""
    batch_id: int
    memories_created: list[str] = Field(default_factory=list)
    memories_updated: list[str] = Field(default_factory=list)
    summary: str = ""
    tool_calls_executed: int = 0
