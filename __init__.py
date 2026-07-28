"""
evsmem — Autonomous Local Memory Agent for EvAgent.

Provides autonomous memory management via a 2B local LLM that
curates memories stored in SQLite with BGE-M3 embeddings.
"""

from evsmem.memory_pipeline import MemoryPipeline
from evsmem.memory_store import MemoryStore
from evsmem.retrieval import RetrievalEngine
from evsmem.embeddings import generate_embedding, embedding_dimension
from evsmem.deriver import Deriver
from evsmem.planner import (
    memory_search,
    ask_memory_agent,
    get_tool_schemas,
    execute_tool,
)
from evsmem.config import get_db_path, get_evagent_db_path
from evsmem.ddl import get_ddl_statements, get_table_names, get_fts_table_name
from evsmem.migrations import run_migration, needs_migration, get_migration_status

__version__ = "0.2.0"
__all__ = [
    "MemoryPipeline",
    "MemoryStore",
    "RetrievalEngine",
    "generate_embedding",
    "embedding_dimension",
    "Deriver",
    "memory_search",
    "ask_memory_agent",
    "get_tool_schemas",
    "execute_tool",
    "get_db_path",
    "get_evagent_db_path",
    "get_ddl_statements",
    "get_table_names",
    "get_fts_table_name",
    "run_migration",
    "needs_migration",
    "get_migration_status",
]
