"""
evsmem/config.py — Centralized configuration for evsmem.

All default paths, table names, and environment variable overrides
are defined here. Import from this module instead of hardcoding paths.

Usage:
    from evsmem.config import get_db_path, get_table_name, EVSMEM_DB_PATH

    # Get the configured database path
    db_path = get_db_path()

    # Get table name for a memory category
    table = get_table_name("hot_memory")  # -> "evsmem_hot_memories"

    # Get LLM configuration
    llm_config = get_llm_config()

    # Get all new table names for schema creation
    tables = get_all_new_table_names()

Environment variables:
    EVSMEM_DB_PATH          — Override the database path
    EVAGENT_DB_PATH         — Override the ev-agent database path
    MEMORY_AGENT_LLM_ENDPOINT — LLM endpoint for the memory agent
    MEMORY_AGENT_LLM_MODEL  — LLM model name
    EVSMEM_BATCH_THRESHOLD  — Batch size threshold (default: 10)
    EVSMEM_POLL_INTERVAL    — Deriver poll interval in seconds (default: 1.0)
    EVSMEM_SCHEDULER_INTERVAL — Scheduler poll interval in seconds (default: 5.0)
    EVSMEM_EMBEDDING_MODEL  — Embedding model name (default: BAAI/bge-m3)
    EVSMEM_EMBEDDING_DIMENSION — Embedding dimension (default: 1024)
"""

import os
from pathlib import Path
from typing import Final, Dict, List

# ============================================================
# Database Paths
# ============================================================

# Default database path — user's existing Honcho-style evsmem database
_DEFAULT_DB_PATH: Final[str] = "C:/Users/Rehan/.evsmem/evsmem.db"

# Override via environment variable
EVSMEM_DB_PATH: str = os.environ.get("EVSMEM_DB_PATH", _DEFAULT_DB_PATH)
EVSMEM_DB_PATH = str(Path(EVSMEM_DB_PATH).resolve())

# EvAgent database (same database in this setup — messages are in Honcho tables)
EVAGENT_DB_PATH: str = os.environ.get("EVAGENT_DB_PATH", EVSMEM_DB_PATH)

# ============================================================
# New EvsMem Table Names (conflict-free with existing 17 Honcho tables)
# ============================================================

# All new tables use the "evsmem_" prefix to avoid collisions with
# existing Honcho tables: workspaces, agents, sessions, peers,
# messages, memories, conclusions, skills, reputation, recommendations,
# notifications, deriver_state

RAW_CONVERSATIONS_TABLE: Final[str] = "evsmem_raw_conversations"
RAW_BATCHES_TABLE: Final[str] = "evsmem_raw_batches"
CURSOR_STATE_TABLE: Final[str] = "evsmem_cursor_state"
HOT_MEMORIES_TABLE: Final[str] = "evsmem_hot_memories"
USER_PREFERENCES_TABLE: Final[str] = "evsmem_user_preferences"
BEHAVIOR_PATTERNS_TABLE: Final[str] = "evsmem_behavior_patterns"
CONCLUSIONS_TABLE: Final[str] = "evsmem_conclusions"
LONG_TERM_MEMORIES_TABLE: Final[str] = "evsmem_long_term_memories"
RELATIONSHIPS_TABLE: Final[str] = "evsmem_relationships"
SCHEMA_VERSION_TABLE: Final[str] = "evsmem_schema_version"

# Map memory categories to table names
CATEGORY_TABLE_MAP: Final[Dict[str, str]] = {
    "hot_memory": HOT_MEMORIES_TABLE,
    "user_preference": USER_PREFERENCES_TABLE,
    "behavior_pattern": BEHAVIOR_PATTERNS_TABLE,
    "conclusion": CONCLUSIONS_TABLE,
    "long_term_memory": LONG_TERM_MEMORIES_TABLE,
    "relationship": RELATIONSHIPS_TABLE,
}

# ============================================================
# LLM / Memory Agent Settings
# ============================================================

MEMORY_AGENT_LLM_ENDPOINT: str = os.environ.get(
    "MEMORY_AGENT_LLM_ENDPOINT",
    "http://localhost:8080/v1/chat/completions"
)
MEMORY_AGENT_LLM_MODEL: str = os.environ.get(
    "MEMORY_AGENT_LLM_MODEL",
    "Qwen2.5-1.5B-Instruct"
)

# ============================================================
# Processing Settings
# ============================================================

BATCH_THRESHOLD: int = int(os.environ.get("EVSMEM_BATCH_THRESHOLD", "10"))
DERIVER_POLL_INTERVAL: float = float(os.environ.get("EVSMEM_POLL_INTERVAL", "1.0"))
SCHEDULER_POLL_INTERVAL: float = float(os.environ.get("EVSMEM_SCHEDULER_INTERVAL", "5.0"))

# ============================================================
# Embedding Settings
# ============================================================

EMBEDDING_MODEL: str = os.environ.get("EVSMEM_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION: int = int(os.environ.get("EVSMEM_EMBEDDING_DIMENSION", "1024"))

# ============================================================
# Helper Functions
# ============================================================


def get_db_path() -> str:
    """Get the configured evsmem database path.

    Returns:
        Absolute path to the evsmem SQLite database as a string.
    """
    return EVSMEM_DB_PATH


def get_evagent_db_path() -> str:
    """Get the configured ev-agent database path (same DB).

    Returns:
        Absolute path to the ev-agent database as a string.
    """
    return EVAGENT_DB_PATH


def get_table_name(category: str) -> str:
    """Get the table name for a memory category.

    Args:
        category: The memory category key (e.g., 'hot_memory', 'conclusion').

    Returns:
        The corresponding evsmem-prefixed table name.
        Falls back to LONG_TERM_MEMORIES_TABLE if the category is unknown.
    """
    return CATEGORY_TABLE_MAP.get(category, LONG_TERM_MEMORIES_TABLE)


def get_llm_config() -> dict:
    """Get LLM configuration for the memory agent.

    Returns:
        Dictionary with 'endpoint' and 'model' keys.
    """
    return {
        "endpoint": MEMORY_AGENT_LLM_ENDPOINT,
        "model": MEMORY_AGENT_LLM_MODEL,
    }


def get_all_new_table_names() -> List[str]:
    """Get all new evsmem table names (for migration/creation).

    Returns:
        List of all evsmem-prefixed table names that should be created
        alongside the existing Honcho tables.
    """
    return [
        RAW_CONVERSATIONS_TABLE,
        RAW_BATCHES_TABLE,
        CURSOR_STATE_TABLE,
        HOT_MEMORIES_TABLE,
        USER_PREFERENCES_TABLE,
        BEHAVIOR_PATTERNS_TABLE,
        CONCLUSIONS_TABLE,
        LONG_TERM_MEMORIES_TABLE,
        RELATIONSHIPS_TABLE,
        SCHEMA_VERSION_TABLE,
    ]


__all__ = [
    "EVSMEM_DB_PATH",
    "EVAGENT_DB_PATH",
    "get_db_path",
    "get_evagent_db_path",
    "get_table_name",
    "get_llm_config",
    "get_all_new_table_names",
    "CATEGORY_TABLE_MAP",
    "BATCH_THRESHOLD",
    "DERIVER_POLL_INTERVAL",
    "SCHEDULER_POLL_INTERVAL",
    "MEMORY_AGENT_LLM_ENDPOINT",
    "MEMORY_AGENT_LLM_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
]
