"""
evsmem/memory_pipeline.py — High-level orchestrator for the memory subsystem.

Manages lifecycle: initialise all components, start / stop the scheduler,
handle errors, provide status, and expose a convenience ``query()``
interface for the ``ask_memory_agent`` EvAgent tool.
"""

from __future__ import annotations

import logging
from typing import Any

from evsmem.config import (
    EVSMEM_DB_PATH,
    EVAGENT_DB_PATH,
    MEMORY_AGENT_LLM_ENDPOINT,
    MEMORY_AGENT_LLM_MODEL,
)

from .memory_store import MemoryStore
from .retrieval import RetrievalEngine
from .memory_tools import MemoryTools
from .deriver import Deriver
from .memory_agent import MemoryAgent
from .memory_scheduler import MemoryScheduler

logger = logging.getLogger(__name__)


class MemoryPipeline:
    """High-level orchestrator for the evsmem memory subsystem.

    Manages the full pipeline lifecycle:

    *   **Deriver** — polls the ev-agent SQLite DB for new messages.
    *   **MemoryStore** — SQLite persistence for the raw-message queue
        and all memory categories.
    *   **RetrievalEngine** — hybrid dense + FTS5 retrieval for RAG.
    *   **MemoryTools** — tool wrappers with pre-write duplicate detection.
    *   **MemoryAgent** — 2B LLM-powered autonomous curator.
    *   **MemoryScheduler** — periodic trigger that connects Deriver → Agent.

    Usage
    -----
    .. code-block:: python

        pipeline = MemoryPipeline()
        pipeline.initialize()
        pipeline.start()                     # blocking loop
        # — or —
        result = pipeline.process_now()      # single cycle

        # Query the agent from an external tool
        answer = pipeline.query("What does the user prefer?")
    """

    def __init__(
        self,
        evagent_db_path: str | None = None,
        evsmem_db_path: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.evagent_db_path = evagent_db_path or EVAGENT_DB_PATH
        self.evsmem_db_path = evsmem_db_path or EVSMEM_DB_PATH
        self.llm_endpoint = llm_endpoint or MEMORY_AGENT_LLM_ENDPOINT
        self.llm_model = llm_model or MEMORY_AGENT_LLM_MODEL

        # Lazy-initialised components
        self.store: MemoryStore | None = None
        self.retrieval: RetrievalEngine | None = None
        self.tools: MemoryTools | None = None
        self.deriver: Deriver | None = None
        self.memory_agent: MemoryAgent | None = None
        self.scheduler: MemoryScheduler | None = None
        self._initialized = False

    # ── Initialisation ─────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create and wire all pipeline components.

        Safe to call multiple times — subsequent calls are no-ops once
        ``_initialized`` is ``True``.
        """
        if self._initialized:
            return

        # Run DB migration on startup (creates evsmem-prefixed tables)
        from evsmem.migrations import run_migration

        run_migration(self.evsmem_db_path)

        logger.info(
            "Initialising memory pipeline (evsmem DB: %s, agent DB: %s)",
            self.evsmem_db_path,
            self.evagent_db_path or "(default)",
        )

        # Persistence
        self.store = MemoryStore(self.evsmem_db_path)

        # Retrieval engine (hybrid dense + FTS5)
        self.retrieval = RetrievalEngine(self.store, self.evsmem_db_path)

        # Tool wrappers for the agent
        self.tools = MemoryTools(self.store, self.retrieval)

        # Ingestion watcher
        self.deriver = Deriver(
            evagent_db_path=self.evagent_db_path,
            evsmem_db_path=self.evsmem_db_path,
        )

        # Autonomous memory curator
        self.memory_agent = MemoryAgent(
            llm_endpoint=self.llm_endpoint,
            llm_model=self.llm_model,
            store=self.store,
            retrieval=self.retrieval,
            tools=self.tools,
        )

        # Scheduler that ties Deriver → Agent
        self.scheduler = MemoryScheduler(
            deriver=self.deriver,
            memory_agent=self.memory_agent,
            store=self.store,
        )

        self._initialized = True
        logger.info("Memory pipeline initialised")

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler (blocking).

        This call does not return until ``stop()`` is called from
        another thread.
        """
        if not self._initialized:
            self.initialize()
        logger.info("Starting memory pipeline scheduler (blocking)")
        self.scheduler.start()

    def stop(self) -> None:
        """Signal the scheduler to stop on its next cycle."""
        if self.scheduler is not None:
            self.scheduler.stop()
        logger.info("Memory pipeline stopped")

    def process_now(self) -> dict[str, Any]:
        """Trigger a single non-blocking processing cycle.

        Returns the same status dict as ``MemoryScheduler.cycle()``.
        """
        if not self._initialized:
            self.initialize()
        return self.scheduler.run_once()

    # ── Query interface ────────────────────────────────────────────────

    def query(self, question: str, context: str | None = None) -> str:
        """Ask the memory agent a question (backed by RAG).

        This is the implementation backing the ``ask_memory_agent``
        EvAgent tool call.

        Args:
            question: Natural-language question to answer.
            context: Optional additional context to include in the prompt.

        Returns:
            The agent's answer as a string.
        """
        if not self._initialized:
            self.initialize()
        return self.memory_agent.ask(question, context)

    def search(
        self, query_text: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search memory via the hybrid retrieval engine.

        Args:
            query_text: Free-text search query.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts with keys such as ``id``, ``content``,
            ``_category``, ``_rrf_score``.
        """
        if not self._initialized:
            self.initialize()
        return self.retrieval.retrieve_relevant_context(
            query_text, top_k=top_k
        )

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get a snapshot of pipeline and agent state."""
        if not self._initialized:
            return {"initialized": False}

        state = self.memory_agent.state
        return {
            "initialized": True,
            "status": state.status,
            "cycle_count": state.cycle_count,
            "total_memories_created": state.total_memories_created,
            "total_memories_updated": state.total_memories_updated,
            "last_processed_at": (
                str(state.last_processed_at)
                if state.last_processed_at
                else None
            ),
            "current_batch_id": state.current_batch_id,
        }
