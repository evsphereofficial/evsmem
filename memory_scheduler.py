"""
evsmem/memory_scheduler.py — Orchestrates the Deriver → Agent pipeline.

Triggers the Deriver to poll for new messages, assembles batches,
and invokes the ``MemoryAgent`` when a batch reaches the minimum
message threshold.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .deriver import Deriver
from .memory_agent import MemoryAgent
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Default minimum messages before the agent considers a batch
BATCH_THRESHOLD = 10

# Default sleep interval between scheduler cycles (seconds)
DEFAULT_POLL_INTERVAL = 5.0


class MemoryScheduler:
    """Scheduler that manages the ingestion → processing pipeline.

    Flow
    ----
    1. Run ``Deriver.poll_once()`` to check for new messages in the
       ev-agent database and ingest them into the evsmem raw queue.
    2. Query ``MemoryStore.get_unprocessed_batches()`` for batches
       that have reached ``batch_threshold`` messages.
    3. For each ready batch, call ``MemoryAgent.process_batch()``.
    4. Mark the batch as processed in the store.
    5. Sleep for ``poll_interval`` seconds, then repeat.

    Parameters
    ----------
    deriver:
        Ingestion component that watches the ev-agent DB.
    memory_agent:
        Autonomous LLM-powered memory curator.
    store:
        Persistence layer holding both the raw-message queue
        and the memory tables.
    poll_interval:
        Seconds to sleep between scheduler cycles.
    batch_threshold:
        Minimum number of messages in a batch before the agent
        processes it.
    """

    def __init__(
        self,
        deriver: Deriver,
        memory_agent: MemoryAgent,
        store: MemoryStore,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        batch_threshold: int = BATCH_THRESHOLD,
    ) -> None:
        self.deriver = deriver
        self.memory_agent = memory_agent
        self.store = store
        self.poll_interval = poll_interval
        self.batch_threshold = batch_threshold
        self._running = False

    # ── Single cycle ───────────────────────────────────────────────────

    def cycle(self) -> dict[str, Any]:
        """Execute one full processing cycle.

        Returns a status dict with keys:

        * ``ingested`` — number of new raw messages ingested
        * ``batches_processed`` — count of batches handled this cycle
        * ``results`` — list of per-batch processing summaries
        """
        # 1. Ingest new messages from the ev-agent DB
        ingested = self.deriver.run_once()

        # 2. Fetch batches that have reached the threshold
        batches = self.store.get_unprocessed_batches(
            min_size=self.batch_threshold
        )

        processed: list[dict[str, Any]] = []
        for batch in batches:
            logger.info(
                "Processing batch %d (%d messages, conversation=%s)",
                batch.id,
                len(batch.messages),
                batch.conversation_id,
            )

            # 3. Run the agent
            result = self.memory_agent.process_batch(batch)

            # 4. Mark as consumed
            self.store.mark_batch_processed(batch.id)

            processed.append({
                "batch_id": batch.id,
                "conversation_id": batch.conversation_id,
                "message_count": len(batch.messages),
                "memories_created": len(result.memories_created),
                "memories_updated": len(result.memories_updated),
                "tool_calls": result.tool_calls_executed,
                "summary": (
                    result.summary[:120]
                    if result.summary
                    else ""
                ),
            })

            logger.info(
                "Batch %d done: %d created, %d updated, %d tool calls",
                batch.id,
                len(result.memories_created),
                len(result.memories_updated),
                result.tool_calls_executed,
            )

        return {
            "ingested": ingested,
            "batches_processed": len(processed),
            "results": processed,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the blocking scheduling loop.

        Call ``stop()`` from another thread to exit gracefully.
        """
        self._running = True
        logger.info(
            "Scheduler started (poll_interval=%.1fs, threshold=%d)",
            self.poll_interval,
            self.batch_threshold,
        )
        while self._running:
            try:
                status = self.cycle()
                if status["batches_processed"] > 0:
                    logger.debug(
                        "Cycle complete: %s",
                        {k: v for k, v in status.items() if k != "results"},
                    )
            except Exception as exc:
                logger.error("Scheduler cycle error: %s", exc, exc_info=True)
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Signal the scheduler loop to exit on the next iteration."""
        self._running = False
        logger.info("Scheduler stopped")

    def run_once(self) -> dict[str, Any]:
        """Execute a single non-blocking cycle.

        Useful for testing or for triggering processing from an
        external API endpoint (e.g. ``/process-now``).
        """
        return self.cycle()
