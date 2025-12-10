"""
Murph - Memory Consolidator
Main orchestrator for all memory consolidation services.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..memory_types import InsightMemory

if TYPE_CHECKING:
    from server.cognition.memory.long_term_memory import LongTermMemory
    from server.cognition.memory.memory_system import MemorySystem
    from server.cognition.memory.short_term_memory import ShortTermMemory
    from server.llm.services.llm_service import LLMService

    from .config import ConsolidationConfig

from .event_summarizer import EventSummarizer
from .experience_reflector import ExperienceReflector
from .relationship_builder import RelationshipBuilder

logger = logging.getLogger("murph.consolidation")


class MemoryConsolidator:
    """
    Orchestrates memory consolidation processes.

    Manages timing and coordination of:
    - Event summarization (hourly)
    - Relationship narrative updates (daily or on significant change)
    - Experience reflection (after notable behaviors)

    Runs as a background component integrated with the orchestrator.

    Usage:
        consolidator = MemoryConsolidator(llm_service, memory_system, ltm, config)
        await consolidator.tick()  # Called periodically from orchestrator
        await consolidator.on_behavior_complete(...)  # After behavior ends
        await consolidator.consolidate_session()  # On shutdown
    """

    def __init__(
        self,
        llm_service: LLMService,
        memory_system: MemorySystem | None = None,
        long_term_memory: LongTermMemory | None = None,
        config: ConsolidationConfig | None = None,
    ) -> None:
        """
        Initialize the memory consolidator.

        Args:
            llm_service: LLM service for generating insights
            memory_system: Memory system for accessing memory
            long_term_memory: Long-term memory for storing insights
            config: Consolidation configuration
        """
        from .config import ConsolidationConfig

        self._llm = llm_service
        self._memory = memory_system
        self._ltm = long_term_memory
        self._config = config or ConsolidationConfig()

        # Initialize sub-services
        self._event_summarizer = EventSummarizer(llm_service, self._config)
        self._relationship_builder = RelationshipBuilder(llm_service, self._config)
        self._experience_reflector = ExperienceReflector(llm_service, self._config)

        # State
        self._enabled = self._config.enabled
        self._last_tick_time: float = 0.0
        self._last_event_summary_time: float = 0.0
        self._last_relationship_update_time: float = 0.0

        # Stats
        self._ticks = 0
        self._insights_saved = 0

        logger.info(f"MemoryConsolidator initialized (enabled: {self._enabled})")

    async def tick(self) -> None:
        """
        Called periodically from orchestrator.

        Checks if any consolidation tasks are due and runs them.
        """
        if not self._enabled:
            return

        current_time = time.time()

        # Rate limit ticks
        if current_time - self._last_tick_time < self._config.consolidation_tick_interval:
            return

        self._last_tick_time = current_time
        self._ticks += 1

        # Run consolidation tasks
        await self._run_event_summarization()
        await self._run_relationship_updates()

        # Periodic insight decay
        if self._ltm and self._ticks % 60 == 0:  # Every ~60 ticks
            await self._decay_insights()

    async def _run_event_summarization(self) -> None:
        """Run event summarization if due."""
        if not self._memory:
            return

        current_time = time.time()
        if current_time - self._last_event_summary_time < self._config.event_summarization_interval:
            return

        # Get recent events from memory
        try:
            events = self._memory.get_recent_events(limit=50)
            if not events:
                return

            insights = await self._event_summarizer.summarize_if_ready(events, force=True)

            # Save insights
            for insight in insights:
                if self._ltm:
                    saved = await self._ltm.save_insight(insight)
                    if saved:
                        self._insights_saved += 1

            self._last_event_summary_time = current_time

        except Exception as e:
            logger.error(f"Error in event summarization: {e}")

    async def _run_relationship_updates(self) -> None:
        """Run relationship narrative updates if due."""
        if not self._memory or not self._ltm:
            return

        current_time = time.time()
        if current_time - self._last_relationship_update_time < self._config.relationship_update_interval:
            return

        try:
            # Get familiar people
            people = self._memory.get_familiar_people()
            if not people:
                return

            # Get events and previous narratives for each person
            for person in people.values():
                events = self._memory.get_events_with_person(person.person_id)

                # Get previous narrative
                previous_narratives = await self._ltm.get_insights_for_subject(
                    subject_type="person",
                    subject_id=person.person_id,
                    insight_type="relationship_narrative",
                    limit=1,
                )
                previous = previous_narratives[0] if previous_narratives else None

                # Build new narrative
                insight = await self._relationship_builder.build_relationship_narrative(
                    person, events, previous, force=True
                )

                if insight and self._ltm:
                    saved = await self._ltm.save_insight(insight)
                    if saved:
                        self._insights_saved += 1

            self._last_relationship_update_time = current_time

        except Exception as e:
            logger.error(f"Error in relationship updates: {e}")

    async def _decay_insights(self) -> None:
        """Decay insight relevance and prune stale ones."""
        if not self._ltm:
            return

        try:
            # Decay relevance
            await self._ltm.decay_insight_relevance(self._config.insight_decay_rate)

            # Prune stale insights
            await self._ltm.prune_stale_insights(self._config.min_relevance_score)

        except Exception as e:
            logger.warning(f"Error decaying insights: {e}")

    async def on_behavior_complete(
        self,
        behavior_name: str,
        result: str,
        duration: float,
        context_snapshot: dict[str, Any],
        need_changes: dict[str, float],
    ) -> InsightMemory | None:
        """
        Called when a behavior completes.

        Triggers experience reflection if warranted.

        Args:
            behavior_name: Name of completed behavior
            result: Result status ("completed", "interrupted", "failed")
            duration: How long the behavior ran
            context_snapshot: WorldContext state at behavior start
            need_changes: How needs changed after behavior

        Returns:
            Generated insight if reflection occurred, None otherwise
        """
        if not self._enabled:
            return None

        # Reflect on the experience
        insight = await self._experience_reflector.reflect_on_behavior(
            behavior_name=behavior_name,
            result=result,
            duration=duration,
            context_snapshot=context_snapshot,
            need_changes=need_changes,
        )

        # Save insight if generated
        if insight and self._ltm:
            saved = await self._ltm.save_insight(insight)
            if saved:
                self._insights_saved += 1
                logger.debug(f"Saved behavior reflection: {insight.summary}")

        return insight

    async def consolidate_session(self) -> dict[str, int]:
        """
        Full consolidation at session end.

        Summarizes all pending events and updates relationships.

        Returns:
            Dict with counts of insights generated
        """
        logger.info("Running session consolidation...")

        counts = {
            "event_summaries": 0,
            "relationship_narratives": 0,
            "reflections": 0,
        }

        if not self._enabled:
            return counts

        # Force event summarization
        if self._memory:
            events = self._memory.get_recent_events(limit=100)
            if events:
                insights = await self._event_summarizer.summarize_if_ready(events, force=True)
                for insight in insights:
                    if self._ltm:
                        saved = await self._ltm.save_insight(insight)
                        if saved:
                            counts["event_summaries"] += 1
                            self._insights_saved += 1

        # Force relationship updates
        if self._memory and self._ltm:
            people = self._memory.get_familiar_people()
            for person in people.values():
                events = self._memory.get_events_with_person(person.person_id)

                previous_narratives = await self._ltm.get_insights_for_subject(
                    subject_type="person",
                    subject_id=person.person_id,
                    insight_type="relationship_narrative",
                    limit=1,
                )
                previous = previous_narratives[0] if previous_narratives else None

                insight = await self._relationship_builder.build_relationship_narrative(
                    person, events, previous, force=True
                )

                if insight:
                    saved = await self._ltm.save_insight(insight)
                    if saved:
                        counts["relationship_narratives"] += 1
                        self._insights_saved += 1

        logger.info(f"Session consolidation complete: {counts}")
        return counts

    def get_stats(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "enabled": self._enabled,
            "ticks": self._ticks,
            "insights_saved": self._insights_saved,
            "last_tick_time": self._last_tick_time,
            "event_summarizer": self._event_summarizer.get_stats(),
            "relationship_builder": self._relationship_builder.get_stats(),
            "experience_reflector": self._experience_reflector.get_stats(),
        }

    def enable(self) -> None:
        """Enable consolidation."""
        self._enabled = True
        logger.info("Memory consolidation enabled")

    def disable(self) -> None:
        """Disable consolidation."""
        self._enabled = False
        logger.info("Memory consolidation disabled")
