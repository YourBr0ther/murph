"""
Murph - Event Summarizer
Summarizes sequences of events into meaningful insights using LLM.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from server.llm.prompts import EVENT_SUMMARY_SYSTEM_PROMPT, EVENT_SUMMARY_USER_PROMPT
from server.llm.types import LLMMessage

from ..memory_types import EventMemory, InsightMemory

if TYPE_CHECKING:
    from server.llm.services.llm_service import LLMService

    from .config import ConsolidationConfig

logger = logging.getLogger("murph.consolidation.event_summarizer")


class EventSummarizer:
    """
    Summarizes sequences of events into meaningful insights.

    Example transformations:
    - 3x "petting" events with person_1 -> "Alice frequently pets Murph"
    - "bump" + "retreat" + "explore" -> "Murph learned caution in the corner area"
    - Multiple "greeting" events -> "Murph greeted 5 visitors today"

    Usage:
        summarizer = EventSummarizer(llm_service, config)
        insights = await summarizer.summarize_if_ready(events)
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: ConsolidationConfig,
    ) -> None:
        """
        Initialize the event summarizer.

        Args:
            llm_service: LLM service for generating summaries
            config: Consolidation configuration
        """
        self._llm = llm_service
        self._config = config
        self._last_summarization_time: float = 0.0
        self._summaries_generated = 0

    async def summarize_if_ready(
        self,
        events: list[EventMemory],
        force: bool = False,
    ) -> list[InsightMemory]:
        """
        Summarize events if enough time has passed.

        Groups events and generates summaries for each meaningful cluster.

        Args:
            events: List of events to potentially summarize
            force: Force summarization even if interval hasn't passed

        Returns:
            List of generated insights
        """
        if not self._config.event_summarization_enabled:
            return []

        # Check if it's time to summarize
        current_time = time.time()
        if not force and (
            current_time - self._last_summarization_time
            < self._config.event_summarization_interval
        ):
            return []

        # Check minimum events
        if len(events) < self._config.min_events_for_summary:
            return []

        self._last_summarization_time = current_time

        # Cluster events by type and participant
        clusters = self._cluster_events(events)

        # Generate summaries for each cluster
        insights: list[InsightMemory] = []
        for cluster_key, cluster_events in clusters.items():
            if len(cluster_events) >= self._config.min_events_for_summary:
                insight = await self._summarize_event_cluster(
                    cluster_events, cluster_key
                )
                if insight:
                    insights.append(insight)

        if insights:
            logger.info(f"Generated {len(insights)} event summaries")
            self._summaries_generated += len(insights)

        return insights

    def _cluster_events(
        self,
        events: list[EventMemory],
    ) -> dict[str, list[EventMemory]]:
        """
        Group events into clusters by type and participant.

        Args:
            events: Events to cluster

        Returns:
            Dictionary of cluster_key -> list of events
        """
        clusters: dict[str, list[EventMemory]] = defaultdict(list)

        for event in events:
            # Filter to events within the time window
            age = time.time() - event.timestamp
            if age > self._config.event_cluster_window_seconds:
                continue

            # Create cluster key based on event type and primary participant
            participant = event.participants[0] if event.participants else "general"
            cluster_key = f"{event.event_type}:{participant}"
            clusters[cluster_key].append(event)

        return dict(clusters)

    async def _summarize_event_cluster(
        self,
        events: list[EventMemory],
        cluster_key: str,
    ) -> InsightMemory | None:
        """
        Generate a summary for a cluster of related events.

        Args:
            events: Events to summarize
            cluster_key: Key identifying the cluster (event_type:participant)

        Returns:
            Generated insight, or None if failed
        """
        if not self._llm.is_available:
            return None

        # Parse cluster key
        parts = cluster_key.split(":", 1)
        event_type = parts[0]
        participant = parts[1] if len(parts) > 1 else None

        # Calculate time window
        timestamps = [e.timestamp for e in events]
        time_window = f"{int((max(timestamps) - min(timestamps)) / 60)} minutes"

        # Format events for prompt
        events_text = "\n".join(
            f"- {e.event_type}: {e.outcome}, {int(time.time() - e.timestamp)}s ago"
            for e in sorted(events, key=lambda x: x.timestamp)
        )

        # Build prompt
        user_prompt = EVENT_SUMMARY_USER_PROMPT.format(
            event_type=event_type,
            time_window=time_window,
            event_count=len(events),
            events=events_text,
            participant=participant if participant != "general" else "None",
        )

        messages = [
            LLMMessage(role="system", content=EVENT_SUMMARY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Generate summary
        cache_key = f"event_summary:{cluster_key}:{len(events)}"
        response = await self._llm.complete(
            messages,
            cache_key=cache_key,
            temperature=0.7,
            max_tokens=300,
        )

        if not response:
            logger.warning(f"Failed to generate summary for cluster: {cluster_key}")
            return None

        # Parse response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON response for event summary: {response.content[:100]}")
            return None

        # Create insight
        source_event_ids = [e.event_id for e in events]

        insight = InsightMemory(
            insight_type="event_summary",
            subject_type="person" if participant and participant != "general" else "session",
            subject_id=participant if participant != "general" else None,
            content=data.get("content", ""),
            summary=data.get("summary", "")[:100],
            source_event_ids=source_event_ids,
            confidence=data.get("confidence", 0.7),
            tags=set(data.get("tags", [])),
        )

        logger.debug(f"Generated event summary: {insight.summary}")
        return insight

    async def summarize_specific_events(
        self,
        events: list[EventMemory],
        context: str = "",
    ) -> InsightMemory | None:
        """
        Generate a summary for a specific list of events (not clustered).

        Useful for summarizing arbitrary event sequences.

        Args:
            events: Events to summarize
            context: Additional context for the summary

        Returns:
            Generated insight, or None if failed
        """
        if not events:
            return None

        if not self._llm.is_available:
            return None

        # Determine primary event type
        event_types = [e.event_type for e in events]
        primary_type = max(set(event_types), key=event_types.count)

        # Get participant if any
        all_participants = []
        for e in events:
            all_participants.extend(e.participants)
        participant = all_participants[0] if all_participants else None

        # Calculate time window
        timestamps = [e.timestamp for e in events]
        duration_mins = int((max(timestamps) - min(timestamps)) / 60)
        time_window = f"{duration_mins} minutes" if duration_mins > 0 else "a moment"

        # Format events
        events_text = "\n".join(
            f"- {e.event_type}: {e.outcome}, {int(time.time() - e.timestamp)}s ago"
            for e in sorted(events, key=lambda x: x.timestamp)
        )

        # Build prompt
        user_prompt = EVENT_SUMMARY_USER_PROMPT.format(
            event_type=primary_type,
            time_window=time_window,
            event_count=len(events),
            events=events_text,
            participant=participant if participant else "None",
        )

        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        messages = [
            LLMMessage(role="system", content=EVENT_SUMMARY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.complete(
            messages,
            skip_cache=True,  # Don't cache specific summaries
            temperature=0.7,
            max_tokens=300,
        )

        if not response:
            return None

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for specific event summary: {response.content[:100]}")
            return None

        insight = InsightMemory(
            insight_type="event_summary",
            subject_type="person" if participant else "session",
            subject_id=participant,
            content=data.get("content", ""),
            summary=data.get("summary", "")[:100],
            source_event_ids=[e.event_id for e in events],
            confidence=data.get("confidence", 0.7),
            tags=set(data.get("tags", [])),
        )

        self._summaries_generated += 1
        return insight

    def get_stats(self) -> dict[str, Any]:
        """Get summarizer statistics."""
        return {
            "summaries_generated": self._summaries_generated,
            "last_summarization_time": self._last_summarization_time,
            "enabled": self._config.event_summarization_enabled,
        }
