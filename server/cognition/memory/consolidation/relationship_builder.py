"""
Murph - Relationship Builder
Builds and maintains relationship narratives for known people using LLM.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from server.llm.prompts import (
    RELATIONSHIP_NARRATIVE_SYSTEM_PROMPT,
    RELATIONSHIP_NARRATIVE_USER_PROMPT,
)
from server.llm.types import LLMMessage

from ..memory_types import EventMemory, InsightMemory, PersonMemory

if TYPE_CHECKING:
    from server.llm.services.llm_service import LLMService

    from .config import ConsolidationConfig

logger = logging.getLogger("murph.consolidation.relationship_builder")


class RelationshipBuilder:
    """
    Builds and maintains relationship narratives for known people.

    Analyzes:
    - Interaction history (frequency, types)
    - Sentiment trajectory
    - Shared experiences
    - Trust level evolution

    Generates:
    - Relationship summaries ("Alice is Murph's favorite human")
    - Trajectory analysis ("Relationship with Bob has cooled recently")
    - Preference insights ("Chris likes to play fetch")

    Usage:
        builder = RelationshipBuilder(llm_service, config)
        insight = await builder.build_relationship_narrative(person, events)
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: ConsolidationConfig,
    ) -> None:
        """
        Initialize the relationship builder.

        Args:
            llm_service: LLM service for generating narratives
            config: Consolidation configuration
        """
        self._llm = llm_service
        self._config = config
        self._last_update_times: dict[str, float] = {}
        self._narratives_generated = 0

    async def build_relationship_narrative(
        self,
        person: PersonMemory,
        events: list[EventMemory],
        previous_narrative: InsightMemory | None = None,
        force: bool = False,
    ) -> InsightMemory | None:
        """
        Generate or update relationship narrative for a person.

        Args:
            person: Person memory to build narrative for
            events: Recent events involving this person
            previous_narrative: Previous narrative to consider for trajectory
            force: Force update even if interval hasn't passed

        Returns:
            Generated insight, or None if not ready/failed
        """
        if not self._config.relationship_building_enabled:
            return None

        # Check if it's time to update this person's narrative
        person_id = person.person_id
        current_time = time.time()
        last_update = self._last_update_times.get(person_id, 0)

        if not force and (current_time - last_update < self._config.relationship_update_interval):
            return None

        if not self._llm.is_available:
            return None

        # Build the narrative
        insight = await self._generate_narrative(person, events, previous_narrative)

        if insight:
            self._last_update_times[person_id] = current_time
            self._narratives_generated += 1
            logger.info(f"Generated relationship narrative for {person.name or person.person_id}")

        return insight

    async def _generate_narrative(
        self,
        person: PersonMemory,
        events: list[EventMemory],
        previous_narrative: InsightMemory | None,
    ) -> InsightMemory | None:
        """
        Generate the actual narrative using LLM.

        Args:
            person: Person to generate narrative for
            events: Events involving this person
            previous_narrative: Previous narrative for trajectory comparison

        Returns:
            Generated insight, or None if failed
        """
        # Format timestamps
        first_seen = datetime.fromtimestamp(person.first_seen).strftime("%Y-%m-%d")
        last_seen = datetime.fromtimestamp(person.last_seen).strftime("%Y-%m-%d %H:%M")

        # Format events
        if events:
            events_text = "\n".join(
                f"- {e.event_type}: {e.outcome}, {self._format_time_ago(e.timestamp)}"
                for e in sorted(events, key=lambda x: x.timestamp, reverse=True)[:10]
            )
        else:
            events_text = "No recent events"

        # Previous narrative
        previous_text = (
            previous_narrative.content if previous_narrative else "None (first narrative)"
        )

        # Build prompt
        user_prompt = RELATIONSHIP_NARRATIVE_USER_PROMPT.format(
            person_name=person.name or "Unknown",
            person_id=person.person_id,
            familiarity=f"{person.familiarity_score:.0f}/100",
            sentiment=self._format_sentiment(person.sentiment),
            trust=f"{person.trust_score:.0f}/100",
            first_seen=first_seen,
            last_seen=last_seen,
            interaction_count=person.interaction_count,
            tags=", ".join(person.tags) if person.tags else "none",
            events=events_text,
            previous_narrative=previous_text,
        )

        messages = [
            LLMMessage(role="system", content=RELATIONSHIP_NARRATIVE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Generate narrative
        cache_key = f"relationship:{person.person_id}:{person.interaction_count}"
        response = await self._llm.complete(
            messages,
            cache_key=cache_key,
            temperature=0.7,
            max_tokens=400,
        )

        if not response:
            logger.warning(f"Failed to generate narrative for {person.person_id}")
            return None

        # Parse response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for relationship narrative: {response.content[:100]}")
            return None

        # Create insight
        insight = InsightMemory(
            insight_type="relationship_narrative",
            subject_type="person",
            subject_id=person.person_id,
            content=data.get("narrative", ""),
            summary=f"{person.name or 'Person'}: {data.get('trajectory', 'stable')}"[:100],
            source_event_ids=[e.event_id for e in events[:10]],
            confidence=data.get("confidence", 0.8),
            tags=set(data.get("key_traits", [])),
        )

        return insight

    async def update_all_relationships(
        self,
        people: dict[str, PersonMemory],
        events_by_person: dict[str, list[EventMemory]],
        previous_narratives: dict[str, InsightMemory],
    ) -> list[InsightMemory]:
        """
        Update narratives for all familiar people.

        Called periodically to refresh relationship narratives.

        Args:
            people: Dictionary of person_id -> PersonMemory
            events_by_person: Events grouped by person_id
            previous_narratives: Previous narratives by person_id

        Returns:
            List of updated narratives
        """
        insights: list[InsightMemory] = []

        for person_id, person in people.items():
            if not person.is_familiar:
                continue

            events = events_by_person.get(person_id, [])
            previous = previous_narratives.get(person_id)

            insight = await self.build_relationship_narrative(
                person, events, previous, force=False
            )
            if insight:
                insights.append(insight)

        return insights

    def analyze_trajectory(
        self,
        current_sentiment: float,
        previous_sentiment: float | None,
    ) -> str:
        """
        Analyze relationship trajectory from sentiment changes.

        Args:
            current_sentiment: Current sentiment value (-1 to 1)
            previous_sentiment: Previous sentiment value, or None

        Returns:
            Trajectory string: "improving", "stable", or "declining"
        """
        if previous_sentiment is None:
            return "stable"

        delta = current_sentiment - previous_sentiment

        if delta > 0.2:
            return "improving"
        elif delta < -0.2:
            return "declining"
        else:
            return "stable"

    def _format_sentiment(self, sentiment: float) -> str:
        """Format sentiment value as descriptive string."""
        if sentiment > 0.5:
            return f"very positive ({sentiment:.2f})"
        elif sentiment > 0.1:
            return f"positive ({sentiment:.2f})"
        elif sentiment < -0.5:
            return f"very negative ({sentiment:.2f})"
        elif sentiment < -0.1:
            return f"negative ({sentiment:.2f})"
        else:
            return f"neutral ({sentiment:.2f})"

    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable time ago string."""
        age = time.time() - timestamp
        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        else:
            return f"{int(age / 86400)}d ago"

    def get_stats(self) -> dict[str, Any]:
        """Get builder statistics."""
        return {
            "narratives_generated": self._narratives_generated,
            "tracked_people": len(self._last_update_times),
            "enabled": self._config.relationship_building_enabled,
        }
