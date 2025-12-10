"""
Murph - Context Builder
Builds rich context for LLM prompts using memory and insights.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..prompts import CONTEXT_SUMMARY_SYSTEM_PROMPT, CONTEXT_SUMMARY_USER_PROMPT
from ..types import LLMMessage

if TYPE_CHECKING:
    from server.cognition.behavior.context import WorldContext
    from server.cognition.memory.long_term_memory import LongTermMemory
    from server.cognition.memory.memory_system import MemorySystem
    from server.cognition.memory.memory_types import InsightMemory, PersonMemory

    from ..config import LLMConfig
    from .llm_service import LLMService

logger = logging.getLogger("murph.llm.context_builder")


class ContextBuilder:
    """
    Builds rich context for LLM prompts using memory and insights.

    Provides:
    - Current situation summary
    - Relevant historical patterns
    - Person relationship context
    - Recent behavioral insights
    - Environmental awareness

    Usage:
        builder = ContextBuilder(memory_system, long_term_memory, config)
        context = await builder.build_behavior_context(world_context, needs)
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        long_term_memory: LongTermMemory | None = None,
        llm_service: LLMService | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            memory_system: Memory system for accessing memory
            long_term_memory: Long-term memory for accessing insights
            llm_service: Optional LLM service for generating summaries
            config: LLM configuration
        """
        self._memory = memory_system
        self._ltm = long_term_memory
        self._llm = llm_service
        self._config = config

        # Cache for person contexts (short TTL)
        self._person_context_cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = 60.0  # 1 minute

    async def build_behavior_context(
        self,
        world_context: WorldContext,
        needs_summary: dict[str, Any] | None = None,
        max_tokens: int = 500,
    ) -> str:
        """
        Build rich context for behavior selection.

        Includes:
        - Current triggers and state
        - Active person relationship summary
        - Recent relevant insights
        - Historical patterns for similar situations

        Args:
            world_context: Current world context
            needs_summary: Optional needs system summary
            max_tokens: Maximum tokens for the context

        Returns:
            Formatted context string
        """
        sections = []

        # 1. Current state summary
        state_summary = self._build_state_summary(world_context, needs_summary)
        sections.append(f"Current state:\n{state_summary}")

        # 2. Person context if someone is present
        if world_context.person_detected:
            person_context = await self._build_person_context(world_context)
            if person_context:
                sections.append(f"Person:\n{person_context}")

        # 3. Recent insights (if LTM available)
        if self._ltm:
            insights_context = await self._build_insights_context(world_context)
            if insights_context:
                sections.append(f"Insights:\n{insights_context}")

        # Combine sections
        context = "\n\n".join(sections)

        # Trim if too long (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "..."

        return context

    async def build_conversation_context(
        self,
        world_context: WorldContext,
        person_id: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """
        Build context for conversational responses.

        Includes:
        - Relationship history with person
        - Recent interactions
        - Known preferences

        Args:
            world_context: Current world context
            person_id: Optional specific person ID
            max_tokens: Maximum tokens for the context

        Returns:
            Formatted context string
        """
        sections = []

        # 1. Brief state
        brief_state = self._build_brief_state(world_context)
        sections.append(brief_state)

        # 2. Person relationship
        if person_id or world_context.person_detected:
            target_id = person_id or world_context.remembered_person_name
            if target_id and self._ltm:
                relationship = await self._get_relationship_narrative(target_id)
                if relationship:
                    sections.append(f"Relationship: {relationship}")

        return " | ".join(sections)

    def _build_state_summary(
        self,
        context: WorldContext,
        needs: dict[str, Any] | None,
    ) -> str:
        """Build summary of current state."""
        lines = []

        # Physical state
        if context.is_being_petted:
            lines.append("- Being petted")
        if context.is_being_held:
            lines.append("- Being held")
        if context.near_edge:
            lines.append("- Near edge (caution)")

        # Environment
        if context.is_dark:
            lines.append("- Dark environment")
        if context.is_loud:
            lines.append("- Loud environment")

        # Person
        if context.person_detected:
            dist = context.person_distance
            familiar = "familiar" if context.person_is_familiar else "unfamiliar"
            lines.append(f"- {familiar.capitalize()} person detected ({dist}cm away)")

        # Objects
        if context.objects_in_view:
            lines.append(f"- Objects in view: {', '.join(context.objects_in_view[:3])}")

        # Current behavior
        if context.current_behavior:
            lines.append(f"- Current behavior: {context.current_behavior}")

        # Needs summary
        if needs:
            urgent = needs.get("most_urgent")
            if urgent:
                lines.append(f"- Most urgent need: {urgent}")
            happiness = needs.get("happiness", 50)
            lines.append(f"- Happiness: {happiness:.0f}%")

        return "\n".join(lines) if lines else "- Normal conditions"

    def _build_brief_state(self, context: WorldContext) -> str:
        """Build very brief state summary."""
        states = []

        if context.person_detected:
            states.append("person nearby")
        if context.is_being_petted:
            states.append("being petted")
        if context.is_being_held:
            states.append("being held")
        if context.current_behavior:
            states.append(f"doing: {context.current_behavior}")

        return ", ".join(states) if states else "idle"

    async def _build_person_context(
        self,
        context: WorldContext,
    ) -> str | None:
        """Build context about the currently detected person."""
        # Check cache
        person_name = context.remembered_person_name
        if person_name:
            cached = self._person_context_cache.get(person_name)
            if cached and time.time() - cached[1] < self._cache_ttl:
                return cached[0]

        # Build context
        lines = []

        if context.remembered_person_name:
            lines.append(f"Name: {context.remembered_person_name}")
        elif context.person_is_familiar:
            lines.append("Familiar person (name unknown)")
        else:
            lines.append("Unfamiliar person")

        # Sentiment
        sentiment = context.person_sentiment
        if sentiment > 0.3:
            lines.append("Relationship: positive")
        elif sentiment < -0.3:
            lines.append("Relationship: tense")
        else:
            lines.append("Relationship: neutral")

        # Interaction history
        interaction_count = context.person_interaction_count
        if interaction_count > 10:
            lines.append(f"Many past interactions ({interaction_count})")
        elif interaction_count > 3:
            lines.append(f"Some past interactions ({interaction_count})")

        # Get relationship narrative if available
        if person_name and self._ltm:
            narrative = await self._get_relationship_narrative(person_name)
            if narrative:
                lines.append(f"History: {narrative}")

        result = "\n".join(lines) if lines else None

        # Cache result
        if person_name and result:
            self._person_context_cache[person_name] = (result, time.time())

        return result

    async def _build_insights_context(
        self,
        context: WorldContext,
    ) -> str | None:
        """Build context from relevant insights."""
        if not self._ltm:
            return None

        insights: list[InsightMemory] = []

        # Get relevant insights based on context
        try:
            # Person-specific insights
            if context.remembered_person_name:
                person_insights = await self._ltm.get_insights_for_subject(
                    subject_type="person",
                    subject_id=context.remembered_person_name,
                    limit=3,
                )
                insights.extend(person_insights)

            # Behavior insights
            if context.current_behavior:
                behavior_insights = await self._ltm.get_insights_for_subject(
                    subject_type="behavior",
                    subject_id=context.current_behavior,
                    limit=2,
                )
                insights.extend(behavior_insights)

            # Recent general insights
            if len(insights) < 5:
                recent = await self._ltm.get_recent_insights(limit=5 - len(insights))
                insights.extend(recent)

        except Exception as e:
            logger.warning(f"Error fetching insights: {e}")
            return None

        if not insights:
            return None

        # Format insights
        lines = []
        for insight in insights[:5]:  # Max 5 insights
            lines.append(f"- {insight.summary}")

        return "\n".join(lines)

    async def _get_relationship_narrative(
        self,
        person_id: str,
    ) -> str | None:
        """Get relationship narrative for a person."""
        if not self._ltm:
            return None

        try:
            narratives = await self._ltm.get_insights_for_subject(
                subject_type="person",
                subject_id=person_id,
                insight_type="relationship_narrative",
                limit=1,
            )
            if narratives:
                return narratives[0].content
        except Exception as e:
            logger.warning(f"Error fetching relationship narrative: {e}")

        return None

    async def get_relevant_insights(
        self,
        context: WorldContext,
        max_insights: int = 5,
    ) -> list[InsightMemory]:
        """
        Retrieve insights relevant to current situation.

        Args:
            context: Current world context
            max_insights: Maximum number of insights to return

        Returns:
            List of relevant insights
        """
        if not self._ltm:
            return []

        insights: list[InsightMemory] = []

        try:
            # Get relevant insights
            relevant = await self._ltm.get_relevant_insights(
                min_relevance=0.2,
                limit=max_insights,
            )
            insights.extend(relevant)

        except Exception as e:
            logger.warning(f"Error fetching relevant insights: {e}")

        return insights[:max_insights]

    def clear_cache(self) -> None:
        """Clear the person context cache."""
        self._person_context_cache.clear()
