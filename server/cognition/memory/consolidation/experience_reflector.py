"""
Murph - Experience Reflector
Reflects on behavior outcomes to improve future decisions using LLM.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from server.llm.prompts import (
    EXPERIENCE_REFLECTION_SYSTEM_PROMPT,
    EXPERIENCE_REFLECTION_USER_PROMPT,
)
from server.llm.types import LLMMessage

from ..memory_types import InsightMemory

if TYPE_CHECKING:
    from server.llm.services.llm_service import LLMService

    from .config import ConsolidationConfig

logger = logging.getLogger("murph.consolidation.experience_reflector")


@dataclass
class BehaviorOutcome:
    """Record of a behavior outcome for reflection."""

    behavior_name: str
    result: str  # "completed", "interrupted", "failed"
    duration: float
    was_interrupted: bool
    context_snapshot: dict[str, Any]
    need_changes: dict[str, float]
    timestamp: float = field(default_factory=time.time)


class ExperienceReflector:
    """
    Reflects on behavior outcomes to improve future decisions.

    After a behavior completes, analyzes:
    - Was the outcome positive/negative/neutral?
    - Did it satisfy the intended need?
    - Was the context appropriate?
    - What could be learned?

    Generates insights like:
    - "Playing when tired leads to interrupted behaviors"
    - "Greeting familiar people in the morning is well-received"

    Usage:
        reflector = ExperienceReflector(llm_service, config)
        insight = await reflector.reflect_on_behavior(...)
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: ConsolidationConfig,
    ) -> None:
        """
        Initialize the experience reflector.

        Args:
            llm_service: LLM service for generating reflections
            config: Consolidation configuration
        """
        self._llm = llm_service
        self._config = config
        self._reflection_queue: deque[BehaviorOutcome] = deque(maxlen=20)
        self._reflections_generated = 0
        self._outcomes_seen = 0

    async def reflect_on_behavior(
        self,
        behavior_name: str,
        result: str,
        duration: float,
        context_snapshot: dict[str, Any],
        need_changes: dict[str, float],
        force: bool = False,
    ) -> InsightMemory | None:
        """
        Reflect on a completed behavior and potentially generate insights.

        Only reflects on notable outcomes (probabilistically).

        Args:
            behavior_name: Name of the behavior that completed
            result: Result string ("completed", "interrupted", "failed")
            duration: How long the behavior ran
            context_snapshot: WorldContext state when behavior started
            need_changes: How needs changed after behavior
            force: Force reflection (bypass probability)

        Returns:
            Generated insight, or None if not reflectable/failed
        """
        self._outcomes_seen += 1

        if not self._config.experience_reflection_enabled:
            return None

        # Create outcome record
        outcome = BehaviorOutcome(
            behavior_name=behavior_name,
            result=result,
            duration=duration,
            was_interrupted=result == "interrupted",
            context_snapshot=context_snapshot,
            need_changes=need_changes,
        )

        # Store in queue for potential pattern analysis
        self._reflection_queue.append(outcome)

        # Check if we should reflect on this outcome
        if not force and not self._should_reflect(outcome):
            return None

        if not self._llm.is_available:
            return None

        # Generate reflection
        insight = await self._generate_reflection(outcome)

        if insight:
            self._reflections_generated += 1
            logger.info(f"Generated reflection for behavior '{behavior_name}'")

        return insight

    def _should_reflect(self, outcome: BehaviorOutcome) -> bool:
        """
        Determine if this behavior outcome warrants reflection.

        Criteria:
        - Random probability check
        - OR unexpected outcome (failed when conditions seemed good)
        - OR very positive outcome
        - OR novel situation

        Args:
            outcome: The behavior outcome to evaluate

        Returns:
            True if we should reflect on this outcome
        """
        # Failed behaviors are more interesting to reflect on
        if outcome.result == "failed":
            return random.random() < (self._config.reflection_probability * 2)

        # Interrupted behaviors might reveal patterns
        if outcome.was_interrupted:
            return random.random() < (self._config.reflection_probability * 1.5)

        # Check for notable need changes (very effective or very ineffective)
        total_need_change = sum(outcome.need_changes.values())
        if abs(total_need_change) > 30:  # Notable change
            return random.random() < (self._config.reflection_probability * 1.5)

        # Random selection for regular outcomes
        return random.random() < self._config.reflection_probability

    async def _generate_reflection(
        self,
        outcome: BehaviorOutcome,
    ) -> InsightMemory | None:
        """
        Generate LLM reflection on behavior outcome.

        Args:
            outcome: The outcome to reflect on

        Returns:
            Generated insight, or None if failed
        """
        # Format context for prompt
        context_text = self._format_context(outcome.context_snapshot)
        need_changes_text = self._format_need_changes(outcome.need_changes)

        # Build prompt
        user_prompt = EXPERIENCE_REFLECTION_USER_PROMPT.format(
            behavior_name=outcome.behavior_name,
            result=outcome.result,
            duration=f"{outcome.duration:.1f}",
            was_interrupted=outcome.was_interrupted,
            context=context_text,
            need_changes=need_changes_text,
        )

        messages = [
            LLMMessage(role="system", content=EXPERIENCE_REFLECTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Generate reflection (no caching since context varies)
        response = await self._llm.complete(
            messages,
            skip_cache=True,
            temperature=0.8,  # Slightly higher for more varied reflections
            max_tokens=300,
        )

        if not response:
            logger.warning(f"Failed to generate reflection for {outcome.behavior_name}")
            return None

        # Parse response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for reflection: {response.content[:100]}")
            return None

        # Create insight
        was_good = data.get("was_good_choice", True)
        lesson = data.get("lesson", "")

        insight = InsightMemory(
            insight_type="behavior_reflection",
            subject_type="behavior",
            subject_id=outcome.behavior_name,
            content=data.get("reasoning", ""),
            summary=f"{outcome.behavior_name}: {lesson[:80]}" if lesson else f"{outcome.behavior_name}: {outcome.result}",
            source_event_ids=[],  # Reflections don't have source events
            confidence=data.get("confidence", 0.7),
            tags={"good_choice" if was_good else "poor_choice", outcome.result},
        )

        return insight

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context snapshot for prompt."""
        # Extract key context elements
        lines = []

        if context.get("person_detected"):
            familiar = "familiar" if context.get("person_is_familiar") else "unfamiliar"
            lines.append(f"- Person detected: {familiar}")

        if context.get("is_being_petted"):
            lines.append("- Being petted")

        if context.get("is_being_held"):
            lines.append("- Being held")

        if context.get("near_edge"):
            lines.append("- Near edge")

        if context.get("is_dark"):
            lines.append("- Dark environment")

        if context.get("is_loud"):
            lines.append("- Loud environment")

        # Add time since interaction
        time_since = context.get("time_since_last_interaction", 0)
        if time_since > 60:
            lines.append(f"- Time since last interaction: {int(time_since / 60)}m")

        if not lines:
            return "- Normal conditions"

        return "\n".join(lines)

    def _format_need_changes(self, need_changes: dict[str, float]) -> str:
        """Format need changes for prompt."""
        if not need_changes:
            return "No significant changes"

        lines = []
        for need, change in need_changes.items():
            if change > 0:
                lines.append(f"- {need}: +{change:.0f}")
            elif change < 0:
                lines.append(f"- {need}: {change:.0f}")

        return "\n".join(lines) if lines else "No significant changes"

    def get_recent_outcomes(self, limit: int = 10) -> list[BehaviorOutcome]:
        """Get recent behavior outcomes."""
        return list(self._reflection_queue)[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get reflector statistics."""
        return {
            "reflections_generated": self._reflections_generated,
            "outcomes_seen": self._outcomes_seen,
            "reflection_rate": (
                self._reflections_generated / self._outcomes_seen
                if self._outcomes_seen > 0
                else 0.0
            ),
            "queued_outcomes": len(self._reflection_queue),
            "enabled": self._config.experience_reflection_enabled,
        }
