"""
Murph - Behavior Reasoner
LLM-assisted behavior selection for ambiguous situations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from ..prompts.reasoning_prompts import (
    BEHAVIOR_REASONING_SYSTEM_PROMPT,
    BEHAVIOR_REASONING_USER_PROMPT,
)
from ..types import BehaviorRecommendation, LLMMessage

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..services.llm_service import LLMService
    from server.cognition.behavior.context import WorldContext
    from server.cognition.behavior.evaluator import ScoredBehavior

logger = logging.getLogger(__name__)


class BehaviorReasoner:
    """
    Uses LLM to help select behaviors in ambiguous situations.

    Only consulted when the top behaviors have similar scores,
    indicating the utility AI is uncertain about the best choice.

    Features:
    - Configurable score threshold for consultation
    - Context summarization for LLM
    - Caching based on context hash
    - Graceful fallback to top utility score

    Usage:
        reasoner = BehaviorReasoner(llm_service)
        if await reasoner.should_consult(scored_behaviors):
            recommendation = await reasoner.recommend_behavior(
                context, scored_behaviors, recent_behaviors
            )
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize behavior reasoner.

        Args:
            llm_service: LLM service for API calls
            config: LLM configuration (uses service config if None)
        """
        self._llm = llm_service
        self._config = config or llm_service._config
        self._threshold = self._config.reasoning_score_threshold

        # Stats
        self._consultations = 0
        self._recommendations_used = 0
        self._parse_errors = 0

        # Recent behaviors tracking
        self._recent_behaviors: list[str] = []
        self._max_recent = 10

        logger.info(f"BehaviorReasoner created (threshold={self._threshold})")

    async def should_consult(
        self,
        scored_behaviors: list[ScoredBehavior],
    ) -> bool:
        """
        Determine if LLM should be consulted for behavior selection.

        Consults when:
        - Multiple behaviors have similar high scores (ambiguous)
        - LLM service is available
        - Reasoning is enabled

        Args:
            scored_behaviors: Behaviors sorted by score (highest first)

        Returns:
            True if LLM consultation is recommended
        """
        if not self._config.reasoning_enabled:
            return False

        if not self._llm.is_available:
            return False

        if len(scored_behaviors) < 2:
            return False

        # Check if top behaviors are close in score
        top_score = scored_behaviors[0].total_score
        second_score = scored_behaviors[1].total_score

        if top_score == 0:
            return False

        # Calculate relative score difference
        score_diff = (top_score - second_score) / top_score
        return score_diff < self._threshold

    async def recommend_behavior(
        self,
        context: WorldContext,
        scored_behaviors: list[ScoredBehavior],
        recent_behaviors: list[str] | None = None,
    ) -> BehaviorRecommendation | None:
        """
        Get LLM recommendation for which behavior to select.

        Args:
            context: Current world context
            scored_behaviors: Behaviors sorted by score (highest first)
            recent_behaviors: Recently executed behavior names

        Returns:
            BehaviorRecommendation, or None if service unavailable
        """
        if not await self._llm._ensure_initialized():
            return None

        self._consultations += 1

        # Build context summary
        context_summary = self._build_context_summary(context)
        behavior_options = self._build_behavior_options(scored_behaviors[:5])
        recent = recent_behaviors or self._recent_behaviors

        # Build messages
        messages = [
            LLMMessage(role="system", content=BEHAVIOR_REASONING_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=BEHAVIOR_REASONING_USER_PROMPT.format(
                    context=context_summary,
                    behaviors=behavior_options,
                    recent=", ".join(recent[-5:]) if recent else "none",
                ),
            ),
        ]

        # Use cache based on context state hash
        cache_key = f"behavior_{hash(str(context.get_state())[:200])}"

        response = await self._llm.complete(messages, cache_key=cache_key)
        if response is None:
            return None

        return self._parse_response(response.content, scored_behaviors)

    def record_behavior_used(self, behavior_name: str) -> None:
        """
        Record that a behavior was executed.

        Args:
            behavior_name: Name of behavior that was used
        """
        self._recent_behaviors.append(behavior_name)
        if len(self._recent_behaviors) > self._max_recent:
            self._recent_behaviors.pop(0)

    def _build_context_summary(self, context: WorldContext) -> str:
        """
        Build a concise context summary for the LLM.

        Args:
            context: WorldContext to summarize

        Returns:
            Human-readable context summary
        """
        triggers = context.get_active_triggers()

        lines = [
            "Current situation:",
            f"- Person detected: {context.person_detected}",
        ]

        if context.person_detected:
            lines.append(f"  - Familiar: {context.person_is_familiar}")
            if context.remembered_person_name:
                lines.append(f"  - Name: {context.remembered_person_name}")
            if context.person_distance:
                lines.append(f"  - Distance: {context.person_distance:.0f}cm")

        lines.extend([
            f"- Time since interaction: {context.time_since_last_interaction:.0f}s",
            f"- Being held: {context.is_being_held}",
            f"- Being petted: {context.is_being_petted}",
            f"- Zone safety: {context.current_zone_safety:.1f}",
            f"- Position confidence: {context.position_confidence:.1f}",
        ])

        if triggers:
            lines.append(f"- Active triggers: {', '.join(triggers[:10])}")

        return "\n".join(lines)

    def _build_behavior_options(
        self,
        behaviors: list[ScoredBehavior],
    ) -> str:
        """
        Build behavior options list for LLM.

        Args:
            behaviors: Top scored behaviors

        Returns:
            Formatted behavior options string
        """
        lines = []
        for i, sb in enumerate(behaviors, 1):
            tags = ", ".join(sb.behavior.tags[:3]) if sb.behavior.tags else "none"
            needs = ", ".join(sb.behavior.driven_by_needs[:2]) if sb.behavior.driven_by_needs else "none"
            lines.append(
                f"{i}. {sb.behavior.display_name} "
                f"(score: {sb.total_score:.2f}, needs: {needs}, tags: {tags})"
            )
        return "\n".join(lines)

    def _parse_response(
        self,
        content: str,
        scored_behaviors: list[ScoredBehavior],
    ) -> BehaviorRecommendation:
        """
        Parse LLM response into BehaviorRecommendation.

        Args:
            content: Raw LLM response
            scored_behaviors: Original scored behaviors for fallback

        Returns:
            Parsed or fallback recommendation
        """
        # Try to extract JSON from response
        json_str = content.strip()

        # Handle markdown code blocks
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        try:
            data = json.loads(json_str)
            behavior_name = data.get("behavior", scored_behaviors[0].behavior.name)

            # Validate behavior exists in options
            valid_names = {sb.behavior.name for sb in scored_behaviors}
            if behavior_name not in valid_names:
                # Try to match display name
                for sb in scored_behaviors:
                    if sb.behavior.display_name.lower() == behavior_name.lower():
                        behavior_name = sb.behavior.name
                        break
                else:
                    # Fall back to top behavior
                    behavior_name = scored_behaviors[0].behavior.name

            return BehaviorRecommendation(
                recommended_behavior=behavior_name,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
                alternative_behaviors=data.get("alternatives", []),
            )

        except json.JSONDecodeError as e:
            self._parse_errors += 1
            logger.warning(f"Failed to parse behavior recommendation: {e}")
            # Return top utility behavior as fallback
            return BehaviorRecommendation(
                recommended_behavior=scored_behaviors[0].behavior.name,
                reasoning=f"Fallback to top utility score. LLM response: {content[:100]}",
                confidence=0.5,
                alternative_behaviors=[
                    sb.behavior.name for sb in scored_behaviors[1:3]
                ],
            )

    def get_stats(self) -> dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "consultations": self._consultations,
            "recommendations_used": self._recommendations_used,
            "parse_errors": self._parse_errors,
            "threshold": self._threshold,
            "recent_behaviors": self._recent_behaviors.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"BehaviorReasoner(threshold={self._threshold}, "
            f"consultations={self._consultations})"
        )
