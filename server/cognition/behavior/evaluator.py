"""
Murph - Behavior Evaluator
Utility AI behavior selection engine.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from .behavior import Behavior
from .behavior_registry import BehaviorRegistry
from .context import WorldContext
from ..needs import NeedsSystem


@dataclass
class ScoredBehavior:
    """
    A behavior with its computed score and score breakdown.

    Used to track why a behavior received its score for debugging
    and to select the highest-scoring behavior.
    """

    behavior: Behavior
    total_score: float
    base_value: float
    need_modifier: float
    personality_modifier: float
    opportunity_bonus: float

    def get_breakdown(self) -> dict[str, float]:
        """Return score breakdown for debugging."""
        return {
            "base_value": self.base_value,
            "need_modifier": self.need_modifier,
            "personality_modifier": self.personality_modifier,
            "opportunity_bonus": self.opportunity_bonus,
            "total_score": self.total_score,
        }

    def __str__(self) -> str:
        return (
            f"ScoredBehavior({self.behavior.name}, "
            f"score={self.total_score:.3f}, "
            f"base={self.base_value:.2f}*need={self.need_modifier:.2f}*"
            f"pers={self.personality_modifier:.2f}*opp={self.opportunity_bonus:.2f})"
        )


class BehaviorEvaluator:
    """
    Utility AI behavior selection engine.

    Evaluates all available behaviors against current needs, personality,
    and world context to select the highest-scoring action.

    The scoring formula is:
        score = base_value * need_modifier * personality_modifier * opportunity_bonus

    Where:
        - base_value: Intrinsic desirability of the behavior (0.5-2.0)
        - need_modifier: How urgent the driving needs are (0.5-2.0)
        - personality_modifier: How personality affects preference (0.3-2.0)
        - opportunity_bonus: Context-appropriate bonuses (1.0-2.0)
    """

    def __init__(
        self,
        needs_system: NeedsSystem,
        registry: BehaviorRegistry | None = None,
    ) -> None:
        """
        Initialize the behavior evaluator.

        Args:
            needs_system: The needs system to read need states from
            registry: Behavior registry to use. If None, creates default.
        """
        self.needs_system = needs_system
        self.registry = registry or BehaviorRegistry()
        self._cooldowns: dict[str, float] = {}  # behavior_name -> last_used_time
        self._last_evaluation_time: float = time.time()

    def evaluate(
        self,
        context: WorldContext | None = None,
        exclude_behaviors: list[str] | None = None,
    ) -> list[ScoredBehavior]:
        """
        Evaluate all behaviors and return them sorted by score.

        Args:
            context: Current world state for opportunity bonuses
            exclude_behaviors: Behavior names to exclude from evaluation

        Returns:
            List of ScoredBehavior sorted by total_score (highest first)
        """
        self._last_evaluation_time = time.time()
        exclude = set(exclude_behaviors or [])
        scored: list[ScoredBehavior] = []

        for behavior in self.registry.get_all():
            # Skip excluded behaviors
            if behavior.name in exclude:
                continue

            # Skip behaviors on cooldown
            if self._is_on_cooldown(behavior):
                continue

            # Calculate all score components
            base_value = behavior.base_value
            need_modifier = self._calculate_need_modifier(behavior)
            personality_modifier = self._calculate_personality_modifier(behavior)
            opportunity_bonus = self._calculate_opportunity_bonus(behavior, context)

            # Calculate total score
            total_score = (
                base_value * need_modifier * personality_modifier * opportunity_bonus
            )

            scored.append(
                ScoredBehavior(
                    behavior=behavior,
                    total_score=total_score,
                    base_value=base_value,
                    need_modifier=need_modifier,
                    personality_modifier=personality_modifier,
                    opportunity_bonus=opportunity_bonus,
                )
            )

        # Sort by score (highest first)
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    def select_best(
        self,
        context: WorldContext | None = None,
        exclude_behaviors: list[str] | None = None,
    ) -> ScoredBehavior | None:
        """
        Select the highest-scoring behavior.

        Args:
            context: Current world state for opportunity bonuses
            exclude_behaviors: Behavior names to exclude

        Returns:
            The highest-scoring behavior, or None if no behaviors available
        """
        scored = self.evaluate(context, exclude_behaviors)
        return scored[0] if scored else None

    def select_top_n(
        self,
        n: int,
        context: WorldContext | None = None,
        exclude_behaviors: list[str] | None = None,
    ) -> list[ScoredBehavior]:
        """
        Select the top N highest-scoring behaviors.

        Args:
            n: Number of behaviors to return
            context: Current world state for opportunity bonuses
            exclude_behaviors: Behavior names to exclude

        Returns:
            List of up to N highest-scoring behaviors
        """
        scored = self.evaluate(context, exclude_behaviors)
        return scored[:n]

    def _calculate_need_modifier(self, behavior: Behavior) -> float:
        """
        Calculate need modifier based on how urgent the driving needs are.

        The modifier scales from 0.5 (needs satisfied) to 2.0 (needs critical).
        If the behavior has no driving needs, returns 1.0 (neutral).

        Args:
            behavior: The behavior to calculate modifier for

        Returns:
            Need modifier in range 0.5-2.0
        """
        if not behavior.driven_by_needs:
            return 1.0  # Neutral if no driving needs

        total_urgency = 0.0
        valid_needs = 0

        for need_name in behavior.driven_by_needs:
            need = self.needs_system.get_need(need_name)
            if need:
                # Get urgency (0-1) weighted by personality importance
                importance = self.needs_system.personality.get_need_importance_modifier(
                    need_name
                )
                total_urgency += need.urgency() * importance
                valid_needs += 1

        if valid_needs == 0:
            return 1.0

        avg_urgency = total_urgency / valid_needs

        # Scale urgency (0-1) to modifier range (0.5-2.0)
        # urgency=0 -> modifier=0.5 (need satisfied, behavior less attractive)
        # urgency=0.5 -> modifier=1.25
        # urgency=1 -> modifier=2.0 (need critical, behavior very attractive)
        return 0.5 + (avg_urgency * 1.5)

    def _calculate_personality_modifier(self, behavior: Behavior) -> float:
        """
        Get personality modifier from the needs system's personality.

        Uses the existing Personality.get_behavior_modifier() method
        which returns a value in range 0.3-2.0.

        Args:
            behavior: The behavior to calculate modifier for

        Returns:
            Personality modifier in range 0.3-2.0
        """
        return self.needs_system.personality.get_behavior_modifier(behavior.name)

    def _calculate_opportunity_bonus(
        self,
        behavior: Behavior,
        context: WorldContext | None,
    ) -> float:
        """
        Calculate opportunity bonus based on context triggers.

        Each active trigger that matches the behavior's opportunity_triggers
        adds 0.3 to the bonus, capped at 2.0 total.

        Args:
            behavior: The behavior to calculate bonus for
            context: Current world context

        Returns:
            Opportunity bonus in range 1.0-2.0
        """
        if context is None or not behavior.opportunity_triggers:
            return 1.0  # No bonus without context or triggers

        active_count = sum(
            1 for trigger in behavior.opportunity_triggers
            if context.has_trigger(trigger)
        )

        if active_count == 0:
            return 1.0

        # Each active trigger adds 0.3, capped at 2.0
        bonus = 1.0 + (active_count * 0.3)
        return min(2.0, bonus)

    def _is_on_cooldown(self, behavior: Behavior) -> bool:
        """
        Check if behavior is still on cooldown.

        Args:
            behavior: The behavior to check

        Returns:
            True if behavior is on cooldown and cannot be selected
        """
        if behavior.cooldown_seconds <= 0:
            return False

        last_used = self._cooldowns.get(behavior.name)
        if last_used is None:
            return False

        elapsed = time.time() - last_used
        return elapsed < behavior.cooldown_seconds

    def mark_behavior_used(self, behavior_name: str) -> None:
        """
        Mark a behavior as used for cooldown tracking.

        Should be called when a behavior starts executing.

        Args:
            behavior_name: Name of the behavior that was used
        """
        self._cooldowns[behavior_name] = time.time()

    def clear_cooldown(self, behavior_name: str) -> None:
        """
        Clear cooldown for a specific behavior.

        Args:
            behavior_name: Name of the behavior to clear cooldown for
        """
        self._cooldowns.pop(behavior_name, None)

    def clear_all_cooldowns(self) -> None:
        """Clear all behavior cooldowns."""
        self._cooldowns.clear()

    def get_cooldown_remaining(self, behavior_name: str) -> float:
        """
        Get remaining cooldown time for a behavior.

        Args:
            behavior_name: Name of the behavior to check

        Returns:
            Remaining cooldown in seconds, or 0 if not on cooldown
        """
        behavior = self.registry.get(behavior_name)
        if behavior is None or behavior.cooldown_seconds <= 0:
            return 0.0

        last_used = self._cooldowns.get(behavior_name)
        if last_used is None:
            return 0.0

        elapsed = time.time() - last_used
        remaining = behavior.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "cooldowns": self._cooldowns.copy(),
            "last_evaluation_time": self._last_evaluation_time,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        needs_system: NeedsSystem,
        registry: BehaviorRegistry | None = None,
    ) -> "BehaviorEvaluator":
        """
        Restore a BehaviorEvaluator from saved state.

        Args:
            state: Previously saved state dict
            needs_system: The needs system to use
            registry: Behavior registry to use

        Returns:
            Restored BehaviorEvaluator instance
        """
        evaluator = cls(needs_system=needs_system, registry=registry)
        evaluator._cooldowns = state.get("cooldowns", {}).copy()
        evaluator._last_evaluation_time = state.get(
            "last_evaluation_time", time.time()
        )
        return evaluator

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "total_behaviors": len(self.registry),
            "behaviors_on_cooldown": [
                name for name in self._cooldowns
                if self._is_on_cooldown(self.registry.get(name) or Behavior(name="", display_name=""))
            ],
            "last_evaluation": self._last_evaluation_time,
        }

    def __str__(self) -> str:
        return f"BehaviorEvaluator(behaviors={len(self.registry)})"
