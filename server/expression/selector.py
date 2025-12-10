"""
Murph - Expression Selector
Selects appropriate expressions based on the robot's internal state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import ExpressionRegistry
from .types import ExpressionCategory, ExpressionType

if TYPE_CHECKING:
    from server.cognition.behavior.context import WorldContext


# Mapping of needs to most relevant expressions when the need is low (critical)
_CRITICAL_NEED_EXPRESSIONS: dict[str, ExpressionType] = {
    "energy": ExpressionType.SLEEPY,
    "curiosity": ExpressionType.CURIOUS,
    "play": ExpressionType.PLAYFUL,
    "social": ExpressionType.SAD,
    "affection": ExpressionType.SAD,
    "comfort": ExpressionType.SLEEPY,
    "safety": ExpressionType.SCARED,
}

# Mapping of needs to expressions when the need is satisfied
_SATISFIED_NEED_EXPRESSIONS: dict[str, ExpressionType] = {
    "energy": ExpressionType.HAPPY,
    "curiosity": ExpressionType.HAPPY,
    "play": ExpressionType.PLAYFUL,
    "social": ExpressionType.HAPPY,
    "affection": ExpressionType.LOVE,
    "comfort": ExpressionType.NEUTRAL,
    "safety": ExpressionType.NEUTRAL,
}

# Mapping of behavior names to appropriate expressions
_BEHAVIOR_EXPRESSIONS: dict[str, ExpressionType] = {
    # Social behaviors
    "greet": ExpressionType.HAPPY,
    "follow": ExpressionType.HAPPY,
    "interact": ExpressionType.PLAYFUL,
    "approach": ExpressionType.CURIOUS,
    # Play behaviors
    "play": ExpressionType.PLAYFUL,
    "chase": ExpressionType.PLAYFUL,
    "bounce": ExpressionType.PLAYFUL,
    "pounce": ExpressionType.ALERT,
    # Affection behaviors
    "nuzzle": ExpressionType.LOVE,
    "be_petted": ExpressionType.HAPPY,
    "cuddle": ExpressionType.LOVE,
    "request_attention": ExpressionType.CURIOUS,
    # Exploration behaviors
    "explore": ExpressionType.CURIOUS,
    "investigate": ExpressionType.CURIOUS,
    "observe": ExpressionType.CURIOUS,
    "wander": ExpressionType.NEUTRAL,
    # Rest behaviors
    "rest": ExpressionType.SLEEPY,
    "sleep": ExpressionType.SLEEPY,
    "charge": ExpressionType.SLEEPY,
    "find_cozy_spot": ExpressionType.NEUTRAL,
    "settle": ExpressionType.NEUTRAL,
    "adjust_position": ExpressionType.NEUTRAL,
    # Safety behaviors
    "retreat": ExpressionType.SCARED,
    "hide": ExpressionType.SCARED,
    "approach_trusted": ExpressionType.HAPPY,
    "scan": ExpressionType.ALERT,
    # Loneliness behaviors
    "sigh": ExpressionType.SAD,
    "mope": ExpressionType.SAD,
    "perk_up_hopeful": ExpressionType.CURIOUS,
    # Idle
    "idle": ExpressionType.NEUTRAL,
}


class ExpressionSelector:
    """
    Selects appropriate expressions based on the robot's internal state.

    Uses needs values, world context, and behavior names to determine
    the most appropriate facial expression.
    """

    def __init__(self) -> None:
        """Initialize the expression selector."""
        self._registry = ExpressionRegistry

    def select_for_needs(
        self,
        needs: dict[str, float],
        context: WorldContext | None = None,
    ) -> ExpressionType:
        """
        Select an expression based on current needs state.

        The algorithm:
        1. Check for context-based overrides (person detected, danger, etc.)
        2. Find the most critical need (lowest value relative to threshold)
        3. If any need is critical, return the expression for that need
        4. Otherwise, return a positive expression based on most satisfied need

        Args:
            needs: Dictionary mapping need names to values (0-100)
            context: Optional world context for situational overrides

        Returns:
            The most appropriate ExpressionType
        """
        # Check context-based overrides first
        if context is not None:
            override = self._check_context_overrides(context)
            if override is not None:
                return override

        if not needs:
            return ExpressionType.NEUTRAL

        # Find the most critical need (lowest value)
        most_critical_need = None
        lowest_value = float("inf")

        for need_name, value in needs.items():
            if value < lowest_value:
                lowest_value = value
                most_critical_need = need_name

        # If any need is critical (below 30%), show that expression
        if lowest_value < 30.0 and most_critical_need:
            return _CRITICAL_NEED_EXPRESSIONS.get(
                most_critical_need, ExpressionType.NEUTRAL
            )

        # If all needs are reasonably satisfied, show positive expression
        # based on the most satisfied need
        most_satisfied_need = max(needs.items(), key=lambda x: x[1])
        if most_satisfied_need[1] > 70.0:
            return _SATISFIED_NEED_EXPRESSIONS.get(
                most_satisfied_need[0], ExpressionType.HAPPY
            )

        # Default to neutral for moderate states
        return ExpressionType.NEUTRAL

    def select_for_behavior(self, behavior_name: str) -> ExpressionType:
        """
        Select an expression appropriate for a specific behavior.

        Args:
            behavior_name: Name of the behavior being executed

        Returns:
            The most appropriate ExpressionType for that behavior
        """
        # Handle compound behavior names (e.g., "explore_room")
        base_name = behavior_name.split("_")[0]

        # First try exact match
        if behavior_name in _BEHAVIOR_EXPRESSIONS:
            return _BEHAVIOR_EXPRESSIONS[behavior_name]

        # Then try base name
        if base_name in _BEHAVIOR_EXPRESSIONS:
            return _BEHAVIOR_EXPRESSIONS[base_name]

        return ExpressionType.NEUTRAL

    def select_for_emotion(
        self,
        valence: float,
        arousal: float,
    ) -> ExpressionType:
        """
        Select an expression based on emotional valence and arousal.

        Uses the circumplex model of affect to map emotions to expressions.

        Args:
            valence: -1.0 (negative) to 1.0 (positive)
            arousal: 0.0 (calm) to 1.0 (excited)

        Returns:
            The closest matching ExpressionType
        """
        # Find the expression with the closest emotional properties
        all_expressions = self._registry.get_all()
        best_match = ExpressionType.NEUTRAL
        best_distance = float("inf")

        for metadata in all_expressions.values():
            # Calculate Euclidean distance in valence-arousal space
            distance = (
                (metadata.valence - valence) ** 2
                + (metadata.arousal - arousal) ** 2
            ) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_match = ExpressionType(metadata.name)

        return best_match

    def _check_context_overrides(self, context: WorldContext) -> ExpressionType | None:
        """
        Check for context-based expression overrides.

        Args:
            context: Current world context

        Returns:
            An expression override, or None if no override applies
        """
        # Check for person detection - be happy/curious
        if hasattr(context, "person_detected") and context.person_detected:
            return ExpressionType.HAPPY

        # Check for danger/edge detection - be scared/alert
        if hasattr(context, "near_edge") and context.near_edge:
            return ExpressionType.SCARED

        # Check for being picked up - be curious
        if hasattr(context, "is_held") and context.is_held:
            return ExpressionType.CURIOUS

        # Check for being touched - be happy
        if hasattr(context, "is_touched") and context.is_touched:
            return ExpressionType.HAPPY

        return None

    def get_category_expressions(
        self,
        category: ExpressionCategory,
    ) -> list[ExpressionType]:
        """
        Get all expressions in a specific category.

        Args:
            category: The category to filter by

        Returns:
            List of ExpressionTypes in that category
        """
        metadata_list = self._registry.list_by_category(category)
        return [ExpressionType(m.name) for m in metadata_list]
