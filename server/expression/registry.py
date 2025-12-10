"""
Murph - Expression Registry
Central catalog of all available expressions with their metadata.
"""

from __future__ import annotations

from .types import ExpressionCategory, ExpressionMetadata, ExpressionType


# Expression metadata definitions
# Valence: -1.0 (negative) to 1.0 (positive)
# Arousal: 0.0 (calm) to 1.0 (excited)
_EXPRESSION_METADATA: dict[ExpressionType, ExpressionMetadata] = {
    ExpressionType.NEUTRAL: ExpressionMetadata(
        name="neutral",
        display_name="Neutral",
        category=ExpressionCategory.NEUTRAL,
        valence=0.0,
        arousal=0.3,
    ),
    ExpressionType.HAPPY: ExpressionMetadata(
        name="happy",
        display_name="Happy",
        category=ExpressionCategory.SOCIAL,
        valence=0.8,
        arousal=0.6,
    ),
    ExpressionType.SAD: ExpressionMetadata(
        name="sad",
        display_name="Sad",
        category=ExpressionCategory.COMFORT,
        valence=-0.6,
        arousal=0.2,
    ),
    ExpressionType.CURIOUS: ExpressionMetadata(
        name="curious",
        display_name="Curious",
        category=ExpressionCategory.PLAY,
        valence=0.4,
        arousal=0.5,
    ),
    ExpressionType.SURPRISED: ExpressionMetadata(
        name="surprised",
        display_name="Surprised",
        category=ExpressionCategory.NEUTRAL,
        valence=0.2,
        arousal=0.8,
    ),
    ExpressionType.SLEEPY: ExpressionMetadata(
        name="sleepy",
        display_name="Sleepy",
        category=ExpressionCategory.COMFORT,
        valence=0.1,
        arousal=0.1,
    ),
    ExpressionType.PLAYFUL: ExpressionMetadata(
        name="playful",
        display_name="Playful",
        category=ExpressionCategory.PLAY,
        valence=0.7,
        arousal=0.8,
    ),
    ExpressionType.LOVE: ExpressionMetadata(
        name="love",
        display_name="Love",
        category=ExpressionCategory.SOCIAL,
        valence=0.9,
        arousal=0.5,
    ),
    ExpressionType.SCARED: ExpressionMetadata(
        name="scared",
        display_name="Scared",
        category=ExpressionCategory.SAFETY,
        valence=-0.7,
        arousal=0.9,
    ),
    ExpressionType.ALERT: ExpressionMetadata(
        name="alert",
        display_name="Alert",
        category=ExpressionCategory.SAFETY,
        valence=0.0,
        arousal=0.7,
    ),
}


class ExpressionRegistry:
    """
    Central catalog of all available expressions.

    Provides methods to retrieve expression metadata, validate expression names,
    and filter expressions by category.
    """

    @staticmethod
    def get(name: str) -> ExpressionMetadata | None:
        """
        Get metadata for an expression by name.

        Args:
            name: Expression name (e.g., "happy", "sad")

        Returns:
            ExpressionMetadata if found, None otherwise
        """
        try:
            expr_type = ExpressionType(name)
            return _EXPRESSION_METADATA.get(expr_type)
        except ValueError:
            return None

    @staticmethod
    def get_by_type(expr_type: ExpressionType) -> ExpressionMetadata:
        """
        Get metadata for an expression by type.

        Args:
            expr_type: The ExpressionType enum value

        Returns:
            ExpressionMetadata for the given type

        Raises:
            KeyError: If the expression type is not registered
        """
        return _EXPRESSION_METADATA[expr_type]

    @staticmethod
    def validate(name: str) -> bool:
        """
        Check if an expression name is valid.

        Args:
            name: Expression name to validate

        Returns:
            True if the name is a valid expression, False otherwise
        """
        try:
            ExpressionType(name)
            return True
        except ValueError:
            return False

    @staticmethod
    def list_by_category(category: ExpressionCategory) -> list[ExpressionMetadata]:
        """
        Get all expressions in a specific category.

        Args:
            category: The category to filter by

        Returns:
            List of ExpressionMetadata for expressions in that category
        """
        return [
            meta
            for meta in _EXPRESSION_METADATA.values()
            if meta.category == category
        ]

    @staticmethod
    def get_all() -> dict[str, ExpressionMetadata]:
        """
        Get all registered expressions.

        Returns:
            Dictionary mapping expression names to their metadata
        """
        return {meta.name: meta for meta in _EXPRESSION_METADATA.values()}

    @staticmethod
    def get_all_types() -> list[ExpressionType]:
        """
        Get all expression types.

        Returns:
            List of all ExpressionType enum values
        """
        return list(ExpressionType)

    @staticmethod
    def get_by_valence_range(
        min_valence: float,
        max_valence: float,
    ) -> list[ExpressionMetadata]:
        """
        Get expressions within a valence range.

        Args:
            min_valence: Minimum valence (-1.0 to 1.0)
            max_valence: Maximum valence (-1.0 to 1.0)

        Returns:
            List of expressions within the valence range
        """
        return [
            meta
            for meta in _EXPRESSION_METADATA.values()
            if min_valence <= meta.valence <= max_valence
        ]

    @staticmethod
    def get_by_arousal_range(
        min_arousal: float,
        max_arousal: float,
    ) -> list[ExpressionMetadata]:
        """
        Get expressions within an arousal range.

        Args:
            min_arousal: Minimum arousal (0.0 to 1.0)
            max_arousal: Maximum arousal (0.0 to 1.0)

        Returns:
            List of expressions within the arousal range
        """
        return [
            meta
            for meta in _EXPRESSION_METADATA.values()
            if min_arousal <= meta.arousal <= max_arousal
        ]
