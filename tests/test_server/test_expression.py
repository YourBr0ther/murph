"""Tests for the expression system."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.expression import (
    ExpressionCategory,
    ExpressionMetadata,
    ExpressionRegistry,
    ExpressionSelector,
    ExpressionType,
)


class TestExpressionType:
    """Tests for ExpressionType enum."""

    def test_all_expression_types_exist(self):
        """Test all expected expression types are defined."""
        expected = [
            "neutral",
            "happy",
            "sad",
            "curious",
            "surprised",
            "sleepy",
            "playful",
            "love",
            "scared",
            "alert",
        ]
        for name in expected:
            assert ExpressionType(name).value == name

    def test_expression_type_values_match_pi_display(self):
        """Test expression type values match Pi display expectations."""
        # These must match the Pi display controller's expression names
        assert ExpressionType.NEUTRAL.value == "neutral"
        assert ExpressionType.HAPPY.value == "happy"
        assert ExpressionType.SAD.value == "sad"
        assert ExpressionType.CURIOUS.value == "curious"
        assert ExpressionType.SURPRISED.value == "surprised"
        assert ExpressionType.SLEEPY.value == "sleepy"
        assert ExpressionType.PLAYFUL.value == "playful"
        assert ExpressionType.LOVE.value == "love"
        assert ExpressionType.SCARED.value == "scared"
        assert ExpressionType.ALERT.value == "alert"


class TestExpressionCategory:
    """Tests for ExpressionCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories are defined."""
        assert ExpressionCategory.NEUTRAL.value == "neutral"
        assert ExpressionCategory.SOCIAL.value == "social"
        assert ExpressionCategory.PLAY.value == "play"
        assert ExpressionCategory.COMFORT.value == "comfort"
        assert ExpressionCategory.SAFETY.value == "safety"


class TestExpressionMetadata:
    """Tests for ExpressionMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating expression metadata."""
        meta = ExpressionMetadata(
            name="happy",
            display_name="Happy",
            category=ExpressionCategory.SOCIAL,
            valence=0.8,
            arousal=0.6,
        )

        assert meta.name == "happy"
        assert meta.display_name == "Happy"
        assert meta.category == ExpressionCategory.SOCIAL
        assert meta.valence == 0.8
        assert meta.arousal == 0.6

    def test_metadata_valence_validation(self):
        """Test valence must be between -1.0 and 1.0."""
        with pytest.raises(ValueError):
            ExpressionMetadata(
                name="test",
                display_name="Test",
                category=ExpressionCategory.NEUTRAL,
                valence=1.5,  # Invalid
                arousal=0.5,
            )

        with pytest.raises(ValueError):
            ExpressionMetadata(
                name="test",
                display_name="Test",
                category=ExpressionCategory.NEUTRAL,
                valence=-1.5,  # Invalid
                arousal=0.5,
            )

    def test_metadata_arousal_validation(self):
        """Test arousal must be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            ExpressionMetadata(
                name="test",
                display_name="Test",
                category=ExpressionCategory.NEUTRAL,
                valence=0.5,
                arousal=1.5,  # Invalid
            )

        with pytest.raises(ValueError):
            ExpressionMetadata(
                name="test",
                display_name="Test",
                category=ExpressionCategory.NEUTRAL,
                valence=0.5,
                arousal=-0.1,  # Invalid
            )

    def test_metadata_is_frozen(self):
        """Test metadata is immutable."""
        meta = ExpressionMetadata(
            name="happy",
            display_name="Happy",
            category=ExpressionCategory.SOCIAL,
            valence=0.8,
            arousal=0.6,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            meta.name = "sad"


class TestExpressionRegistry:
    """Tests for ExpressionRegistry."""

    def test_get_valid_expression(self):
        """Test getting metadata for a valid expression."""
        meta = ExpressionRegistry.get("happy")

        assert meta is not None
        assert meta.name == "happy"
        assert meta.category == ExpressionCategory.SOCIAL

    def test_get_invalid_expression(self):
        """Test getting metadata for an invalid expression returns None."""
        meta = ExpressionRegistry.get("nonexistent")
        assert meta is None

    def test_get_by_type(self):
        """Test getting metadata by ExpressionType."""
        meta = ExpressionRegistry.get_by_type(ExpressionType.SCARED)

        assert meta.name == "scared"
        assert meta.category == ExpressionCategory.SAFETY

    def test_validate_valid_name(self):
        """Test validating a valid expression name."""
        assert ExpressionRegistry.validate("happy") is True
        assert ExpressionRegistry.validate("neutral") is True
        assert ExpressionRegistry.validate("scared") is True

    def test_validate_invalid_name(self):
        """Test validating an invalid expression name."""
        assert ExpressionRegistry.validate("invalid") is False
        assert ExpressionRegistry.validate("") is False
        assert ExpressionRegistry.validate("HAPPY") is False  # Case-sensitive

    def test_list_by_category_social(self):
        """Test listing expressions by social category."""
        social = ExpressionRegistry.list_by_category(ExpressionCategory.SOCIAL)

        names = [m.name for m in social]
        assert "happy" in names
        assert "love" in names

    def test_list_by_category_safety(self):
        """Test listing expressions by safety category."""
        safety = ExpressionRegistry.list_by_category(ExpressionCategory.SAFETY)

        names = [m.name for m in safety]
        assert "scared" in names
        assert "alert" in names

    def test_get_all(self):
        """Test getting all expressions."""
        all_expr = ExpressionRegistry.get_all()

        assert len(all_expr) == 10
        assert "happy" in all_expr
        assert "sad" in all_expr
        assert "neutral" in all_expr

    def test_get_all_types(self):
        """Test getting all expression types."""
        types = ExpressionRegistry.get_all_types()

        assert len(types) == 10
        assert ExpressionType.HAPPY in types
        assert ExpressionType.SAD in types

    def test_get_by_valence_range(self):
        """Test getting expressions by valence range."""
        positive = ExpressionRegistry.get_by_valence_range(0.5, 1.0)
        names = [m.name for m in positive]

        assert "happy" in names
        assert "playful" in names
        assert "love" in names
        assert "sad" not in names
        assert "scared" not in names

    def test_get_by_arousal_range(self):
        """Test getting expressions by arousal range."""
        calm = ExpressionRegistry.get_by_arousal_range(0.0, 0.3)
        names = [m.name for m in calm]

        assert "sleepy" in names
        assert "sad" in names
        assert "playful" not in names


class TestExpressionSelector:
    """Tests for ExpressionSelector."""

    @pytest.fixture
    def selector(self):
        """Create an expression selector."""
        return ExpressionSelector()

    def test_select_for_needs_empty(self, selector):
        """Test selecting expression with empty needs returns neutral."""
        result = selector.select_for_needs({})
        assert result == ExpressionType.NEUTRAL

    def test_select_for_needs_critical_energy(self, selector):
        """Test low energy returns sleepy expression."""
        needs = {"energy": 15.0, "social": 80.0, "play": 70.0}
        result = selector.select_for_needs(needs)
        assert result == ExpressionType.SLEEPY

    def test_select_for_needs_critical_social(self, selector):
        """Test low social returns sad expression."""
        needs = {"energy": 80.0, "social": 10.0, "play": 70.0}
        result = selector.select_for_needs(needs)
        assert result == ExpressionType.SAD

    def test_select_for_needs_critical_safety(self, selector):
        """Test low safety returns scared expression."""
        needs = {"energy": 80.0, "social": 80.0, "safety": 20.0}
        result = selector.select_for_needs(needs)
        assert result == ExpressionType.SCARED

    def test_select_for_needs_satisfied(self, selector):
        """Test satisfied needs returns positive expression."""
        needs = {"energy": 90.0, "social": 80.0, "play": 75.0, "affection": 95.0}
        result = selector.select_for_needs(needs)
        # Most satisfied is affection -> love
        assert result == ExpressionType.LOVE

    def test_select_for_needs_moderate(self, selector):
        """Test moderate needs returns neutral expression."""
        needs = {"energy": 50.0, "social": 50.0, "play": 50.0}
        result = selector.select_for_needs(needs)
        assert result == ExpressionType.NEUTRAL

    def test_select_for_behavior_greet(self, selector):
        """Test selecting expression for greet behavior."""
        result = selector.select_for_behavior("greet")
        assert result == ExpressionType.HAPPY

    def test_select_for_behavior_explore(self, selector):
        """Test selecting expression for explore behavior."""
        result = selector.select_for_behavior("explore")
        assert result == ExpressionType.CURIOUS

    def test_select_for_behavior_sleep(self, selector):
        """Test selecting expression for sleep behavior."""
        result = selector.select_for_behavior("sleep")
        assert result == ExpressionType.SLEEPY

    def test_select_for_behavior_retreat(self, selector):
        """Test selecting expression for retreat behavior."""
        result = selector.select_for_behavior("retreat")
        assert result == ExpressionType.SCARED

    def test_select_for_behavior_unknown(self, selector):
        """Test selecting expression for unknown behavior returns neutral."""
        result = selector.select_for_behavior("unknown_behavior")
        assert result == ExpressionType.NEUTRAL

    def test_select_for_behavior_compound_name(self, selector):
        """Test selecting expression for compound behavior name."""
        result = selector.select_for_behavior("explore_room")
        assert result == ExpressionType.CURIOUS

    def test_select_for_emotion_positive_high_arousal(self, selector):
        """Test selecting expression for positive high arousal."""
        result = selector.select_for_emotion(valence=0.8, arousal=0.8)
        # Should match playful (0.7, 0.8) or happy (0.8, 0.6)
        assert result in [ExpressionType.PLAYFUL, ExpressionType.HAPPY]

    def test_select_for_emotion_negative_high_arousal(self, selector):
        """Test selecting expression for negative high arousal."""
        result = selector.select_for_emotion(valence=-0.7, arousal=0.9)
        assert result == ExpressionType.SCARED

    def test_select_for_emotion_positive_low_arousal(self, selector):
        """Test selecting expression for positive low arousal."""
        result = selector.select_for_emotion(valence=0.1, arousal=0.1)
        assert result == ExpressionType.SLEEPY

    def test_select_for_emotion_neutral(self, selector):
        """Test selecting expression for neutral emotion."""
        result = selector.select_for_emotion(valence=0.0, arousal=0.3)
        assert result == ExpressionType.NEUTRAL

    def test_select_for_needs_with_person_context(self, selector):
        """Test context override when person detected."""
        needs = {"energy": 50.0, "social": 50.0}
        context = MagicMock()
        context.person_detected = True
        context.near_edge = False
        context.is_held = False
        context.is_touched = False

        result = selector.select_for_needs(needs, context)
        assert result == ExpressionType.HAPPY

    def test_select_for_needs_with_edge_context(self, selector):
        """Test context override when near edge."""
        needs = {"energy": 90.0, "social": 90.0}
        context = MagicMock()
        context.person_detected = False
        context.near_edge = True
        context.is_held = False
        context.is_touched = False

        result = selector.select_for_needs(needs, context)
        assert result == ExpressionType.SCARED

    def test_select_for_needs_with_held_context(self, selector):
        """Test context override when being held."""
        needs = {"energy": 50.0}
        context = MagicMock()
        context.person_detected = False
        context.near_edge = False
        context.is_held = True
        context.is_touched = False

        result = selector.select_for_needs(needs, context)
        assert result == ExpressionType.CURIOUS

    def test_get_category_expressions(self, selector):
        """Test getting expressions for a category."""
        social = selector.get_category_expressions(ExpressionCategory.SOCIAL)

        assert ExpressionType.HAPPY in social
        assert ExpressionType.LOVE in social
        assert ExpressionType.SCARED not in social
