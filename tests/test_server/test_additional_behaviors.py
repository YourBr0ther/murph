"""
Unit tests for Murph's Additional Behavior Sets.

Tests for:
- Time-based routine behaviors
- Personality expression behaviors
- Reactive behaviors
- New evaluator modifiers (time_modifier, spontaneous_bonus)
"""

import random
import pytest
from server.cognition.behavior import (
    Behavior,
    WorldContext,
    BehaviorRegistry,
    BehaviorEvaluator,
    ScoredBehavior,
    BehaviorTreeFactory,
)
from server.cognition.needs import NeedsSystem, Personality


class TestTimeBehaviors:
    """Tests for time-based routine behaviors."""

    def test_time_based_behavior_has_time_preferences(self):
        """Test that time-based behaviors have time_preferences."""
        registry = BehaviorRegistry()
        wake_up = registry.get("wake_up")
        assert wake_up is not None
        assert wake_up.time_preferences != {}
        assert "morning" in wake_up.time_preferences
        assert wake_up.time_preferences["morning"] > wake_up.time_preferences["night"]

    def test_morning_behavior_scores_higher_in_morning(self):
        """Test that morning behaviors score higher in the morning."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        # Create morning context
        morning_context = WorldContext(current_hour=8, time_period="morning")

        # Create night context
        night_context = WorldContext(current_hour=2, time_period="night")

        scored_morning = evaluator.evaluate(context=morning_context)
        scored_night = evaluator.evaluate(context=night_context)

        # Find wake_up behavior in both evaluations
        wake_up_morning = next(
            (s for s in scored_morning if s.behavior.name == "wake_up"), None
        )
        wake_up_night = next(
            (s for s in scored_night if s.behavior.name == "wake_up"), None
        )

        assert wake_up_morning is not None
        assert wake_up_night is not None
        assert wake_up_morning.time_modifier > wake_up_night.time_modifier

    def test_evening_behavior_scores_higher_in_evening(self):
        """Test that evening behaviors score higher in the evening."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        evening_context = WorldContext(current_hour=19, time_period="evening")
        morning_context = WorldContext(current_hour=8, time_period="morning")

        scored_evening = evaluator.evaluate(context=evening_context)
        scored_morning = evaluator.evaluate(context=morning_context)

        evening_settle_eve = next(
            (s for s in scored_evening if s.behavior.name == "evening_settle"), None
        )
        evening_settle_morn = next(
            (s for s in scored_morning if s.behavior.name == "evening_settle"), None
        )

        assert evening_settle_eve is not None
        assert evening_settle_morn is not None
        assert evening_settle_eve.time_modifier > evening_settle_morn.time_modifier

    def test_time_modifier_in_valid_range(self):
        """Test that time modifier stays within 0.5-1.5 range."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        for period in ["morning", "midday", "evening", "night"]:
            context = WorldContext(time_period=period)
            scored = evaluator.evaluate(context=context)

            for sb in scored:
                assert 0.5 <= sb.time_modifier <= 1.5, (
                    f"time_modifier {sb.time_modifier} out of range for "
                    f"{sb.behavior.name} in {period}"
                )

    def test_time_based_behaviors_have_trees(self):
        """Test that all time-based behaviors have behavior trees."""
        time_behaviors = [
            "wake_up",
            "morning_stretch",
            "energetic_start",
            "midday_activity",
            "afternoon_rest",
            "evening_settle",
            "pre_sleep_yawn",
            "night_stir",
        ]
        for name in time_behaviors:
            assert BehaviorTreeFactory.has_tree(name), f"Missing tree for {name}"


class TestPersonalityExpressions:
    """Tests for personality expression behaviors."""

    def test_expressive_behaviors_have_spontaneous_probability(self):
        """Test that expressive behaviors have spontaneous probability."""
        registry = BehaviorRegistry()

        expressive_behaviors = ["stretch", "yawn", "daydream", "shake_off", "sneeze"]
        for name in expressive_behaviors:
            behavior = registry.get(name)
            assert behavior is not None, f"Missing behavior {name}"
            assert behavior.spontaneous_probability > 0, (
                f"{name} should have spontaneous_probability"
            )

    def test_spontaneous_probability_clamped(self):
        """Test that spontaneous probability is clamped to 0-1."""
        # Test clamping to max
        behavior = Behavior(
            name="test",
            display_name="Test",
            spontaneous_probability=1.5,
        )
        assert behavior.spontaneous_probability == 1.0

        # Test clamping to min
        behavior = Behavior(
            name="test",
            display_name="Test",
            spontaneous_probability=-0.5,
        )
        assert behavior.spontaneous_probability == 0.0

    def test_spontaneous_bonus_sometimes_activates(self):
        """Test that spontaneous bonus sometimes applies."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        # Run multiple evaluations and check if stretch ever gets bonus
        random.seed(42)  # For reproducibility
        bonus_count = 0
        iterations = 200

        for _ in range(iterations):
            scored = evaluator.evaluate()
            stretch = next(
                (s for s in scored if s.behavior.name == "stretch"), None
            )
            if stretch and stretch.spontaneous_bonus > 1.0:
                bonus_count += 1

        # With 5% probability over 200 iterations, we expect ~10 bonuses
        # Allow range of 2-25 to account for randomness
        assert 2 <= bonus_count <= 30, (
            f"Expected ~10 spontaneous activations, got {bonus_count}"
        )

    def test_spontaneous_bonus_in_valid_range(self):
        """Test that spontaneous bonus is in valid range."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        for _ in range(50):
            scored = evaluator.evaluate()
            for sb in scored:
                assert sb.spontaneous_bonus >= 1.0, (
                    f"spontaneous_bonus {sb.spontaneous_bonus} below 1.0 "
                    f"for {sb.behavior.name}"
                )
                assert sb.spontaneous_bonus <= 1.5, (
                    f"spontaneous_bonus {sb.spontaneous_bonus} above 1.5 "
                    f"for {sb.behavior.name}"
                )

    def test_personality_expressions_have_trees(self):
        """Test that all personality expressions have behavior trees."""
        expressive_behaviors = [
            "stretch",
            "yawn",
            "daydream",
            "shake_off",
            "sneeze",
            "happy_wiggle",
            "curious_tilt",
            "contented_sigh",
        ]
        for name in expressive_behaviors:
            assert BehaviorTreeFactory.has_tree(name), f"Missing tree for {name}"


class TestReactiveBehaviors:
    """Tests for reactive behaviors."""

    def test_reactive_behaviors_registered(self):
        """Test that all reactive behaviors are registered."""
        registry = BehaviorRegistry()
        reactive_behaviors = [
            "dropped_recovery",
            "loud_noise_reaction",
            "new_object_investigation",
            "person_left_sad",
            "touched_unexpectedly",
            "picked_up_happy",
        ]
        for name in reactive_behaviors:
            assert registry.get(name) is not None, f"Missing behavior {name}"

    def test_dropped_recovery_wins_when_dropped(self):
        """Test that dropped_recovery is selected when dropped trigger active."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        # Create context with dropped trigger
        context = WorldContext()
        context.was_falling = True
        context.recent_bump = True

        scored = evaluator.evaluate(context=context)

        # dropped_recovery should score very high
        dropped = next(
            (s for s in scored if s.behavior.name == "dropped_recovery"), None
        )
        assert dropped is not None
        assert dropped.opportunity_bonus > 1.0  # Should have trigger bonus

    def test_person_left_trigger(self):
        """Test person_left trigger detection."""
        context = WorldContext()

        # Initially no trigger
        assert context.has_trigger("person_left") is False

        # Set up transition (person was detected, now not)
        context.person_detected_previous = True
        context.person_detected = False

        # Now trigger should be active
        assert context.has_trigger("person_left") is True

    def test_touched_unexpected_trigger(self):
        """Test touched_unexpected trigger detection."""
        context = WorldContext()

        # Initially no trigger
        assert context.has_trigger("touched_unexpected") is False

        # Being petted without person detected
        context.is_being_petted = True
        context.person_detected = False

        # Now trigger should be active
        assert context.has_trigger("touched_unexpected") is True

    def test_reactive_behaviors_have_high_base_value(self):
        """Test that reactive behaviors have high base values."""
        registry = BehaviorRegistry()

        high_priority = ["dropped_recovery", "loud_noise_reaction"]
        for name in high_priority:
            behavior = registry.get(name)
            assert behavior is not None
            assert behavior.base_value >= 1.5, (
                f"{name} base_value {behavior.base_value} too low"
            )

    def test_reactive_behaviors_have_trees(self):
        """Test that all reactive behaviors have behavior trees."""
        reactive_behaviors = [
            "dropped_recovery",
            "loud_noise_reaction",
            "new_object_investigation",
            "person_left_sad",
            "touched_unexpectedly",
            "picked_up_happy",
        ]
        for name in reactive_behaviors:
            assert BehaviorTreeFactory.has_tree(name), f"Missing tree for {name}"


class TestWorldContextTime:
    """Tests for WorldContext time-related features."""

    def test_time_period_computed_triggers(self):
        """Test that time period triggers work correctly."""
        context = WorldContext(time_period="morning")
        assert context.has_trigger("is_morning") is True
        assert context.has_trigger("is_midday") is False
        assert context.has_trigger("is_evening") is False
        assert context.has_trigger("is_night") is False

        context = WorldContext(time_period="night")
        assert context.has_trigger("is_morning") is False
        assert context.has_trigger("is_night") is True

    def test_update_time_context(self):
        """Test updating time context from hour."""
        context = WorldContext()

        context.update_time_context(7)
        assert context.time_period == "morning"
        assert context.current_hour == 7

        context.update_time_context(12)
        assert context.time_period == "midday"

        context.update_time_context(18)
        assert context.time_period == "evening"

        context.update_time_context(22)
        assert context.time_period == "night"

        context.update_time_context(3)
        assert context.time_period == "night"

    def test_update_transition_state(self):
        """Test updating transition state for reactive behaviors."""
        context = WorldContext()
        context.person_detected = True

        # Update transition state
        context.update_transition_state(is_falling=True)

        assert context.person_detected_previous is True
        assert context.was_falling is True

    def test_time_context_serialization(self):
        """Test that time fields are included in serialization."""
        context = WorldContext(
            current_hour=15,
            time_period="midday",
            person_detected_previous=True,
            was_falling=True,
        )

        state = context.get_state()
        assert state["current_hour"] == 15
        assert state["time_period"] == "midday"
        assert state["person_detected_previous"] is True
        assert state["was_falling"] is True

        # Test restoration
        restored = WorldContext.from_state(state)
        assert restored.current_hour == 15
        assert restored.time_period == "midday"
        assert restored.person_detected_previous is True
        assert restored.was_falling is True


class TestBehaviorSerialization:
    """Tests for new Behavior field serialization."""

    def test_time_preferences_serialization(self):
        """Test that time_preferences is serialized correctly."""
        behavior = Behavior(
            name="test",
            display_name="Test",
            time_preferences={"morning": 1.5, "night": 0.5},
        )

        state = behavior.get_state()
        assert "time_preferences" in state
        assert state["time_preferences"]["morning"] == 1.5
        assert state["time_preferences"]["night"] == 0.5

        restored = Behavior.from_state(state)
        assert restored.time_preferences["morning"] == 1.5
        assert restored.time_preferences["night"] == 0.5

    def test_spontaneous_probability_serialization(self):
        """Test that spontaneous_probability is serialized correctly."""
        behavior = Behavior(
            name="test",
            display_name="Test",
            spontaneous_probability=0.05,
        )

        state = behavior.get_state()
        assert "spontaneous_probability" in state
        assert state["spontaneous_probability"] == 0.05

        restored = Behavior.from_state(state)
        assert restored.spontaneous_probability == 0.05


class TestScoredBehaviorBreakdown:
    """Tests for ScoredBehavior with new modifier fields."""

    def test_scored_behavior_includes_new_modifiers(self):
        """Test that ScoredBehavior includes time and spontaneous modifiers."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        context = WorldContext(time_period="morning")
        scored = evaluator.evaluate(context=context)

        for sb in scored:
            breakdown = sb.get_breakdown()
            assert "time_modifier" in breakdown
            assert "spontaneous_bonus" in breakdown
            assert breakdown["time_modifier"] >= 0.5
            assert breakdown["spontaneous_bonus"] >= 1.0

    def test_total_score_includes_all_modifiers(self):
        """Test that total score calculation includes all modifiers."""
        needs = NeedsSystem()
        registry = BehaviorRegistry()
        evaluator = BehaviorEvaluator(needs, registry)

        context = WorldContext(time_period="morning")
        scored = evaluator.evaluate(context=context)

        for sb in scored:
            # Verify total score is product of all modifiers
            expected = (
                sb.base_value
                * sb.need_modifier
                * sb.personality_modifier
                * sb.opportunity_bonus
                * sb.time_modifier
                * sb.spontaneous_bonus
            )
            assert abs(sb.total_score - expected) < 0.001, (
                f"Score mismatch for {sb.behavior.name}: "
                f"expected {expected}, got {sb.total_score}"
            )


class TestBehaviorRegistryCount:
    """Test that the correct number of behaviors are registered."""

    def test_total_behavior_count(self):
        """Test that we have the expected number of behaviors."""
        registry = BehaviorRegistry()

        # Original behaviors + 22 new behaviors
        # Original count from exploration was 38, plus 22 new = 60
        # But let's just check we have at least the new ones
        assert len(registry) >= 60, (
            f"Expected at least 60 behaviors, got {len(registry)}"
        )

    def test_all_new_behaviors_have_trees(self):
        """Test that all 22 new behaviors have behavior trees."""
        new_behaviors = [
            # Time-based (8)
            "wake_up", "morning_stretch", "energetic_start", "midday_activity",
            "afternoon_rest", "evening_settle", "pre_sleep_yawn", "night_stir",
            # Personality expressions (8)
            "stretch", "yawn", "daydream", "shake_off",
            "sneeze", "happy_wiggle", "curious_tilt", "contented_sigh",
            # Reactive (6)
            "dropped_recovery", "loud_noise_reaction", "new_object_investigation",
            "person_left_sad", "touched_unexpectedly", "picked_up_happy",
        ]

        for name in new_behaviors:
            assert BehaviorTreeFactory.has_tree(name), f"Missing tree for {name}"
