"""
Unit tests for Murph's Behavior Evaluator system.
"""

import time
import pytest
from server.cognition.behavior import (
    Behavior,
    WorldContext,
    BehaviorRegistry,
    DEFAULT_BEHAVIORS,
    BehaviorEvaluator,
    ScoredBehavior,
)
from server.cognition.needs import NeedsSystem, Personality


class TestBehavior:
    """Tests for the Behavior dataclass."""

    def test_behavior_creation(self):
        """Test creating a behavior with default values."""
        behavior = Behavior(name="test", display_name="Test Behavior")
        assert behavior.name == "test"
        assert behavior.display_name == "Test Behavior"
        assert behavior.base_value == 1.0
        assert behavior.need_effects == {}
        assert behavior.driven_by_needs == []
        assert behavior.opportunity_triggers == []
        assert behavior.duration_seconds == 5.0
        assert behavior.interruptible is True
        assert behavior.cooldown_seconds == 0.0
        assert behavior.energy_cost == 0.0
        assert behavior.tags == []

    def test_behavior_with_effects(self):
        """Test creating a behavior with need effects."""
        behavior = Behavior(
            name="explore",
            display_name="Explore",
            base_value=1.1,
            need_effects={"curiosity": 15.0, "energy": -10.0},
            driven_by_needs=["curiosity"],
            opportunity_triggers=["unknown_object"],
            tags=["active", "movement"],
        )
        assert behavior.base_value == 1.1
        assert behavior.need_effects["curiosity"] == 15.0
        assert behavior.need_effects["energy"] == -10.0
        assert "curiosity" in behavior.driven_by_needs
        assert "unknown_object" in behavior.opportunity_triggers
        assert "active" in behavior.tags

    def test_behavior_base_value_clamping(self):
        """Test that base_value is clamped to valid range."""
        behavior = Behavior(name="test", display_name="Test", base_value=5.0)
        assert behavior.base_value == 3.0  # Clamped to max

        behavior = Behavior(name="test", display_name="Test", base_value=-1.0)
        assert behavior.base_value == 0.1  # Clamped to min

    def test_behavior_satisfies_need(self):
        """Test checking if behavior satisfies a need."""
        behavior = Behavior(
            name="test",
            display_name="Test",
            need_effects={"curiosity": 15.0, "energy": -10.0},
        )
        assert behavior.satisfies_need("curiosity") is True
        assert behavior.satisfies_need("energy") is False
        assert behavior.satisfies_need("social") is False

    def test_behavior_depletes_need(self):
        """Test checking if behavior depletes a need."""
        behavior = Behavior(
            name="test",
            display_name="Test",
            need_effects={"curiosity": 15.0, "energy": -10.0},
        )
        assert behavior.depletes_need("energy") is True
        assert behavior.depletes_need("curiosity") is False
        assert behavior.depletes_need("social") is False

    def test_behavior_has_tag(self):
        """Test checking behavior tags."""
        behavior = Behavior(
            name="test",
            display_name="Test",
            tags=["active", "movement"],
        )
        assert behavior.has_tag("active") is True
        assert behavior.has_tag("passive") is False

    def test_behavior_state_serialization(self):
        """Test saving and loading behavior state."""
        behavior = Behavior(
            name="explore",
            display_name="Explore",
            base_value=1.1,
            need_effects={"curiosity": 15.0},
            driven_by_needs=["curiosity"],
            opportunity_triggers=["unknown_object"],
            duration_seconds=20.0,
            interruptible=False,
            cooldown_seconds=5.0,
            energy_cost=10.0,
            tags=["active"],
        )

        state = behavior.get_state()
        restored = Behavior.from_state(state)

        assert restored.name == behavior.name
        assert restored.display_name == behavior.display_name
        assert restored.base_value == behavior.base_value
        assert restored.need_effects == behavior.need_effects
        assert restored.driven_by_needs == behavior.driven_by_needs
        assert restored.opportunity_triggers == behavior.opportunity_triggers
        assert restored.duration_seconds == behavior.duration_seconds
        assert restored.interruptible == behavior.interruptible
        assert restored.cooldown_seconds == behavior.cooldown_seconds
        assert restored.energy_cost == behavior.energy_cost
        assert restored.tags == behavior.tags


class TestWorldContext:
    """Tests for the WorldContext dataclass."""

    def test_context_creation(self):
        """Test creating a context with default values."""
        context = WorldContext()
        assert context.person_detected is False
        assert context.person_is_familiar is False
        assert context.objects_in_view == []
        assert context.is_dark is False
        assert context.active_triggers == set()

    def test_context_with_person(self):
        """Test context with person detection."""
        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_familiarity_score=85.0,
            person_distance=30.0,
        )
        assert context.person_detected is True
        assert context.person_is_familiar is True
        assert context.person_familiarity_score == 85.0
        assert context.person_distance == 30.0

    def test_context_has_trigger_explicit(self):
        """Test explicit trigger checking."""
        context = WorldContext()
        context.add_trigger("custom_trigger")
        assert context.has_trigger("custom_trigger") is True
        assert context.has_trigger("other_trigger") is False

    def test_context_has_trigger_computed_person_nearby(self):
        """Test computed trigger: person_nearby."""
        context = WorldContext(person_detected=True, person_distance=30.0)
        assert context.has_trigger("person_nearby") is True

        context = WorldContext(person_detected=True, person_distance=80.0)
        assert context.has_trigger("person_nearby") is False

    def test_context_has_trigger_computed_familiar_person(self):
        """Test computed trigger: familiar_person."""
        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
        )
        assert context.has_trigger("familiar_person") is True
        assert context.has_trigger("stranger") is False

    def test_context_has_trigger_computed_stranger(self):
        """Test computed trigger: stranger."""
        context = WorldContext(
            person_detected=True,
            person_is_familiar=False,
        )
        assert context.has_trigger("stranger") is True
        assert context.has_trigger("familiar_person") is False

    def test_context_has_trigger_computed_lonely(self):
        """Test computed trigger: lonely."""
        context = WorldContext(time_since_last_interaction=100.0)
        assert context.has_trigger("lonely") is False

        context = WorldContext(time_since_last_interaction=350.0)
        assert context.has_trigger("lonely") is True

    def test_context_has_trigger_computed_being_held(self):
        """Test computed trigger: being_held."""
        context = WorldContext(is_being_held=True)
        assert context.has_trigger("being_held") is True

    def test_context_trigger_management(self):
        """Test adding and removing triggers."""
        context = WorldContext()
        context.add_trigger("test1")
        context.add_trigger("test2")
        assert context.has_trigger("test1") is True
        assert context.has_trigger("test2") is True

        context.remove_trigger("test1")
        assert context.has_trigger("test1") is False
        assert context.has_trigger("test2") is True

        context.clear_triggers()
        assert context.has_trigger("test2") is False

    def test_context_get_active_triggers(self):
        """Test getting all active triggers."""
        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
        )
        context.add_trigger("custom")

        active = context.get_active_triggers()
        assert "custom" in active
        assert "person_nearby" in active
        assert "familiar_person" in active
        assert "person_detected" in active

    def test_context_state_serialization(self):
        """Test saving and loading context state."""
        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
            objects_in_view=["ball", "cup"],
            is_dark=True,
            time_since_last_interaction=100.0,
        )
        context.add_trigger("custom")

        state = context.get_state()
        restored = WorldContext.from_state(state)

        assert restored.person_detected == context.person_detected
        assert restored.person_is_familiar == context.person_is_familiar
        assert restored.person_distance == context.person_distance
        assert restored.objects_in_view == context.objects_in_view
        assert restored.is_dark == context.is_dark
        assert restored.time_since_last_interaction == context.time_since_last_interaction
        assert "custom" in restored.active_triggers


class TestBehaviorRegistry:
    """Tests for the BehaviorRegistry."""

    def test_registry_creation(self):
        """Test creating a registry with default behaviors."""
        registry = BehaviorRegistry()
        assert len(registry) > 0
        assert len(registry) == len(DEFAULT_BEHAVIORS)

    def test_registry_creation_empty(self):
        """Test creating an empty registry."""
        registry = BehaviorRegistry(load_defaults=False)
        assert len(registry) == 0

    def test_registry_register(self):
        """Test registering a custom behavior."""
        registry = BehaviorRegistry(load_defaults=False)
        behavior = Behavior(name="custom", display_name="Custom", tags=["test"])
        registry.register(behavior)

        assert len(registry) == 1
        assert "custom" in registry
        assert registry.get("custom") is behavior

    def test_registry_unregister(self):
        """Test unregistering a behavior."""
        registry = BehaviorRegistry(load_defaults=False)
        behavior = Behavior(name="custom", display_name="Custom", tags=["test"])
        registry.register(behavior)

        assert registry.unregister("custom") is True
        assert "custom" not in registry
        assert registry.unregister("custom") is False  # Already removed

    def test_registry_get(self):
        """Test getting a behavior by name."""
        registry = BehaviorRegistry()
        behavior = registry.get("explore")
        assert behavior is not None
        assert behavior.name == "explore"

        assert registry.get("nonexistent") is None

    def test_registry_get_all(self):
        """Test getting all behaviors."""
        registry = BehaviorRegistry()
        all_behaviors = registry.get_all()
        assert len(all_behaviors) == len(DEFAULT_BEHAVIORS)
        assert all(isinstance(b, Behavior) for b in all_behaviors)

    def test_registry_get_names(self):
        """Test getting all behavior names."""
        registry = BehaviorRegistry()
        names = registry.get_names()
        assert "explore" in names
        assert "rest" in names
        assert "greet" in names

    def test_registry_get_by_tag(self):
        """Test getting behaviors by tag."""
        registry = BehaviorRegistry()
        active_behaviors = registry.get_by_tag("active")
        assert len(active_behaviors) > 0
        assert all(b.has_tag("active") for b in active_behaviors)

    def test_registry_get_by_tags_any(self):
        """Test getting behaviors matching any of multiple tags."""
        registry = BehaviorRegistry()
        behaviors = registry.get_by_tags(["active", "passive"], match_all=False)
        assert len(behaviors) > 0
        assert all(
            b.has_tag("active") or b.has_tag("passive")
            for b in behaviors
        )

    def test_registry_get_by_tags_all(self):
        """Test getting behaviors matching all tags."""
        registry = BehaviorRegistry()
        behaviors = registry.get_by_tags(["active", "movement"], match_all=True)
        assert len(behaviors) > 0
        assert all(
            b.has_tag("active") and b.has_tag("movement")
            for b in behaviors
        )

    def test_registry_get_for_need(self):
        """Test getting behaviors that satisfy a need."""
        registry = BehaviorRegistry()
        curiosity_behaviors = registry.get_for_need("curiosity")
        assert len(curiosity_behaviors) > 0
        assert all(b.satisfies_need("curiosity") for b in curiosity_behaviors)

    def test_registry_get_driven_by_need(self):
        """Test getting behaviors driven by a need."""
        registry = BehaviorRegistry()
        energy_driven = registry.get_driven_by_need("energy")
        assert len(energy_driven) > 0
        assert all("energy" in b.driven_by_needs for b in energy_driven)

    def test_registry_contains(self):
        """Test the 'in' operator."""
        registry = BehaviorRegistry()
        assert "explore" in registry
        assert "nonexistent" not in registry

    def test_registry_state_serialization(self):
        """Test saving and loading registry state."""
        registry = BehaviorRegistry()
        custom = Behavior(name="custom", display_name="Custom")
        registry.register(custom)

        state = registry.get_state()
        restored = BehaviorRegistry.from_state(state)

        # Should have default behaviors plus custom
        assert "explore" in restored  # Default
        assert "custom" in restored   # Custom


class TestBehaviorEvaluator:
    """Tests for the BehaviorEvaluator."""

    def test_evaluator_creation(self):
        """Test creating an evaluator."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        assert evaluator.needs_system is needs
        assert evaluator.registry is not None
        assert len(evaluator.registry) > 0

    def test_evaluator_evaluate_returns_sorted_list(self):
        """Test that evaluate returns behaviors sorted by score."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate()

        assert len(scored) > 0
        # Check that scores are in descending order
        for i in range(len(scored) - 1):
            assert scored[i].total_score >= scored[i + 1].total_score

    def test_evaluator_select_best(self):
        """Test selecting the best behavior."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        best = evaluator.select_best()

        assert best is not None
        assert isinstance(best, ScoredBehavior)

        # Should be the same as first in evaluate list
        scored = evaluator.evaluate()
        assert best.behavior.name == scored[0].behavior.name

    def test_evaluator_select_top_n(self):
        """Test selecting top N behaviors."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        top_3 = evaluator.select_top_n(3)

        assert len(top_3) == 3
        # Should be in descending order
        assert top_3[0].total_score >= top_3[1].total_score >= top_3[2].total_score

    def test_evaluator_exclude_behaviors(self):
        """Test excluding behaviors from evaluation."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        # Get best without exclusion
        best_original = evaluator.select_best()

        # Exclude the best and get next best
        scored = evaluator.evaluate(exclude_behaviors=[best_original.behavior.name])
        assert all(s.behavior.name != best_original.behavior.name for s in scored)


class TestNeedModifier:
    """Tests for need modifier calculation."""

    def test_need_modifier_satisfied_need(self):
        """Test need modifier when needs are satisfied."""
        needs = NeedsSystem()
        # All needs start at 100 (satisfied)
        evaluator = BehaviorEvaluator(needs)

        # Find a behavior driven by a need
        explore = evaluator.registry.get("explore")
        assert explore is not None
        assert "curiosity" in explore.driven_by_needs

        scored = evaluator.evaluate()
        explore_scored = next(s for s in scored if s.behavior.name == "explore")

        # Satisfied needs should give low modifier (close to 0.5)
        assert explore_scored.need_modifier < 1.0

    def test_need_modifier_urgent_need(self):
        """Test need modifier when need is urgent."""
        needs = NeedsSystem()
        needs.needs["curiosity"].value = 10.0  # Very low
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate()
        explore_scored = next(s for s in scored if s.behavior.name == "explore")

        # Urgent need should give high modifier (close to 2.0)
        assert explore_scored.need_modifier > 1.5

    def test_need_modifier_no_driving_needs(self):
        """Test need modifier for behavior with no driving needs."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        # Idle has no driving needs
        idle = evaluator.registry.get("idle")
        assert idle is not None
        assert len(idle.driven_by_needs) == 0

        scored = evaluator.evaluate()
        idle_scored = next(s for s in scored if s.behavior.name == "idle")

        # Should be neutral (1.0)
        assert idle_scored.need_modifier == 1.0

    def test_need_modifier_multiple_driving_needs(self):
        """Test need modifier with multiple driving needs."""
        needs = NeedsSystem()
        needs.needs["affection"].value = 20.0  # Low
        needs.needs["social"].value = 80.0     # High
        evaluator = BehaviorEvaluator(needs)

        # request_attention is driven by both affection and social
        request = evaluator.registry.get("request_attention")
        assert request is not None
        assert "affection" in request.driven_by_needs
        assert "social" in request.driven_by_needs

        scored = evaluator.evaluate()
        request_scored = next(s for s in scored if s.behavior.name == "request_attention")

        # Should be between values for each need individually
        assert 0.5 < request_scored.need_modifier < 2.0


class TestPersonalityModifier:
    """Tests for personality modifier calculation."""

    def test_personality_modifier_playful(self):
        """Test personality modifier for playful personality."""
        personality = Personality.from_preset("playful")
        needs = NeedsSystem(personality=personality)
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate()
        play_scored = next(s for s in scored if s.behavior.name == "play")

        # Playful personality should boost play behaviors
        assert play_scored.personality_modifier > 1.0

    def test_personality_modifier_shy(self):
        """Test personality modifier for shy personality."""
        personality = Personality.from_preset("shy")
        needs = NeedsSystem(personality=personality)
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate()
        greet_scored = next(s for s in scored if s.behavior.name == "greet")

        # Shy personality should reduce social behaviors
        assert greet_scored.personality_modifier < 1.0

    def test_personality_modifier_explorer(self):
        """Test personality modifier for explorer personality."""
        personality = Personality.from_preset("explorer")
        needs = NeedsSystem(personality=personality)
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate()
        explore_scored = next(s for s in scored if s.behavior.name == "explore")

        # Explorer personality should boost exploration behaviors
        assert explore_scored.personality_modifier > 1.0


class TestOpportunityBonus:
    """Tests for opportunity bonus calculation."""

    def test_opportunity_bonus_no_context(self):
        """Test opportunity bonus without context."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        scored = evaluator.evaluate(context=None)
        greet_scored = next(s for s in scored if s.behavior.name == "greet")

        # No context means no bonus
        assert greet_scored.opportunity_bonus == 1.0

    def test_opportunity_bonus_no_triggers(self):
        """Test opportunity bonus with context but no matching triggers."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        context = WorldContext(is_dark=True)  # No person

        scored = evaluator.evaluate(context=context)
        greet_scored = next(s for s in scored if s.behavior.name == "greet")

        # Greet needs person_nearby or familiar_person, which aren't active
        assert greet_scored.opportunity_bonus == 1.0

    def test_opportunity_bonus_single_trigger(self):
        """Test opportunity bonus with single matching trigger."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        context = WorldContext(
            person_detected=True,
            person_distance=30.0,  # person_nearby
        )

        scored = evaluator.evaluate(context=context)
        greet_scored = next(s for s in scored if s.behavior.name == "greet")

        # One trigger = 1.0 + 0.3 = 1.3
        assert greet_scored.opportunity_bonus == pytest.approx(1.3, abs=0.01)

    def test_opportunity_bonus_multiple_triggers(self):
        """Test opportunity bonus with multiple matching triggers."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,  # Both person_nearby AND familiar_person
        )

        scored = evaluator.evaluate(context=context)
        greet_scored = next(s for s in scored if s.behavior.name == "greet")

        # Two triggers = 1.0 + 0.6 = 1.6
        assert greet_scored.opportunity_bonus == pytest.approx(1.6, abs=0.01)

    def test_opportunity_bonus_caps_at_two(self):
        """Test that opportunity bonus caps at 2.0."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        # Create behavior with many triggers
        behavior = Behavior(
            name="test",
            display_name="Test",
            opportunity_triggers=[
                "person_nearby", "familiar_person", "being_held",
                "being_petted", "unknown_object",
            ],
        )
        registry.register(behavior)
        evaluator = BehaviorEvaluator(needs, registry=registry)

        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
            is_being_held=True,
            is_being_petted=True,
            unknown_object_detected=True,
        )

        scored = evaluator.evaluate(context=context)
        assert scored[0].opportunity_bonus == 2.0


class TestCooldowns:
    """Tests for behavior cooldown tracking."""

    def test_cooldown_excludes_behavior(self):
        """Test that behaviors on cooldown are excluded."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        behavior = Behavior(
            name="test",
            display_name="Test",
            cooldown_seconds=10.0,
        )
        registry.register(behavior)
        evaluator = BehaviorEvaluator(needs, registry=registry)

        # Before cooldown
        scored = evaluator.evaluate()
        assert len(scored) == 1

        # Mark as used
        evaluator.mark_behavior_used("test")

        # Should be excluded
        scored = evaluator.evaluate()
        assert len(scored) == 0

    def test_cooldown_expires(self):
        """Test that cooldowns expire after time passes."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        behavior = Behavior(
            name="test",
            display_name="Test",
            cooldown_seconds=0.1,  # Very short for testing
        )
        registry.register(behavior)
        evaluator = BehaviorEvaluator(needs, registry=registry)

        evaluator.mark_behavior_used("test")

        # Wait for cooldown
        time.sleep(0.15)

        scored = evaluator.evaluate()
        assert len(scored) == 1

    def test_cooldown_clear(self):
        """Test clearing cooldowns."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        behavior = Behavior(
            name="test",
            display_name="Test",
            cooldown_seconds=60.0,
        )
        registry.register(behavior)
        evaluator = BehaviorEvaluator(needs, registry=registry)

        evaluator.mark_behavior_used("test")

        # Clear specific cooldown
        evaluator.clear_cooldown("test")
        scored = evaluator.evaluate()
        assert len(scored) == 1

    def test_cooldown_clear_all(self):
        """Test clearing all cooldowns."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        for i in range(3):
            behavior = Behavior(
                name=f"test{i}",
                display_name=f"Test {i}",
                cooldown_seconds=60.0,
            )
            registry.register(behavior)

        evaluator = BehaviorEvaluator(needs, registry=registry)

        for i in range(3):
            evaluator.mark_behavior_used(f"test{i}")

        scored = evaluator.evaluate()
        assert len(scored) == 0

        evaluator.clear_all_cooldowns()

        scored = evaluator.evaluate()
        assert len(scored) == 3

    def test_get_cooldown_remaining(self):
        """Test getting remaining cooldown time."""
        needs = NeedsSystem()
        registry = BehaviorRegistry(load_defaults=False)

        behavior = Behavior(
            name="test",
            display_name="Test",
            cooldown_seconds=10.0,
        )
        registry.register(behavior)
        evaluator = BehaviorEvaluator(needs, registry=registry)

        assert evaluator.get_cooldown_remaining("test") == 0.0

        evaluator.mark_behavior_used("test")

        remaining = evaluator.get_cooldown_remaining("test")
        assert 9.0 < remaining <= 10.0


class TestScoringIntegration:
    """Integration tests for complete scoring scenarios."""

    def test_hungry_robot_selects_rest(self):
        """When energy is critical, rest/sleep should score highest."""
        needs = NeedsSystem()
        needs.needs["energy"].value = 10.0  # Critical
        evaluator = BehaviorEvaluator(needs)

        best = evaluator.select_best()

        # go_to_charger is now a valid energy recovery behavior
        assert best.behavior.name in ["rest", "sleep", "go_to_charger"]

    def test_bored_robot_selects_explore(self):
        """When curiosity is low, exploration behaviors should score higher."""
        needs = NeedsSystem()
        needs.needs["curiosity"].value = 15.0  # Critical
        # Keep energy high so exploration is viable
        needs.needs["energy"].value = 80.0
        evaluator = BehaviorEvaluator(needs)

        best = evaluator.select_best()

        assert best.behavior.name in [
            "explore", "investigate", "wander", "observe", "new_object_investigation"
        ]

    def test_social_robot_greets_person(self):
        """When person is nearby and social need is moderate, social behaviors win."""
        needs = NeedsSystem()
        needs.needs["social"].value = 40.0  # Moderate need
        evaluator = BehaviorEvaluator(needs)

        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
        )

        best = evaluator.select_best(context=context)

        # With opportunity bonus, social behaviors should score high
        assert best.behavior.name in ["greet", "nuzzle", "follow", "approach", "interact"]

    def test_critical_safety_overrides_play(self):
        """Safety behaviors should override play when safety is critical."""
        needs = NeedsSystem()
        needs.needs["safety"].value = 15.0  # Critical
        needs.needs["play"].value = 30.0    # Also low but less critical
        evaluator = BehaviorEvaluator(needs)

        best = evaluator.select_best()

        # Safety behaviors have high base values and should win
        # Navigation safety behaviors (flee_danger, retreat_to_safe) are also valid
        assert best.behavior.name in [
            "retreat", "hide", "scan", "approach_trusted",
            "flee_danger", "retreat_to_safe", "reorient"
        ]

    def test_opportunity_swings_decision(self):
        """Opportunity bonus should swing decision to appropriate behavior."""
        needs = NeedsSystem()
        # All needs moderately satisfied
        for need in needs.needs.values():
            need.value = 70.0
        evaluator = BehaviorEvaluator(needs)

        # Without context, no strong preference
        best_no_context = evaluator.select_best()

        # With person nearby, social behaviors should be boosted
        context = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
        )
        best_with_person = evaluator.select_best(context=context)

        # The presence of a person should change the selection
        # (specific behavior depends on exact scoring, but should be social)
        scored = evaluator.evaluate(context=context)
        social_behaviors = ["greet", "nuzzle", "follow", "approach", "interact"]
        top_5_names = [s.behavior.name for s in scored[:5]]

        # At least some social behaviors should be in top 5
        assert any(name in social_behaviors for name in top_5_names)

    def test_personality_preference_emerges(self):
        """Different personalities should prefer different behaviors."""
        # Playful personality
        playful_needs = NeedsSystem(personality=Personality.from_preset("playful"))
        for need in playful_needs.needs.values():
            need.value = 50.0  # All moderately low
        playful_eval = BehaviorEvaluator(playful_needs)

        # Explorer personality
        explorer_needs = NeedsSystem(personality=Personality.from_preset("explorer"))
        for need in explorer_needs.needs.values():
            need.value = 50.0
        explorer_eval = BehaviorEvaluator(explorer_needs)

        # Get top 3 for each
        playful_top = playful_eval.select_top_n(3)
        explorer_top = explorer_eval.select_top_n(3)

        playful_names = [s.behavior.name for s in playful_top]
        explorer_names = [s.behavior.name for s in explorer_top]

        # They should have different preferences
        # (at least partially different rankings)
        assert playful_names != explorer_names


class TestStateSerialization:
    """Tests for evaluator state serialization."""

    def test_evaluator_state_serialization(self):
        """Test saving and loading evaluator state."""
        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)

        evaluator.mark_behavior_used("explore")
        evaluator.mark_behavior_used("greet")

        state = evaluator.get_state()
        restored = BehaviorEvaluator.from_state(state, needs)

        assert "explore" in restored._cooldowns
        assert "greet" in restored._cooldowns


class TestScoredBehavior:
    """Tests for ScoredBehavior dataclass."""

    def test_scored_behavior_breakdown(self):
        """Test getting score breakdown."""
        behavior = Behavior(name="test", display_name="Test")
        scored = ScoredBehavior(
            behavior=behavior,
            total_score=2.4,
            base_value=1.0,
            need_modifier=1.5,
            personality_modifier=1.0,
            opportunity_bonus=1.6,
        )

        breakdown = scored.get_breakdown()

        assert breakdown["base_value"] == 1.0
        assert breakdown["need_modifier"] == 1.5
        assert breakdown["personality_modifier"] == 1.0
        assert breakdown["opportunity_bonus"] == 1.6
        assert breakdown["total_score"] == 2.4

    def test_scored_behavior_str(self):
        """Test string representation."""
        behavior = Behavior(name="explore", display_name="Explore")
        scored = ScoredBehavior(
            behavior=behavior,
            total_score=2.4,
            base_value=1.0,
            need_modifier=1.5,
            personality_modifier=1.0,
            opportunity_bonus=1.6,
        )

        string = str(scored)
        assert "explore" in string
        assert "2.4" in string


class TestDefaultBehaviors:
    """Tests for the default behavior set."""

    def test_default_behaviors_exist(self):
        """Test that essential default behaviors exist."""
        registry = BehaviorRegistry()

        essential = [
            "rest", "sleep", "explore", "investigate", "greet",
            "follow", "play", "nuzzle", "retreat", "idle",
        ]

        for name in essential:
            assert name in registry, f"Missing essential behavior: {name}"

    def test_default_behaviors_have_effects(self):
        """Test that behaviors have appropriate need effects."""
        registry = BehaviorRegistry()

        # Rest should boost energy
        rest = registry.get("rest")
        assert rest.need_effects.get("energy", 0) > 0

        # Explore should boost curiosity but cost energy
        explore = registry.get("explore")
        assert explore.need_effects.get("curiosity", 0) > 0
        assert explore.need_effects.get("energy", 0) < 0

        # Greet should boost social
        greet = registry.get("greet")
        assert greet.need_effects.get("social", 0) > 0

    def test_safety_behaviors_have_high_base_value(self):
        """Test that safety behaviors have high base values."""
        registry = BehaviorRegistry()

        safety_behaviors = registry.get_by_tag("safety")
        assert len(safety_behaviors) > 0

        for behavior in safety_behaviors:
            assert behavior.base_value >= 1.5, f"{behavior.name} should have high base value"
