"""
Unit tests for Murph's loneliness behaviors and conditional trees.
"""

import pytest
from py_trees.composites import Selector, Sequence

from server.cognition.behavior import (
    BehaviorRegistry,
    BehaviorEvaluator,
    WorldContext,
    BehaviorTreeFactory,
)
from server.cognition.needs import NeedsSystem


class TestLonelinessBehaviors:
    """Tests for loneliness behavior definitions."""

    def test_sigh_behavior_exists(self):
        """Test sigh behavior is registered."""
        registry = BehaviorRegistry()
        sigh = registry.get("sigh")
        assert sigh is not None
        assert sigh.display_name == "Sigh"
        assert "lonely" in sigh.opportunity_triggers
        assert sigh.has_tag("loneliness")
        assert sigh.cooldown_seconds == 60.0

    def test_mope_behavior_exists(self):
        """Test mope behavior is registered."""
        registry = BehaviorRegistry()
        mope = registry.get("mope")
        assert mope is not None
        assert mope.display_name == "Mope Around"
        assert "very_lonely" in mope.opportunity_triggers
        assert mope.has_tag("loneliness")
        assert mope.cooldown_seconds == 120.0

    def test_perk_up_hopeful_behavior_exists(self):
        """Test perk_up_hopeful behavior is registered."""
        registry = BehaviorRegistry()
        perk = registry.get("perk_up_hopeful")
        assert perk is not None
        assert perk.display_name == "Perk Up Hopeful"
        assert "lonely" in perk.opportunity_triggers
        assert perk.has_tag("loneliness")
        assert perk.has_tag("expressive")

    def test_seek_company_behavior_exists(self):
        """Test seek_company behavior is registered."""
        registry = BehaviorRegistry()
        seek = registry.get("seek_company")
        assert seek is not None
        assert seek.display_name == "Seek Company"
        assert "lonely" in seek.opportunity_triggers
        assert "very_lonely" in seek.opportunity_triggers
        assert seek.has_tag("loneliness")
        assert seek.has_tag("movement")

    def test_loneliness_behaviors_driven_by_social_needs(self):
        """Test loneliness behaviors are driven by social and affection needs."""
        registry = BehaviorRegistry()
        loneliness_behaviors = registry.get_by_tag("loneliness")

        assert len(loneliness_behaviors) == 4

        for behavior in loneliness_behaviors:
            assert "social" in behavior.driven_by_needs or "affection" in behavior.driven_by_needs


class TestLonelinessTriggersEvaluation:
    """Tests for lonely trigger evaluation in behavior scoring."""

    def test_lonely_trigger_activates_at_300_seconds(self):
        """Test lonely trigger activates after 5 minutes."""
        context = WorldContext(time_since_last_interaction=299.0)
        assert context.has_trigger("lonely") is False

        context = WorldContext(time_since_last_interaction=301.0)
        assert context.has_trigger("lonely") is True

    def test_very_lonely_trigger_activates_at_600_seconds(self):
        """Test very_lonely trigger activates after 10 minutes."""
        context = WorldContext(time_since_last_interaction=599.0)
        assert context.has_trigger("very_lonely") is False

        context = WorldContext(time_since_last_interaction=601.0)
        assert context.has_trigger("very_lonely") is True

    def test_lonely_trigger_boosts_loneliness_behaviors(self):
        """Test that lonely trigger gives opportunity bonus to loneliness behaviors."""
        needs = NeedsSystem()
        needs.needs["social"].value = 50.0  # Moderate need
        needs.needs["affection"].value = 50.0
        evaluator = BehaviorEvaluator(needs)

        # Context with lonely trigger active
        context = WorldContext(time_since_last_interaction=350.0)
        assert context.has_trigger("lonely") is True

        scored = evaluator.evaluate(context=context)

        # Find sigh and perk_up_hopeful which have "lonely" trigger
        sigh_scored = next((s for s in scored if s.behavior.name == "sigh"), None)
        perk_scored = next((s for s in scored if s.behavior.name == "perk_up_hopeful"), None)

        assert sigh_scored is not None
        assert perk_scored is not None
        assert sigh_scored.opportunity_bonus > 1.0
        assert perk_scored.opportunity_bonus > 1.0

    def test_very_lonely_trigger_boosts_mope(self):
        """Test that very_lonely trigger gives opportunity bonus to mope."""
        needs = NeedsSystem()
        needs.needs["social"].value = 30.0  # Low need
        needs.needs["affection"].value = 30.0
        evaluator = BehaviorEvaluator(needs)

        # Context with very_lonely trigger active
        context = WorldContext(time_since_last_interaction=650.0)
        assert context.has_trigger("very_lonely") is True

        scored = evaluator.evaluate(context=context)

        mope_scored = next((s for s in scored if s.behavior.name == "mope"), None)
        assert mope_scored is not None
        assert mope_scored.opportunity_bonus > 1.0

    def test_lonely_robot_prioritizes_loneliness_behaviors(self):
        """Test lonely robot with low social needs selects loneliness behaviors."""
        needs = NeedsSystem()
        # Set all needs to moderate except social and affection
        for need in needs.needs.values():
            need.value = 70.0
        needs.needs["social"].value = 25.0
        needs.needs["affection"].value = 25.0
        evaluator = BehaviorEvaluator(needs)

        # Very lonely context
        context = WorldContext(time_since_last_interaction=700.0)

        # Get top behaviors
        top_5 = evaluator.select_top_n(5, context=context)
        top_5_names = [s.behavior.name for s in top_5]

        # Should have at least one loneliness behavior in top 5
        loneliness_behaviors = ["sigh", "mope", "perk_up_hopeful", "seek_company"]
        assert any(name in loneliness_behaviors for name in top_5_names), \
            f"No loneliness behavior in top 5: {top_5_names}"


class TestLonelinessTrees:
    """Tests for loneliness behavior tree definitions."""

    def test_sigh_tree_exists(self):
        """Test sigh tree is registered."""
        assert BehaviorTreeFactory.has_tree("sigh")
        tree = BehaviorTreeFactory.create_tree("sigh")
        assert tree is not None
        assert tree.name == "sigh"

    def test_mope_tree_exists(self):
        """Test mope tree is registered."""
        assert BehaviorTreeFactory.has_tree("mope")
        tree = BehaviorTreeFactory.create_tree("mope")
        assert tree is not None
        assert tree.name == "mope"

    def test_perk_up_hopeful_tree_exists(self):
        """Test perk_up_hopeful tree is registered."""
        assert BehaviorTreeFactory.has_tree("perk_up_hopeful")
        tree = BehaviorTreeFactory.create_tree("perk_up_hopeful")
        assert tree is not None
        assert tree.name == "perk_up_hopeful"

    def test_seek_company_tree_exists(self):
        """Test seek_company tree is registered."""
        assert BehaviorTreeFactory.has_tree("seek_company")
        tree = BehaviorTreeFactory.create_tree("seek_company")
        assert tree is not None
        assert tree.name == "seek_company"


class TestConditionalTrees:
    """Tests for conditional behavior tree structures."""

    def test_explore_tree_uses_selector(self):
        """Test explore tree uses Selector for conditional branching."""
        tree = BehaviorTreeFactory.create_tree("explore")
        assert tree is not None
        assert isinstance(tree, Selector), f"Expected Selector, got {type(tree).__name__}"
        assert tree.name == "explore"

    def test_explore_tree_has_person_detection_branch(self):
        """Test explore tree has a branch for person detection."""
        tree = BehaviorTreeFactory.create_tree("explore")

        # Should have two children: person_detected branch and default branch
        assert len(tree.children) == 2

        # First child should be the person detection branch
        person_branch = tree.children[0]
        assert isinstance(person_branch, Sequence)
        assert "person_detected" in person_branch.name

    def test_wander_tree_uses_selector(self):
        """Test wander tree uses Selector for conditional branching."""
        tree = BehaviorTreeFactory.create_tree("wander")
        assert tree is not None
        assert isinstance(tree, Selector), f"Expected Selector, got {type(tree).__name__}"
        assert tree.name == "wander"

    def test_wander_tree_has_edge_detection_branch(self):
        """Test wander tree has a branch for edge detection."""
        tree = BehaviorTreeFactory.create_tree("wander")

        # Should have two children: edge_detected branch and default branch
        assert len(tree.children) == 2

        # First child should be the edge detection branch
        edge_branch = tree.children[0]
        assert isinstance(edge_branch, Sequence)
        assert "edge" in edge_branch.name

    def test_seek_company_tree_uses_selector(self):
        """Test seek_company tree uses Selector for conditional branching."""
        tree = BehaviorTreeFactory.create_tree("seek_company")
        assert tree is not None
        assert isinstance(tree, Selector), f"Expected Selector, got {type(tree).__name__}"
        assert tree.name == "seek_company"

    def test_seek_company_tree_has_found_person_branch(self):
        """Test seek_company tree has a branch for when person is found."""
        tree = BehaviorTreeFactory.create_tree("seek_company")

        # Should have two children: found branch and searching branch
        assert len(tree.children) == 2

        # First child should be the found branch
        found_branch = tree.children[0]
        assert isinstance(found_branch, Sequence)
        assert "found" in found_branch.name

    def test_selector_memory_is_false(self):
        """Test that Selector-based trees have memory=False for condition re-evaluation."""
        explore_tree = BehaviorTreeFactory.create_tree("explore")
        wander_tree = BehaviorTreeFactory.create_tree("wander")
        seek_tree = BehaviorTreeFactory.create_tree("seek_company")

        # memory=False means conditions are re-evaluated each tick
        assert explore_tree.memory is False
        assert wander_tree.memory is False
        assert seek_tree.memory is False

    def test_inner_sequences_have_memory_true(self):
        """Test that inner Sequences have memory=True to complete their actions."""
        tree = BehaviorTreeFactory.create_tree("explore")

        for child in tree.children:
            if isinstance(child, Sequence):
                assert child.memory is True, f"Sequence {child.name} should have memory=True"


class TestSighSoundDuration:
    """Tests for sigh sound support."""

    def test_sigh_sound_duration_defined(self):
        """Test that sigh sound has a defined duration."""
        from server.cognition.behavior.actions import PlaySoundAction

        assert "sigh" in PlaySoundAction.SOUND_DURATIONS
        assert PlaySoundAction.SOUND_DURATIONS["sigh"] == 1.5
