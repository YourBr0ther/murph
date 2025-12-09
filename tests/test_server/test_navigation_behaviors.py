"""
Unit tests for Murph's spatial navigation behaviors.
"""

import pytest
from py_trees.composites import Selector, Sequence

from server.cognition.behavior import (
    BehaviorRegistry,
    BehaviorTreeFactory,
    WorldContext,
    NavigateToLandmarkAction,
    ReorientAction,
    MoveTowardSafetyAction,
    AtLandmarkCondition,
    ZoneSafetyCondition,
    HasPathCondition,
    HasUnexploredZonesCondition,
    PositionKnownCondition,
)
from server.cognition.memory.spatial_types import (
    SpatialMapMemory,
    SpatialLandmark,
    SpatialZone,
)


class TestNavigationBehaviorDefinitions:
    """Tests for navigation behavior definitions."""

    def test_go_home_behavior_exists(self):
        """Test go_home behavior is registered."""
        registry = BehaviorRegistry()
        go_home = registry.get("go_home")
        assert go_home is not None
        assert go_home.display_name == "Go Home"
        assert "position_known" in go_home.opportunity_triggers
        assert go_home.has_tag("navigation")
        assert go_home.has_tag("goal")

    def test_go_to_charger_behavior_exists(self):
        """Test go_to_charger behavior is registered."""
        registry = BehaviorRegistry()
        go_charger = registry.get("go_to_charger")
        assert go_charger is not None
        assert go_charger.display_name == "Go To Charger"
        assert go_charger.interruptible is False
        assert go_charger.has_tag("charging")
        assert "energy" in go_charger.driven_by_needs

    def test_go_to_landmark_behavior_exists(self):
        """Test go_to_landmark behavior is registered."""
        registry = BehaviorRegistry()
        go_landmark = registry.get("go_to_landmark")
        assert go_landmark is not None
        assert go_landmark.has_tag("navigation")
        assert "curiosity" in go_landmark.driven_by_needs

    def test_explore_unfamiliar_behavior_exists(self):
        """Test explore_unfamiliar behavior is registered."""
        registry = BehaviorRegistry()
        explore = registry.get("explore_unfamiliar")
        assert explore is not None
        assert explore.display_name == "Explore Unfamiliar Area"
        assert "has_unfamiliar_zones" in explore.opportunity_triggers
        assert explore.has_tag("exploration")

    def test_patrol_behavior_exists(self):
        """Test patrol behavior is registered."""
        registry = BehaviorRegistry()
        patrol = registry.get("patrol")
        assert patrol is not None
        assert patrol.has_tag("navigation")
        assert "in_safe_zone" in patrol.opportunity_triggers

    def test_flee_danger_behavior_exists(self):
        """Test flee_danger behavior is registered."""
        registry = BehaviorRegistry()
        flee = registry.get("flee_danger")
        assert flee is not None
        assert flee.base_value >= 2.0  # High priority
        assert flee.interruptible is False
        assert "in_danger_zone" in flee.opportunity_triggers
        assert flee.has_tag("urgent")

    def test_retreat_to_safe_behavior_exists(self):
        """Test retreat_to_safe behavior is registered."""
        registry = BehaviorRegistry()
        retreat = registry.get("retreat_to_safe")
        assert retreat is not None
        assert "safety" in retreat.driven_by_needs
        assert retreat.has_tag("safety")

    def test_reorient_behavior_exists(self):
        """Test reorient behavior is registered."""
        registry = BehaviorRegistry()
        reorient = registry.get("reorient")
        assert reorient is not None
        assert "position_lost" in reorient.opportunity_triggers
        assert reorient.has_tag("recovery")

    def test_navigation_behaviors_count(self):
        """Test that all 8 navigation behaviors are registered."""
        registry = BehaviorRegistry()
        navigation_behaviors = registry.get_by_tag("navigation")
        assert len(navigation_behaviors) == 8


class TestNavigationTrees:
    """Tests for navigation behavior tree structures."""

    def test_go_home_tree_exists(self):
        """Test go_home tree is registered."""
        assert BehaviorTreeFactory.has_tree("go_home")
        tree = BehaviorTreeFactory.create_tree("go_home")
        assert isinstance(tree, Selector)

    def test_go_home_tree_has_position_lost_branch(self):
        """Test go_home tree handles position_lost."""
        tree = BehaviorTreeFactory.create_tree("go_home")
        branch_names = [c.name for c in tree.children]
        assert any("lost" in name for name in branch_names)

    def test_go_to_charger_tree_exists(self):
        """Test go_to_charger tree is registered."""
        assert BehaviorTreeFactory.has_tree("go_to_charger")
        tree = BehaviorTreeFactory.create_tree("go_to_charger")
        assert isinstance(tree, Selector)

    def test_flee_danger_tree_exists(self):
        """Test flee_danger tree is registered."""
        assert BehaviorTreeFactory.has_tree("flee_danger")
        tree = BehaviorTreeFactory.create_tree("flee_danger")
        assert isinstance(tree, Sequence)

    def test_patrol_tree_exists(self):
        """Test patrol tree is registered."""
        assert BehaviorTreeFactory.has_tree("patrol")
        tree = BehaviorTreeFactory.create_tree("patrol")
        assert isinstance(tree, Selector)

    def test_all_navigation_trees_exist(self):
        """Test all navigation behavior trees are registered."""
        navigation_trees = [
            "go_home",
            "go_to_charger",
            "go_to_landmark",
            "explore_unfamiliar",
            "patrol",
            "flee_danger",
            "retreat_to_safe",
            "reorient",
        ]
        for tree_name in navigation_trees:
            assert BehaviorTreeFactory.has_tree(tree_name), f"Missing tree: {tree_name}"


class TestNavigationTriggers:
    """Tests for navigation-related triggers in WorldContext."""

    def test_position_known_trigger(self):
        """Test position_known trigger activates with high confidence."""
        context = WorldContext(position_confidence=0.3)
        assert context.has_trigger("position_known") is False

        context = WorldContext(position_confidence=0.7)
        assert context.has_trigger("position_known") is True

    def test_position_lost_trigger(self):
        """Test position_lost trigger activates with low confidence."""
        context = WorldContext(position_confidence=0.3)
        assert context.has_trigger("position_lost") is False

        context = WorldContext(position_confidence=0.1)
        assert context.has_trigger("position_lost") is True

    def test_in_danger_zone_trigger(self):
        """Test in_danger_zone trigger activates with low safety."""
        context = WorldContext(current_zone_safety=0.5)
        assert context.has_trigger("in_danger_zone") is False

        context = WorldContext(current_zone_safety=0.2)
        assert context.has_trigger("in_danger_zone") is True

    def test_in_safe_zone_trigger(self):
        """Test in_safe_zone trigger activates with high safety."""
        context = WorldContext(current_zone_safety=0.5)
        assert context.has_trigger("in_safe_zone") is False

        context = WorldContext(current_zone_safety=0.8)
        assert context.has_trigger("in_safe_zone") is True

    def test_has_unfamiliar_zones_trigger(self):
        """Test has_unfamiliar_zones trigger."""
        context = WorldContext(has_unfamiliar_zones=False)
        assert context.has_trigger("has_unfamiliar_zones") is False

        context = WorldContext(has_unfamiliar_zones=True)
        assert context.has_trigger("has_unfamiliar_zones") is True

    def test_has_path_home_trigger(self):
        """Test has_path_home trigger."""
        context = WorldContext(has_path_to_home=False)
        assert context.has_trigger("has_path_home") is False

        context = WorldContext(has_path_to_home=True)
        assert context.has_trigger("has_path_home") is True

    def test_in_unfamiliar_zone_trigger(self):
        """Test in_unfamiliar_zone trigger based on zone familiarity."""
        context = WorldContext(zone_familiarity=0.7)
        assert context.has_trigger("in_unfamiliar_zone") is False

        context = WorldContext(zone_familiarity=0.3)
        assert context.has_trigger("in_unfamiliar_zone") is True


class TestSpatialMapPathfinding:
    """Tests for spatial map pathfinding methods."""

    def test_find_path_simple(self):
        """Test pathfinding between connected landmarks."""
        spatial_map = SpatialMapMemory()

        # Create landmarks with connections: A -> B -> C
        lm_a = SpatialLandmark("lm_a", "home_base")
        lm_b = SpatialLandmark("lm_b", "corner")
        lm_c = SpatialLandmark("lm_c", "charging_station")

        lm_a.add_connection("lm_b", 100.0)
        lm_b.add_connection("lm_a", 100.0)
        lm_b.add_connection("lm_c", 50.0)
        lm_c.add_connection("lm_b", 50.0)

        spatial_map.add_landmark(lm_a)
        spatial_map.add_landmark(lm_b)
        spatial_map.add_landmark(lm_c)
        spatial_map.update_current_location("lm_a")

        # Find path from A to C
        path = spatial_map.find_path_to("charging_station")
        assert path is not None
        assert path == ["lm_a", "lm_b", "lm_c"]

    def test_find_path_already_at_target(self):
        """Test pathfinding when already at target returns empty list."""
        spatial_map = SpatialMapMemory()

        lm_a = SpatialLandmark("lm_a", "home_base")
        spatial_map.add_landmark(lm_a)
        spatial_map.update_current_location("lm_a")
        spatial_map.set_home("lm_a")

        path = spatial_map.find_path_to("home_base")
        assert path == []  # Empty list = already there

    def test_find_path_no_path_exists(self):
        """Test pathfinding returns None when no path exists."""
        spatial_map = SpatialMapMemory()

        # Disconnected landmarks
        lm_a = SpatialLandmark("lm_a", "home_base")
        lm_b = SpatialLandmark("lm_b", "charging_station")

        spatial_map.add_landmark(lm_a)
        spatial_map.add_landmark(lm_b)
        spatial_map.update_current_location("lm_a")

        path = spatial_map.find_path_to("charging_station")
        assert path is None

    def test_find_path_position_unknown(self):
        """Test pathfinding returns None when position is unknown."""
        spatial_map = SpatialMapMemory()
        path = spatial_map.find_path_to("home_base")
        assert path is None

    def test_has_path_to(self):
        """Test has_path_to convenience method."""
        spatial_map = SpatialMapMemory()

        lm_a = SpatialLandmark("lm_a", "home_base")
        lm_b = SpatialLandmark("lm_b", "charging_station")
        lm_a.add_connection("lm_b", 100.0)
        lm_b.add_connection("lm_a", 100.0)

        spatial_map.add_landmark(lm_a)
        spatial_map.add_landmark(lm_b)
        spatial_map.update_current_location("lm_a")

        assert spatial_map.has_path_to("charging_station") is True
        assert spatial_map.has_path_to("nonexistent_type") is False

    def test_get_nearest_safe_zone(self):
        """Test finding nearest safe zone."""
        spatial_map = SpatialMapMemory()

        # Create landmarks
        lm_a = SpatialLandmark("lm_a", "corner")
        lm_b = SpatialLandmark("lm_b", "corner")
        lm_a.add_connection("lm_b", 100.0)
        lm_b.add_connection("lm_a", 100.0)

        # Create zones - lm_b is safe, lm_a is not
        zone_a = SpatialZone("zone_a", "dangerous", "lm_a", safety_score=0.2)
        zone_b = SpatialZone("zone_b", "safe", "lm_b", safety_score=0.9)

        spatial_map.add_landmark(lm_a)
        spatial_map.add_landmark(lm_b)
        spatial_map.add_zone(zone_a)
        spatial_map.add_zone(zone_b)
        spatial_map.update_current_location("lm_a")

        safe = spatial_map.get_nearest_safe_zone()
        assert safe == "lm_b"

    def test_get_unfamiliar_zones(self):
        """Test finding unfamiliar zones."""
        spatial_map = SpatialMapMemory()

        # Create zones with different familiarity
        zone_a = SpatialZone("zone_a", "play_area", "lm_a", familiarity=0.2)
        zone_b = SpatialZone("zone_b", "safe", "lm_b", familiarity=0.8)
        zone_c = SpatialZone("zone_c", "rest_area", "lm_c", familiarity=0.1)

        spatial_map.add_zone(zone_a)
        spatial_map.add_zone(zone_b)
        spatial_map.add_zone(zone_c)

        unfamiliar = spatial_map.get_unfamiliar_zones()
        assert len(unfamiliar) == 2  # zone_a and zone_c
        assert unfamiliar[0] == "lm_c"  # Most unfamiliar first

    def test_has_unfamiliar_zones(self):
        """Test has_unfamiliar_zones check."""
        spatial_map = SpatialMapMemory()

        # All zones are familiar
        zone_a = SpatialZone("zone_a", "safe", "lm_a", familiarity=0.9)
        spatial_map.add_zone(zone_a)
        assert spatial_map.has_unfamiliar_zones() is False

        # Add an unfamiliar zone
        zone_b = SpatialZone("zone_b", "safe", "lm_b", familiarity=0.2)
        spatial_map.add_zone(zone_b)
        assert spatial_map.has_unfamiliar_zones() is True


class TestNavigationActions:
    """Tests for navigation action nodes."""

    def test_navigate_to_landmark_action_creation(self):
        """Test NavigateToLandmarkAction can be created."""
        action = NavigateToLandmarkAction(target_type="home_base", timeout=30.0)
        assert action is not None
        assert "NavigateTo" in action.name

    def test_reorient_action_creation(self):
        """Test ReorientAction can be created."""
        action = ReorientAction(max_attempts=4)
        assert action is not None
        assert "Reorient" in action.name

    def test_move_toward_safety_action_creation(self):
        """Test MoveTowardSafetyAction can be created."""
        action = MoveTowardSafetyAction(retreat_duration=1.5, retreat_speed=0.5)
        assert action is not None
        assert action.name == "MoveTowardSafety"


class TestNavigationConditions:
    """Tests for navigation condition nodes."""

    def test_at_landmark_condition_creation(self):
        """Test AtLandmarkCondition can be created."""
        condition = AtLandmarkCondition(landmark_type="home_base")
        assert condition is not None
        assert "AtLandmark" in condition.name

    def test_zone_safety_condition_creation(self):
        """Test ZoneSafetyCondition can be created."""
        condition = ZoneSafetyCondition(min_safety=0.7)
        assert condition is not None
        assert "ZoneSafety" in condition.name

    def test_zone_is_dangerous_condition_creation(self):
        """Test ZoneSafetyCondition with check_dangerous flag."""
        condition = ZoneSafetyCondition(check_dangerous=True)
        assert condition is not None
        assert "Dangerous" in condition.name

    def test_has_path_condition_creation(self):
        """Test HasPathCondition can be created."""
        condition = HasPathCondition(target_type="charging_station")
        assert condition is not None
        assert "HasPathTo" in condition.name

    def test_has_unexplored_zones_condition_creation(self):
        """Test HasUnexploredZonesCondition can be created."""
        condition = HasUnexploredZonesCondition(familiarity_threshold=0.5)
        assert condition is not None
        assert "HasUnexploredZones" in condition.name

    def test_position_known_condition_creation(self):
        """Test PositionKnownCondition can be created."""
        condition = PositionKnownCondition()
        assert condition is not None
        assert condition.name == "PositionKnown"


class TestWorldContextNavigationFields:
    """Tests for navigation-related fields in WorldContext."""

    def test_navigation_fields_in_get_state(self):
        """Test navigation fields are serialized correctly."""
        context = WorldContext(
            has_unfamiliar_zones=True,
            has_path_to_home=True,
            has_path_to_charger=False,
            zone_familiarity=0.3,
        )
        state = context.get_state()
        assert state["has_unfamiliar_zones"] is True
        assert state["has_path_to_home"] is True
        assert state["has_path_to_charger"] is False
        assert state["zone_familiarity"] == 0.3

    def test_navigation_fields_from_state(self):
        """Test navigation fields are deserialized correctly."""
        state = {
            "has_unfamiliar_zones": True,
            "has_path_to_home": False,
            "has_path_to_charger": True,
            "zone_familiarity": 0.7,
        }
        context = WorldContext.from_state(state)
        assert context.has_unfamiliar_zones is True
        assert context.has_path_to_home is False
        assert context.has_path_to_charger is True
        assert context.zone_familiarity == 0.7

    def test_navigation_fields_in_summary(self):
        """Test navigation fields appear in summary."""
        context = WorldContext(
            has_unfamiliar_zones=True,
            has_path_to_home=True,
            zone_familiarity=0.4,
        )
        summary = context.summary()
        assert "spatial" in summary
        assert summary["spatial"]["has_unfamiliar_zones"] is True
        assert summary["spatial"]["has_path_home"] is True
        assert summary["spatial"]["zone_familiarity"] == 0.4
