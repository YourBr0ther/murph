"""
Unit tests for Murph's Needs System.
"""

import pytest
from server.cognition.needs import Need, Personality, NeedsSystem, PRESETS


class TestNeed:
    """Tests for the Need base class."""

    def test_need_creation(self):
        """Test creating a need with default values."""
        need = Need(name="test")
        assert need.name == "test"
        assert need.value == 100.0
        assert need.decay_rate == 1.0
        assert need.critical_threshold == 20.0

    def test_need_decay(self):
        """Test that needs decay over time."""
        need = Need(name="test", value=100.0, decay_rate=60.0)  # 60 per minute = 1 per second
        need.decay(10.0)  # 10 seconds
        assert need.value == pytest.approx(90.0, abs=0.1)

    def test_need_decay_stops_at_zero(self):
        """Test that needs don't go below zero."""
        need = Need(name="test", value=5.0, decay_rate=60.0)
        need.decay(60.0)  # Should decay more than 5 points
        assert need.value == 0.0

    def test_need_satisfy(self):
        """Test satisfying a need."""
        need = Need(name="test", value=50.0)
        need.satisfy(30.0)
        assert need.value == 80.0

    def test_need_satisfy_caps_at_max(self):
        """Test that satisfaction doesn't exceed max."""
        need = Need(name="test", value=90.0)
        need.satisfy(50.0)
        assert need.value == 100.0

    def test_need_deplete(self):
        """Test depleting a need."""
        need = Need(name="test", value=100.0)
        need.deplete(30.0)
        assert need.value == 70.0

    def test_need_is_critical(self):
        """Test critical threshold detection."""
        need = Need(name="test", critical_threshold=20.0, value=25.0)
        assert not need.is_critical()

        need.value = 15.0
        assert need.is_critical()

    def test_need_is_satisfied(self):
        """Test satisfaction detection."""
        need = Need(name="test", value=85.0)
        assert need.is_satisfied(threshold=80.0)

        need.value = 75.0
        assert not need.is_satisfied(threshold=80.0)

    def test_need_urgency(self):
        """Test urgency calculation."""
        need = Need(name="test", value=100.0)
        assert need.urgency() == 0.0

        need.value = 0.0
        assert need.urgency() == 1.0

        need.value = 50.0
        assert need.urgency() == 0.5

    def test_need_weighted_urgency(self):
        """Test weighted urgency calculation."""
        need = Need(name="test", value=50.0, happiness_weight=2.0)
        assert need.weighted_urgency() == 1.0  # 0.5 urgency * 2.0 weight

    def test_need_state_serialization(self):
        """Test saving and loading need state."""
        need = Need(
            name="test",
            value=75.0,
            decay_rate=1.5,
            critical_threshold=25.0,
            happiness_weight=1.2,
            satisfaction_behaviors=["behavior1", "behavior2"],
        )

        state = need.get_state()
        restored = Need.from_state(state)

        assert restored.name == need.name
        assert restored.value == need.value
        assert restored.decay_rate == need.decay_rate
        assert restored.critical_threshold == need.critical_threshold
        assert restored.happiness_weight == need.happiness_weight
        assert restored.satisfaction_behaviors == need.satisfaction_behaviors

    def test_need_zero_decay_rate(self):
        """Test that zero decay rate doesn't decay."""
        need = Need(name="safety", value=100.0, decay_rate=0.0)
        need.decay(3600.0)  # 1 hour
        assert need.value == 100.0


class TestPersonality:
    """Tests for the Personality class."""

    def test_personality_creation(self):
        """Test creating a personality with default values."""
        personality = Personality()
        assert personality.playfulness == 0.0
        assert personality.boldness == 0.0
        assert personality.sociability == 0.0
        assert personality.curiosity == 0.0
        assert personality.affectionate == 0.0
        assert personality.energy_level == 0.0

    def test_personality_clamping(self):
        """Test that traits are clamped to valid range."""
        personality = Personality(playfulness=2.0, boldness=-2.0)
        assert personality.playfulness == 1.0
        assert personality.boldness == -1.0

    def test_personality_from_preset(self):
        """Test creating personality from preset."""
        personality = Personality.from_preset("playful")
        assert personality.playfulness == 0.8
        assert personality.energy_level == 0.7

    def test_personality_from_invalid_preset(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError):
            Personality.from_preset("nonexistent")

    def test_personality_random(self):
        """Test random personality generation."""
        p1 = Personality.random()
        p2 = Personality.random()

        # Should be different (extremely unlikely to be same)
        assert p1.get_state() != p2.get_state()

        # All values should be in valid range
        for trait in p1.TRAIT_NAMES:
            value = p1.get_trait(trait)
            assert -1.0 <= value <= 1.0

    def test_behavior_modifier(self):
        """Test behavior modifier calculation."""
        # High playfulness should boost play behavior
        playful = Personality(playfulness=1.0)
        modifier = playful.get_behavior_modifier("play")
        assert modifier > 1.0

        # Low playfulness should reduce play behavior
        serious = Personality(playfulness=-1.0)
        modifier = serious.get_behavior_modifier("play")
        assert modifier < 1.0

    def test_behavior_modifier_unknown(self):
        """Test that unknown behaviors return 1.0 modifier."""
        personality = Personality()
        assert personality.get_behavior_modifier("unknown_behavior") == 1.0

    def test_need_decay_modifier(self):
        """Test need decay modifier calculation."""
        # High curiosity should make curiosity decay faster
        curious = Personality(curiosity=1.0)
        modifier = curious.get_need_decay_modifier("curiosity")
        assert modifier > 1.0

        # Low curiosity should make curiosity decay slower
        content = Personality(curiosity=-1.0)
        modifier = content.get_need_decay_modifier("curiosity")
        assert modifier < 1.0

    def test_personality_state_serialization(self):
        """Test saving and loading personality state."""
        personality = Personality(
            playfulness=0.5,
            boldness=-0.3,
            sociability=0.8,
            curiosity=0.2,
            affectionate=-0.1,
            energy_level=0.6,
        )

        state = personality.get_state()
        restored = Personality.from_state(state)

        assert restored.playfulness == personality.playfulness
        assert restored.boldness == personality.boldness
        assert restored.sociability == personality.sociability
        assert restored.curiosity == personality.curiosity
        assert restored.affectionate == personality.affectionate
        assert restored.energy_level == personality.energy_level

    def test_personality_describe(self):
        """Test personality description."""
        playful = Personality.from_preset("playful")
        description = playful.describe()
        assert "playful" in description.lower() or len(description) > 0

        balanced = Personality()
        description = balanced.describe()
        assert description == "balanced personality"

    def test_available_presets(self):
        """Test getting available presets."""
        presets = Personality.available_presets()
        assert "playful" in presets
        assert "shy" in presets
        assert "explorer" in presets
        assert "cuddly" in presets
        assert "hyper" in presets


class TestNeedsSystem:
    """Tests for the NeedsSystem class."""

    def test_needs_system_creation(self):
        """Test creating a needs system."""
        system = NeedsSystem()
        assert len(system.needs) == 7  # All default needs
        assert "energy" in system.needs
        assert "curiosity" in system.needs
        assert "play" in system.needs
        assert "social" in system.needs
        assert "affection" in system.needs
        assert "comfort" in system.needs
        assert "safety" in system.needs

    def test_needs_system_with_personality(self):
        """Test creating a needs system with specific personality."""
        personality = Personality.from_preset("playful")
        system = NeedsSystem(personality=personality)
        assert system.personality.playfulness == 0.8

    def test_needs_start_satisfied(self):
        """Test that all needs start at 100."""
        system = NeedsSystem()
        for need in system.needs.values():
            assert need.value == 100.0

    def test_needs_system_update(self):
        """Test updating needs over time."""
        system = NeedsSystem()
        initial_curiosity = system.needs["curiosity"].value

        # Simulate 60 seconds
        system.update(60.0)

        # Curiosity should have decayed (decay_rate=1.5/min)
        assert system.needs["curiosity"].value < initial_curiosity

        # Safety shouldn't decay (decay_rate=0)
        assert system.needs["safety"].value == 100.0

    def test_get_need(self):
        """Test getting a specific need."""
        system = NeedsSystem()
        energy = system.get_need("energy")
        assert energy is not None
        assert energy.name == "energy"

        unknown = system.get_need("unknown")
        assert unknown is None

    def test_satisfy_need(self):
        """Test satisfying a need."""
        system = NeedsSystem()
        system.needs["energy"].value = 50.0

        result = system.satisfy_need("energy", 30.0)
        assert result is True
        assert system.needs["energy"].value == 80.0

        # Unknown need
        result = system.satisfy_need("unknown", 30.0)
        assert result is False

    def test_deplete_need(self):
        """Test depleting a need."""
        system = NeedsSystem()
        result = system.deplete_need("energy", 30.0)
        assert result is True
        assert system.needs["energy"].value == 70.0

    def test_calculate_happiness(self):
        """Test happiness calculation."""
        system = NeedsSystem()

        # All needs at 100 = max happiness
        happiness = system.calculate_happiness()
        assert happiness == pytest.approx(100.0, abs=0.1)

        # Lower some needs
        system.needs["energy"].value = 50.0
        system.needs["play"].value = 50.0
        happiness = system.calculate_happiness()
        assert happiness < 100.0
        assert happiness > 0.0

    def test_get_critical_needs(self):
        """Test getting critical needs."""
        system = NeedsSystem()

        # No critical needs initially
        critical = system.get_critical_needs()
        assert len(critical) == 0

        # Make some needs critical
        system.needs["energy"].value = 10.0  # Below 20 threshold
        system.needs["play"].value = 15.0  # Below 25 threshold

        critical = system.get_critical_needs()
        assert len(critical) == 2
        assert any(n.name == "energy" for n in critical)
        assert any(n.name == "play" for n in critical)

    def test_get_most_urgent_need(self):
        """Test getting most urgent need."""
        system = NeedsSystem()

        # Lower one need significantly
        system.needs["affection"].value = 10.0

        most_urgent = system.get_most_urgent_need()
        assert most_urgent is not None
        assert most_urgent.name == "affection"

    def test_get_behaviors_for_need(self):
        """Test getting behaviors that satisfy a need."""
        system = NeedsSystem()

        behaviors = system.get_behaviors_for_need("energy")
        assert "rest" in behaviors
        assert "charge" in behaviors

        behaviors = system.get_behaviors_for_need("play")
        assert "play" in behaviors
        assert "chase" in behaviors

    def test_get_suggested_behaviors(self):
        """Test getting behavior suggestions."""
        system = NeedsSystem()

        # Make curiosity very low
        system.needs["curiosity"].value = 10.0

        suggestions = system.get_suggested_behaviors(count=3)
        assert len(suggestions) <= 3

        # Curiosity-related behaviors should be highly suggested
        behavior_names = [s[0] for s in suggestions]
        assert any(b in ["explore", "investigate", "observe", "wander"] for b in behavior_names)

    def test_get_mood(self):
        """Test mood calculation."""
        system = NeedsSystem()

        # All satisfied = happy
        assert system.get_mood() == "happy"

        # Lower happiness
        for need in system.needs.values():
            need.value = 60.0
        assert system.get_mood() == "content"

        # Make needs critical
        system.needs["energy"].value = 10.0
        system.needs["play"].value = 10.0
        system.needs["social"].value = 10.0
        assert system.get_mood() == "distressed"

    def test_needs_system_state_serialization(self):
        """Test saving and loading needs system state."""
        personality = Personality.from_preset("playful")
        system = NeedsSystem(personality=personality)

        # Modify some needs
        system.needs["energy"].value = 75.0
        system.needs["play"].value = 50.0

        # Save and restore
        state = system.get_state()
        restored = NeedsSystem.from_state(state)

        assert restored.personality.playfulness == personality.playfulness
        assert restored.needs["energy"].value == 75.0
        assert restored.needs["play"].value == 50.0

    def test_summary(self):
        """Test getting a summary of needs state."""
        system = NeedsSystem()
        system.needs["energy"].value = 10.0

        summary = system.summary()
        assert "happiness" in summary
        assert "mood" in summary
        assert "critical_needs" in summary
        assert "energy" in summary["critical_needs"]
        assert "needs" in summary

    def test_personality_affects_decay(self):
        """Test that personality modifies need decay."""
        # High curiosity personality = curiosity decays faster
        curious = Personality(curiosity=1.0)
        system_curious = NeedsSystem(personality=curious)

        # Balanced personality
        balanced = Personality()
        system_balanced = NeedsSystem(personality=balanced)

        # Update both for same time
        system_curious.update(60.0)
        system_balanced.update(60.0)

        # Curious personality should have lower curiosity
        assert system_curious.needs["curiosity"].value < system_balanced.needs["curiosity"].value


class TestIntegration:
    """Integration tests for the needs system."""

    def test_simulation_over_time(self):
        """Test simulating needs over an extended period."""
        system = NeedsSystem(personality=Personality.from_preset("playful"))

        # Simulate 10 minutes without any satisfaction
        for _ in range(10):
            system.update(60.0)  # 1 minute

        # Some needs should be low
        assert system.needs["curiosity"].value < 100.0
        assert system.needs["play"].value < 100.0

        # Energy should still be relatively high (slow decay)
        assert system.needs["energy"].value > 50.0

        # Safety should be unchanged (no decay)
        assert system.needs["safety"].value == 100.0

    def test_behavior_cycle(self):
        """Test a typical behavior cycle."""
        system = NeedsSystem()

        # Simulate time passing
        system.update(300.0)  # 5 minutes

        # Get suggestions
        suggestions = system.get_suggested_behaviors(count=3)
        assert len(suggestions) > 0

        # "Do" the top suggested behavior
        top_behavior, need_name, _ = suggestions[0]
        original_value = system.needs[need_name].value

        # Satisfy that need
        system.satisfy_need(need_name, 50.0)

        # Need should be higher now
        assert system.needs[need_name].value > original_value

    def test_critical_need_recovery(self):
        """Test recovering from critical needs."""
        system = NeedsSystem()

        # Make multiple needs critical
        system.needs["energy"].value = 5.0
        system.needs["social"].value = 5.0

        assert system.get_mood() == "uneasy" or system.get_mood() == "distressed"

        # Satisfy the critical needs
        system.satisfy_need("energy", 80.0)
        system.satisfy_need("social", 80.0)

        # Should be happier now
        assert system.get_mood() in ["happy", "content"]
