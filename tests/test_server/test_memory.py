"""
Unit tests for Murph's Memory System.
"""

import time
import pytest
from server.cognition.memory import (
    PersonMemory,
    ObjectMemory,
    EventMemory,
    WorkingMemory,
    ShortTermMemory,
    MemorySystem,
)
from server.cognition.behavior import WorldContext


class TestPersonMemory:
    """Tests for the PersonMemory dataclass."""

    def test_person_memory_creation(self):
        """Test creating a person memory with default values."""
        person = PersonMemory(person_id="person_1")
        assert person.person_id == "person_1"
        assert person.name is None
        assert person.familiarity_score == 0.0
        assert person.is_familiar is False
        assert person.interaction_count == 0
        assert person.sentiment == 0.0
        assert len(person.tags) == 0

    def test_person_memory_with_name(self):
        """Test creating a person memory with a name."""
        person = PersonMemory(person_id="person_1", name="Alice")
        assert person.name == "Alice"

    def test_familiarity_threshold(self):
        """Test that is_familiar triggers at score >= 50."""
        person = PersonMemory(person_id="person_1", familiarity_score=49.9)
        assert person.is_familiar is False

        person = PersonMemory(person_id="person_1", familiarity_score=50.0)
        assert person.is_familiar is True

        person = PersonMemory(person_id="person_1", familiarity_score=100.0)
        assert person.is_familiar is True

    def test_record_sighting(self):
        """Test recording a sighting updates last_seen."""
        person = PersonMemory(person_id="person_1")
        original_last_seen = person.last_seen
        time.sleep(0.01)
        person.record_sighting()
        assert person.last_seen > original_last_seen
        assert person.interaction_count == 0  # No interaction

    def test_record_sighting_with_interaction(self):
        """Test recording an interaction increments count."""
        person = PersonMemory(person_id="person_1")
        person.record_sighting(interaction=True)
        assert person.interaction_count == 1
        person.record_sighting(interaction=True)
        assert person.interaction_count == 2

    def test_adjust_sentiment(self):
        """Test sentiment adjustment with clamping."""
        person = PersonMemory(person_id="person_1")
        person.adjust_sentiment(0.5)
        assert person.sentiment == 0.5

        person.adjust_sentiment(0.7)
        assert person.sentiment == 1.0  # Clamped to max

        person.adjust_sentiment(-2.5)
        assert person.sentiment == -1.0  # Clamped to min

    def test_person_memory_serialization(self):
        """Test state serialization and restoration."""
        person = PersonMemory(
            person_id="person_1",
            name="Alice",
            familiarity_score=75.0,
            interaction_count=10,
            sentiment=0.5,
        )
        person.add_tag("friendly")

        state = person.get_state()
        restored = PersonMemory.from_state(state)

        assert restored.person_id == "person_1"
        assert restored.name == "Alice"
        assert restored.familiarity_score == 75.0
        assert restored.interaction_count == 10
        assert restored.sentiment == 0.5
        assert "friendly" in restored.tags


class TestObjectMemory:
    """Tests for the ObjectMemory dataclass."""

    def test_object_memory_creation(self):
        """Test creating an object memory."""
        obj = ObjectMemory(object_id="ball_1", object_type="ball")
        assert obj.object_id == "ball_1"
        assert obj.object_type == "ball"
        assert obj.times_seen == 1
        assert obj.last_position is None
        assert obj.interesting is False

    def test_record_sighting(self):
        """Test recording object sightings."""
        obj = ObjectMemory(object_id="ball_1", object_type="ball")
        obj.record_sighting(position=(10.0, 20.0))
        assert obj.times_seen == 2
        assert obj.last_position == (10.0, 20.0)

    def test_mark_interesting(self):
        """Test marking object as interesting."""
        obj = ObjectMemory(object_id="ball_1", object_type="ball")
        assert obj.interesting is False
        obj.mark_interesting()
        assert obj.interesting is True

    def test_object_memory_serialization(self):
        """Test state serialization and restoration."""
        obj = ObjectMemory(
            object_id="ball_1",
            object_type="ball",
            times_seen=5,
            last_position=(15.0, 25.0),
            interesting=True,
        )

        state = obj.get_state()
        restored = ObjectMemory.from_state(state)

        assert restored.object_id == "ball_1"
        assert restored.object_type == "ball"
        assert restored.times_seen == 5
        assert restored.last_position == (15.0, 25.0)
        assert restored.interesting is True


class TestEventMemory:
    """Tests for the EventMemory dataclass."""

    def test_event_memory_creation(self):
        """Test creating an event memory."""
        event = EventMemory(event_type="greeting")
        assert event.event_type == "greeting"
        assert event.strength == 1.0
        assert event.outcome == "neutral"
        assert len(event.participants) == 0
        assert len(event.objects) == 0
        assert event.event_id is not None

    def test_event_with_participants(self):
        """Test creating event with participants and objects."""
        event = EventMemory(
            event_type="play",
            participants=["person_1"],
            objects=["ball_1"],
            outcome="positive",
        )
        assert "person_1" in event.participants
        assert "ball_1" in event.objects
        assert event.is_positive is True
        assert event.is_negative is False

    def test_event_decay(self):
        """Test event strength decay."""
        event = EventMemory(event_type="greeting")
        assert event.strength == 1.0
        event.decay(0.3)
        assert event.strength == pytest.approx(0.7)
        event.decay(0.9)
        assert event.strength == 0.0  # Clamped to 0

    def test_event_memory_serialization(self):
        """Test state serialization and restoration."""
        event = EventMemory(
            event_type="petting",
            participants=["person_1"],
            outcome="positive",
            strength=0.8,
        )

        state = event.get_state()
        restored = EventMemory.from_state(state)

        assert restored.event_type == "petting"
        assert "person_1" in restored.participants
        assert restored.outcome == "positive"
        assert restored.strength == 0.8


class TestWorkingMemory:
    """Tests for the WorkingMemory class."""

    def test_working_memory_creation(self):
        """Test creating working memory."""
        wm = WorkingMemory()
        assert wm.current_behavior is None
        assert wm.current_goal is None
        assert wm.active_person_id is None
        assert len(wm.active_objects) == 0
        assert wm.attention_target is None
        assert wm.was_interrupted is False

    def test_start_behavior(self):
        """Test starting a behavior."""
        wm = WorkingMemory()
        wm.start_behavior("explore", "curiosity")
        assert wm.current_behavior == "explore"
        assert wm.current_goal == "curiosity"
        assert wm.behavior_start_time is not None

    def test_end_behavior(self):
        """Test ending a behavior adds to history."""
        wm = WorkingMemory()
        wm.start_behavior("explore", "curiosity")
        wm.end_behavior("completed")
        assert wm.current_behavior is None
        assert wm.previous_behavior == "explore"
        assert "explore" in wm.behavior_history

    def test_behavior_history_limit(self):
        """Test behavior history has a limit of 5."""
        wm = WorkingMemory()
        for i in range(7):
            wm.start_behavior(f"behavior_{i}", None)
            wm.end_behavior("completed")

        assert len(wm.behavior_history) == 5
        # Most recent should be present
        assert "behavior_6" in wm.behavior_history
        # Oldest should be gone
        assert "behavior_0" not in wm.behavior_history

    def test_interrupt_behavior(self):
        """Test interrupting a behavior."""
        wm = WorkingMemory()
        wm.start_behavior("explore", "curiosity")
        wm.interrupt_behavior("person_detected")
        assert wm.current_behavior is None
        assert wm.was_interrupted is True
        assert wm.interruption_reason == "person_detected"

    def test_attention_tracking(self):
        """Test attention target and duration."""
        wm = WorkingMemory()
        wm.set_attention("person_1")
        assert wm.attention_target == "person_1"
        assert wm.attention_start_time is not None
        time.sleep(0.01)
        assert wm.attention_duration > 0

    def test_active_objects(self):
        """Test managing active objects."""
        wm = WorkingMemory()
        wm.add_active_object("ball_1")
        wm.add_active_object("cup_1")
        assert "ball_1" in wm.active_objects
        assert "cup_1" in wm.active_objects

        wm.remove_active_object("ball_1")
        assert "ball_1" not in wm.active_objects

        wm.clear_active_objects()
        assert len(wm.active_objects) == 0

    def test_was_doing(self):
        """Test checking recent behaviors."""
        wm = WorkingMemory()
        wm.start_behavior("explore", None)
        wm.end_behavior("completed")
        wm.start_behavior("greet", None)

        assert wm.was_doing("explore") is True
        assert wm.was_doing("sleep") is False

    def test_working_memory_serialization(self):
        """Test state serialization and restoration."""
        wm = WorkingMemory()
        wm.start_behavior("explore", "curiosity")
        wm.set_active_person("person_1")
        wm.add_active_object("ball_1")
        wm.set_attention("person_1")

        state = wm.get_state()
        restored = WorkingMemory.from_state(state)

        assert restored.current_behavior == "explore"
        assert restored.current_goal == "curiosity"
        assert restored.active_person_id == "person_1"
        assert "ball_1" in restored.active_objects
        assert restored.attention_target == "person_1"


class TestShortTermMemory:
    """Tests for the ShortTermMemory class."""

    def test_short_term_memory_creation(self):
        """Test creating short-term memory."""
        stm = ShortTermMemory()
        assert len(stm.get_all_people()) == 0
        assert len(stm.get_recent_events()) == 0

    def test_record_person_seen(self):
        """Test recording person sightings."""
        stm = ShortTermMemory()
        person = stm.record_person_seen("person_1")
        assert person.person_id == "person_1"
        assert person.familiarity_score > 0

    def test_person_familiarity_growth(self):
        """Test familiarity grows with repeated sightings."""
        stm = ShortTermMemory(familiarity_growth=5.0)
        stm.record_person_seen("person_1")
        initial_score = stm.get_person("person_1").familiarity_score

        # More sightings increase familiarity
        for _ in range(5):
            stm.record_person_seen("person_1")

        final_score = stm.get_person("person_1").familiarity_score
        assert final_score > initial_score

    def test_person_familiarity_with_interaction(self):
        """Test interactions increase familiarity more."""
        stm = ShortTermMemory(familiarity_growth=5.0)
        stm.record_person_seen("person_1", interaction=False)
        score_without_interaction = stm.get_person("person_1").familiarity_score

        stm2 = ShortTermMemory(familiarity_growth=5.0)
        stm2.record_person_seen("person_1", interaction=True)
        score_with_interaction = stm2.get_person("person_1").familiarity_score

        assert score_with_interaction > score_without_interaction

    def test_person_familiarity_with_proximity(self):
        """Test proximity increases familiarity more."""
        stm = ShortTermMemory(familiarity_growth=5.0)
        stm.record_person_seen("person_1", distance=100.0)
        score_far = stm.get_person("person_1").familiarity_score

        stm2 = ShortTermMemory(familiarity_growth=5.0)
        stm2.record_person_seen("person_1", distance=30.0)
        score_close = stm2.get_person("person_1").familiarity_score

        assert score_close > score_far

    def test_record_object_seen(self):
        """Test recording object sightings."""
        stm = ShortTermMemory()
        obj = stm.record_object_seen("ball_1", "ball")
        assert obj.object_id == "ball_1"
        assert obj.object_type == "ball"

        # Record again
        obj = stm.record_object_seen("ball_1", "ball")
        assert obj.times_seen == 2

    def test_record_event(self):
        """Test recording events."""
        stm = ShortTermMemory()
        event = stm.record_event("greeting", ["person_1"], [], "positive")
        assert event.event_type == "greeting"
        assert event.strength == 1.0

    def test_event_updates_sentiment(self):
        """Test events update person sentiment."""
        stm = ShortTermMemory()
        stm.record_person_seen("person_1")
        initial_sentiment = stm.get_person("person_1").sentiment

        stm.record_event("petting", ["person_1"], [], "positive")
        final_sentiment = stm.get_person("person_1").sentiment

        assert final_sentiment > initial_sentiment

    def test_event_decay(self):
        """Test events decay over time."""
        stm = ShortTermMemory(event_decay_rate=1.0)  # 1 point/minute
        stm.record_event("greeting", [], [], "neutral")

        events = stm.get_recent_events()
        assert events[0].strength == 1.0

        # Simulate 30 seconds passing
        stm.update(30.0)
        events = stm.get_recent_events()
        assert events[0].strength == pytest.approx(0.5, abs=0.1)

    def test_event_pruning(self):
        """Test events are pruned when strength falls below threshold."""
        stm = ShortTermMemory(event_decay_rate=60.0, event_threshold=0.1)
        stm.record_event("old_event", [], [], "neutral")

        assert len(stm.get_recent_events()) == 1

        # Decay to below threshold
        stm.update(60.0)  # 60 points decay

        assert len(stm.get_recent_events()) == 0

    def test_was_event_recent(self):
        """Test checking for recent events."""
        stm = ShortTermMemory()
        stm.record_event("greeting", [], [], "neutral")

        assert stm.was_event_recent("greeting", within_seconds=60.0) is True
        assert stm.was_event_recent("play", within_seconds=60.0) is False

    def test_get_familiar_people(self):
        """Test getting familiar people."""
        stm = ShortTermMemory(familiarity_growth=60.0)  # Fast growth
        stm.record_person_seen("person_1")  # Will be familiar (60 > 50)
        stm.record_person_seen("person_2")

        # Make person_2 not familiar by resetting score
        stm._people["person_2"].familiarity_score = 30.0

        familiar = stm.get_familiar_people()
        assert len(familiar) == 1
        assert familiar[0].person_id == "person_1"

    def test_short_term_memory_serialization(self):
        """Test state serialization and restoration."""
        stm = ShortTermMemory()
        stm.record_person_seen("person_1")
        stm.record_object_seen("ball_1", "ball")
        stm.record_event("greeting", ["person_1"], [], "positive")

        state = stm.get_state()
        restored = ShortTermMemory.from_state(state)

        assert restored.get_person("person_1") is not None
        assert restored.get_object("ball_1") is not None
        assert len(restored.get_recent_events()) == 1


class TestMemorySystem:
    """Tests for the unified MemorySystem."""

    def test_memory_system_creation(self):
        """Test creating the memory system."""
        ms = MemorySystem()
        assert ms.working is not None
        assert ms.short_term is not None

    def test_process_perception_person(self):
        """Test processing perception with person detection."""
        ms = MemorySystem()
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            person_distance=50.0,
        )

        assert ms.short_term.get_person("person_1") is not None

    def test_process_perception_interaction(self):
        """Test processing perception during interaction."""
        ms = MemorySystem()
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            person_distance=20.0,  # Close = interaction
        )

        assert ms.working.active_person_id == "person_1"
        assert ms.working.attention_target == "person_1"

    def test_process_perception_petting(self):
        """Test petting creates positive event."""
        ms = MemorySystem()
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            is_being_petted=True,
        )

        assert ms.short_term.was_event_recent("petting", 60.0) is True

    def test_record_behavior(self):
        """Test recording behavior start and end."""
        ms = MemorySystem()
        ms.record_behavior_start("explore", "curiosity")
        assert ms.working.current_behavior == "explore"

        ms.record_behavior_end("completed")
        assert ms.working.current_behavior is None
        assert ms.working.previous_behavior == "explore"

    def test_get_behavior_context(self):
        """Test getting behavior context."""
        ms = MemorySystem(familiarity_growth=60.0)
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            person_distance=20.0,
        )
        ms.record_behavior_start("greet", "social")

        context = ms.get_behavior_context()
        assert context["current_behavior"] == "greet"
        assert context["active_person_familiar"] is True  # Score > 50

    def test_get_memory_triggers(self):
        """Test getting memory-derived triggers."""
        ms = MemorySystem(familiarity_growth=60.0)
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            person_distance=20.0,
        )

        # Record a greeting event
        ms.record_event("greeting", ["person_1"], [], "positive")

        triggers = ms.get_memory_triggers()
        assert triggers["familiar_person_remembered"] is True
        assert triggers["recently_greeted"] is True

    def test_memory_system_serialization(self):
        """Test state serialization and restoration."""
        ms = MemorySystem()
        ms.process_perception(
            person_detected=True,
            person_id="person_1",
            person_distance=50.0,
        )
        ms.record_behavior_start("explore", "curiosity")
        ms.record_event("greeting", ["person_1"], [], "positive")

        state = ms.get_state()
        restored = MemorySystem.from_state(state)

        assert restored.working.current_behavior == "explore"
        assert restored.short_term.get_person("person_1") is not None
        assert len(restored.short_term.get_recent_events()) == 1


class TestWorldContextMemoryIntegration:
    """Tests for WorldContext memory-derived triggers."""

    def test_memory_derived_triggers_default(self):
        """Test memory triggers with default values."""
        context = WorldContext()
        assert context.has_trigger("familiar_person_remembered") is False
        assert context.has_trigger("positive_history") is False
        assert context.has_trigger("recently_greeted") is False

    def test_familiar_person_remembered_trigger(self):
        """Test familiar_person_remembered trigger."""
        context = WorldContext(
            person_detected=True,
            remembered_person_name="Alice",
        )
        assert context.has_trigger("familiar_person_remembered") is True

    def test_positive_history_trigger(self):
        """Test positive_history trigger."""
        context = WorldContext(person_interaction_count=4)
        assert context.has_trigger("positive_history") is False

        context = WorldContext(person_interaction_count=5)
        assert context.has_trigger("positive_history") is True

    def test_sentiment_triggers(self):
        """Test sentiment-based triggers."""
        context = WorldContext(person_sentiment=-0.5)
        assert context.has_trigger("negative_sentiment") is True
        assert context.has_trigger("positive_sentiment") is False

        context = WorldContext(person_sentiment=0.5)
        assert context.has_trigger("negative_sentiment") is False
        assert context.has_trigger("positive_sentiment") is True

    def test_recent_event_triggers(self):
        """Test recent event type triggers."""
        context = WorldContext(recent_event_types=["greeting", "play"])
        assert context.has_trigger("recently_greeted") is True
        assert context.has_trigger("recently_played") is True
        assert context.has_trigger("recently_petted") is False

    def test_context_serialization_with_memory_fields(self):
        """Test WorldContext serialization includes memory fields."""
        context = WorldContext(
            remembered_person_name="Alice",
            person_interaction_count=10,
            person_sentiment=0.5,
            recent_event_types=["greeting"],
        )

        state = context.get_state()
        restored = WorldContext.from_state(state)

        assert restored.remembered_person_name == "Alice"
        assert restored.person_interaction_count == 10
        assert restored.person_sentiment == 0.5
        assert "greeting" in restored.recent_event_types
