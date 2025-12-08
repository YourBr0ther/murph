"""
Unit tests for Murph's Long-Term Memory System.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from server.cognition.memory import (
    EventMemory,
    LongTermMemory,
    MemorySystem,
    ObjectMemory,
    PersonMemory,
)
from server.storage import Database


@pytest.fixture
async def database():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        db = Database(db_path)
        await db.initialize()
        yield db
        await db.close()


@pytest.fixture
async def long_term_memory(database):
    """Create a long-term memory instance."""
    ltm = LongTermMemory(database)
    await ltm.initialize()
    return ltm


class TestDatabaseSetup:
    """Tests for database initialization."""

    async def test_database_creates_file(self):
        """Test that database file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            await db.initialize()
            assert db_path.exists()
            await db.close()

    async def test_database_creates_directory(self):
        """Test that database directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "test.db"
            db = Database(db_path)
            await db.initialize()
            assert db_path.exists()
            await db.close()

    async def test_database_not_initialized_error(self):
        """Test error when using uninitialized database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            with pytest.raises(RuntimeError, match="not initialized"):
                async with db.session():
                    pass


class TestLongTermMemoryPerson:
    """Tests for person-related long-term memory operations."""

    async def test_save_and_load_person(self, long_term_memory):
        """Test saving and loading a person."""
        person = PersonMemory(
            person_id="test_person_1",
            name="Alice",
            familiarity_score=75.0,
            interaction_count=10,
            sentiment=0.5,
        )
        person.add_tag("friendly")

        await long_term_memory.save_person(person, trust_score=80.0)
        loaded = await long_term_memory.get_person("test_person_1")

        assert loaded is not None
        assert loaded.person_id == "test_person_1"
        assert loaded.name == "Alice"
        assert loaded.familiarity_score == 75.0
        assert loaded.interaction_count == 10
        assert loaded.sentiment == 0.5
        assert "friendly" in loaded.tags

    async def test_person_not_found(self, long_term_memory):
        """Test getting a non-existent person returns None."""
        result = await long_term_memory.get_person("nonexistent")
        assert result is None

    async def test_update_existing_person(self, long_term_memory):
        """Test updating an existing person."""
        person = PersonMemory(person_id="test_person_1", familiarity_score=50.0)
        await long_term_memory.save_person(person)

        # Update
        person.familiarity_score = 80.0
        person.name = "Bob"
        await long_term_memory.save_person(person)

        loaded = await long_term_memory.get_person("test_person_1")
        assert loaded.familiarity_score == 80.0
        assert loaded.name == "Bob"

    async def test_get_familiar_people(self, long_term_memory):
        """Test getting all familiar people."""
        # Create familiar person
        familiar = PersonMemory(person_id="familiar_1", familiarity_score=75.0)
        await long_term_memory.save_person(familiar)

        # Create non-familiar person
        non_familiar = PersonMemory(person_id="stranger_1", familiarity_score=30.0)
        await long_term_memory.save_person(non_familiar)

        familiar_list = await long_term_memory.get_all_familiar_people()
        assert len(familiar_list) == 1
        assert familiar_list[0].person_id == "familiar_1"

    async def test_get_all_people(self, long_term_memory):
        """Test getting all people."""
        p1 = PersonMemory(person_id="p1", familiarity_score=75.0)
        p2 = PersonMemory(person_id="p2", familiarity_score=30.0)
        await long_term_memory.save_person(p1)
        await long_term_memory.save_person(p2)

        all_people = await long_term_memory.get_all_people()
        assert len(all_people) == 2

    async def test_get_person_by_name(self, long_term_memory):
        """Test finding a person by name."""
        person = PersonMemory(
            person_id="named_1", name="Charlie", familiarity_score=60.0
        )
        await long_term_memory.save_person(person)

        found = await long_term_memory.get_person_by_name("Charlie")
        assert found is not None
        assert found.person_id == "named_1"

        not_found = await long_term_memory.get_person_by_name("Nobody")
        assert not_found is None

    async def test_should_persist_person(self, long_term_memory):
        """Test persistence criteria for people."""
        # Familiar should persist
        familiar = PersonMemory(person_id="p1", familiarity_score=50.0)
        assert long_term_memory.should_persist_person(familiar) is True

        # Named should persist
        named = PersonMemory(person_id="p2", name="Chris", familiarity_score=10.0)
        assert long_term_memory.should_persist_person(named) is True

        # Unknown stranger should not
        stranger = PersonMemory(person_id="p3", familiarity_score=20.0)
        assert long_term_memory.should_persist_person(stranger) is False


class TestLongTermMemoryFaceEmbedding:
    """Tests for face embedding operations."""

    async def test_save_and_load_face_embedding(self, long_term_memory):
        """Test saving and loading face embeddings."""
        # First create a person
        person = PersonMemory(person_id="face_test_1", familiarity_score=60.0)
        await long_term_memory.save_person(person)

        # Save embedding
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        result = await long_term_memory.save_face_embedding("face_test_1", embedding)
        assert result is True

        # Load embeddings
        embeddings = await long_term_memory.get_face_embeddings("face_test_1")
        assert len(embeddings) == 1
        assert embeddings[0].shape == (128,)
        np.testing.assert_array_almost_equal(embeddings[0], embedding)

    async def test_save_embedding_invalid_shape(self, long_term_memory):
        """Test that invalid embedding shape raises error."""
        person = PersonMemory(person_id="shape_test", familiarity_score=60.0)
        await long_term_memory.save_person(person)

        wrong_shape = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError, match="128-dim"):
            await long_term_memory.save_face_embedding("shape_test", wrong_shape)

    async def test_save_embedding_person_not_found(self, long_term_memory):
        """Test saving embedding for non-existent person."""
        embedding = np.random.randn(128).astype(np.float32)
        result = await long_term_memory.save_face_embedding("nonexistent", embedding)
        assert result is False

    async def test_find_person_by_embedding(self, long_term_memory):
        """Test finding a person by face embedding."""
        # Create person and save embedding
        person = PersonMemory(person_id="face_match_1", familiarity_score=60.0)
        await long_term_memory.save_person(person)

        original_embedding = np.random.randn(128).astype(np.float32)
        original_embedding = original_embedding / np.linalg.norm(original_embedding)
        await long_term_memory.save_face_embedding("face_match_1", original_embedding)

        # Search with similar embedding (add very small noise)
        noise = np.random.randn(128).astype(np.float32) * 0.05
        search_embedding = original_embedding + noise
        search_embedding = search_embedding / np.linalg.norm(search_embedding)

        found_id, similarity = await long_term_memory.find_person_by_embedding(
            search_embedding, threshold=0.6
        )

        assert found_id == "face_match_1"
        assert similarity > 0.6

    async def test_no_match_below_threshold(self, long_term_memory):
        """Test that dissimilar embeddings don't match."""
        person = PersonMemory(person_id="no_match_1", familiarity_score=60.0)
        await long_term_memory.save_person(person)

        stored = np.random.randn(128).astype(np.float32)
        stored = stored / np.linalg.norm(stored)
        await long_term_memory.save_face_embedding("no_match_1", stored)

        # Completely different embedding
        different = np.random.randn(128).astype(np.float32)
        different = different / np.linalg.norm(different)

        found_id, similarity = await long_term_memory.find_person_by_embedding(
            different, threshold=0.9
        )

        assert found_id is None

    async def test_multiple_embeddings_per_person(self, long_term_memory):
        """Test storing multiple embeddings for one person."""
        person = PersonMemory(person_id="multi_emb", familiarity_score=60.0)
        await long_term_memory.save_person(person)

        # Save three embeddings
        for _ in range(3):
            emb = np.random.randn(128).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            await long_term_memory.save_face_embedding("multi_emb", emb)

        embeddings = await long_term_memory.get_face_embeddings("multi_emb")
        assert len(embeddings) == 3


class TestLongTermMemoryObject:
    """Tests for object-related long-term memory operations."""

    async def test_save_and_load_object(self, long_term_memory):
        """Test saving and loading an object."""
        obj = ObjectMemory(
            object_id="ball_1",
            object_type="ball",
            times_seen=15,
            last_position=(10.0, 20.0),
            interesting=True,
        )

        await long_term_memory.save_object(obj)
        loaded = await long_term_memory.get_object("ball_1")

        assert loaded is not None
        assert loaded.object_id == "ball_1"
        assert loaded.object_type == "ball"
        assert loaded.times_seen == 15
        assert loaded.last_position == (10.0, 20.0)
        assert loaded.interesting is True

    async def test_object_not_found(self, long_term_memory):
        """Test getting a non-existent object returns None."""
        result = await long_term_memory.get_object("nonexistent")
        assert result is None

    async def test_update_object(self, long_term_memory):
        """Test updating an existing object."""
        obj = ObjectMemory(object_id="cup_1", object_type="cup", times_seen=5)
        await long_term_memory.save_object(obj)

        obj.times_seen = 20
        obj.interesting = True
        await long_term_memory.save_object(obj)

        loaded = await long_term_memory.get_object("cup_1")
        assert loaded.times_seen == 20
        assert loaded.interesting is True

    async def test_get_interesting_objects(self, long_term_memory):
        """Test getting all interesting objects."""
        interesting = ObjectMemory(
            object_id="toy_1", object_type="toy", interesting=True
        )
        boring = ObjectMemory(object_id="box_1", object_type="box", interesting=False)
        await long_term_memory.save_object(interesting)
        await long_term_memory.save_object(boring)

        interesting_list = await long_term_memory.get_interesting_objects()
        assert len(interesting_list) == 1
        assert interesting_list[0].object_id == "toy_1"

    async def test_should_persist_object(self, long_term_memory):
        """Test persistence criteria for objects."""
        # Interesting should persist
        interesting = ObjectMemory(object_id="o1", object_type="toy", interesting=True)
        assert long_term_memory.should_persist_object(interesting) is True

        # Frequently seen should persist
        frequent = ObjectMemory(object_id="o2", object_type="cup", times_seen=15)
        assert long_term_memory.should_persist_object(frequent) is True

        # Rarely seen, not interesting should not
        rare = ObjectMemory(object_id="o3", object_type="thing", times_seen=3)
        assert long_term_memory.should_persist_object(rare) is False


class TestLongTermMemoryEvent:
    """Tests for event-related long-term memory operations."""

    async def test_save_and_load_event(self, long_term_memory):
        """Test saving and loading an event."""
        event = EventMemory(
            event_type="first_meeting",
            participants=["person_1"],
            outcome="positive",
        )

        result = await long_term_memory.save_event(event, significance=1.0)
        assert result is True

        loaded = await long_term_memory.get_event(event.event_id)
        assert loaded is not None
        assert loaded.event_type == "first_meeting"
        assert "person_1" in loaded.participants
        assert loaded.outcome == "positive"

    async def test_event_not_found(self, long_term_memory):
        """Test getting a non-existent event returns None."""
        result = await long_term_memory.get_event("nonexistent")
        assert result is None

    async def test_duplicate_event_not_saved(self, long_term_memory):
        """Test that duplicate events are not saved."""
        event = EventMemory(event_type="greeting", participants=["p1"])

        first_save = await long_term_memory.save_event(event)
        assert first_save is True

        second_save = await long_term_memory.save_event(event)
        assert second_save is False

    async def test_get_events_by_type(self, long_term_memory):
        """Test getting events by type."""
        e1 = EventMemory(event_type="greeting", participants=["p1"])
        e2 = EventMemory(event_type="greeting", participants=["p2"])
        e3 = EventMemory(event_type="petting", participants=["p1"])

        await long_term_memory.save_event(e1)
        await long_term_memory.save_event(e2)
        await long_term_memory.save_event(e3)

        greetings = await long_term_memory.get_events_by_type("greeting")
        assert len(greetings) == 2

    async def test_get_events_with_person(self, long_term_memory):
        """Test getting events involving a specific person."""
        e1 = EventMemory(event_type="greeting", participants=["person_a"])
        e2 = EventMemory(event_type="play", participants=["person_a", "person_b"])
        e3 = EventMemory(event_type="greeting", participants=["person_b"])

        await long_term_memory.save_event(e1)
        await long_term_memory.save_event(e2)
        await long_term_memory.save_event(e3)

        person_a_events = await long_term_memory.get_events_with_person("person_a")
        assert len(person_a_events) == 2

    async def test_should_persist_event(self, long_term_memory):
        """Test persistence criteria for events."""
        # Milestone should persist
        milestone = EventMemory(event_type="first_meeting", participants=["p1"])
        assert long_term_memory.should_persist_event(milestone, False) is True

        # Event with familiar person should persist
        familiar_event = EventMemory(event_type="petting", participants=["p1"])
        assert long_term_memory.should_persist_event(familiar_event, True) is True

        # Strong emotional event should persist
        strong = EventMemory(
            event_type="bump", outcome="negative", strength=0.9, participants=[]
        )
        assert long_term_memory.should_persist_event(strong, False) is True

        # Weak event with stranger should not
        weak = EventMemory(
            event_type="observed", outcome="neutral", strength=0.3, participants=[]
        )
        assert long_term_memory.should_persist_event(weak, False) is False


class TestLongTermMemoryBulkOperations:
    """Tests for bulk operations."""

    async def test_load_familiar_to_dict(self, long_term_memory):
        """Test loading familiar people as a dictionary."""
        p1 = PersonMemory(person_id="fam1", familiarity_score=75.0)
        p2 = PersonMemory(person_id="fam2", familiarity_score=60.0)
        p3 = PersonMemory(person_id="stranger", familiarity_score=30.0)

        await long_term_memory.save_person(p1)
        await long_term_memory.save_person(p2)
        await long_term_memory.save_person(p3)

        familiar_dict = await long_term_memory.load_all_familiar_to_dict()
        assert len(familiar_dict) == 2
        assert "fam1" in familiar_dict
        assert "fam2" in familiar_dict
        assert "stranger" not in familiar_dict

    async def test_sync_from_short_term(self, long_term_memory):
        """Test syncing from short-term memory."""
        people = {
            "familiar_1": PersonMemory(person_id="familiar_1", familiarity_score=75.0),
            "stranger_1": PersonMemory(person_id="stranger_1", familiarity_score=20.0),
        }
        objects = {
            "interesting_1": ObjectMemory(
                object_id="interesting_1", object_type="toy", interesting=True
            ),
            "boring_1": ObjectMemory(
                object_id="boring_1", object_type="box", times_seen=2
            ),
        }

        people_synced, objects_synced = await long_term_memory.sync_from_short_term(
            people, objects
        )

        assert people_synced == 1  # Only familiar
        assert objects_synced == 1  # Only interesting

    async def test_get_stats(self, long_term_memory):
        """Test getting memory statistics."""
        # Add some data
        p1 = PersonMemory(person_id="p1", familiarity_score=75.0)
        p2 = PersonMemory(person_id="p2", familiarity_score=30.0)
        o1 = ObjectMemory(object_id="o1", object_type="ball")
        e1 = EventMemory(event_type="greeting", participants=["p1"])

        await long_term_memory.save_person(p1)
        await long_term_memory.save_person(p2)
        await long_term_memory.save_object(o1)
        await long_term_memory.save_event(e1)

        stats = await long_term_memory.get_stats()
        assert stats["people_total"] == 2
        assert stats["people_familiar"] == 1
        assert stats["objects"] == 1
        assert stats["events"] == 1


class TestMemorySystemIntegration:
    """Tests for MemorySystem integration with LongTermMemory."""

    async def test_memory_system_without_long_term(self):
        """Test MemorySystem works without long-term memory."""
        memory = MemorySystem()
        assert memory.long_term is None

        # Should not error
        count = await memory.initialize_from_database()
        assert count == 0

        synced = await memory.sync_to_long_term()
        assert synced == (0, 0)

    async def test_memory_system_with_long_term(self, database):
        """Test MemorySystem with long-term memory."""
        ltm = LongTermMemory(database)
        memory = MemorySystem(long_term=ltm)

        # Initialize from database
        count = await memory.initialize_from_database()
        assert count == 0  # Empty database

    async def test_initialize_loads_familiar_people(self, database):
        """Test that initialization loads familiar people."""
        # Pre-populate database
        ltm = LongTermMemory(database)
        await ltm.initialize()
        person = PersonMemory(person_id="preexisting", familiarity_score=75.0)
        await ltm.save_person(person)

        # Create new memory system with same database
        memory = MemorySystem(long_term=ltm)
        count = await memory.initialize_from_database()

        assert count == 1
        assert "preexisting" in memory.short_term._people

    async def test_sync_persists_familiar_people(self, database):
        """Test that sync persists familiar people."""
        ltm = LongTermMemory(database)
        memory = MemorySystem(long_term=ltm)
        await memory.initialize_from_database()

        # Add a familiar person to short-term memory
        memory.short_term._people["new_person"] = PersonMemory(
            person_id="new_person", familiarity_score=75.0
        )

        # Force sync
        await memory.sync_to_long_term(force=True)

        # Verify persisted
        loaded = await ltm.get_person("new_person")
        assert loaded is not None
        assert loaded.familiarity_score == 75.0

    async def test_shutdown_syncs_all(self, database):
        """Test that shutdown syncs all qualifying memories."""
        ltm = LongTermMemory(database)
        memory = MemorySystem(long_term=ltm)
        await memory.initialize_from_database()

        # Add data
        memory.short_term._people["shutdown_person"] = PersonMemory(
            person_id="shutdown_person", familiarity_score=80.0
        )

        await memory.shutdown()

        # Verify persisted
        loaded = await ltm.get_person("shutdown_person")
        assert loaded is not None

    async def test_lookup_person_by_face(self, database):
        """Test face lookup through MemorySystem."""
        ltm = LongTermMemory(database)
        await ltm.initialize()

        # Create person with face embedding
        person = PersonMemory(person_id="face_person", familiarity_score=60.0)
        await ltm.save_person(person)

        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        await ltm.save_face_embedding("face_person", embedding)

        # Now use MemorySystem to look up
        memory = MemorySystem(long_term=ltm)

        found_id, similarity = await memory.lookup_person_by_face(embedding)
        assert found_id == "face_person"
        assert similarity > 0.99  # Same embedding should be nearly identical


class TestPersonMemoryTrustScore:
    """Tests for trust_score field in PersonMemory."""

    def test_default_trust_score(self):
        """Test that trust_score defaults to 50."""
        person = PersonMemory(person_id="test")
        assert person.trust_score == 50.0

    def test_trust_score_in_state(self):
        """Test that trust_score is included in serialized state."""
        person = PersonMemory(person_id="test", trust_score=80.0)
        state = person.get_state()
        assert "trust_score" in state
        assert state["trust_score"] == 80.0

    def test_trust_score_from_state(self):
        """Test that trust_score is restored from state."""
        state = {
            "person_id": "test",
            "trust_score": 90.0,
        }
        person = PersonMemory.from_state(state)
        assert person.trust_score == 90.0

    def test_trust_score_from_state_defaults(self):
        """Test that missing trust_score defaults to 50."""
        state = {"person_id": "test"}
        person = PersonMemory.from_state(state)
        assert person.trust_score == 50.0
