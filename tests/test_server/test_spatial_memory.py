"""
Unit tests for Murph's Spatial Memory System.
"""

import tempfile
import time
from pathlib import Path

import pytest

from server.cognition.memory import (
    LongTermMemory,
    MemorySystem,
    SpatialLandmark,
    SpatialMapMemory,
    SpatialObservation,
    SpatialZone,
)
from server.storage import Database


# ==================== Fixtures ====================


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


@pytest.fixture
def spatial_map():
    """Create a SpatialMapMemory instance."""
    return SpatialMapMemory()


@pytest.fixture
async def memory_system(database):
    """Create a MemorySystem with long-term memory."""
    ltm = LongTermMemory(database)
    ms = MemorySystem(long_term=ltm)
    await ms.initialize_from_database()
    return ms


# ==================== SpatialLandmark Tests ====================


class TestSpatialLandmark:
    """Tests for SpatialLandmark dataclass."""

    def test_create_landmark(self):
        """Test creating a landmark with required fields."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="charging_station",
        )
        assert landmark.landmark_id == "lm_001"
        assert landmark.landmark_type == "charging_station"
        assert landmark.name is None
        assert landmark.times_visited == 1
        assert landmark.confidence == 0.5

    def test_landmark_with_all_fields(self):
        """Test creating a landmark with all fields."""
        landmark = SpatialLandmark(
            landmark_id="lm_002",
            landmark_type="corner",
            name="Kitchen Corner",
            times_visited=10,
            confidence=0.9,
            connections={"lm_001": 50.0, "lm_003": 30.0},
        )
        assert landmark.name == "Kitchen Corner"
        assert landmark.times_visited == 10
        assert landmark.confidence == 0.9
        assert len(landmark.connections) == 2

    def test_record_visit(self):
        """Test recording a visit to a landmark."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="edge",
        )
        initial_visits = landmark.times_visited
        initial_time = landmark.last_seen

        time.sleep(0.01)  # Small delay to ensure time changes
        landmark.record_visit()

        assert landmark.times_visited == initial_visits + 1
        assert landmark.last_seen > initial_time

    def test_add_connection(self):
        """Test adding a connection to another landmark."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="corner",
        )
        landmark.add_connection("lm_002", 45.5)

        assert "lm_002" in landmark.connections
        assert landmark.connections["lm_002"] == 45.5

    def test_remove_connection(self):
        """Test removing a connection."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="corner",
            connections={"lm_002": 50.0},
        )
        landmark.remove_connection("lm_002")
        assert "lm_002" not in landmark.connections

        # Removing non-existent connection shouldn't raise
        landmark.remove_connection("nonexistent")

    def test_adjust_confidence(self):
        """Test adjusting confidence with clamping."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="corner",
            confidence=0.5,
        )

        landmark.adjust_confidence(0.3)
        assert landmark.confidence == 0.8

        landmark.adjust_confidence(0.5)
        assert landmark.confidence == 1.0  # Clamped

        landmark.adjust_confidence(-1.5)
        assert landmark.confidence == 0.0  # Clamped

    def test_get_state_and_from_state(self):
        """Test serialization roundtrip."""
        original = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="home_base",
            name="My Spot",
            times_visited=5,
            confidence=0.8,
            connections={"lm_002": 30.0},
        )
        state = original.get_state()
        restored = SpatialLandmark.from_state(state)

        assert restored.landmark_id == original.landmark_id
        assert restored.landmark_type == original.landmark_type
        assert restored.name == original.name
        assert restored.times_visited == original.times_visited
        assert restored.confidence == original.confidence
        assert restored.connections == original.connections


# ==================== SpatialZone Tests ====================


class TestSpatialZone:
    """Tests for SpatialZone dataclass."""

    def test_create_zone(self):
        """Test creating a zone with required fields."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )
        assert zone.zone_id == "z_001"
        assert zone.zone_type == "safe"
        assert zone.primary_landmark_id == "lm_001"
        assert zone.safety_score == 0.5
        assert zone.familiarity == 0.0

    def test_record_visit(self):
        """Test recording a visit increases familiarity."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
            familiarity=0.0,
        )

        zone.record_visit()
        assert zone.familiarity > 0.0
        initial_familiarity = zone.familiarity

        zone.record_visit()
        assert zone.familiarity > initial_familiarity

    def test_record_event(self):
        """Test recording events in a zone."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )

        zone.record_event("petting")
        assert "petting" in zone.associated_events

        # Duplicate event shouldn't be added twice
        zone.record_event("petting")
        assert zone.associated_events.count("petting") == 1

    def test_adjust_safety(self):
        """Test adjusting safety score with clamping."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="unknown",
            primary_landmark_id="lm_001",
            safety_score=0.5,
        )

        zone.adjust_safety(0.3)
        assert zone.safety_score == 0.8

        zone.adjust_safety(0.5)
        assert zone.safety_score == 1.0  # Clamped

        zone.adjust_safety(-1.5)
        assert zone.safety_score == 0.0  # Clamped

    def test_is_safe_property(self):
        """Test is_safe property threshold."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
            safety_score=0.6,
        )
        assert not zone.is_safe

        zone.safety_score = 0.7
        assert zone.is_safe

    def test_is_dangerous_property(self):
        """Test is_dangerous property threshold."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="edge_zone",
            primary_landmark_id="lm_001",
            safety_score=0.4,
        )
        assert not zone.is_dangerous

        zone.safety_score = 0.2
        assert zone.is_dangerous

    def test_get_state_and_from_state(self):
        """Test serialization roundtrip."""
        original = SpatialZone(
            zone_id="z_001",
            zone_type="play_area",
            primary_landmark_id="lm_001",
            name="Play Zone",
            safety_score=0.9,
            familiarity=0.7,
            associated_events=["play", "petting"],
        )
        state = original.get_state()
        restored = SpatialZone.from_state(state)

        assert restored.zone_id == original.zone_id
        assert restored.zone_type == original.zone_type
        assert restored.primary_landmark_id == original.primary_landmark_id
        assert restored.name == original.name
        assert restored.safety_score == original.safety_score
        assert restored.familiarity == original.familiarity
        assert restored.associated_events == original.associated_events


# ==================== SpatialObservation Tests ====================


class TestSpatialObservation:
    """Tests for SpatialObservation dataclass."""

    def test_create_observation(self):
        """Test creating an observation with required fields."""
        obs = SpatialObservation(
            entity_type="object",
            entity_id="ball_001",
            landmark_id="lm_001",
            relative_direction=45.0,
            relative_distance=30.0,
        )
        assert obs.entity_type == "object"
        assert obs.entity_id == "ball_001"
        assert obs.landmark_id == "lm_001"
        assert obs.relative_direction == 45.0
        assert obs.relative_distance == 30.0
        assert obs.confidence == 0.5
        assert obs.observation_id is not None

    def test_get_state_and_from_state(self):
        """Test serialization roundtrip."""
        original = SpatialObservation(
            entity_type="person",
            entity_id="person_001",
            landmark_id="lm_002",
            relative_direction=90.0,
            relative_distance=100.0,
            confidence=0.8,
        )
        state = original.get_state()
        restored = SpatialObservation.from_state(state)

        assert restored.entity_type == original.entity_type
        assert restored.entity_id == original.entity_id
        assert restored.landmark_id == original.landmark_id
        assert restored.relative_direction == original.relative_direction
        assert restored.relative_distance == original.relative_distance
        assert restored.confidence == original.confidence


# ==================== SpatialMapMemory Tests ====================


class TestSpatialMapMemory:
    """Tests for SpatialMapMemory container."""

    def test_create_empty_map(self, spatial_map):
        """Test creating an empty spatial map."""
        assert len(spatial_map.landmarks) == 0
        assert len(spatial_map.zones) == 0
        assert len(spatial_map.observations) == 0
        assert spatial_map.current_landmark_id is None
        assert spatial_map.current_zone_id is None
        assert spatial_map.home_landmark_id is None

    def test_add_and_get_landmark(self, spatial_map):
        """Test adding and retrieving a landmark."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="charging_station",
        )
        spatial_map.add_landmark(landmark)

        retrieved = spatial_map.get_landmark("lm_001")
        assert retrieved is not None
        assert retrieved.landmark_id == "lm_001"

    def test_remove_landmark(self, spatial_map):
        """Test removing a landmark and its connections."""
        lm1 = SpatialLandmark(landmark_id="lm_001", landmark_type="corner")
        lm2 = SpatialLandmark(
            landmark_id="lm_002",
            landmark_type="corner",
            connections={"lm_001": 50.0},
        )
        spatial_map.add_landmark(lm1)
        spatial_map.add_landmark(lm2)

        spatial_map.remove_landmark("lm_001")

        assert spatial_map.get_landmark("lm_001") is None
        assert "lm_001" not in spatial_map.landmarks["lm_002"].connections

    def test_add_and_get_zone(self, spatial_map):
        """Test adding and retrieving a zone."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )
        spatial_map.add_zone(zone)

        retrieved = spatial_map.get_zone("z_001")
        assert retrieved is not None
        assert retrieved.zone_id == "z_001"

    def test_get_zone_for_landmark(self, spatial_map):
        """Test finding zone by primary landmark."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )
        spatial_map.add_zone(zone)

        found = spatial_map.get_zone_for_landmark("lm_001")
        assert found is not None
        assert found.zone_id == "z_001"

        not_found = spatial_map.get_zone_for_landmark("nonexistent")
        assert not_found is None

    def test_add_observation_fifo(self, spatial_map):
        """Test that observations follow FIFO when exceeding max."""
        from server.cognition.memory.spatial_types import MAX_OBSERVATIONS

        # Add more than max observations
        for i in range(MAX_OBSERVATIONS + 10):
            obs = SpatialObservation(
                entity_type="object",
                entity_id=f"obj_{i}",
                landmark_id="lm_001",
                relative_direction=0.0,
                relative_distance=10.0,
            )
            spatial_map.add_observation(obs)

        assert len(spatial_map.observations) == MAX_OBSERVATIONS

        # First observation should have been removed
        entity_ids = [obs.entity_id for obs in spatial_map.observations]
        assert "obj_0" not in entity_ids
        assert f"obj_{MAX_OBSERVATIONS + 9}" in entity_ids

    def test_get_observations_for_entity(self, spatial_map):
        """Test filtering observations by entity."""
        for i in range(5):
            obs = SpatialObservation(
                entity_type="object",
                entity_id="ball",
                landmark_id="lm_001",
                relative_direction=i * 10.0,
                relative_distance=10.0,
            )
            spatial_map.add_observation(obs)

        obs_other = SpatialObservation(
            entity_type="object",
            entity_id="cup",
            landmark_id="lm_001",
            relative_direction=0.0,
            relative_distance=20.0,
        )
        spatial_map.add_observation(obs_other)

        ball_obs = spatial_map.get_observations_for_entity("ball")
        assert len(ball_obs) == 5

    def test_update_current_location(self, spatial_map):
        """Test updating current location."""
        lm = SpatialLandmark(landmark_id="lm_001", landmark_type="corner")
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )
        spatial_map.add_landmark(lm)
        spatial_map.add_zone(zone)

        spatial_map.update_current_location("lm_001")

        assert spatial_map.current_landmark_id == "lm_001"
        assert spatial_map.current_zone_id == "z_001"

    def test_set_home(self, spatial_map):
        """Test setting home landmark."""
        lm = SpatialLandmark(landmark_id="lm_001", landmark_type="home_base")
        spatial_map.add_landmark(lm)
        spatial_map.set_home("lm_001")

        assert spatial_map.home_landmark_id == "lm_001"

    def test_is_at_home_property(self, spatial_map):
        """Test is_at_home property."""
        lm = SpatialLandmark(landmark_id="lm_001", landmark_type="home_base")
        spatial_map.add_landmark(lm)
        spatial_map.set_home("lm_001")

        assert not spatial_map.is_at_home  # Not at home yet

        spatial_map.update_current_location("lm_001")
        assert spatial_map.is_at_home

    def test_is_position_known_property(self, spatial_map):
        """Test is_position_known property."""
        assert not spatial_map.is_position_known

        lm = SpatialLandmark(landmark_id="lm_001", landmark_type="corner")
        spatial_map.add_landmark(lm)
        spatial_map.update_current_location("lm_001")

        assert spatial_map.is_position_known

    def test_get_state_and_from_state(self, spatial_map):
        """Test full serialization roundtrip."""
        lm = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="home_base",
            name="Home",
        )
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )
        obs = SpatialObservation(
            entity_type="object",
            entity_id="ball",
            landmark_id="lm_001",
            relative_direction=45.0,
            relative_distance=30.0,
        )
        spatial_map.add_landmark(lm)
        spatial_map.add_zone(zone)
        spatial_map.add_observation(obs)
        spatial_map.set_home("lm_001")
        spatial_map.update_current_location("lm_001")

        state = spatial_map.get_state()
        restored = SpatialMapMemory.from_state(state)

        assert len(restored.landmarks) == 1
        assert len(restored.zones) == 1
        assert len(restored.observations) == 1
        assert restored.home_landmark_id == "lm_001"
        assert restored.current_landmark_id == "lm_001"


# ==================== LongTermMemory Spatial Tests ====================


class TestLongTermMemorySpatial:
    """Tests for spatial persistence in LongTermMemory."""

    async def test_save_and_load_landmark(self, long_term_memory):
        """Test saving and loading a landmark."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="charging_station",
            name="Main Charger",
            times_visited=10,
            confidence=0.9,
            connections={"lm_002": 50.0},
        )

        await long_term_memory.save_landmark(landmark)
        loaded = await long_term_memory.get_landmark("lm_001")

        assert loaded is not None
        assert loaded.landmark_id == "lm_001"
        assert loaded.landmark_type == "charging_station"
        assert loaded.name == "Main Charger"
        assert loaded.times_visited == 10
        assert loaded.confidence == 0.9
        assert loaded.connections == {"lm_002": 50.0}

    async def test_landmark_not_found(self, long_term_memory):
        """Test getting a non-existent landmark returns None."""
        result = await long_term_memory.get_landmark("nonexistent")
        assert result is None

    async def test_update_existing_landmark(self, long_term_memory):
        """Test updating an existing landmark."""
        landmark = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="corner",
            times_visited=1,
        )
        await long_term_memory.save_landmark(landmark)

        landmark.times_visited = 10
        landmark.confidence = 0.9
        await long_term_memory.save_landmark(landmark)

        loaded = await long_term_memory.get_landmark("lm_001")
        assert loaded.times_visited == 10
        assert loaded.confidence == 0.9

    async def test_get_all_landmarks(self, long_term_memory):
        """Test getting all landmarks."""
        for i in range(3):
            lm = SpatialLandmark(
                landmark_id=f"lm_{i}",
                landmark_type="corner",
            )
            await long_term_memory.save_landmark(lm)

        all_landmarks = await long_term_memory.get_all_landmarks()
        assert len(all_landmarks) == 3

    async def test_get_landmarks_by_type(self, long_term_memory):
        """Test filtering landmarks by type."""
        lm1 = SpatialLandmark(landmark_id="lm_001", landmark_type="corner")
        lm2 = SpatialLandmark(landmark_id="lm_002", landmark_type="charging_station")
        lm3 = SpatialLandmark(landmark_id="lm_003", landmark_type="corner")

        await long_term_memory.save_landmark(lm1)
        await long_term_memory.save_landmark(lm2)
        await long_term_memory.save_landmark(lm3)

        corners = await long_term_memory.get_landmarks_by_type("corner")
        assert len(corners) == 2

    async def test_should_persist_landmark(self, long_term_memory):
        """Test landmark persistence criteria."""
        # Low confidence, few visits, non-critical type
        lm1 = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="visual_marker",
            confidence=0.3,
            times_visited=2,
        )
        assert not long_term_memory.should_persist_landmark(lm1)

        # High confidence
        lm2 = SpatialLandmark(
            landmark_id="lm_002",
            landmark_type="visual_marker",
            confidence=0.7,
        )
        assert long_term_memory.should_persist_landmark(lm2)

        # Many visits
        lm3 = SpatialLandmark(
            landmark_id="lm_003",
            landmark_type="visual_marker",
            confidence=0.3,
            times_visited=6,
        )
        assert long_term_memory.should_persist_landmark(lm3)

        # Critical type
        lm4 = SpatialLandmark(
            landmark_id="lm_004",
            landmark_type="charging_station",
            confidence=0.1,
            times_visited=1,
        )
        assert long_term_memory.should_persist_landmark(lm4)

    async def test_save_and_load_zone(self, long_term_memory):
        """Test saving and loading a zone."""
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="play_area",
            primary_landmark_id="lm_001",
            name="Play Zone",
            safety_score=0.9,
            familiarity=0.8,
            associated_events=["play", "petting"],
        )

        await long_term_memory.save_zone(zone)
        loaded = await long_term_memory.get_zone("z_001")

        assert loaded is not None
        assert loaded.zone_id == "z_001"
        assert loaded.zone_type == "play_area"
        assert loaded.name == "Play Zone"
        assert loaded.safety_score == 0.9
        assert loaded.familiarity == 0.8
        assert loaded.associated_events == ["play", "petting"]

    async def test_should_persist_zone(self, long_term_memory):
        """Test zone persistence criteria."""
        # Low familiarity, non-critical type
        z1 = SpatialZone(
            zone_id="z_001",
            zone_type="unexplored",
            primary_landmark_id="lm_001",
            familiarity=0.3,
        )
        assert not long_term_memory.should_persist_zone(z1)

        # High familiarity
        z2 = SpatialZone(
            zone_id="z_002",
            zone_type="unexplored",
            primary_landmark_id="lm_001",
            familiarity=0.6,
        )
        assert long_term_memory.should_persist_zone(z2)

        # Critical type
        z3 = SpatialZone(
            zone_id="z_003",
            zone_type="edge_zone",
            primary_landmark_id="lm_001",
            familiarity=0.1,
        )
        assert long_term_memory.should_persist_zone(z3)

    async def test_save_observation(self, long_term_memory):
        """Test saving an observation."""
        obs = SpatialObservation(
            entity_type="person",
            entity_id="person_001",
            landmark_id="lm_001",
            relative_direction=45.0,
            relative_distance=100.0,
            confidence=0.8,
        )

        await long_term_memory.save_observation(obs)

        # Retrieve it
        observations = await long_term_memory.get_observations_for_entity("person_001")
        assert len(observations) == 1
        assert observations[0].entity_id == "person_001"

    async def test_get_observations_near_landmark(self, long_term_memory):
        """Test getting observations near a landmark."""
        for i in range(3):
            obs = SpatialObservation(
                entity_type="object",
                entity_id=f"obj_{i}",
                landmark_id="lm_001",
                relative_direction=i * 30.0,
                relative_distance=50.0,
            )
            await long_term_memory.save_observation(obs)

        obs_other = SpatialObservation(
            entity_type="object",
            entity_id="obj_other",
            landmark_id="lm_002",
            relative_direction=0.0,
            relative_distance=50.0,
        )
        await long_term_memory.save_observation(obs_other)

        near_lm1 = await long_term_memory.get_observations_near_landmark("lm_001")
        assert len(near_lm1) == 3

    async def test_load_spatial_map(self, long_term_memory):
        """Test loading full spatial map."""
        # Save some data
        lm1 = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="home_base",
            name="Home",
        )
        lm2 = SpatialLandmark(
            landmark_id="lm_002",
            landmark_type="corner",
        )
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )

        await long_term_memory.save_landmark(lm1)
        await long_term_memory.save_landmark(lm2)
        await long_term_memory.save_zone(zone)

        # Load spatial map
        spatial_map = await long_term_memory.load_spatial_map()

        assert len(spatial_map.landmarks) == 2
        assert len(spatial_map.zones) == 1
        assert spatial_map.home_landmark_id == "lm_001"

    async def test_sync_spatial_map(self, long_term_memory):
        """Test syncing spatial map to long-term memory."""
        spatial_map = SpatialMapMemory()

        # Add landmarks with different persistence criteria
        lm_persist = SpatialLandmark(
            landmark_id="lm_001",
            landmark_type="charging_station",  # Critical type
        )
        lm_no_persist = SpatialLandmark(
            landmark_id="lm_002",
            landmark_type="visual_marker",
            confidence=0.3,
            times_visited=1,
        )

        spatial_map.add_landmark(lm_persist)
        spatial_map.add_landmark(lm_no_persist)

        counts = await long_term_memory.sync_spatial_map(spatial_map)

        assert counts["landmarks"] == 1  # Only critical type persisted
        assert await long_term_memory.get_landmark("lm_001") is not None
        assert await long_term_memory.get_landmark("lm_002") is None

    async def test_stats_include_spatial(self, long_term_memory):
        """Test that stats include spatial counts."""
        lm = SpatialLandmark(landmark_id="lm_001", landmark_type="corner")
        zone = SpatialZone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )

        await long_term_memory.save_landmark(lm)
        await long_term_memory.save_zone(zone)

        stats = await long_term_memory.get_stats()
        assert stats["landmarks"] == 1
        assert stats["zones"] == 1


# ==================== MemorySystem Integration Tests ====================


class TestMemorySystemSpatialIntegration:
    """Tests for spatial memory integration in MemorySystem."""

    async def test_record_landmark(self, memory_system):
        """Test recording a landmark through MemorySystem."""
        landmark = memory_system.record_landmark(
            landmark_id="lm_001",
            landmark_type="corner",
            name="Test Corner",
            confidence=0.7,
        )

        assert landmark.landmark_id == "lm_001"
        assert landmark.name == "Test Corner"
        assert memory_system.spatial_map.get_landmark("lm_001") is not None

    async def test_record_existing_landmark(self, memory_system):
        """Test that recording existing landmark updates it."""
        memory_system.record_landmark(
            landmark_id="lm_001",
            landmark_type="corner",
            confidence=0.5,
        )
        memory_system.record_landmark(
            landmark_id="lm_001",
            landmark_type="corner",
            name="Named Corner",
            confidence=0.8,
        )

        landmark = memory_system.spatial_map.get_landmark("lm_001")
        assert landmark.times_visited == 2
        assert landmark.confidence == 0.8
        assert landmark.name == "Named Corner"

    async def test_record_zone(self, memory_system):
        """Test recording a zone through MemorySystem."""
        zone = memory_system.record_zone(
            zone_id="z_001",
            zone_type="play_area",
            primary_landmark_id="lm_001",
            name="Play Area",
            safety_score=0.9,
        )

        assert zone.zone_id == "z_001"
        assert zone.name == "Play Area"
        assert memory_system.spatial_map.get_zone("z_001") is not None

    async def test_update_current_location(self, memory_system):
        """Test updating current location."""
        memory_system.record_landmark(
            landmark_id="lm_001",
            landmark_type="corner",
        )
        memory_system.record_zone(
            zone_id="z_001",
            zone_type="safe",
            primary_landmark_id="lm_001",
        )

        memory_system.update_current_location("lm_001")

        assert memory_system.spatial_map.current_landmark_id == "lm_001"
        assert memory_system.spatial_map.current_zone_id == "z_001"

    async def test_record_spatial_observation(self, memory_system):
        """Test recording a spatial observation."""
        obs = memory_system.record_spatial_observation(
            entity_type="object",
            entity_id="ball_001",
            landmark_id="lm_001",
            relative_direction=45.0,
            relative_distance=30.0,
            confidence=0.8,
        )

        assert obs.entity_id == "ball_001"
        assert len(memory_system.spatial_map.observations) == 1

    async def test_record_zone_event(self, memory_system):
        """Test recording events in current zone."""
        memory_system.record_landmark("lm_001", "corner")
        memory_system.record_zone("z_001", "safe", "lm_001", safety_score=0.5)
        memory_system.update_current_location("lm_001")

        initial_safety = memory_system.spatial_map.current_zone.safety_score

        memory_system.record_zone_event("bump")

        assert "bump" in memory_system.spatial_map.current_zone.associated_events
        assert memory_system.spatial_map.current_zone.safety_score < initial_safety

    async def test_set_home_landmark(self, memory_system):
        """Test setting home landmark."""
        memory_system.record_landmark("lm_001", "home_base")
        memory_system.set_home_landmark("lm_001")

        assert memory_system.spatial_map.home_landmark_id == "lm_001"

    async def test_get_spatial_context(self, memory_system):
        """Test getting spatial context."""
        memory_system.record_landmark("lm_001", "charging_station")
        memory_system.record_zone("z_001", "safe", "lm_001", safety_score=0.9)
        memory_system.update_current_location("lm_001")

        context = memory_system.get_spatial_context()

        assert context["current_landmark_id"] == "lm_001"
        assert context["current_landmark_type"] == "charging_station"
        assert context["current_zone_type"] == "safe"
        assert context["current_zone_safety"] == 0.9

    async def test_get_memory_triggers_spatial(self, memory_system):
        """Test spatial triggers in memory triggers."""
        memory_system.record_landmark("lm_001", "home_base")
        memory_system.record_zone("z_001", "safe", "lm_001", safety_score=0.9)
        memory_system.set_home_landmark("lm_001")
        memory_system.update_current_location("lm_001")

        triggers = memory_system.get_memory_triggers()

        assert triggers["at_home"] is True
        assert triggers["in_safe_zone"] is True
        assert triggers["position_known"] is True

    async def test_spatial_in_summary(self, memory_system):
        """Test spatial info in summary."""
        memory_system.record_landmark("lm_001", "corner")

        summary = memory_system.summary()

        assert "spatial" in summary
        assert summary["spatial"]["landmarks"] == 1

    async def test_clear_resets_spatial(self, memory_system):
        """Test that clear() resets spatial map."""
        memory_system.record_landmark("lm_001", "corner")
        memory_system.clear()

        assert len(memory_system.spatial_map.landmarks) == 0

    async def test_get_state_includes_spatial(self, memory_system):
        """Test that get_state includes spatial map."""
        memory_system.record_landmark("lm_001", "corner")

        state = memory_system.get_state()

        assert "spatial_map" in state
        assert "landmarks" in state["spatial_map"]

    async def test_from_state_restores_spatial(self, memory_system):
        """Test that from_state restores spatial map."""
        memory_system.record_landmark("lm_001", "corner")
        state = memory_system.get_state()

        restored = MemorySystem.from_state(state)

        assert len(restored.spatial_map.landmarks) == 1
        assert restored.spatial_map.get_landmark("lm_001") is not None

    async def test_initialize_loads_spatial(self, database):
        """Test that initialize_from_database loads spatial map."""
        # First memory system saves data
        ltm1 = LongTermMemory(database)
        ms1 = MemorySystem(long_term=ltm1)
        await ms1.initialize_from_database()

        ms1.record_landmark("lm_001", "charging_station")  # Critical type
        await ms1.sync_to_long_term(force=True)

        # Second memory system loads it
        ltm2 = LongTermMemory(database)
        ms2 = MemorySystem(long_term=ltm2)
        counts = await ms2.initialize_from_database()

        assert counts["landmarks"] == 1
        assert ms2.spatial_map.get_landmark("lm_001") is not None

    async def test_sync_includes_spatial(self, memory_system):
        """Test that sync_to_long_term includes spatial data."""
        memory_system.record_landmark("lm_001", "charging_station")

        counts = await memory_system.sync_to_long_term(force=True)

        assert counts["spatial"] >= 1
