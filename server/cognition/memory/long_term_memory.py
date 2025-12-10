"""
Murph - Long-Term Memory
Persistent memory storage backed by SQLite.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import func, select

from server.storage import (
    Database,
    EventModel,
    FaceEmbeddingModel,
    InsightModel,
    ObjectModel,
    PersonModel,
    SpatialLandmarkModel,
    SpatialObservationModel,
    SpatialZoneModel,
)

from .memory_types import EventMemory, InsightMemory, ObjectMemory, PersonMemory
from .spatial_types import (
    SpatialLandmark,
    SpatialMapMemory,
    SpatialObservation,
    SpatialZone,
)

logger = logging.getLogger("murph.memory.long_term")


# Thresholds for promoting to long-term memory
FAMILIARITY_THRESHOLD = 50.0  # Matches PersonMemory.is_familiar
OBJECT_SIGHTING_THRESHOLD = 10  # Times seen before considered noteworthy
EVENT_SIGNIFICANCE_THRESHOLD = 0.7  # Minimum significance to store

# Spatial memory thresholds
LANDMARK_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to persist
LANDMARK_VISIT_THRESHOLD = 5  # Visits before noteworthy
ZONE_FAMILIARITY_THRESHOLD = 0.5  # Minimum familiarity to persist
OBSERVATION_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for observations

# Critical types that are always persisted
CRITICAL_LANDMARK_TYPES = {"charging_station", "home_base", "edge"}
CRITICAL_ZONE_TYPES = {"charging_zone", "edge_zone"}


class LongTermMemory:
    """
    Long-term memory with SQLite persistence.

    Stores:
    - Familiar people (familiarity >= 50 or explicitly named)
    - Interesting objects (investigated or frequently seen)
    - Significant events (milestones, strong emotional valence)

    Usage:
        db = Database()
        ltm = LongTermMemory(db)
        await ltm.initialize()

        # Load person from database
        person = await ltm.get_person("person_123")

        # Save person to database
        await ltm.save_person(person_memory)

        # Check if person should be persisted
        if ltm.should_persist_person(person_memory):
            await ltm.save_person(person_memory)
    """

    def __init__(self, database: Database) -> None:
        """
        Initialize long-term memory.

        Args:
            database: Database instance (not yet initialized is OK)
        """
        self._db = database
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the long-term memory (ensure DB is ready)."""
        if not self._db.is_initialized:
            await self._db.initialize()
        self._initialized = True
        logger.info("Long-term memory initialized")

    # ==================== Person Operations ====================

    async def get_person(self, person_id: str) -> PersonMemory | None:
        """
        Load a person from long-term memory.

        Args:
            person_id: Unique identifier for the person

        Returns:
            PersonMemory if found, None otherwise
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(PersonModel).where(PersonModel.person_id == person_id)
            )
            model = result.scalar_one_or_none()

            if model is None:
                return None

            return PersonMemory.from_state(model.to_dict())

    async def save_person(
        self, person: PersonMemory, trust_score: float | None = None
    ) -> None:
        """
        Save or update a person in long-term memory.

        Args:
            person: PersonMemory to save
            trust_score: Optional trust score (0-100), defaults to existing or 50.0
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(PersonModel).where(PersonModel.person_id == person.person_id)
            )
            model = result.scalar_one_or_none()

            if model is None:
                # Create new record
                model = PersonModel(
                    person_id=person.person_id,
                    name=person.name,
                    familiarity_score=person.familiarity_score,
                    trust_score=(
                        trust_score
                        if trust_score is not None
                        else getattr(person, "trust_score", 50.0)
                    ),
                    sentiment=person.sentiment,
                    first_seen=datetime.fromtimestamp(person.first_seen),
                    last_seen=datetime.fromtimestamp(person.last_seen),
                    interaction_count=person.interaction_count,
                    tags=list(person.tags),
                )
                session.add(model)
                logger.debug(f"Created new person record: {person.person_id}")
            else:
                # Update existing record
                model.name = person.name
                model.familiarity_score = person.familiarity_score
                if trust_score is not None:
                    model.trust_score = trust_score
                model.sentiment = person.sentiment
                model.last_seen = datetime.fromtimestamp(person.last_seen)
                model.interaction_count = person.interaction_count
                model.tags = list(person.tags)
                logger.debug(f"Updated person record: {person.person_id}")

            await session.commit()

    async def get_all_familiar_people(self) -> list[PersonMemory]:
        """Get all familiar people from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(PersonModel).where(
                    PersonModel.familiarity_score >= FAMILIARITY_THRESHOLD
                )
            )
            models = result.scalars().all()
            return [PersonMemory.from_state(m.to_dict()) for m in models]

    async def get_all_people(self) -> list[PersonMemory]:
        """Get all people from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(select(PersonModel))
            models = result.scalars().all()
            return [PersonMemory.from_state(m.to_dict()) for m in models]

    async def get_person_by_name(self, name: str) -> PersonMemory | None:
        """Look up a person by name."""
        async with self._db.session() as session:
            result = await session.execute(
                select(PersonModel).where(PersonModel.name == name)
            )
            model = result.scalar_one_or_none()
            return PersonMemory.from_state(model.to_dict()) if model else None

    def should_persist_person(self, person: PersonMemory) -> bool:
        """
        Determine if a person should be persisted to long-term memory.

        Criteria:
        - Familiarity >= 50 (is_familiar)
        - OR has a name set
        """
        return person.is_familiar or person.name is not None

    # ==================== Face Embedding Operations ====================

    async def save_face_embedding(
        self, person_id: str, embedding: np.ndarray, quality_score: float = 1.0
    ) -> bool:
        """
        Save a face embedding for a person.

        Args:
            person_id: Person's unique identifier
            embedding: 128-dim FaceNet embedding
            quality_score: Quality of the source image (0-1)

        Returns:
            True if saved successfully, False if person not found
        """
        if embedding.shape != (128,):
            raise ValueError(f"Expected 128-dim embedding, got {embedding.shape}")

        async with self._db.session() as session:
            # Get person's database ID
            result = await session.execute(
                select(PersonModel).where(PersonModel.person_id == person_id)
            )
            person_model = result.scalar_one_or_none()

            if person_model is None:
                logger.warning(f"Cannot save embedding: person {person_id} not found")
                return False

            # Store embedding as bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()

            face_model = FaceEmbeddingModel(
                person_id=person_model.id,
                embedding=embedding_bytes,
                quality_score=quality_score,
            )
            session.add(face_model)
            await session.commit()
            logger.debug(f"Saved face embedding for {person_id}")
            return True

    async def get_face_embeddings(self, person_id: str) -> list[np.ndarray]:
        """Get all face embeddings for a person."""
        async with self._db.session() as session:
            result = await session.execute(
                select(PersonModel).where(PersonModel.person_id == person_id)
            )
            person_model = result.scalar_one_or_none()

            if person_model is None:
                return []

            result = await session.execute(
                select(FaceEmbeddingModel).where(
                    FaceEmbeddingModel.person_id == person_model.id
                )
            )
            embeddings = result.scalars().all()

            return [np.frombuffer(e.embedding, dtype=np.float32) for e in embeddings]

    async def find_person_by_embedding(
        self, embedding: np.ndarray, threshold: float = 0.6
    ) -> tuple[str | None, float]:
        """
        Find a person by face embedding similarity.

        Args:
            embedding: 128-dim FaceNet embedding to match
            threshold: Minimum cosine similarity (0.6 is reasonable)

        Returns:
            Tuple of (person_id, similarity) or (None, 0.0) if no match
        """
        async with self._db.session() as session:
            result = await session.execute(select(PersonModel))
            persons = result.scalars().all()

            best_match: str | None = None
            best_similarity = 0.0

            for person in persons:
                # Get embeddings for this person
                emb_result = await session.execute(
                    select(FaceEmbeddingModel).where(
                        FaceEmbeddingModel.person_id == person.id
                    )
                )
                person_embeddings = emb_result.scalars().all()

                for stored in person_embeddings:
                    stored_emb = np.frombuffer(stored.embedding, dtype=np.float32)
                    # Cosine similarity
                    norm_product = np.linalg.norm(embedding) * np.linalg.norm(
                        stored_emb
                    )
                    if norm_product > 0:
                        similarity = np.dot(embedding, stored_emb) / norm_product
                    else:
                        similarity = 0.0

                    if similarity > best_similarity:
                        best_similarity = float(similarity)
                        best_match = person.person_id

            if best_similarity >= threshold:
                return best_match, best_similarity
            return None, 0.0

    # ==================== Object Operations ====================

    async def get_object(self, object_id: str) -> ObjectMemory | None:
        """Load an object from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(ObjectModel).where(ObjectModel.object_id == object_id)
            )
            model = result.scalar_one_or_none()
            return ObjectMemory.from_state(model.to_dict()) if model else None

    async def save_object(self, obj: ObjectMemory) -> None:
        """Save or update an object in long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(ObjectModel).where(ObjectModel.object_id == obj.object_id)
            )
            model = result.scalar_one_or_none()

            pos_x = obj.last_position[0] if obj.last_position else None
            pos_y = obj.last_position[1] if obj.last_position else None

            if model is None:
                model = ObjectModel(
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    first_seen=datetime.fromtimestamp(obj.first_seen),
                    last_seen=datetime.fromtimestamp(obj.last_seen),
                    times_seen=obj.times_seen,
                    interesting=obj.interesting,
                    last_position_x=pos_x,
                    last_position_y=pos_y,
                )
                session.add(model)
                logger.debug(f"Created new object record: {obj.object_id}")
            else:
                model.object_type = obj.object_type
                model.last_seen = datetime.fromtimestamp(obj.last_seen)
                model.times_seen = obj.times_seen
                model.interesting = obj.interesting
                model.last_position_x = pos_x
                model.last_position_y = pos_y
                logger.debug(f"Updated object record: {obj.object_id}")

            await session.commit()

    async def get_all_objects(self) -> list[ObjectMemory]:
        """Get all objects from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(select(ObjectModel))
            models = result.scalars().all()
            return [ObjectMemory.from_state(m.to_dict()) for m in models]

    async def get_interesting_objects(self) -> list[ObjectMemory]:
        """Get all interesting objects from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(ObjectModel).where(ObjectModel.interesting == True)  # noqa: E712
            )
            models = result.scalars().all()
            return [ObjectMemory.from_state(m.to_dict()) for m in models]

    def should_persist_object(self, obj: ObjectMemory) -> bool:
        """
        Determine if an object should be persisted.

        Criteria:
        - Has been investigated (interesting == True)
        - OR seen many times (times_seen >= threshold)
        """
        return obj.interesting or obj.times_seen >= OBJECT_SIGHTING_THRESHOLD

    # ==================== Event Operations ====================

    async def get_event(self, event_id: str) -> EventMemory | None:
        """Load an event from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(EventModel).where(EventModel.event_id == event_id)
            )
            model = result.scalar_one_or_none()
            return EventMemory.from_state(model.to_dict()) if model else None

    async def save_event(
        self,
        event: EventMemory,
        significance: float = 1.0,
        description: str | None = None,
    ) -> bool:
        """
        Save an event to long-term memory.

        Args:
            event: EventMemory to save
            significance: How significant/memorable (0-1)
            description: Optional description

        Returns:
            True if saved (new event), False if already exists
        """
        async with self._db.session() as session:
            # Check if already exists
            result = await session.execute(
                select(EventModel).where(EventModel.event_id == event.event_id)
            )
            if result.scalar_one_or_none() is not None:
                return False  # Already saved

            model = EventModel(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=datetime.fromtimestamp(event.timestamp),
                participants=event.participants,
                objects=event.objects,
                outcome=event.outcome,
                significance=significance,
                description=description,
            )
            session.add(model)
            await session.commit()
            logger.debug(f"Saved event: {event.event_type} ({event.event_id})")
            return True

    async def get_events_with_person(
        self, person_id: str, limit: int = 20
    ) -> list[EventMemory]:
        """Get recent events involving a specific person."""
        async with self._db.session() as session:
            # Fetch all events and filter in Python since JSON querying varies by SQLite version
            result = await session.execute(
                select(EventModel).order_by(EventModel.timestamp.desc()).limit(limit * 5)
            )
            models = result.scalars().all()

            # Filter for events with the person
            filtered = [m for m in models if person_id in (m.participants or [])]
            return [EventMemory.from_state(m.to_dict()) for m in filtered[:limit]]

    async def get_events_by_type(
        self, event_type: str, limit: int = 20
    ) -> list[EventMemory]:
        """Get recent events of a specific type."""
        async with self._db.session() as session:
            result = await session.execute(
                select(EventModel)
                .where(EventModel.event_type == event_type)
                .order_by(EventModel.timestamp.desc())
                .limit(limit)
            )
            models = result.scalars().all()
            return [EventMemory.from_state(m.to_dict()) for m in models]

    async def get_recent_events(self, limit: int = 20) -> list[EventMemory]:
        """Get the most recent events."""
        async with self._db.session() as session:
            result = await session.execute(
                select(EventModel).order_by(EventModel.timestamp.desc()).limit(limit)
            )
            models = result.scalars().all()
            return [EventMemory.from_state(m.to_dict()) for m in models]

    def should_persist_event(self, event: EventMemory, participants_familiar: bool) -> bool:
        """
        Determine if an event should be persisted.

        Criteria:
        - Involves a familiar person
        - OR is a milestone event type (first_meeting, etc.)
        - OR has strong emotional valence
        """
        milestone_types = {
            "first_meeting",
            "first_play",
            "first_petting",
            "return_after_absence",
        }

        if event.event_type in milestone_types:
            return True

        if participants_familiar:
            return True

        # Strong emotional events (both positive and negative)
        if event.outcome in ("positive", "negative") and event.strength >= 0.8:
            return True

        return False

    # ==================== Bulk Operations ====================

    async def load_all_familiar_to_dict(self) -> dict[str, PersonMemory]:
        """
        Load all familiar people for session startup.

        Returns:
            Dictionary of person_id -> PersonMemory
        """
        people = await self.get_all_familiar_people()
        return {p.person_id: p for p in people}

    async def sync_from_short_term(
        self,
        people: dict[str, PersonMemory],
        objects: dict[str, ObjectMemory],
    ) -> tuple[int, int]:
        """
        Sync qualifying memories from short-term to long-term.

        Called periodically or on shutdown to persist important memories.

        Args:
            people: Dictionary of person_id -> PersonMemory
            objects: Dictionary of object_id -> ObjectMemory

        Returns:
            Tuple of (people_synced, objects_synced) counts
        """
        people_synced = 0
        objects_synced = 0

        # Sync people
        for person in people.values():
            if self.should_persist_person(person):
                await self.save_person(person)
                people_synced += 1

        # Sync objects
        for obj in objects.values():
            if self.should_persist_object(obj):
                await self.save_object(obj)
                objects_synced += 1

        if people_synced or objects_synced:
            logger.info(
                f"Synced {people_synced} people and {objects_synced} objects to long-term memory"
            )

        return people_synced, objects_synced

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about long-term memory."""
        async with self._db.session() as session:
            people_count = await session.scalar(
                select(func.count()).select_from(PersonModel)
            )
            familiar_count = await session.scalar(
                select(func.count())
                .select_from(PersonModel)
                .where(PersonModel.familiarity_score >= FAMILIARITY_THRESHOLD)
            )
            objects_count = await session.scalar(
                select(func.count()).select_from(ObjectModel)
            )
            events_count = await session.scalar(
                select(func.count()).select_from(EventModel)
            )
            embeddings_count = await session.scalar(
                select(func.count()).select_from(FaceEmbeddingModel)
            )
            landmarks_count = await session.scalar(
                select(func.count()).select_from(SpatialLandmarkModel)
            )
            zones_count = await session.scalar(
                select(func.count()).select_from(SpatialZoneModel)
            )
            observations_count = await session.scalar(
                select(func.count()).select_from(SpatialObservationModel)
            )

            return {
                "people_total": people_count or 0,
                "people_familiar": familiar_count or 0,
                "objects": objects_count or 0,
                "events": events_count or 0,
                "face_embeddings": embeddings_count or 0,
                "landmarks": landmarks_count or 0,
                "zones": zones_count or 0,
                "spatial_observations": observations_count or 0,
            }

    # ==================== Spatial Landmark Operations ====================

    async def get_landmark(self, landmark_id: str) -> SpatialLandmark | None:
        """Load a landmark from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialLandmarkModel).where(
                    SpatialLandmarkModel.landmark_id == landmark_id
                )
            )
            model = result.scalar_one_or_none()
            return SpatialLandmark.from_state(model.to_dict()) if model else None

    async def save_landmark(self, landmark: SpatialLandmark) -> None:
        """Save or update a landmark in long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialLandmarkModel).where(
                    SpatialLandmarkModel.landmark_id == landmark.landmark_id
                )
            )
            model = result.scalar_one_or_none()

            if model is None:
                model = SpatialLandmarkModel(
                    landmark_id=landmark.landmark_id,
                    landmark_type=landmark.landmark_type,
                    name=landmark.name,
                    first_seen=datetime.fromtimestamp(landmark.first_seen),
                    last_seen=datetime.fromtimestamp(landmark.last_seen),
                    times_visited=landmark.times_visited,
                    confidence=landmark.confidence,
                    connections=landmark.connections,
                )
                session.add(model)
                logger.debug(f"Created new landmark record: {landmark.landmark_id}")
            else:
                model.landmark_type = landmark.landmark_type
                model.name = landmark.name
                model.last_seen = datetime.fromtimestamp(landmark.last_seen)
                model.times_visited = landmark.times_visited
                model.confidence = landmark.confidence
                model.connections = landmark.connections
                logger.debug(f"Updated landmark record: {landmark.landmark_id}")

            await session.commit()

    async def get_all_landmarks(self) -> list[SpatialLandmark]:
        """Get all landmarks from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(select(SpatialLandmarkModel))
            models = result.scalars().all()
            return [SpatialLandmark.from_state(m.to_dict()) for m in models]

    async def get_landmarks_by_type(self, landmark_type: str) -> list[SpatialLandmark]:
        """Get landmarks of a specific type."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialLandmarkModel).where(
                    SpatialLandmarkModel.landmark_type == landmark_type
                )
            )
            models = result.scalars().all()
            return [SpatialLandmark.from_state(m.to_dict()) for m in models]

    def should_persist_landmark(self, landmark: SpatialLandmark) -> bool:
        """
        Determine if a landmark should be persisted.

        Criteria:
        - Confidence >= 0.6 (reliably recognizable)
        - OR times_visited >= 5 (frequently visited)
        - OR is a critical type (charging_station, home_base, edge)
        """
        return (
            landmark.confidence >= LANDMARK_CONFIDENCE_THRESHOLD
            or landmark.times_visited >= LANDMARK_VISIT_THRESHOLD
            or landmark.landmark_type in CRITICAL_LANDMARK_TYPES
        )

    # ==================== Spatial Zone Operations ====================

    async def get_zone(self, zone_id: str) -> SpatialZone | None:
        """Load a zone from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialZoneModel).where(SpatialZoneModel.zone_id == zone_id)
            )
            model = result.scalar_one_or_none()
            return SpatialZone.from_state(model.to_dict()) if model else None

    async def save_zone(self, zone: SpatialZone) -> None:
        """Save or update a zone in long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialZoneModel).where(SpatialZoneModel.zone_id == zone.zone_id)
            )
            model = result.scalar_one_or_none()

            if model is None:
                model = SpatialZoneModel(
                    zone_id=zone.zone_id,
                    zone_type=zone.zone_type,
                    name=zone.name,
                    primary_landmark_id=zone.primary_landmark_id,
                    safety_score=zone.safety_score,
                    familiarity=zone.familiarity,
                    associated_events=zone.associated_events,
                    last_visited=datetime.fromtimestamp(zone.last_visited),
                )
                session.add(model)
                logger.debug(f"Created new zone record: {zone.zone_id}")
            else:
                model.zone_type = zone.zone_type
                model.name = zone.name
                model.primary_landmark_id = zone.primary_landmark_id
                model.safety_score = zone.safety_score
                model.familiarity = zone.familiarity
                model.associated_events = zone.associated_events
                model.last_visited = datetime.fromtimestamp(zone.last_visited)
                logger.debug(f"Updated zone record: {zone.zone_id}")

            await session.commit()

    async def get_all_zones(self) -> list[SpatialZone]:
        """Get all zones from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(select(SpatialZoneModel))
            models = result.scalars().all()
            return [SpatialZone.from_state(m.to_dict()) for m in models]

    async def get_zones_by_type(self, zone_type: str) -> list[SpatialZone]:
        """Get zones of a specific type."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialZoneModel).where(SpatialZoneModel.zone_type == zone_type)
            )
            models = result.scalars().all()
            return [SpatialZone.from_state(m.to_dict()) for m in models]

    def should_persist_zone(self, zone: SpatialZone) -> bool:
        """
        Determine if a zone should be persisted.

        Criteria:
        - Familiarity >= 0.5 (reasonably explored)
        - OR is a critical type (charging_zone, edge_zone)
        """
        return (
            zone.familiarity >= ZONE_FAMILIARITY_THRESHOLD
            or zone.zone_type in CRITICAL_ZONE_TYPES
        )

    # ==================== Spatial Observation Operations ====================

    async def save_observation(self, observation: SpatialObservation) -> None:
        """Save a spatial observation to long-term memory."""
        async with self._db.session() as session:
            # Check if already exists
            result = await session.execute(
                select(SpatialObservationModel).where(
                    SpatialObservationModel.observation_id == observation.observation_id
                )
            )
            if result.scalar_one_or_none() is not None:
                return  # Already saved

            model = SpatialObservationModel(
                observation_id=observation.observation_id,
                entity_type=observation.entity_type,
                entity_id=observation.entity_id,
                landmark_id=observation.landmark_id,
                relative_direction=observation.relative_direction,
                relative_distance=observation.relative_distance,
                timestamp=datetime.fromtimestamp(observation.timestamp),
                confidence=observation.confidence,
            )
            session.add(model)
            await session.commit()
            logger.debug(
                f"Saved observation: {observation.entity_type}:{observation.entity_id}"
            )

    async def get_observations_for_entity(
        self, entity_id: str, limit: int = 10
    ) -> list[SpatialObservation]:
        """Get recent observations of a specific entity."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialObservationModel)
                .where(SpatialObservationModel.entity_id == entity_id)
                .order_by(SpatialObservationModel.timestamp.desc())
                .limit(limit)
            )
            models = result.scalars().all()
            return [SpatialObservation.from_state(m.to_dict()) for m in models]

    async def get_observations_near_landmark(
        self, landmark_id: str, limit: int = 20
    ) -> list[SpatialObservation]:
        """Get observations near a specific landmark."""
        async with self._db.session() as session:
            result = await session.execute(
                select(SpatialObservationModel)
                .where(SpatialObservationModel.landmark_id == landmark_id)
                .order_by(SpatialObservationModel.timestamp.desc())
                .limit(limit)
            )
            models = result.scalars().all()
            return [SpatialObservation.from_state(m.to_dict()) for m in models]

    def should_persist_observation(
        self, observation: SpatialObservation, entity_is_important: bool
    ) -> bool:
        """
        Determine if an observation should be persisted.

        Criteria:
        - Entity is important (familiar person or interesting object)
        - AND confidence >= 0.5
        """
        return (
            entity_is_important
            and observation.confidence >= OBSERVATION_CONFIDENCE_THRESHOLD
        )

    # ==================== Spatial Map Bulk Operations ====================

    async def load_spatial_map(self) -> SpatialMapMemory:
        """
        Load the full spatial map from long-term storage.

        Returns:
            SpatialMapMemory populated with persisted data
        """
        landmarks = await self.get_all_landmarks()
        zones = await self.get_all_zones()

        # Build dictionaries
        landmarks_dict = {lm.landmark_id: lm for lm in landmarks}
        zones_dict = {z.zone_id: z for z in zones}

        # Find home landmark if any
        home_landmark_id = None
        for lm in landmarks:
            if lm.landmark_type == "home_base":
                home_landmark_id = lm.landmark_id
                break

        spatial_map = SpatialMapMemory(
            landmarks=landmarks_dict,
            zones=zones_dict,
            observations=[],  # Observations not loaded from DB (too transient)
            home_landmark_id=home_landmark_id,
        )

        logger.info(
            f"Loaded spatial map: {len(landmarks)} landmarks, {len(zones)} zones"
        )
        return spatial_map

    async def sync_spatial_map(
        self, spatial_map: SpatialMapMemory
    ) -> dict[str, int]:
        """
        Sync qualifying spatial data to long-term storage.

        Args:
            spatial_map: The spatial map to sync

        Returns:
            Dict with counts: {"landmarks": n, "zones": n, "observations": n}
        """
        landmarks_synced = 0
        zones_synced = 0
        observations_synced = 0

        # Sync landmarks
        for landmark in spatial_map.landmarks.values():
            if self.should_persist_landmark(landmark):
                await self.save_landmark(landmark)
                landmarks_synced += 1

        # Sync zones
        for zone in spatial_map.zones.values():
            if self.should_persist_zone(zone):
                await self.save_zone(zone)
                zones_synced += 1

        # Note: Observations are typically not bulk-synced since they're transient
        # They can be saved individually when deemed important

        if landmarks_synced or zones_synced:
            logger.info(
                f"Synced spatial map: {landmarks_synced} landmarks, {zones_synced} zones"
            )

        return {
            "landmarks": landmarks_synced,
            "zones": zones_synced,
            "observations": observations_synced,
        }

    # ==================== Insight Operations ====================

    async def save_insight(self, insight: InsightMemory) -> bool:
        """
        Save an insight to long-term memory.

        Args:
            insight: InsightMemory to save

        Returns:
            True if saved (new insight), False if already exists
        """
        async with self._db.session() as session:
            # Check if already exists
            result = await session.execute(
                select(InsightModel).where(InsightModel.insight_id == insight.insight_id)
            )
            if result.scalar_one_or_none() is not None:
                return False  # Already saved

            model = InsightModel(
                insight_id=insight.insight_id,
                insight_type=insight.insight_type,
                subject_type=insight.subject_type,
                subject_id=insight.subject_id,
                content=insight.content,
                summary=insight.summary,
                source_event_ids=insight.source_event_ids,
                created_at=datetime.fromtimestamp(insight.created_at),
                confidence=insight.confidence,
                relevance_score=insight.relevance_score,
                tags=list(insight.tags),
            )
            session.add(model)
            await session.commit()
            logger.debug(f"Saved insight: {insight.insight_type} ({insight.insight_id})")
            return True

    async def get_insight(self, insight_id: str) -> InsightMemory | None:
        """Load an insight from long-term memory."""
        async with self._db.session() as session:
            result = await session.execute(
                select(InsightModel).where(InsightModel.insight_id == insight_id)
            )
            model = result.scalar_one_or_none()
            return InsightMemory.from_state(model.to_dict()) if model else None

    async def get_insights_for_subject(
        self,
        subject_type: str,
        subject_id: str | None = None,
        insight_type: str | None = None,
        limit: int = 10,
    ) -> list[InsightMemory]:
        """
        Get insights related to a subject.

        Args:
            subject_type: Type of subject ("person", "behavior", etc.)
            subject_id: Optional specific subject ID
            insight_type: Optional filter by insight type
            limit: Maximum number of insights to return

        Returns:
            List of InsightMemory objects
        """
        async with self._db.session() as session:
            query = select(InsightModel).where(InsightModel.subject_type == subject_type)

            if subject_id is not None:
                query = query.where(InsightModel.subject_id == subject_id)

            if insight_type is not None:
                query = query.where(InsightModel.insight_type == insight_type)

            query = query.order_by(InsightModel.created_at.desc()).limit(limit)

            result = await session.execute(query)
            models = result.scalars().all()
            return [InsightMemory.from_state(m.to_dict()) for m in models]

    async def get_recent_insights(
        self,
        insight_type: str | None = None,
        limit: int = 20,
    ) -> list[InsightMemory]:
        """
        Get the most recent insights.

        Args:
            insight_type: Optional filter by insight type
            limit: Maximum number of insights to return

        Returns:
            List of InsightMemory objects
        """
        async with self._db.session() as session:
            query = select(InsightModel)

            if insight_type is not None:
                query = query.where(InsightModel.insight_type == insight_type)

            query = query.order_by(InsightModel.created_at.desc()).limit(limit)

            result = await session.execute(query)
            models = result.scalars().all()
            return [InsightMemory.from_state(m.to_dict()) for m in models]

    async def get_relevant_insights(
        self,
        min_relevance: float = 0.1,
        limit: int = 20,
    ) -> list[InsightMemory]:
        """
        Get insights that are still relevant (above relevance threshold).

        Args:
            min_relevance: Minimum relevance score
            limit: Maximum number of insights to return

        Returns:
            List of InsightMemory objects ordered by relevance
        """
        async with self._db.session() as session:
            query = (
                select(InsightModel)
                .where(InsightModel.relevance_score >= min_relevance)
                .order_by(InsightModel.relevance_score.desc())
                .limit(limit)
            )

            result = await session.execute(query)
            models = result.scalars().all()
            return [InsightMemory.from_state(m.to_dict()) for m in models]

    async def decay_insight_relevance(self, decay_rate: float = 0.01) -> int:
        """
        Decay relevance scores for all insights.

        Called periodically to reduce relevance of old insights.

        Args:
            decay_rate: Amount to decay each insight's relevance

        Returns:
            Number of insights decayed
        """
        async with self._db.session() as session:
            result = await session.execute(select(InsightModel))
            models = result.scalars().all()

            decayed_count = 0
            for model in models:
                new_relevance = max(0.0, model.relevance_score - decay_rate)
                if new_relevance != model.relevance_score:
                    model.relevance_score = new_relevance
                    decayed_count += 1

            await session.commit()

            if decayed_count > 0:
                logger.debug(f"Decayed relevance for {decayed_count} insights")

            return decayed_count

    async def prune_stale_insights(self, min_relevance: float = 0.1) -> int:
        """
        Remove insights that have fallen below the relevance threshold.

        Args:
            min_relevance: Minimum relevance score to keep

        Returns:
            Number of insights deleted
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(InsightModel).where(InsightModel.relevance_score < min_relevance)
            )
            stale_insights = result.scalars().all()

            deleted_count = len(stale_insights)
            for model in stale_insights:
                await session.delete(model)

            await session.commit()

            if deleted_count > 0:
                logger.info(f"Pruned {deleted_count} stale insights")

            return deleted_count

    async def get_insight_stats(self) -> dict[str, Any]:
        """Get statistics about stored insights."""
        async with self._db.session() as session:
            total_count = await session.scalar(
                select(func.count()).select_from(InsightModel)
            )

            # Count by type
            type_counts: dict[str, int] = {}
            for insight_type in ["event_summary", "relationship_narrative", "behavior_reflection"]:
                count = await session.scalar(
                    select(func.count())
                    .select_from(InsightModel)
                    .where(InsightModel.insight_type == insight_type)
                )
                type_counts[insight_type] = count or 0

            # Average relevance
            avg_relevance = await session.scalar(
                select(func.avg(InsightModel.relevance_score))
            )

            return {
                "total": total_count or 0,
                "by_type": type_counts,
                "avg_relevance": float(avg_relevance) if avg_relevance else 0.0,
            }
