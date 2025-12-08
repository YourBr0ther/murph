"""
Murph - Memory System
Unified facade combining working, short-term, and long-term memory.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .memory_types import EventMemory, ObjectMemory, PersonMemory
from .short_term_memory import ShortTermMemory
from .spatial_types import (
    SpatialLandmark,
    SpatialMapMemory,
    SpatialObservation,
    SpatialZone,
)
from .working_memory import WorkingMemory

if TYPE_CHECKING:
    from .long_term_memory import LongTermMemory

logger = logging.getLogger("murph.memory")


class MemorySystem:
    """
    Unified memory system for Murph.

    Combines working memory (immediate context), short-term memory
    (decaying memories of people, objects, events), and optionally
    long-term memory (persistent SQLite storage) into a single interface.

    Usage:
        memory = MemorySystem()

        # Process perception each cycle
        memory.process_perception(context, person_id="person_1")

        # Update decay
        memory.update(delta_seconds)

        # Query for behavior evaluation
        context_data = memory.get_behavior_context()

        # With long-term memory:
        memory = MemorySystem(long_term=ltm)
        await memory.initialize_from_database()  # Load familiar people
        # ... during operation ...
        await memory.sync_to_long_term()  # Periodic sync
        await memory.shutdown()  # Final sync on shutdown
    """

    def __init__(
        self,
        event_decay_rate: float = 0.5,
        event_threshold: float = 0.1,
        max_events: int = 50,
        familiarity_growth: float = 1.0,
        familiarity_decay: float = 0.1,
        long_term: LongTermMemory | None = None,
    ) -> None:
        """
        Initialize the memory system.

        Args:
            event_decay_rate: How fast events decay (points/minute)
            event_threshold: Below this strength, events are forgotten
            max_events: Maximum events to keep in memory
            familiarity_growth: How much familiarity grows per interaction
            familiarity_decay: How fast familiarity decays (points/hour)
            long_term: Optional LongTermMemory for persistent storage
        """
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(
            event_decay_rate=event_decay_rate,
            event_threshold=event_threshold,
            max_events=max_events,
            familiarity_growth=familiarity_growth,
            familiarity_decay=familiarity_decay,
        )
        self.spatial_map = SpatialMapMemory()
        self.long_term = long_term
        self._last_update_time: float = time.time()
        self._last_sync_time: float = time.time()
        self._sync_interval: float = 60.0  # Sync to database every 60 seconds

    def update(self, delta_seconds: float | None = None) -> None:
        """
        Update memory systems (decay, pruning).

        Args:
            delta_seconds: Time elapsed. If None, calculates from last update.
        """
        if delta_seconds is None:
            current_time = time.time()
            delta_seconds = current_time - self._last_update_time
            self._last_update_time = current_time
        else:
            self._last_update_time = time.time()

        self.short_term.update(delta_seconds)

    def process_perception(
        self,
        person_detected: bool = False,
        person_id: str | None = None,
        person_is_familiar: bool = False,
        person_distance: float | None = None,
        objects_in_view: list[str] | None = None,
        is_being_petted: bool = False,
        is_being_held: bool = False,
    ) -> None:
        """
        Process perception data and update memories.

        Called each perception/cognition cycle to update memory state
        based on what Murph perceives.

        Args:
            person_detected: Whether a person is in view
            person_id: Identifier for detected person
            person_is_familiar: Whether perception thinks person is familiar
            person_distance: Distance to person in cm
            objects_in_view: List of object IDs/types visible
            is_being_petted: Whether being petted right now
            is_being_held: Whether being held right now
        """
        # Process person perception
        if person_detected and person_id:
            # Determine if this is an interaction
            is_interacting = (
                is_being_petted
                or is_being_held
                or (person_distance is not None and person_distance < 30)
            )

            # Record the sighting
            self.short_term.record_person_seen(
                person_id=person_id,
                is_familiar=person_is_familiar,
                distance=person_distance,
                interaction=is_interacting,
            )

            # Update working memory
            if is_interacting:
                self.working.set_active_person(person_id)
                self.working.set_attention(person_id)
        else:
            # No person detected, clear active person
            self.working.set_active_person(None)

        # Process object perception
        if objects_in_view:
            for obj_id in objects_in_view:
                # For now, use the ID as the type if it looks like a type
                self.short_term.record_object_seen(
                    object_id=obj_id,
                    object_type=obj_id,
                )
            self.working.active_objects = objects_in_view.copy()
        else:
            self.working.clear_active_objects()

        # Record significant events
        if is_being_petted:
            # Only record if not recently recorded
            if not self.short_term.was_event_recent("petting", within_seconds=5.0):
                participants = [person_id] if person_id else []
                self.short_term.record_event(
                    event_type="petting",
                    participants=participants,
                    outcome="positive",
                )

        if is_being_held:
            if not self.short_term.was_event_recent("held", within_seconds=10.0):
                participants = [person_id] if person_id else []
                self.short_term.record_event(
                    event_type="held",
                    participants=participants,
                    outcome="positive",
                )

    def record_behavior_start(self, behavior: str, goal: str | None = None) -> None:
        """
        Record starting a behavior.

        Args:
            behavior: Name of the behavior
            goal: The need/goal being addressed
        """
        self.working.start_behavior(behavior, goal)

    def record_behavior_end(self, result: str = "completed") -> None:
        """
        Record ending a behavior.

        Args:
            result: How it ended ("completed", "interrupted", "failed")
        """
        self.working.end_behavior(result)

    def record_event(
        self,
        event_type: str,
        participants: list[str] | None = None,
        objects: list[str] | None = None,
        outcome: str = "neutral",
    ) -> EventMemory:
        """
        Manually record an event.

        Args:
            event_type: Type of event
            participants: Person IDs involved
            objects: Object IDs involved
            outcome: "positive", "negative", or "neutral"

        Returns:
            The created event memory
        """
        return self.short_term.record_event(
            event_type=event_type,
            participants=participants,
            objects=objects,
            outcome=outcome,
        )

    def get_active_person(self) -> PersonMemory | None:
        """Get memory of the currently active person."""
        if self.working.active_person_id:
            return self.short_term.get_person(self.working.active_person_id)
        return None

    def get_active_person_name(self) -> str | None:
        """Get name of currently active person if known."""
        person = self.get_active_person()
        return person.name if person else None

    def is_active_person_familiar(self) -> bool:
        """Check if the currently active person is familiar."""
        person = self.get_active_person()
        return person.is_familiar if person else False

    def get_behavior_context(self) -> dict[str, Any]:
        """
        Get memory context for behavior evaluation.

        Returns:
            Dictionary with memory-derived context information
        """
        active_person = self.get_active_person()

        return {
            # Working memory context
            **self.working.get_context_summary(),
            # Person context
            "active_person_name": active_person.name if active_person else None,
            "active_person_familiar": active_person.is_familiar if active_person else False,
            "active_person_sentiment": active_person.sentiment if active_person else 0.0,
            "active_person_interactions": (
                active_person.interaction_count if active_person else 0
            ),
            # Memory summary
            "familiar_people_count": len(self.short_term.get_familiar_people()),
            "recent_event_types": self.short_term.get_recent_event_types(60.0),
        }

    def get_memory_triggers(self) -> dict[str, bool]:
        """
        Get memory-derived triggers for WorldContext.

        Returns:
            Dictionary of trigger name -> bool
        """
        active_person = self.get_active_person()
        recent_events = self.short_term.get_recent_event_types(60.0)

        # Get spatial triggers
        current_zone = self.spatial_map.current_zone
        current_landmark = self.spatial_map.current_landmark

        return {
            "familiar_person_remembered": (
                active_person.is_familiar if active_person else False
            ),
            "positive_history": (
                active_person is not None and active_person.interaction_count >= 5
            ),
            "negative_sentiment": (
                active_person is not None and active_person.sentiment < -0.3
            ),
            "positive_sentiment": (
                active_person is not None and active_person.sentiment > 0.3
            ),
            "recently_greeted": "greeting" in recent_events,
            "recently_played": "play" in recent_events,
            "recently_petted": "petting" in recent_events,
            "was_interrupted": self.working.was_interrupted,
            # Spatial triggers
            "at_home": self.spatial_map.is_at_home,
            "at_charger": (
                current_landmark is not None
                and current_landmark.landmark_type == "charging_station"
            ),
            "in_safe_zone": current_zone is not None and current_zone.is_safe,
            "in_danger_zone": current_zone is not None and current_zone.is_dangerous,
            "position_known": self.spatial_map.is_position_known,
            "position_lost": not self.spatial_map.is_position_known,
            "near_edge": (
                current_landmark is not None
                and current_landmark.landmark_type == "edge"
            ),
        }

    # ==================== Spatial Memory Operations ====================

    def update_current_location(
        self,
        landmark_id: str | None,
        zone_id: str | None = None,
    ) -> None:
        """
        Update the robot's current spatial location.

        Args:
            landmark_id: Nearest landmark ID (or None if position unknown)
            zone_id: Current zone ID (optional, will be inferred from landmark)
        """
        self.spatial_map.update_current_location(landmark_id, zone_id)

        # Record visit to landmark if known
        if landmark_id:
            landmark = self.spatial_map.get_landmark(landmark_id)
            if landmark:
                landmark.record_visit()

            # Record visit to zone if known
            zone = self.spatial_map.current_zone
            if zone:
                zone.record_visit()

    def record_landmark(
        self,
        landmark_id: str,
        landmark_type: str,
        name: str | None = None,
        confidence: float = 0.5,
    ) -> SpatialLandmark:
        """
        Record discovering or visiting a landmark.

        Args:
            landmark_id: Unique ID for the landmark
            landmark_type: Type ("charging_station", "edge", "corner", etc.)
            name: Optional human-readable name
            confidence: Recognition confidence (0-1)

        Returns:
            The created or updated landmark
        """
        existing = self.spatial_map.get_landmark(landmark_id)
        if existing:
            existing.record_visit()
            existing.confidence = max(existing.confidence, confidence)
            if name:
                existing.name = name
            return existing

        landmark = SpatialLandmark(
            landmark_id=landmark_id,
            landmark_type=landmark_type,
            name=name,
            confidence=confidence,
        )
        self.spatial_map.add_landmark(landmark)
        return landmark

    def record_zone(
        self,
        zone_id: str,
        zone_type: str,
        primary_landmark_id: str,
        name: str | None = None,
        safety_score: float = 0.5,
    ) -> SpatialZone:
        """
        Record discovering or visiting a zone.

        Args:
            zone_id: Unique ID for the zone
            zone_type: Type ("safe", "dangerous", "play_area", etc.)
            primary_landmark_id: The landmark that anchors this zone
            name: Optional human-readable name
            safety_score: Initial safety score (0-1)

        Returns:
            The created or updated zone
        """
        existing = self.spatial_map.get_zone(zone_id)
        if existing:
            existing.record_visit()
            if name:
                existing.name = name
            return existing

        zone = SpatialZone(
            zone_id=zone_id,
            zone_type=zone_type,
            primary_landmark_id=primary_landmark_id,
            name=name,
            safety_score=safety_score,
        )
        self.spatial_map.add_zone(zone)
        return zone

    def record_spatial_observation(
        self,
        entity_type: str,
        entity_id: str,
        landmark_id: str,
        relative_direction: float,
        relative_distance: float,
        confidence: float = 0.5,
    ) -> SpatialObservation:
        """
        Record seeing an entity at a location relative to a landmark.

        Args:
            entity_type: "object" or "person"
            entity_id: The object_id or person_id
            landmark_id: Nearest landmark
            relative_direction: Direction from landmark (degrees, 0=forward)
            relative_distance: Distance from landmark (cm)
            confidence: Location confidence (0-1)

        Returns:
            The created observation
        """
        observation = SpatialObservation(
            entity_type=entity_type,
            entity_id=entity_id,
            landmark_id=landmark_id,
            relative_direction=relative_direction,
            relative_distance=relative_distance,
            confidence=confidence,
        )
        self.spatial_map.add_observation(observation)
        return observation

    def record_zone_event(self, event_type: str) -> None:
        """
        Record an event occurring in the current zone.

        Args:
            event_type: Type of event (e.g., "bump", "petting", "play")
        """
        zone = self.spatial_map.current_zone
        if zone:
            zone.record_event(event_type)

            # Adjust safety based on event
            if event_type in ("bump", "fall", "drop"):
                zone.adjust_safety(-0.1)  # Negative events reduce safety
            elif event_type in ("petting", "play", "rest"):
                zone.adjust_safety(0.05)  # Positive events increase safety

    def set_home_landmark(self, landmark_id: str) -> None:
        """Set a landmark as the home base."""
        self.spatial_map.set_home(landmark_id)

    def get_spatial_context(self) -> dict[str, Any]:
        """
        Get spatial context for behavior evaluation.

        Returns:
            Dictionary with spatial awareness information
        """
        current_zone = self.spatial_map.current_zone
        current_landmark = self.spatial_map.current_landmark

        return {
            "current_landmark_id": self.spatial_map.current_landmark_id,
            "current_landmark_type": (
                current_landmark.landmark_type if current_landmark else None
            ),
            "current_zone_id": self.spatial_map.current_zone_id,
            "current_zone_type": current_zone.zone_type if current_zone else None,
            "current_zone_safety": current_zone.safety_score if current_zone else 0.5,
            "at_home": self.spatial_map.is_at_home,
            "position_known": self.spatial_map.is_position_known,
            "landmarks_count": len(self.spatial_map.landmarks),
            "zones_count": len(self.spatial_map.zones),
        }

    def clear(self) -> None:
        """Clear all memory state."""
        self.working.clear()
        self.short_term.clear()
        self.spatial_map = SpatialMapMemory()

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "working": self.working.get_state(),
            "short_term": self.short_term.get_state(),
            "spatial_map": self.spatial_map.get_state(),
            "last_update_time": self._last_update_time,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MemorySystem":
        """Create a MemorySystem from saved state."""
        # Get config from short_term state if available
        st_config = state.get("short_term", {}).get("config", {})

        system = cls(
            event_decay_rate=st_config.get("event_decay_rate", 0.5),
            event_threshold=st_config.get("event_threshold", 0.1),
            max_events=st_config.get("max_events", 50),
            familiarity_growth=st_config.get("familiarity_growth", 1.0),
            familiarity_decay=st_config.get("familiarity_decay", 0.1),
        )

        # Restore subsystem states
        if "working" in state:
            system.working = WorkingMemory.from_state(state["working"])
        if "short_term" in state:
            system.short_term = ShortTermMemory.from_state(state["short_term"])
        if "spatial_map" in state:
            system.spatial_map = SpatialMapMemory.from_state(state["spatial_map"])

        system._last_update_time = state.get("last_update_time", time.time())
        return system

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "working": {
                "current_behavior": self.working.current_behavior,
                "active_person": self.working.active_person_id,
                "attention": self.working.attention_target,
            },
            "short_term": self.short_term.summary(),
            "spatial": {
                "landmarks": len(self.spatial_map.landmarks),
                "zones": len(self.spatial_map.zones),
                "current_landmark": self.spatial_map.current_landmark_id,
                "current_zone": self.spatial_map.current_zone_id,
                "at_home": self.spatial_map.is_at_home,
            },
        }

    # ==================== Long-term Memory Integration ====================

    async def initialize_from_database(self) -> dict[str, int]:
        """
        Initialize memory by loading data from long-term storage.

        Call this at startup when using persistent storage.

        Returns:
            Dictionary with counts of loaded items
        """
        if self.long_term is None:
            return {"people": 0, "landmarks": 0, "zones": 0}

        await self.long_term.initialize()

        # Load all familiar people into short-term memory
        familiar = await self.long_term.load_all_familiar_to_dict()
        for person_id, person in familiar.items():
            self.short_term._people[person_id] = person

        # Load spatial map from long-term memory
        self.spatial_map = await self.long_term.load_spatial_map()

        logger.info(
            f"Loaded {len(familiar)} familiar people, "
            f"{len(self.spatial_map.landmarks)} landmarks, "
            f"{len(self.spatial_map.zones)} zones from long-term memory"
        )
        return {
            "people": len(familiar),
            "landmarks": len(self.spatial_map.landmarks),
            "zones": len(self.spatial_map.zones),
        }

    async def sync_to_long_term(self, force: bool = False) -> dict[str, int]:
        """
        Sync qualifying memories to long-term storage.

        Args:
            force: If True, sync immediately regardless of interval

        Returns:
            Dictionary with sync counts
        """
        if self.long_term is None:
            return {"people": 0, "objects": 0, "events": 0, "spatial": 0}

        current_time = time.time()
        if not force and (current_time - self._last_sync_time) < self._sync_interval:
            return {"people": 0, "objects": 0, "events": 0, "spatial": 0}

        self._last_sync_time = current_time

        # Sync people and objects
        people_synced, objects_synced = await self.long_term.sync_from_short_term(
            self.short_term._people,
            self.short_term._objects,
        )

        # Sync significant events
        events_synced = 0
        for event in self.short_term._events:
            has_familiar = any(
                self.short_term.is_person_familiar(p) for p in event.participants
            )
            if self.long_term.should_persist_event(event, has_familiar):
                if await self.long_term.save_event(event):
                    events_synced += 1

        if events_synced:
            logger.debug(f"Synced {events_synced} events to long-term memory")

        # Sync spatial map
        spatial_counts = await self.long_term.sync_spatial_map(self.spatial_map)
        spatial_synced = spatial_counts["landmarks"] + spatial_counts["zones"]

        return {
            "people": people_synced,
            "objects": objects_synced,
            "events": events_synced,
            "spatial": spatial_synced,
        }

    async def shutdown(self) -> None:
        """
        Shutdown memory system, ensuring all data is persisted.

        Call this when shutting down the robot to save all important memories.
        """
        await self.sync_to_long_term(force=True)
        logger.info("Memory system shutdown complete")

    async def lookup_person_by_face(
        self, embedding: Any, threshold: float = 0.6
    ) -> tuple[str | None, float]:
        """
        Look up a person by face embedding in long-term memory.

        Args:
            embedding: 128-dim FaceNet embedding (numpy array)
            threshold: Minimum cosine similarity for a match

        Returns:
            Tuple of (person_id, similarity) or (None, 0.0) if no match
        """
        if self.long_term is None:
            return None, 0.0

        return await self.long_term.find_person_by_embedding(embedding, threshold)

    def __str__(self) -> str:
        ltm_str = ", long_term=enabled" if self.long_term else ""
        spatial_str = f", {len(self.spatial_map.landmarks)} landmarks"
        return f"MemorySystem({self.working}, {self.short_term}{spatial_str}{ltm_str})"
