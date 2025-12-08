"""
Murph - Spatial Memory Types
Data models for spatial awareness and environment mapping.

Uses landmark-based relative positioning since Murph has no absolute
positioning sensors (GPS, LIDAR, depth camera).
"""

from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass
class SpatialLandmark:
    """
    A recognizable spatial reference point.

    Landmarks are features the robot can recognize visually or through
    sensor patterns. They anchor the spatial map since Murph has no
    absolute positioning.

    Attributes:
        landmark_id: Unique identifier (UUID)
        landmark_type: Category ("charging_station", "edge", "corner",
                       "obstacle", "home_base", "visual_marker")
        name: Human-readable name if assigned ("desk corner", "charging spot")
        first_seen: Timestamp of first encounter
        last_seen: Most recent observation
        times_visited: How often robot has been at this landmark
        confidence: How reliably this landmark can be recognized (0-1)
        connections: Dict mapping landmark_id -> estimated_distance (cm)
    """

    landmark_id: str
    landmark_type: str
    name: str | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_visited: int = 1
    confidence: float = 0.5
    connections: dict[str, float] = field(default_factory=dict)

    def record_visit(self) -> None:
        """Record visiting this landmark."""
        self.last_seen = time.time()
        self.times_visited += 1

    def add_connection(self, other_landmark_id: str, distance_cm: float) -> None:
        """
        Add or update a connection to another landmark.

        Args:
            other_landmark_id: The connected landmark's ID
            distance_cm: Estimated distance in centimeters
        """
        self.connections[other_landmark_id] = distance_cm

    def remove_connection(self, other_landmark_id: str) -> None:
        """Remove a connection to another landmark."""
        self.connections.pop(other_landmark_id, None)

    def adjust_confidence(self, delta: float) -> None:
        """
        Adjust recognition confidence.

        Args:
            delta: Amount to adjust (clamped to 0-1)
        """
        self.confidence = max(0.0, min(1.0, self.confidence + delta))

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "landmark_id": self.landmark_id,
            "landmark_type": self.landmark_type,
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "times_visited": self.times_visited,
            "confidence": self.confidence,
            "connections": self.connections.copy(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SpatialLandmark":
        """Create a SpatialLandmark from saved state."""
        return cls(
            landmark_id=state["landmark_id"],
            landmark_type=state.get("landmark_type", "unknown"),
            name=state.get("name"),
            first_seen=state.get("first_seen", time.time()),
            last_seen=state.get("last_seen", time.time()),
            times_visited=state.get("times_visited", 1),
            confidence=state.get("confidence", 0.5),
            connections=state.get("connections", {}),
        )

    def __str__(self) -> str:
        name_str = f" ({self.name})" if self.name else ""
        conn_count = len(self.connections)
        return (
            f"Landmark[{self.landmark_id[:8]}{name_str}]: {self.landmark_type}, "
            f"visits={self.times_visited}, confidence={self.confidence:.2f}, "
            f"connections={conn_count}"
        )


@dataclass
class SpatialZone:
    """
    A region of the environment with behavioral meaning.

    Zones categorize areas by their characteristics and the robot's
    experience with them. They are defined by proximity to a primary landmark.

    Attributes:
        zone_id: Unique identifier
        zone_type: Category ("safe", "dangerous", "play_area", "rest_area",
                   "charging_zone", "edge_zone", "unexplored")
        name: Human-readable name if assigned
        primary_landmark_id: Main landmark that defines this zone
        safety_score: 0 (dangerous) to 1 (safe), based on experience
        familiarity: 0 (unknown) to 1 (well-explored)
        associated_events: Event types that have occurred here
        last_visited: When robot was last in this zone
    """

    zone_id: str
    zone_type: str
    primary_landmark_id: str
    name: str | None = None
    safety_score: float = 0.5
    familiarity: float = 0.0
    associated_events: list[str] = field(default_factory=list)
    last_visited: float = field(default_factory=time.time)

    def record_visit(self) -> None:
        """Record visiting this zone."""
        self.last_visited = time.time()
        # Increase familiarity with each visit, with diminishing returns
        self.familiarity = min(1.0, self.familiarity + 0.1 * (1.0 - self.familiarity))

    def record_event(self, event_type: str) -> None:
        """
        Record an event occurring in this zone.

        Args:
            event_type: The type of event (e.g., "bump", "petting", "play")
        """
        if event_type not in self.associated_events:
            self.associated_events.append(event_type)

    def adjust_safety(self, delta: float) -> None:
        """
        Adjust safety score based on experience.

        Args:
            delta: Amount to adjust (clamped to 0-1)
        """
        self.safety_score = max(0.0, min(1.0, self.safety_score + delta))

    @property
    def is_safe(self) -> bool:
        """Zone is considered safe when safety_score >= 0.7."""
        return self.safety_score >= 0.7

    @property
    def is_dangerous(self) -> bool:
        """Zone is considered dangerous when safety_score < 0.3."""
        return self.safety_score < 0.3

    @property
    def is_familiar(self) -> bool:
        """Zone is considered familiar when familiarity >= 0.5."""
        return self.familiarity >= 0.5

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "primary_landmark_id": self.primary_landmark_id,
            "name": self.name,
            "safety_score": self.safety_score,
            "familiarity": self.familiarity,
            "associated_events": self.associated_events.copy(),
            "last_visited": self.last_visited,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SpatialZone":
        """Create a SpatialZone from saved state."""
        return cls(
            zone_id=state["zone_id"],
            zone_type=state.get("zone_type", "unknown"),
            primary_landmark_id=state["primary_landmark_id"],
            name=state.get("name"),
            safety_score=state.get("safety_score", 0.5),
            familiarity=state.get("familiarity", 0.0),
            associated_events=state.get("associated_events", []),
            last_visited=state.get("last_visited", time.time()),
        )

    def __str__(self) -> str:
        name_str = f" ({self.name})" if self.name else ""
        safety_str = "safe" if self.is_safe else ("dangerous" if self.is_dangerous else "neutral")
        return (
            f"Zone[{self.zone_id[:8]}{name_str}]: {self.zone_type}, "
            f"{safety_str}, familiarity={self.familiarity:.2f}"
        )


@dataclass
class SpatialObservation:
    """
    A remembered observation of something at a location.

    Records where objects or people were seen relative to landmarks.

    Attributes:
        observation_id: Unique identifier
        entity_type: "object" or "person"
        entity_id: The object_id or person_id observed
        landmark_id: Nearest landmark when observed
        relative_direction: Approximate direction from landmark (degrees, 0=forward)
        relative_distance: Estimated distance from landmark (cm)
        timestamp: When observed
        confidence: How confident the location estimate is (0-1)
    """

    entity_type: str
    entity_id: str
    landmark_id: str
    relative_direction: float
    relative_distance: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "observation_id": self.observation_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "landmark_id": self.landmark_id,
            "relative_direction": self.relative_direction,
            "relative_distance": self.relative_distance,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SpatialObservation":
        """Create a SpatialObservation from saved state."""
        return cls(
            observation_id=state.get("observation_id", str(uuid.uuid4())),
            entity_type=state["entity_type"],
            entity_id=state["entity_id"],
            landmark_id=state["landmark_id"],
            relative_direction=state.get("relative_direction", 0.0),
            relative_distance=state.get("relative_distance", 0.0),
            timestamp=state.get("timestamp", time.time()),
            confidence=state.get("confidence", 0.5),
        )

    def __str__(self) -> str:
        age_seconds = time.time() - self.timestamp
        if age_seconds < 60:
            age_str = f"{age_seconds:.0f}s ago"
        elif age_seconds < 3600:
            age_str = f"{age_seconds / 60:.0f}m ago"
        else:
            age_str = f"{age_seconds / 3600:.1f}h ago"
        return (
            f"Observation[{self.entity_type}:{self.entity_id[:8]}]: "
            f"near {self.landmark_id[:8]}, {self.relative_distance:.0f}cm @ "
            f"{self.relative_direction:.0f}°, {age_str}"
        )


# Maximum observations to keep in memory (FIFO)
MAX_OBSERVATIONS = 50


@dataclass
class SpatialMapMemory:
    """
    Murph's spatial understanding of the environment.

    Combines landmarks, zones, and observations into a navigable
    mental map. Uses landmark-based relative positioning since
    the robot has no absolute positioning sensors.

    Attributes:
        landmarks: Dict of landmark_id -> SpatialLandmark
        zones: Dict of zone_id -> SpatialZone
        observations: List of recent SpatialObservations (capped at MAX_OBSERVATIONS)
        current_landmark_id: Nearest known landmark (or None if lost)
        current_zone_id: Current zone (or None)
        home_landmark_id: The "home base" landmark
        last_update: When map was last updated
    """

    landmarks: dict[str, SpatialLandmark] = field(default_factory=dict)
    zones: dict[str, SpatialZone] = field(default_factory=dict)
    observations: list[SpatialObservation] = field(default_factory=list)
    current_landmark_id: str | None = None
    current_zone_id: str | None = None
    home_landmark_id: str | None = None
    last_update: float = field(default_factory=time.time)

    def add_landmark(self, landmark: SpatialLandmark) -> None:
        """Add or update a landmark."""
        self.landmarks[landmark.landmark_id] = landmark
        self.last_update = time.time()

    def get_landmark(self, landmark_id: str) -> SpatialLandmark | None:
        """Get a landmark by ID."""
        return self.landmarks.get(landmark_id)

    def remove_landmark(self, landmark_id: str) -> None:
        """Remove a landmark and its connections."""
        if landmark_id in self.landmarks:
            del self.landmarks[landmark_id]
            # Remove connections to this landmark from others
            for other in self.landmarks.values():
                other.remove_connection(landmark_id)
            self.last_update = time.time()

    def add_zone(self, zone: SpatialZone) -> None:
        """Add or update a zone."""
        self.zones[zone.zone_id] = zone
        self.last_update = time.time()

    def get_zone(self, zone_id: str) -> SpatialZone | None:
        """Get a zone by ID."""
        return self.zones.get(zone_id)

    def get_zone_for_landmark(self, landmark_id: str) -> SpatialZone | None:
        """Get the zone associated with a landmark."""
        for zone in self.zones.values():
            if zone.primary_landmark_id == landmark_id:
                return zone
        return None

    def add_observation(self, observation: SpatialObservation) -> None:
        """
        Add an observation, maintaining the FIFO cap.

        Args:
            observation: The observation to add
        """
        self.observations.append(observation)
        # Trim to max size
        while len(self.observations) > MAX_OBSERVATIONS:
            self.observations.pop(0)
        self.last_update = time.time()

    def get_observations_for_entity(
        self, entity_id: str, limit: int = 10
    ) -> list[SpatialObservation]:
        """Get recent observations of a specific entity."""
        matching = [obs for obs in self.observations if obs.entity_id == entity_id]
        return matching[-limit:]

    def get_observations_near_landmark(
        self, landmark_id: str, limit: int = 20
    ) -> list[SpatialObservation]:
        """Get observations near a specific landmark."""
        matching = [obs for obs in self.observations if obs.landmark_id == landmark_id]
        return matching[-limit:]

    def update_current_location(
        self, landmark_id: str | None, zone_id: str | None = None
    ) -> None:
        """
        Update the robot's current location.

        Args:
            landmark_id: Nearest landmark ID (or None if lost)
            zone_id: Current zone ID (optional, will be inferred if not provided)
        """
        self.current_landmark_id = landmark_id
        if zone_id is not None:
            self.current_zone_id = zone_id
        elif landmark_id is not None:
            # Try to infer zone from landmark
            zone = self.get_zone_for_landmark(landmark_id)
            self.current_zone_id = zone.zone_id if zone else None
        else:
            self.current_zone_id = None
        self.last_update = time.time()

    def set_home(self, landmark_id: str) -> None:
        """Set the home base landmark."""
        self.home_landmark_id = landmark_id

    @property
    def is_at_home(self) -> bool:
        """Check if currently at home base."""
        return (
            self.home_landmark_id is not None
            and self.current_landmark_id == self.home_landmark_id
        )

    @property
    def is_position_known(self) -> bool:
        """Check if current position is known (near a landmark)."""
        return self.current_landmark_id is not None

    @property
    def current_landmark(self) -> SpatialLandmark | None:
        """Get the current landmark object."""
        if self.current_landmark_id:
            return self.landmarks.get(self.current_landmark_id)
        return None

    @property
    def current_zone(self) -> SpatialZone | None:
        """Get the current zone object."""
        if self.current_zone_id:
            return self.zones.get(self.current_zone_id)
        return None

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "landmarks": {
                lid: lm.get_state() for lid, lm in self.landmarks.items()
            },
            "zones": {zid: z.get_state() for zid, z in self.zones.items()},
            "observations": [obs.get_state() for obs in self.observations],
            "current_landmark_id": self.current_landmark_id,
            "current_zone_id": self.current_zone_id,
            "home_landmark_id": self.home_landmark_id,
            "last_update": self.last_update,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SpatialMapMemory":
        """Create a SpatialMapMemory from saved state."""
        landmarks = {
            lid: SpatialLandmark.from_state(lm_state)
            for lid, lm_state in state.get("landmarks", {}).items()
        }
        zones = {
            zid: SpatialZone.from_state(z_state)
            for zid, z_state in state.get("zones", {}).items()
        }
        observations = [
            SpatialObservation.from_state(obs_state)
            for obs_state in state.get("observations", [])
        ]
        return cls(
            landmarks=landmarks,
            zones=zones,
            observations=observations,
            current_landmark_id=state.get("current_landmark_id"),
            current_zone_id=state.get("current_zone_id"),
            home_landmark_id=state.get("home_landmark_id"),
            last_update=state.get("last_update", time.time()),
        )

    def __str__(self) -> str:
        pos_str = (
            f"at {self.current_landmark_id[:8]}"
            if self.current_landmark_id
            else "position unknown"
        )
        return (
            f"SpatialMap: {len(self.landmarks)} landmarks, {len(self.zones)} zones, "
            f"{len(self.observations)} observations, {pos_str}"
        )
