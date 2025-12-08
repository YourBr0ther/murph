# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Spatial map storage (environment awareness) - 2024-12-08

## Next Feature Options (from PROGRESS.md)
1. WebRTC video streaming integration (connect vision to Pi camera)
2. Spatial navigation behaviors (use spatial map for exploration)

## Notes
Spatial map storage is complete with:

### Spatial Types (server/cognition/memory/spatial_types.py)
Four dataclasses for spatial awareness:
- **SpatialLandmark**: Recognizable reference points (charging_station, edge, corner, home_base)
- **SpatialZone**: Behavioral areas with safety scores (safe, dangerous, play_area)
- **SpatialObservation**: Where objects/people were seen relative to landmarks
- **SpatialMapMemory**: Container combining landmarks, zones, and observations

### Database Models (server/storage/models.py)
Three new SQLAlchemy models for persistence:
- SpatialLandmarkModel: Stores landmarks with connections graph
- SpatialZoneModel: Stores zones with safety/familiarity scores
- SpatialObservationModel: Stores location observations

### Persistence Criteria
- Landmarks: confidence >= 0.6 OR visits >= 5 OR critical type (charging_station, home_base, edge)
- Zones: familiarity >= 0.5 OR critical type (charging_zone, edge_zone)
- Observations: entity is important AND confidence >= 0.5

### Integration Points
- MemorySystem.spatial_map: Contains current spatial state
- MemorySystem.record_landmark/zone/observation(): Create spatial entities
- MemorySystem.update_current_location(): Track robot position
- WorldContext: 7 new spatial fields and triggers
- LongTermMemory: load_spatial_map() and sync_spatial_map() for persistence

### WorldContext Spatial Triggers
- at_home, at_charger, in_safe_zone, in_danger_zone
- position_known, position_lost, near_landmark

### Test Coverage
- 56 new tests for spatial memory (458 total passing)
- Tests for all spatial types, database operations, and MemorySystem integration
