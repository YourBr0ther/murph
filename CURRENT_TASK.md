# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Server main loop integration (CognitionOrchestrator) - 2024-12-08

## Next Feature Options (from PROGRESS.md)
1. Spatial map storage (environment awareness)
2. WebRTC video streaming integration (connect vision to Pi camera)

## Notes
Server main loop integration is complete with:

### CognitionOrchestrator (server/orchestrator.py)
Central orchestrator that wires all cognitive components together:
- Three concurrent async loops with different cycle times:
  - Perception loop (100ms/10Hz): Updates WorldContext from sensors
  - Cognition loop (200ms/5Hz): Decays needs, selects behaviors
  - Execution loop (50ms/20Hz): Ticks behavior trees, dispatches actions
- Graceful startup/shutdown with signal handling
- Pi disconnection recovery (resets state, stops behaviors)
- Behavior completion handling (applies need_effects to NeedsSystem)

### Integration Points
- PiConnectionManager → SensorProcessor (sensor callbacks)
- SensorProcessor → WorldContext (update_world_context)
- BehaviorEvaluator.select_best() → BehaviorTreeExecutor.start_behavior()
- BehaviorTreeExecutor → ActionDispatcher.create_callback() → Pi commands
- On completion: behavior.need_effects applied to NeedsSystem

### Main Entry Point (server/main.py)
- Creates CognitionOrchestrator
- Sets up SIGINT/SIGTERM signal handlers
- Runs until shutdown signal received
- Graceful cleanup on exit

### Test Coverage
- 27 new tests for orchestrator (423 total passing)
- Tests for initialization, lifecycle, callbacks, behavior completion
- Integration tests for full cognitive cycle
