# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Core behaviors - Hardware integration with emulator - 2024-12-08

## Next Feature Options (from PROGRESS.md)
1. Spatial map storage (environment awareness)
2. WebRTC video streaming integration (connect vision to Pi camera)
3. Server main loop integration (wire everything together)

## Notes
Hardware integration is complete with:

### Message Types (shared/messages/)
- Protocol Buffer-compatible message definitions
- Command types: MotorCommand, TurnCommand, ExpressionCommand, SoundCommand, ScanCommand, StopCommand
- Sensor types: IMUData, TouchData, MotorState
- Control types: Heartbeat, CommandAck, LocalTrigger, RobotMessage envelope
- Serialization via JSON (binary-ready)

### Pi Hardware Abstraction (pi/actuators/, pi/sensors/)
- Abstract base classes for all hardware
- Mock implementations for development without hardware:
  - MockMotorController (differential drive simulation)
  - MockDisplayController (expression tracking)
  - MockAudioController (sound playback simulation)
  - MockIMUSensor (with pickup/bump/shake simulation)
  - MockTouchSensor (electrode simulation)
- Real hardware placeholders (DRV8833, SSD1306, MPU6050, MPR121)

### WebSocket Communication (server/communication/, pi/communication/)
- PiConnectionManager: Server-side connection handling with ack tracking
- ServerConnection: Pi-side client with auto-reconnection
- CommandHandler: Routes commands to actuators
- ActionDispatcher: Bridges behavior tree actions to WebSocket commands

### Sensor Processing (server/perception/)
- SensorProcessor: Converts IMU/touch data to WorldContext updates
- Integrates with existing WorldContext for behavior decisions

### Web-Based Emulator (emulator/)
- FastAPI application with WebSocket state streaming
- VirtualPi: Full hardware simulation with position tracking
- Web UI for visualizing robot state and simulating sensors
- Manual controls for touch, pickup, bump, shake events

### Local Behaviors (pi/local_behaviors/)
- ReflexController: Fast IMU-triggered responses (< 50ms)
- Reflexes: picked_up, bump, falling, shake, set_down
- Configurable thresholds and cooldowns

### Pi Main Loop (pi/main.py)
- Full client implementation wiring hardware, communication, and sensors
- Command-line args for host/port/real-hardware mode
- Graceful shutdown with signal handlers

### Test Coverage
- 78 new tests for hardware integration (396 total passing)
- Tests for messages, actuators, sensors, and reflexes
