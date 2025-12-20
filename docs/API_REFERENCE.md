# Murph API Reference

Complete protocol and API documentation for Murph robot communication.

## WebSocket Protocol

### Connection

```
ws://{host}:{port}
```

Default: `ws://localhost:6765`

### Message Envelope

All messages use the `RobotMessage` envelope:

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 10,
  "payload": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_ms` | int | Unix timestamp in milliseconds |
| `message_type` | int | Message type enum value |
| `payload` | object | Type-specific payload |

### Message Types

| Type | Value | Direction | Description |
|------|-------|-----------|-------------|
| `COMMAND` | 10 | Server → Pi | Motor/expression/sound commands |
| `COMMAND_ACK` | 11 | Pi → Server | Command acknowledgment |
| `SENSOR_DATA` | 20 | Pi → Server | IMU, touch, motor state |
| `LOCAL_TRIGGER` | 21 | Pi → Server | Reflex event notifications |
| `HEARTBEAT` | 30 | Bidirectional | Connection health check |
| `PI_STATUS` | 31 | Pi → Server | Hardware status report |
| `WEBRTC_OFFER` | 40 | Pi → Server | WebRTC SDP offer |
| `WEBRTC_ANSWER` | 41 | Server → Pi | WebRTC SDP answer |
| `WEBRTC_ICE_CANDIDATE` | 42 | Bidirectional | ICE candidate exchange |
| `SPEECH_COMMAND` | 51 | Server → Pi | TTS audio playback |
| `VOICE_ACTIVITY` | 52 | Pi → Server | VAD state changes |
| `AUDIO_DATA` | 53 | Pi → Server | Raw audio chunks |
| `SIMULATED_TRANSCRIPTION` | 54 | Emulator → Server | Simulated voice text |

---

## Command Messages (Server → Pi)

### MotorCommand

Control robot movement.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 10,
  "payload": {
    "sequence_id": 1,
    "timestamp_ms": 1702234567890,
    "payload": {
      "type": "motor",
      "direction": 1,
      "speed": 0.5,
      "duration_ms": 1000
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `direction` | int | 0=STOP, 1=FORWARD, 2=BACKWARD, 3=LEFT, 4=RIGHT |
| `speed` | float | 0.0-1.0 normalized speed |
| `duration_ms` | float | Duration in ms (0 = until stopped) |

### TurnCommand

Rotate robot by angle.

```json
{
  "payload": {
    "sequence_id": 2,
    "payload": {
      "type": "turn",
      "angle_degrees": 90.0,
      "speed": 0.5
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `angle_degrees` | float | Rotation angle (positive = clockwise) |
| `speed` | float | 0.0-1.0 normalized speed |

### ExpressionCommand

Set facial expression on OLED display.

```json
{
  "payload": {
    "sequence_id": 3,
    "payload": {
      "type": "expression",
      "expression_name": "happy"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `expression_name` | string | Expression: `neutral`, `happy`, `sad`, `curious`, `sleepy`, `startled`, `dizzy`, `love` |

### SoundCommand

Play a sound effect.

```json
{
  "payload": {
    "sequence_id": 4,
    "payload": {
      "type": "sound",
      "sound_name": "chirp",
      "volume": 0.8
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sound_name` | string | Sound file name (without extension) |
| `volume` | float | 0.0-1.0 volume level |

### ScanCommand

Perform scanning motion.

```json
{
  "payload": {
    "sequence_id": 5,
    "payload": {
      "type": "scan",
      "scan_type": 1
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `scan_type` | int | 0=QUICK (90°), 1=PARTIAL (180°), 2=FULL (360°) |

### StopCommand

Emergency stop all motors.

```json
{
  "payload": {
    "sequence_id": 6,
    "payload": {
      "type": "stop"
    }
  }
}
```

### SpeechCommand

Play synthesized speech audio.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 51,
  "payload": {
    "sequence_id": 7,
    "payload": {
      "type": "speech",
      "audio_data": "UklGRi4AAABXQVZFZm10...",
      "audio_format": "wav",
      "sample_rate": 22050,
      "volume": 1.0,
      "emotion": "happy",
      "text": "Hello there!"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_data` | string | Base64-encoded audio bytes |
| `audio_format` | string | `wav`, `mp3`, or `pcm` |
| `sample_rate` | int | Sample rate in Hz (default 22050) |
| `volume` | float | 0.0-1.0 volume |
| `emotion` | string | Emotional tone for playback |
| `text` | string | Original text (for debugging) |

---

## Sensor Messages (Pi → Server)

### IMUData

Accelerometer and gyroscope readings.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 20,
  "payload": {
    "timestamp_ms": 1702234567890,
    "payload": {
      "type": "imu",
      "accel_x": 0.01,
      "accel_y": 0.02,
      "accel_z": -1.0,
      "gyro_x": 0.5,
      "gyro_y": -0.3,
      "gyro_z": 0.1,
      "temperature": 25.5
    }
  }
}
```

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `accel_x` | float | g | X-axis acceleration |
| `accel_y` | float | g | Y-axis acceleration |
| `accel_z` | float | g | Z-axis acceleration (default -1.0 at rest) |
| `gyro_x` | float | deg/s | X-axis angular velocity |
| `gyro_y` | float | deg/s | Y-axis angular velocity |
| `gyro_z` | float | deg/s | Z-axis angular velocity |
| `temperature` | float | °C | IMU temperature |

### TouchData

Capacitive touch sensor state.

```json
{
  "payload": {
    "timestamp_ms": 1702234567890,
    "payload": {
      "type": "touch",
      "touched_electrodes": [0, 2, 5],
      "is_touched": true
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `touched_electrodes` | int[] | List of active electrode indices (0-11) |
| `is_touched` | bool | True if any electrode is touched |

### MotorState

Motor encoder feedback.

```json
{
  "payload": {
    "timestamp_ms": 1702234567890,
    "payload": {
      "type": "motor_state",
      "left_speed": 0.45,
      "right_speed": 0.48,
      "is_moving": true
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `left_speed` | float | Left motor speed (0.0-1.0) |
| `right_speed` | float | Right motor speed (0.0-1.0) |
| `is_moving` | bool | True if any motor is active |

### LocalTrigger

Reflex event notification.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 21,
  "payload": {
    "trigger_name": "picked_up",
    "intensity": 0.8,
    "timestamp_ms": 1702234567890
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trigger_name` | string | Event: `picked_up`, `set_down`, `bump`, `shake`, `falling`, `petted` |
| `intensity` | float | 0.0-1.0 event intensity |

### VoiceActivityMessage

Voice activity detection state.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 52,
  "payload": {
    "is_speaking": true,
    "audio_level": 0.45,
    "timestamp_ms": 1702234567890
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_speaking` | bool | True when voice detected |
| `audio_level` | float | RMS level 0.0-1.0 |

### AudioDataMessage

Streaming audio from microphone.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 53,
  "payload": {
    "audio_data": "AAAA...",
    "sample_rate": 16000,
    "channels": 1,
    "chunk_index": 42,
    "is_final": false,
    "timestamp_ms": 1702234567890
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_data` | string | Base64-encoded PCM audio |
| `sample_rate` | int | 16000 Hz for Whisper |
| `channels` | int | 1 (mono) |
| `chunk_index` | int | Sequence number |
| `is_final` | bool | End of utterance (VAD silence) |

---

## Connection Messages

### Heartbeat

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 30,
  "payload": {
    "timestamp_ms": 1702234567890,
    "sequence": 42
  }
}
```

### PiStatus

Hardware status report.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 31,
  "payload": {
    "cpu_temp": 45.2,
    "cpu_usage": 0.35,
    "memory_usage": 0.42,
    "hardware_ok": true
  }
}
```

### CommandAck

Command acknowledgment.

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 11,
  "payload": {
    "sequence_id": 1,
    "success": true,
    "error_message": ""
  }
}
```

---

## WebRTC Signaling

### WebRTCOffer (Pi → Server)

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 40,
  "payload": {
    "sdp": "v=0\r\no=- 123456789...",
    "type": "offer"
  }
}
```

### WebRTCAnswer (Server → Pi)

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 41,
  "payload": {
    "sdp": "v=0\r\no=- 987654321...",
    "type": "answer"
  }
}
```

### WebRTCIceCandidate (Bidirectional)

```json
{
  "timestamp_ms": 1702234567890,
  "message_type": 42,
  "payload": {
    "candidate": "candidate:1 1 UDP 2130706431...",
    "sdp_mid": "0",
    "sdp_mline_index": 0
  }
}
```

---

## Dashboard REST API

The dashboard runs on port 6081 by default.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard web UI |
| GET | `/api/state` | Current robot state |
| GET | `/api/needs` | Current needs values |
| GET | `/api/behaviors` | Available behaviors |
| POST | `/api/behavior` | Trigger a behavior |
| GET | `/api/memory` | Memory system status |
| GET | `/api/ws` | WebSocket connection for real-time updates |

### GET /api/state

Returns current robot state.

```json
{
  "connected": true,
  "current_behavior": "idle",
  "expression": "neutral",
  "is_moving": false,
  "needs": {
    "energy": 85.2,
    "curiosity": 62.1,
    "play": 45.8,
    "social": 71.3,
    "affection": 55.0,
    "comfort": 90.1
  },
  "person_detected": false,
  "uptime_seconds": 3600
}
```

### POST /api/behavior

Trigger a specific behavior.

```json
{
  "behavior": "play_excited"
}
```

---

## Factory Helper Functions

The `shared/messages/types.py` module provides factory functions for creating messages:

```python
from shared.messages.types import (
    create_motor_command,
    create_turn_command,
    create_expression_command,
    create_sound_command,
    create_stop_command,
    create_sensor_data,
    create_command_ack,
    create_local_trigger,
    create_heartbeat,
    create_webrtc_offer,
    create_webrtc_answer,
    create_webrtc_ice_candidate,
    create_speech_command,
    create_voice_activity,
    create_audio_data,
    create_simulated_transcription,
    MotorDirection,
    ScanType,
)

# Create a motor command
msg = create_motor_command(
    direction=MotorDirection.FORWARD,
    speed=0.5,
    duration_ms=1000,
    sequence_id=1
)

# Serialize for transmission
wire_bytes = msg.serialize()

# Deserialize received message
received = RobotMessage.deserialize(wire_bytes)
```

---

## Error Handling

### Connection Errors

- WebSocket disconnection triggers automatic reconnection with exponential backoff
- Max reconnection attempts: 10
- Base delay: 5 seconds

### Command Errors

Commands that fail return `CommandAck` with `success=false`:

```json
{
  "sequence_id": 1,
  "success": false,
  "error_message": "Motor driver not responding"
}
```

### Rate Limiting

- Sensor data: ~10 messages/second
- Commands: No hard limit, but respect motor safety
- LLM requests: 20 requests/minute (configurable)
