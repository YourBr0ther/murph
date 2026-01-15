# Murph Robot Design Specification

**Date:** 2025-01-15
**Status:** Approved

## 1. System Overview

Murph is a voice-interactive desk robot with a WALL-E-inspired personality. The system uses a client-server architecture where the Raspberry Pi 4B handles hardware interaction and wake word detection, while a Windows PC (10.0.2.192) runs the heavy processing.

### Architecture

```
┌─────────────────────┐         WebSocket          ┌─────────────────────┐
│   Raspberry Pi 4B   │◄──────────────────────────►│   Windows Server    │
│      (Client)       │                            │   (10.0.2.192)      │
├─────────────────────┤                            ├─────────────────────┤
│ - Wake word detect  │                            │ - Speech-to-text    │
│ - Audio capture     │                            │ - LLM (Ollama)      │
│ - Audio playback    │                            │ - Text-to-speech    │
│ - Motor control     │                            │ - Intent parsing    │
│ - Ultrasonic sensor │                            │                     │
└─────────────────────┘                            └─────────────────────┘
```

### Design Principles

- **CPU-light client**: Pi only handles hardware I/O and wake word detection
- **Real-time communication**: WebSocket for bidirectional streaming
- **Free and open source only**: No paid APIs or services
- **Desk-safe operation**: Wheels locked at 25% duty cycle

---

## 2. Hardware & GPIO Pinout

### Components

| Component | Model | Connection |
|-----------|-------|------------|
| Single Board Computer | Raspberry Pi 4B | - |
| Motor Driver HAT | Cokoino Pi Power & 4WD HAT | GPIO header |
| Motors | 4x DC motors (mecanum wheels) | Via DRV8833 drivers |
| Microphone | USB microphone | USB port |
| Speaker | Any speaker/headphones | 3.5mm audio jack (built-in) |
| Ultrasonic Sensor | HC-SR04 | GPIO4 (trig), GPIO5 (echo) |

### GPIO Pin Allocation (BCM Mode)

| Pin | Function | Notes |
|-----|----------|-------|
| GPIO12 | NSLEEP1 | Motor driver 1 PWM enable |
| GPIO13 | NSLEEP2 | Motor driver 2 PWM enable |
| GPIO17 | AN11 | Motor 1 input A (left back) |
| GPIO27 | AN12 | Motor 1 input B |
| GPIO22 | BN11 | Motor 2 input A (right back) |
| GPIO23 | BN12 | Motor 2 input B |
| GPIO24 | AN21 | Motor 3 input A (left front) |
| GPIO25 | AN22 | Motor 3 input B |
| GPIO26 | BN21 | Motor 4 input A (right front) |
| GPIO16 | BN22 | Motor 4 input B |
| GPIO4 | Ultrasonic Trig | HC-SR04 trigger |
| GPIO5 | Ultrasonic Echo | HC-SR04 echo |

### Motor Mapping

```
Front of Robot
    ┌───────────┐
    │  M3   M4  │   M3 = Left Front (AN21/AN22)
    │           │   M4 = Right Front (BN21/BN22)
    │  M1   M2  │   M1 = Left Back (AN11/AN12)
    └───────────┘   M2 = Right Back (BN11/BN12)
Back of Robot
```

### Audio Output

Audio is output through the Pi's built-in 3.5mm headphone jack. Simply connect any speaker or headphones to the jack. No additional circuitry required.

---

## 3. Voice Interaction Flow

### Wake Word to Response Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Idle       │───►│  Wake Word   │───►│  "Yes sir?"  │
│  Listening   │    │  Detected    │    │   Response   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│   Execute    │◄───│   Intent     │◄───│   Listen     │
│   Command    │    │   Parsed     │    │   (3-5 sec)  │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐
│   Speak      │───► Return to Idle
│   Response   │
└──────────────┘
```

### Audio Buffer Strategy

While waiting for LLM responses, Murph plays R2-D2-style thinking sounds:
- Chirps, beeps, and boops
- Conveys "processing" without silence
- Stops immediately when response is ready

### Component Responsibilities

**Pi Client:**
- OpenWakeWord for "Murph?" detection
- PyAudio for microphone capture
- PyAudio for audio playback via 3.5mm jack
- Pre-recorded "Yes sir?" response

**Windows Server:**
- faster-whisper for speech-to-text (GPU accelerated)
- Ollama for LLM queries (configurable model)
- Piper TTS for response synthesis
- Intent parsing and command extraction

---

## 4. Commands & Intent Parsing

### Command Categories

| Category | Examples | Action |
|----------|----------|--------|
| Movement | "go forward 6 inches", "turn left" | Motor control |
| Questions | "what's the weather?", "tell me a joke" | LLM query |
| Greetings | "hello", "how are you?" | Personality response |
| System | "stop", "be quiet" | Control commands |

### Movement Commands

Distance-based movement with calibration:

```python
COMMANDS = {
    "forward": {"motors": [M1, M2, M3, M4], "direction": "forward"},
    "backward": {"motors": [M1, M2, M3, M4], "direction": "reverse"},
    "left": {"motors": "strafe_left"},   # Mecanum capability
    "right": {"motors": "strafe_right"}, # Mecanum capability
    "rotate_left": {"motors": "spin_ccw"},
    "rotate_right": {"motors": "spin_cw"},
}
```

Distance execution:
- Calibrate: measure inches traveled per second at 25% duty cycle
- Convert requested inches to runtime duration
- Execute with obstacle monitoring

### Intent Parsing (Server)

Simple keyword extraction initially:
1. Check for movement keywords first
2. Extract distance if present (default: 3 inches)
3. Fall through to LLM for questions/conversation

---

## 5. Ultrasonic Sensor & Obstacle Avoidance

### Sensor Configuration

| Parameter | Value |
|-----------|-------|
| Trigger Pin | GPIO4 |
| Echo Pin | GPIO5 |
| Detection Range | 2cm - 400cm |
| Trigger Distance | 20cm |
| Backup Distance | 2 inches |

### Avoidance Behavior

```
┌─────────────┐     Object      ┌─────────────┐
│  Moving     │────< 20cm ─────►│    Stop     │
│  Forward    │                 │   Motors    │
└─────────────┘                 └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │  Back up    │
                                │  2 inches   │
                                └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │  Announce   │
                                │  "Obstacle" │
                                └─────────────┘
```

### Implementation Notes

- Continuous monitoring during forward movement
- Check every 50ms while motors active
- No avoidance needed for backward/rotation (user responsible)
- Chirp or say "Oops!" when backing up (personality)

---

## 6. Project Structure

```
murph/
├── README.md
├── docs/
│   └── plans/
│       └── 2025-01-15-murph-design.md
├── server/
│   ├── pyproject.toml
│   └── src/
│       └── murph_server/
│           ├── __init__.py
│           ├── main.py              # FastAPI + WebSocket entry
│           ├── config.py            # Server configuration
│           ├── audio/
│           │   ├── __init__.py
│           │   ├── stt.py           # faster-whisper integration
│           │   └── tts.py           # Piper TTS integration
│           ├── llm/
│           │   ├── __init__.py
│           │   └── ollama.py        # Ollama client
│           └── intent/
│               ├── __init__.py
│               └── parser.py        # Command extraction
├── client/
│   ├── pyproject.toml
│   └── src/
│       └── murph_client/
│           ├── __init__.py
│           ├── main.py              # Main client loop
│           ├── config.py            # Client configuration
│           ├── audio/
│           │   ├── __init__.py
│           │   ├── wakeword.py      # OpenWakeWord integration
│           │   ├── capture.py       # Microphone handling
│           │   └── playback.py      # Audio output via 3.5mm jack
│           ├── motors/
│           │   ├── __init__.py
│           │   └── driver.py        # DRV8833 motor control
│           ├── sensors/
│           │   ├── __init__.py
│           │   └── ultrasonic.py    # HC-SR04 driver
│           └── sounds/
│               ├── yes_sir.wav      # Wake response
│               └── chirps/          # Thinking sounds
│                   ├── beep_01.wav
│                   ├── boop_01.wav
│                   └── ...
└── scripts/
    ├── calibrate_motors.py          # Distance calibration
    └── test_audio.py                # Audio circuit test
```

---

## 7. Dependencies

### Server (pyproject.toml)

```toml
[project]
name = "murph-server"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "faster-whisper>=1.0.0",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12>=9.0",
    "piper-tts>=1.2.0",
    "ollama>=0.3.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

### Client (pyproject.toml)

```toml
[project]
name = "murph-client"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "websockets>=12.0",
    "openwakeword>=0.6.0",
    "RPi.GPIO>=0.7.1",
    "pyaudio>=0.2.14",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]
```

### Verified via Context7

All dependencies verified against current documentation:
- FastAPI 0.115.x - WebSocket support confirmed
- OpenWakeWord 0.6.x - Custom wake word training supported
- Piper TTS - Local inference, no API keys
- Ollama - Python client with streaming
- faster-whisper - GPU acceleration with CUDA 12

---

## 8. Personality Notes

Murph embodies a WALL-E-inspired character:

- **Bright and bubbly**: Enthusiastic responses, helpful attitude
- **Caring**: Makes users feel loved and valued
- **Human-like soul**: Genuinely trying to fit in and be helpful
- **Expressive**: R2-D2-style chirps convey emotion and processing

The LLM system prompt will reinforce this personality in all responses.

---

## Appendix A: Server Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 9950X3D |
| RAM | 64 GB |
| GPU | NVIDIA RTX 3080 Ti (12GB VRAM) |
| OS | Windows |
| IP Address | 10.0.2.192 |

---

## Appendix B: Reference Code

Motor control reference from `CKK0018/Tutorial/Code/Drive_4_motors_run.py`:
- PWM frequency: 1000 Hz
- Direction control: HIGH/LOW pairs on ANx1/ANx2 pins
- Speed control: Duty cycle on NSLEEP pins
