# Murph

A voice-interactive desk robot with a WALL-E-inspired personality.

Murph listens for his name, responds with "Yes sir?", and helps with questions or moves around on command. He chirps and beeps while thinking, backs up when he detects obstacles, and genuinely tries to make your day better.

## Architecture

```
┌─────────────────────┐         WebSocket          ┌─────────────────────┐
│   Raspberry Pi 4B   │◄──────────────────────────►│   Windows Server    │
│      (Client)       │                            │                     │
├─────────────────────┤                            ├─────────────────────┤
│ • Wake word detect  │                            │ • Speech-to-text    │
│ • Audio capture     │                            │ • LLM (Ollama)      │
│ • Audio playback    │                            │ • Text-to-speech    │
│ • Motor control     │                            │ • Intent parsing    │
│ • Obstacle sensing  │                            │                     │
└─────────────────────┘                            └─────────────────────┘
```

## Hardware

| Component | Model |
|-----------|-------|
| Brain | Raspberry Pi 4B |
| Chassis | Cokoino 4WD with mecanum wheels |
| Motor Driver | Pi Power & 4WD HAT (2x DRV8833) |
| Microphone | USB microphone |
| Speaker | 1W 8 Ohm via PWM + RC filter |
| Distance Sensor | HC-SR04 ultrasonic |

## Features

- **Wake word activation**: Say "Murph?" and he responds "Yes sir?"
- **Voice commands**: Ask questions or give movement instructions
- **Thinking sounds**: R2-D2-style chirps while processing
- **Obstacle avoidance**: Backs up 2 inches when detecting objects within 20cm
- **Distance-based movement**: "Go forward 6 inches", "Turn left"

## Quick Start

### Server (Windows)

```bash
cd server
pip install -e .
python -m murph_server
```

### Client (Raspberry Pi)

```bash
cd client
pip install -e .
python -m murph_client
```

## Project Structure

```
murph/
├── server/                 # Windows server (heavy processing)
│   └── src/murph_server/
│       ├── main.py         # FastAPI + WebSocket
│       ├── audio/          # STT (faster-whisper) + TTS (Piper)
│       ├── llm/            # Ollama integration
│       └── intent/         # Command parsing
├── client/                 # Raspberry Pi client
│   └── src/murph_client/
│       ├── main.py         # Main loop
│       ├── audio/          # Wake word + capture + playback
│       ├── motors/         # DRV8833 control
│       ├── sensors/        # HC-SR04 ultrasonic
│       └── sounds/         # Audio files
└── scripts/                # Calibration and testing
```

## Configuration

### Server

Set the Ollama model and other options in `server/src/murph_server/config.py`:

```python
OLLAMA_MODEL = "llama3.2"  # Or any model you have installed
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
```

### Client

Configure the server address in `client/src/murph_client/config.py`:

```python
SERVER_URL = "ws://10.0.2.192:8765"
MOTOR_SPEED = 0.25  # Locked at 25% for desk safety
```

## Commands

| Say | Murph does |
|-----|------------|
| "Go forward 6 inches" | Moves forward 6 inches |
| "Turn left" | Rotates 90 degrees left |
| "What's the weather?" | Asks LLM and speaks response |
| "Tell me a joke" | Tells you a joke |
| "Stop" | Stops all motors |

## Requirements

### Server
- Python 3.10+
- NVIDIA GPU with CUDA 12 (for faster-whisper)
- Ollama installed with at least one model

### Client
- Raspberry Pi 4B
- Python 3.10+
- USB microphone
- Cokoino Pi Power & 4WD HAT

## License

MIT

## Acknowledgments

- Personality inspired by WALL-E
- Thinking sounds inspired by R2-D2
- Motor control based on Cokoino documentation
