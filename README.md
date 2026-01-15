# Murph

A voice-interactive desk robot with a WALL-E-inspired personality and a vintage radio voice.

Say "Hey Murph" and he'll beep to let you know he's listening. Ask questions, give movement commands, or just chat - he'll respond with a warm, old-timey radio voice (think Alastor from Hazbin Hotel). He backs up when he detects obstacles and genuinely tries to make your day better.

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
| Speaker | Any speaker/headphones via 3.5mm jack |
| Distance Sensor | HC-SR04 ultrasonic |

## Features

- **Wake word activation**: Say "Hey Murph" - low beep = recording, high beep = processing
- **Vintage radio voice**: Responses have that warm, old-timey radio effect
- **Voice commands**: Ask questions or give movement instructions
- **Auto-reconnect**: Client automatically reconnects if server restarts
- **Obstacle avoidance**: Backs up 2 inches when detecting objects within 20cm
- **Distance-based movement**: "Go forward 6 inches", "Turn left"

## Quick Start

### Server (Windows)

```bash
cd C:\scripts\murph
python -m venv venv
venv\Scripts\activate
cd server
pip install -e .
python -m murph_server
```

### Client (Raspberry Pi)

```bash
cd ~/murph
python3 -m venv venv
source venv/bin/activate
cd client
pip install -e .
python -m murph_client
```

> **Note:** You'll need to activate the venv each session:
> - Windows: `venv\Scripts\activate`
> - Pi: `source ~/murph/venv/bin/activate`

## Project Structure

```
murph/
├── server/                 # Windows server (heavy processing)
│   └── src/murph_server/
│       ├── main.py         # FastAPI + WebSocket
│       ├── audio/          # STT (faster-whisper) + TTS (Piper + radio effect)
│       ├── llm/            # Ollama integration
│       └── intent/         # Command parsing
├── client/                 # Raspberry Pi client
│   └── src/murph_client/
│       ├── main.py         # Main loop with auto-reconnect
│       ├── audio/          # Wake word + capture + playback
│       ├── models/         # Custom wake word model (hey_murph.onnx)
│       ├── motors/         # DRV8833 control
│       └── sensors/        # HC-SR04 ultrasonic
└── docs/plans/             # Design docs and implementation plans
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
