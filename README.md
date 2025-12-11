# Murph

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-1105%20passing-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An autonomous desktop pet robot with a Sims-like personality and Wall-E voice.

---

## About

Murph is a Raspberry Pi 5-powered desktop robot (~15-20cm) that mimics a small dog or cat. It has autonomous behaviors driven by internal needs, recognizes family members, reacts to physical interaction, and communicates with adorable robot sounds.

### Key Features

- **Autonomous Behavior** - Sims-style needs system (curiosity, play, social, energy, affection) drives behavior selection
- **Person Recognition** - Learns and remembers family members using face recognition
- **Physical Reactions** - Screams when picked up fast, purrs when petted, yelps when bumped
- **Wall-E Communication** - Beeps, chirps, and simple robot vocalizations
- **Environment Awareness** - Identifies objects and scenes via vision LLM
- **Memory System** - Three-tier memory (working, short-term, long-term) with relationship tracking
- **Voice Commands** - Wake word detection and natural language understanding

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER (Brain)                           │
│  Perception │ Cognition │ Expression │ LLM Integration      │
└─────────────────────────────────────────────────────────────┘
                    │ WebSocket + WebRTC │
┌─────────────────────────────────────────────────────────────┐
│                 RASPBERRY PI 5 (Body)                       │
│  Camera │ Mic │ Touch │ IMU │ Motors │ Speaker │ Display    │
└─────────────────────────────────────────────────────────────┘
```

The system uses a thin-client architecture:
- **Server** hosts the AI brain (perception, cognition, behavior trees, LLM integration)
- **Raspberry Pi** handles real-time I/O (sensors, motors, display, audio)
- **Communication** via WebSocket (commands/state) and WebRTC (video/audio streaming)

## Hardware Requirements

| Component | Description |
|-----------|-------------|
| Raspberry Pi 5 | 4GB+ RAM recommended |
| Pi Camera Module 3 | For vision and face recognition |
| 1.3" OLED SSD1306 | 128x64 display for expressions |
| 4x N20 Micro Motors | With DRV8833 dual H-bridge driver |
| MPU6050 IMU | 6-axis accelerometer + gyroscope |
| MPR121 Touch Sensor | 12-channel capacitive touch |
| I2S Audio | SPH0645 microphone + MAX98357A amplifier |

See [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) for complete BOM and wiring diagrams.

## Prerequisites

- **Server:** Python 3.11+, 4GB+ RAM, Ubuntu 22.04+ / macOS 13+ / Windows 11 WSL2
- **Pi:** Raspberry Pi OS Bookworm (64-bit), Python 3.11+
- **Optional:** Ollama for local LLM, or NanoGPT API key for cloud LLM

## Quick Start

### 1. Server Setup

```bash
# Clone the repository
git clone https://github.com/YourBr0ther/murph.git
cd murph

# Install dependencies
pip install poetry
poetry install

# Configure (optional - defaults work for development)
cp .env.example .env

# Run the server
poetry run python -m server.main
```

Dashboard: http://localhost:8081

### 2. Emulator (No Hardware Required)

```bash
# Run the emulator for testing without a Pi
poetry run python -m emulator
```

Emulator UI: http://localhost:8080

### 3. Raspberry Pi Setup

```bash
# On the Raspberry Pi
git clone https://github.com/YourBr0ther/murph.git
cd murph
poetry install

# Run with real hardware
poetry run python -m pi.main --host <SERVER_IP> --real-hardware
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed installation instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design patterns |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Installation and deployment guide |
| [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) | Wiring diagrams, BOM, assembly |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Environment variables reference |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | WebSocket protocol and REST API |

## Project Structure

```
murph/
├── server/             # Server brain (perception, cognition, expression)
│   ├── cognition/      # Behavior trees, needs system, memory
│   ├── perception/     # Vision processing, sensor fusion
│   ├── expression/     # Face animations, sound selection
│   └── llm/            # LLM integration (Ollama, NanoGPT)
├── pi/                 # Raspberry Pi client
│   ├── actuators/      # Motors, display, speaker
│   ├── sensors/        # IMU, touch sensor
│   └── video/          # Camera and WebRTC streaming
├── emulator/           # Virtual Pi for testing
├── shared/             # Shared code (messages, constants)
├── assets/             # Sounds, expression frames
├── tests/              # Test suite (1105 tests)
└── docs/               # Documentation
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=server --cov=pi --cov=emulator --cov=shared

# Run specific test file
poetry run pytest tests/test_server/test_behavior_evaluator.py -v
```

## Development

```bash
# Install dev dependencies
poetry install

# Format code
poetry run black .

# Lint
poetry run ruff check .

# Type check
poetry run mypy server pi shared emulator
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [py-trees](https://github.com/splintered-reality/py_trees) - Behavior tree implementation
- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) - Face recognition
- [Ollama](https://ollama.ai/) - Local LLM inference
- [aiortc](https://github.com/aiortc/aiortc) - WebRTC for Python
