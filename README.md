# Murph

An autonomous desktop pet robot with a Sims-like personality and Wall-E voice.

## Overview

Murph is a Raspberry Pi 5-powered desktop robot (~15-20cm) that mimics a small dog/cat with:

- **Autonomous behavior** driven by needs (curiosity, play, social, energy, affection)
- **Wall-E style communication** - beeps, chirps, simple robot words
- **Person recognition** - learns and remembers family members
- **Environment awareness** - identifies objects via vision LLM
- **Physical reactions** - screams when picked up fast, purrs when petted

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER (Brain)                           │
│  Perception │ Cognition │ Expression │ LLM (NanoGPT)        │
└─────────────────────────────────────────────────────────────┘
                    │ WebSocket + WebRTC │
┌─────────────────────────────────────────────────────────────┐
│                 RASPBERRY PI 5 (Body)                       │
│  Camera │ Mic │ Touch │ IMU │ Motors │ Speaker │ Display    │
└─────────────────────────────────────────────────────────────┘
```

## Hardware

- Raspberry Pi 5 (4GB+)
- Pi Camera Module 3
- 1.3" OLED SSD1306 display (face)
- 4x N20 micro gear motors + DRV8833 driver
- MPU6050 IMU (accelerometer + gyro)
- MPR121 capacitive touch sensor
- I2S microphone + speaker

## Getting Started

### Server Setup

```bash
# Clone the repository
git clone git@github.com:YourBr0ther/murph.git
cd murph

# Install dependencies
poetry install

# Run the server
poetry run python -m server.main
```

### Pi Setup

```bash
# Install dependencies (on Pi)
poetry install --with pi

# Run the Pi client
poetry run python -m pi.main
```

## Development

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### Project Structure

```
murph/
├── pi/                 # Raspberry Pi client
├── server/             # Server brain
├── shared/             # Shared code
├── proto/              # Protocol Buffers
├── assets/             # Sounds, expressions
└── tests/              # Test suite
```

### Development Workflow

1. Check `CURRENT_TASK.md` for current work
2. Implement one feature at a time
3. Write tests
4. Commit with descriptive message
5. Push to GitHub

## License

MIT
