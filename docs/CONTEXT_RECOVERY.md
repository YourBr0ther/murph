# Murph - Context Recovery Guide

**Read this file first after any context clear.**

## Quick Start (Read These Files in Order)

1. **This file** (`docs/CONTEXT_RECOVERY.md`) - You're here
2. **`PROGRESS.md`** - See what phase we're in and what's done
3. **`CURRENT_TASK.md`** - See what we're actively working on
4. **`git status`** - Check for uncommitted work
5. **`git log --oneline -10`** - See recent commits

## Project Overview

**Murph** is an autonomous desktop pet robot (~15-20cm) that mimics a small dog/cat with:
- Sims-like personality traits and needs system
- Wall-E style voice (beeps, chirps, simple words)
- Person recognition and relationship tracking
- Environment awareness via vision LLM (NanoGPT)
- Physical reactions via IMU (screams when picked up fast, etc.)

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER (Brain)                           │
│  - Perception: Vision, Audio, Face Recognition              │
│  - Cognition: Utility AI, Needs System, Memory              │
│  - Expression: Voice Synthesis, Face Animation              │
│  - LLM: NanoGPT integration for reasoning                   │
└─────────────────────────────────────────────────────────────┘
                    │ WebSocket + WebRTC │
┌─────────────────────────────────────────────────────────────┐
│                 RASPBERRY PI 5 (Body)                       │
│  - Sensors: Camera, Mic, Touch, IMU                         │
│  - Actuators: 4 Motors, Speaker, OLED Display               │
│  - Local Behaviors: Safety fallbacks, IMU reactions         │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
1. **Thin client** - Pi handles I/O, server handles AI
2. **Utility AI** - Behaviors selected by scoring needs/opportunities
3. **Local reactions** - IMU events processed on Pi for instant response
4. **NanoGPT** - Pay-as-you-go LLM for vision and reasoning

## Key Files by Area

### Core Structure
- `pyproject.toml` - Dependencies and project config
- `shared/constants.py` - Shared constants and thresholds

### Pi Client (`pi/`)
- `pi/main.py` - Entry point
- `pi/communication/` - WebSocket/WebRTC clients
- `pi/sensors/imu.py` - Accelerometer for physical events
- `pi/local_behaviors/physical_reactions.py` - Instant reactions

### Server Brain (`server/`)
- `server/main.py` - Entry point
- `server/cognition/needs/needs_system.py` - Sims-like needs
- `server/cognition/behavior/utility_system.py` - Decision making
- `server/perception/vision/face_recognizer.py` - Person recognition
- `server/llm/nanogpt_client.py` - LLM integration

### Workflow Files
- `PROGRESS.md` - Overall progress tracking
- `CURRENT_TASK.md` - Current feature being worked on
- `ARCHITECTURE.md` - Detailed architecture decisions

## Development Workflow

**The Golden Rule:** One feature → Test → Commit → Push → Clear context → Repeat

1. Check `CURRENT_TASK.md` status
2. If "In Progress" → Continue implementation
3. If "Testing" → Run tests, then commit
4. If "Ready to Commit" → Commit and push
5. If empty → Start next feature from `PROGRESS.md`

## Common Commands

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Run Pi client (on Pi)
poetry run python -m pi.main

# Run server (on server)
poetry run python -m server.main
```
