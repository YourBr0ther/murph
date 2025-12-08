# Murph - Architecture Document

## System Overview

Murph is a desktop pet robot with a thin-client architecture:
- **Raspberry Pi 5** handles real-time I/O (sensors, motors, display)
- **Home Server** hosts the AI brain (perception, cognition, expression)

This separation allows upgrading the brain without changing hardware.

## Hardware Components

| Component | Part | Interface |
|-----------|------|-----------|
| Brain | Raspberry Pi 5 (4GB+) | - |
| Camera | Pi Camera Module 3 | CSI |
| Display | 1.3" OLED SSD1306 | I2C |
| Speaker | 3W + MAX98357A | I2S |
| Microphone | SPH0645 MEMS | I2S |
| Motors | 4x N20 + DRV8833 | PWM/GPIO |
| Touch | MPR121 Capacitive | I2C |
| IMU | MPU6050 | I2C |
| Power | LiPo + BMS | - |

## Communication Protocol

### WebSocket (Commands & State)
- Persistent bidirectional connection
- Used for: motor commands, state updates, configuration
- Message format: Protocol Buffers
- Reconnection with exponential backoff

### WebRTC (Video/Audio Streaming)
- Low latency P2P streaming (~200ms)
- Hardware H.264 encoding on Pi 5
- DataChannel for ultra-low-latency control

## Cognitive Architecture

### Needs System (Sims-style)

```python
needs = {
    "energy": Need(decay_rate=0.5, critical=20),
    "curiosity": Need(decay_rate=1.5, critical=30),
    "play": Need(decay_rate=1.0, critical=25),
    "social": Need(decay_rate=0.8, critical=20),
    "affection": Need(decay_rate=0.3, critical=15),
    "comfort": Need(decay_rate=0.2, critical=30),
}
```

Each need:
- Ranges from 0 (critical) to 100 (satisfied)
- Decays over time at its decay_rate
- Triggers urgent behaviors when below critical threshold

### Personality Traits

```python
traits = {
    "playfulness": 0.0,   # -1=serious, +1=playful
    "boldness": 0.0,      # -1=timid, +1=adventurous
    "sociability": 0.0,   # -1=independent, +1=needs company
    "curiosity": 0.0,     # -1=content, +1=must explore
    "affectionate": 0.0,  # -1=aloof, +1=cuddly
    "energy_level": 0.0,  # -1=lazy, +1=hyperactive
}
```

Traits modify behavior selection scores.

### Behavior Decision (Utility AI)

Each tick, evaluate all possible behaviors:

```
score = base_value × need_modifier × personality_modifier × opportunity_bonus
```

Select highest-scoring behavior and execute via behavior tree.

### Memory System

Three tiers:
1. **Working Memory** - Current percepts, active behavior (ephemeral)
2. **Short-term Memory** - Session events, recent interactions (FIFO, ~100 items)
3. **Long-term Memory** - Person relationships, spatial map, learned preferences (SQLite)

### Person Recognition

- FaceNet embeddings (128-dim) for face recognition
- Voice embeddings for speaker identification
- Per-person relationship tracking:
  - Familiarity (0-100)
  - Trust (0-100)
  - Interaction history

## Local Behaviors (Pi-side)

These run on the Pi without server round-trip for instant response:

| Trigger | Detection | Response |
|---------|-----------|----------|
| Picked up fast | Accel > 1.5g + tilt | Startled yelp |
| Picked up gently | Slow lift + tilt | Happy chirp |
| Bump/collision | Decel > 2.0g | "Oof!" sound |
| Shaking | Oscillation > 3Hz | Dizzy expression |
| Falling | Accel < 0.3g | SCREAM |
| Set down | Downward + stable | Relief sigh |

## LLM Integration (NanoGPT)

Used for:
1. **Scene Understanding** - Describe what's in view
2. **Object Identification** - What is this thing?
3. **Complex Reasoning** - Ambiguous situations
4. **Speech Understanding** - Via Whisper API

Caching strategy to reduce API costs:
- Scene analysis cached for 5 minutes (scenes change slowly)
- Object identification cached indefinitely
- Rate limit: max 10 calls/minute

## Safety Systems

1. **Motor Safety** - Speed limits enforced on Pi, cannot be overridden
2. **Emergency Stop** - Collision detection triggers immediate stop
3. **Graceful Degradation** - Safe mode when server disconnected
4. **Privacy** - No video storage, face embeddings only

## Data Flow

```
[Sensors] → [Pi Preprocessing] → [WebRTC/WS] → [Server Perception]
                                                      ↓
[Actuators] ← [Pi Command Exec] ← [WebRTC/WS] ← [Server Cognition]
                                                      ↓
                                              [Server Expression]
```

Cycle times:
- Perception: 100ms
- Cognition: 200ms
- Action: 50ms
