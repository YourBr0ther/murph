# Murph Configuration Reference

Complete reference for all configuration options and environment variables.

## Environment Variables

### Communication

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_SERVER_HOST` | `localhost` | Server hostname or IP |
| `MURPH_SERVER_PORT` | `6765` | WebSocket server port |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `nanogpt`, or `mock` |

### NanoGPT Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOGPT_API_KEY` | (none) | API key for NanoGPT (required if using nanogpt provider) |
| `NANOGPT_BASE_URL` | `https://nano-gpt.com/api/v1` | NanoGPT API base URL |
| `NANOGPT_MODEL` | `gpt-4o-mini` | Default text model |
| `NANOGPT_VISION_MODEL` | `gpt-4o-mini` | Vision analysis model |

### Ollama Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default text model |
| `OLLAMA_VISION_MODEL` | `llama3.2-vision` | Vision analysis model |

### LLM Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_LLM_VISION_ENABLED` | `true` | Enable vision analysis |
| `MURPH_LLM_REASONING_ENABLED` | `true` | Enable behavior reasoning |
| `MURPH_SPEECH_ENABLED` | `true` | Enable speech synthesis/recognition |

### LLM Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_LLM_VISION_INTERVAL` | `3.0` | Vision analysis interval (seconds) |
| `MURPH_LLM_MAX_RPM` | `20` | Max LLM requests per minute |
| `MURPH_LLM_CACHE_TTL` | `30.0` | Cache TTL for responses (seconds) |
| `MURPH_LLM_TIMEOUT` | `10.0` | Request timeout (seconds) |
| `MURPH_LLM_REASONING_THRESHOLD` | `0.3` | Score difference threshold for LLM reasoning |

### Speech Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_TTS_MODEL` | `kokoro-82m` | Text-to-speech model (NanoGPT) |
| `MURPH_STT_MODEL` | `whisper-large-v3` | Speech-to-text model (NanoGPT) |

### Voice Commands

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_VOICE_COMMANDS_ENABLED` | `true` | Enable voice command processing |
| `MURPH_VOICE_WAKE_WORDS` | `murph,murphy` | Comma-separated wake words |
| `MURPH_VOICE_LLM_FALLBACK` | `true` | Use LLM for ambiguous commands |

### Memory Consolidation

| Variable | Default | Description |
|----------|---------|-------------|
| `MURPH_CONSOLIDATION_ENABLED` | `true` | Enable memory consolidation |
| `MURPH_CONSOLIDATION_TICK_INTERVAL` | `60.0` | Consolidation check interval (seconds) |
| `MURPH_EVENT_SUMMARIZATION_INTERVAL` | `3600.0` | Event summary interval (seconds, default 1 hour) |
| `MURPH_RELATIONSHIP_UPDATE_INTERVAL` | `86400.0` | Relationship update interval (seconds, default 1 day) |
| `MURPH_REFLECTION_PROBABILITY` | `0.3` | Chance to reflect on behavior outcome (0.0-1.0) |

---

## System Constants

These values are defined in `shared/constants.py` and cannot be changed via environment variables. Modify the source file if needed.

### Communication Timing

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SERVER_HOST` | `localhost` | Default server host |
| `DEFAULT_SERVER_PORT` | `6765` | Default WebSocket port |
| `WEBSOCKET_PING_INTERVAL` | `30` | Heartbeat interval (seconds) |
| `RECONNECT_DELAY` | `5` | Delay between reconnection attempts (seconds) |
| `MAX_RECONNECT_ATTEMPTS` | `10` | Maximum reconnection attempts |

### Processing Cycles

| Constant | Value | Description |
|----------|-------|-------------|
| `PERCEPTION_CYCLE_MS` | `100` | Sensor processing interval (ms) |
| `COGNITION_CYCLE_MS` | `200` | Behavior decision interval (ms) |
| `ACTION_CYCLE_MS` | `50` | Actuator update interval (ms) |

### Needs System

| Constant | Value | Description |
|----------|-------|-------------|
| `NEED_MIN` | `0.0` | Minimum need value (critical) |
| `NEED_MAX` | `100.0` | Maximum need value (satisfied) |
| `NEED_CRITICAL_THRESHOLD` | `20.0` | Threshold for urgent behaviors |
| `HAPPINESS_UPDATE_INTERVAL` | `1.0` | Happiness recalculation interval (seconds) |

### Motor Safety

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_MOTOR_SPEED` | `0.8` | Maximum motor speed (80% of hardware max) |
| `ACCELERATION_LIMIT` | `0.1` | Speed change per tick |
| `EMERGENCY_STOP_DISTANCE_CM` | `10` | Collision detection distance |

### IMU Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `PICKUP_ACCELERATION_THRESHOLD` | `1.5` | Pickup detection (g) |
| `BUMP_DECELERATION_THRESHOLD` | `2.0` | Bump detection (g) |
| `FREEFALL_THRESHOLD` | `0.3` | Freefall detection (g, below this = falling) |
| `SHAKE_FREQUENCY_THRESHOLD` | `3.0` | Shake detection (Hz) |

### Display

| Constant | Value | Description |
|----------|-------|-------------|
| `DISPLAY_WIDTH` | `128` | OLED display width (pixels) |
| `DISPLAY_HEIGHT` | `64` | OLED display height (pixels) |
| `ANIMATION_FPS` | `30` | Expression animation framerate |

### Face Recognition

| Constant | Value | Description |
|----------|-------|-------------|
| `FACE_DETECTION_MIN_SIZE` | `40` | Minimum face size (pixels) |
| `FACE_DETECTION_CONFIDENCE` | `0.9` | Detection confidence threshold |
| `FACE_MATCH_THRESHOLD` | `0.6` | Cosine similarity for identity match |
| `FACE_QUALITY_THRESHOLD` | `0.5` | Minimum quality to save embedding |
| `FACE_CONFIRMATION_FRAMES` | `3` | Consecutive matches for confirmation |
| `FACE_TRACK_TIMEOUT_FRAMES` | `10` | Frames before dropping track |

### Video Streaming

| Constant | Value | Description |
|----------|-------|-------------|
| `VIDEO_WIDTH` | `640` | Frame width (pixels) |
| `VIDEO_HEIGHT` | `480` | Frame height (pixels) |
| `VIDEO_FPS` | `10` | Frames per second |
| `VIDEO_BITRATE_KBPS` | `1500` | H.264 bitrate |
| `VIDEO_CODEC` | `h264` | Video codec |

### Vision Processing

| Constant | Value | Description |
|----------|-------|-------------|
| `VISION_RESULT_TTL_MS` | `500` | Vision result validity (ms) |
| `VISION_FRAME_STALE_MS` | `2000` | Frame staleness threshold (ms) |
| `VISION_FACE_DISTANCE_CALIBRATION_PX` | `160` | Face pixels at 50cm distance |

### LLM Integration

| Constant | Value | Description |
|----------|-------|-------------|
| `LLM_DEFAULT_PROVIDER` | `ollama` | Default LLM provider |
| `LLM_VISION_INTERVAL_MS` | `3000` | Vision analysis interval (ms) |
| `LLM_REQUEST_TIMEOUT_MS` | `10000` | Request timeout (ms) |
| `LLM_CACHE_TTL_MS` | `30000` | Cache TTL (ms) |
| `LLM_MAX_REQUESTS_PER_MINUTE` | `20` | Rate limit |
| `LLM_REASONING_THRESHOLD` | `0.3` | Score diff for reasoning |

### Audio/Speech

| Constant | Value | Description |
|----------|-------|-------------|
| `AUDIO_SAMPLE_RATE_STT` | `16000` | STT sample rate (Hz, Whisper requires 16kHz) |
| `AUDIO_SAMPLE_RATE_TTS` | `22050` | TTS output sample rate (Hz) |
| `AUDIO_CHUNK_SIZE` | `1024` | Samples per chunk (~64ms at 16kHz) |
| `VAD_THRESHOLD` | `0.15` | Voice activity detection threshold |
| `VAD_SILENCE_DURATION_MS` | `500` | Silence before end-of-speech (ms) |
| `TTS_MAX_TEXT_LENGTH` | `200` | Max characters per TTS request |
| `SPEECH_TIMEOUT_MS` | `5000` | TTS request timeout (ms) |
| `SPEECH_CACHE_MAX_ENTRIES` | `50` | Max cached TTS phrases |

---

## Emulator Configuration

The emulator uses `EmulatorConfig` from `emulator/config.py`. These can be passed programmatically:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server_host` | `localhost` | Server hostname |
| `server_port` | `6765` | Server WebSocket port |
| `base_speed` | `50.0` | Units per second at speed=1.0 |
| `acceleration_rate` | `0.1` | Speed change per tick |
| `deceleration_rate` | `0.15` | Braking speed (1.5x accel) |
| `wheel_base_mm` | `100.0` | Distance between wheels |
| `encoder_ticks_per_revolution` | `1440` | Encoder resolution |
| `wheel_diameter_mm` | `60.0` | Wheel diameter |
| `encoder_noise` | `0.01` | Noise fraction |
| `sensor_interval_ms` | `100` | Sensor data rate |
| `imu_noise` | `0.02` | IMU noise level |
| `physics_tick_ms` | `50` | Physics simulation rate |
| `video_enabled` | `true` | Enable webcam streaming |
| `audio_enabled` | `false` | Enable microphone capture |
| `pickup_threshold_g` | `1.5` | Pickup detection threshold |
| `bump_threshold_g` | `2.0` | Bump detection threshold |
| `shake_gyro_threshold` | `100.0` | Shake detection threshold |
| `reflex_cooldown_s` | `1.0` | Cooldown between reflexes |

---

## Configuration Validation

Both `LLMConfig` and `EmulatorConfig` have `.validate()` methods that return a list of issues:

```python
from server.llm.config import LLMConfig

config = LLMConfig.from_env()
issues = config.validate()
if issues:
    print("Configuration errors:", issues)
```

Common validation checks:
- NanoGPT API key required when using nanogpt provider
- Vision interval must be >= 0.5 seconds
- Max requests per minute must be >= 1
- Cache TTL cannot be negative
- Request timeout must be >= 1 second

---

## Recommended Configurations

### Development (Local)

```bash
MURPH_LLM_PROVIDER=mock
MURPH_SERVER_HOST=localhost
```

### Production with Ollama

```bash
MURPH_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_VISION_MODEL=llama3.2-vision
```

### Production with NanoGPT

```bash
MURPH_LLM_PROVIDER=nanogpt
NANOGPT_API_KEY=your-api-key-here
NANOGPT_MODEL=gpt-4o-mini
```

### Raspberry Pi Client

```bash
MURPH_SERVER_HOST=192.168.1.100  # Server IP
MURPH_SERVER_PORT=6765
```
