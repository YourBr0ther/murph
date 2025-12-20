"""
Murph - Shared Constants
Constants used by both Pi client and server brain.
"""

# Communication
DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 6765
WEBSOCKET_PING_INTERVAL = 30  # seconds
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10

# Timing
PERCEPTION_CYCLE_MS = 100  # How often to process sensor data
COGNITION_CYCLE_MS = 200  # How often to run behavior decisions
ACTION_CYCLE_MS = 50  # How often to update actuators

# Needs System
NEED_MIN = 0.0
NEED_MAX = 100.0
NEED_CRITICAL_THRESHOLD = 20.0
HAPPINESS_UPDATE_INTERVAL = 1.0  # seconds

# Motor Safety
MAX_MOTOR_SPEED = 0.8  # 80% of max
ACCELERATION_LIMIT = 0.1  # per tick
EMERGENCY_STOP_DISTANCE_CM = 10

# IMU Thresholds
PICKUP_ACCELERATION_THRESHOLD = 1.5  # g
BUMP_DECELERATION_THRESHOLD = 2.0  # g
FREEFALL_THRESHOLD = 0.3  # g (below this = falling)
SHAKE_FREQUENCY_THRESHOLD = 3.0  # Hz

# Display
DISPLAY_WIDTH = 128
DISPLAY_HEIGHT = 64
ANIMATION_FPS = 30

# Face Recognition
FACE_DETECTION_MIN_SIZE = 40  # Minimum face size in pixels
FACE_DETECTION_CONFIDENCE = 0.9  # Detection confidence threshold
FACE_MATCH_THRESHOLD = 0.6  # Cosine similarity threshold for matches
FACE_QUALITY_THRESHOLD = 0.5  # Minimum quality to save embeddings
FACE_CONFIRMATION_FRAMES = 3  # Consecutive matches needed for identity confirmation
FACE_TRACK_TIMEOUT_FRAMES = 10  # Frames without detection before dropping track

# Video Streaming
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 10
VIDEO_BITRATE_KBPS = 1500
VIDEO_CODEC = "h264"

# Vision Processing
VISION_RESULT_TTL_MS = 500  # How long vision results are valid
VISION_FRAME_STALE_MS = 2000  # Consider frame stale after this time
VISION_FACE_DISTANCE_CALIBRATION_PX = 160  # Face height pixels at 50cm distance

# LLM Integration
LLM_DEFAULT_PROVIDER = "ollama"  # Default LLM provider
LLM_VISION_INTERVAL_MS = 3000  # How often to run LLM vision analysis
LLM_REQUEST_TIMEOUT_MS = 10000  # Timeout for LLM requests
LLM_CACHE_TTL_MS = 30000  # Cache TTL for LLM responses
LLM_MAX_REQUESTS_PER_MINUTE = 20  # Rate limit for LLM requests
LLM_REASONING_THRESHOLD = 0.3  # Score diff threshold for behavior reasoning

# Speech/Audio
AUDIO_SAMPLE_RATE_STT = 16000  # Hz for STT (Whisper requires 16kHz)
AUDIO_SAMPLE_RATE_TTS = 22050  # Hz for TTS output
AUDIO_CHUNK_SIZE = 1024  # Samples per chunk (~64ms at 16kHz)
VAD_THRESHOLD = 0.15  # Voice activity detection threshold
VAD_SILENCE_DURATION_MS = 500  # Silence duration before end-of-speech
TTS_MAX_TEXT_LENGTH = 200  # Max characters per TTS request
SPEECH_TIMEOUT_MS = 5000  # TTS request timeout
SPEECH_CACHE_MAX_ENTRIES = 50  # Max cached TTS phrases
