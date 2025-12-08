"""
Murph - Shared Constants
Constants used by both Pi client and server brain.
"""

# Communication
DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 8765
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
