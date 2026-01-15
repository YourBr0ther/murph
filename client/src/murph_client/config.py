"""Client configuration."""

# Server connection
SERVER_URL = "ws://10.0.2.192:8765"

# Motor pins (BCM mode)
MOTOR_PINS = {
    "NSLEEP1": 12,  # Motor driver 1 PWM enable
    "NSLEEP2": 13,  # Motor driver 2 PWM enable
    "AN11": 17,     # Motor 1 input A (left back)
    "AN12": 27,     # Motor 1 input B
    "BN11": 22,     # Motor 2 input A (right back)
    "BN12": 23,     # Motor 2 input B
    "AN21": 24,     # Motor 3 input A (left front)
    "AN22": 25,     # Motor 3 input B
    "BN21": 26,     # Motor 4 input A (right front)
    "BN22": 16,     # Motor 4 input B
}

# Motor settings
PWM_FREQUENCY = 1000  # Hz
MOTOR_SPEED = 0.25    # 25% duty cycle - locked for desk safety

# Motor mapping
# M1 = left back, M2 = right back, M3 = left front, M4 = right front
MOTORS = {
    "M1": {"in1": "AN11", "in2": "AN12", "enable": "NSLEEP1"},
    "M2": {"in1": "BN11", "in2": "BN12", "enable": "NSLEEP1"},
    "M3": {"in1": "AN21", "in2": "AN22", "enable": "NSLEEP2"},
    "M4": {"in1": "BN21", "in2": "BN22", "enable": "NSLEEP2"},
}

# Ultrasonic sensor pins
ULTRASONIC_TRIGGER = 4
ULTRASONIC_ECHO = 5

# Ultrasonic settings
OBSTACLE_DISTANCE_CM = 20  # Trigger distance
BACKUP_DISTANCE_INCHES = 2  # How far to back up

# Audio
AUDIO_PIN = 18  # PWM audio output

# Calibration (inches per second at 25% duty cycle)
# This needs to be measured and updated
INCHES_PER_SECOND = 2.0  # Placeholder - calibrate this!
