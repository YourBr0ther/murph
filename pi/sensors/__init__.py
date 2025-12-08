"""
Murph - Sensor Module
Hardware sensors for robot input (IMU, touch, etc.).
"""

from .base import Sensor, StreamingSensor
from .imu import MockIMUSensor, MPU6050IMUSensor
from .touch import MockTouchSensor, MPR121TouchSensor

__all__ = [
    # Base classes
    "Sensor",
    "StreamingSensor",
    # Mock implementations
    "MockIMUSensor",
    "MockTouchSensor",
    # Real hardware implementations
    "MPU6050IMUSensor",
    "MPR121TouchSensor",
]
