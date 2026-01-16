# client/src/murph_client/config.py
import os
from dataclasses import dataclass


@dataclass
class GPIOPins:
    NSLEEP1: int = 12
    M1_A: int = 17
    M1_B: int = 27
    M2_A: int = 22
    M2_B: int = 23
    NSLEEP2: int = 13
    M3_A: int = 24
    M3_B: int = 25
    M4_A: int = 26
    M4_B: int = 16
    ULTRASONIC_TRIG: int = 4
    ULTRASONIC_ECHO: int = 5


@dataclass
class ClientConfig:
    server_host: str = "10.0.2.192"
    server_port: int = 8765
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_device_index: int = 1  # USB mic is typically device 1 on Pi
    audio_device_sample_rate: int = 44100  # USB mic native rate
    max_duty_cycle: int = 100
    pwm_frequency: int = 1000
    obstacle_threshold_cm: float = 20.0
    backup_distance_inches: float = 2.0

    def __post_init__(self):
        self.server_host = os.getenv("MURPH_SERVER_HOST", self.server_host)
        self.server_port = int(os.getenv("MURPH_SERVER_PORT", self.server_port))
        self.audio_device_index = int(os.getenv("MURPH_AUDIO_DEVICE", self.audio_device_index))

    @property
    def server_uri(self) -> str:
        return f"ws://{self.server_host}:{self.server_port}/ws"
