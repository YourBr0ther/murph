"""
Murph - Emulator Video Module
USB webcam capture and WebRTC video streaming.
"""

from .streamer import EmulatorVideoStreamer
from .webcam import MockWebcamCamera, WebcamCamera

__all__ = [
    "WebcamCamera",
    "MockWebcamCamera",
    "EmulatorVideoStreamer",
]
