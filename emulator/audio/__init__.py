"""
Murph - Emulator Audio Module
Audio capture and streaming for the emulator.
"""

from .microphone import BaseMicrophoneCapture, MicrophoneCapture, MockMicrophoneCapture
from .track import MicrophoneAudioTrack

__all__ = [
    "BaseMicrophoneCapture",
    "MicrophoneCapture",
    "MockMicrophoneCapture",
    "MicrophoneAudioTrack",
]
