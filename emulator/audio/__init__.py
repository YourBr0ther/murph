"""
Murph - Emulator Audio Module
Audio capture and streaming for the emulator.
"""

from .microphone import MicrophoneCapture, MockMicrophoneCapture

__all__ = ["MicrophoneCapture", "MockMicrophoneCapture"]
