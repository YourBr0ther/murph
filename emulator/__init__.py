"""
Murph - Emulator
Web-based robot simulator for testing without hardware.
"""

from .config import EmulatorConfig
from .virtual_pi import VirtualPi, VirtualRobotState

# Optional import of FastAPI app (requires fastapi to be installed)
try:
    from .app import create_app
except ImportError:
    create_app = None

# Optional import of video module (requires aiortc/opencv to be installed)
try:
    from .video import EmulatorVideoStreamer, MockWebcamCamera, WebcamCamera
except ImportError:
    EmulatorVideoStreamer = None
    MockWebcamCamera = None
    WebcamCamera = None

__all__ = [
    "create_app",
    "EmulatorConfig",
    "VirtualPi",
    "VirtualRobotState",
    "EmulatorVideoStreamer",
    "MockWebcamCamera",
    "WebcamCamera",
]
