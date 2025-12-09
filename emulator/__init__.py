"""
Murph - Emulator
Web-based robot simulator for testing without hardware.
"""

from .virtual_pi import VirtualPi, VirtualRobotState

# Optional import of FastAPI app (requires fastapi to be installed)
try:
    from .app import create_app
    __all__ = [
        "create_app",
        "VirtualPi",
        "VirtualRobotState",
    ]
except ImportError:
    # FastAPI not installed, only export VirtualPi components
    __all__ = [
        "VirtualPi",
        "VirtualRobotState",
    ]
