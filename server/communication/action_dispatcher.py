"""
Murph - Action Dispatcher
Bridges behavior tree actions to Pi via WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from shared.messages import (
    Command,
    ExpressionCommand,
    MotorCommand,
    MotorDirection,
    ScanCommand,
    ScanType,
    SoundCommand,
    StopCommand,
    TurnCommand,
)
from .websocket_server import PiConnectionManager

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """
    Converts behavior tree action callbacks to Protocol Buffer commands
    and sends them to the Pi via WebSocket.

    This is the glue between BehaviorTreeExecutor and the Pi client.
    The executor calls dispatch() when an action node executes.
    """

    def __init__(self, connection: PiConnectionManager) -> None:
        """
        Initialize the action dispatcher.

        Args:
            connection: WebSocket connection manager
        """
        self._connection = connection

    async def dispatch(self, action_name: str, params: dict[str, Any]) -> bool:
        """
        Dispatch an action to the Pi.

        This is the callback used by BehaviorTreeExecutor.

        Args:
            action_name: Action node name (e.g., "Move(forward, 0.5, 1.0s)")
            params: Action parameters dict from ActionNode._params

        Returns:
            True if command was acknowledged successfully
        """
        command = self._build_command(params)
        if command is None:
            logger.debug(f"Action '{action_name}' doesn't require Pi command")
            return True  # Some actions (like Wait) don't need Pi commands

        if not self._connection.is_connected:
            logger.warning(f"No Pi connection - action '{action_name}' dropped")
            return False

        return await self._connection.send_command(command)

    def _build_command(self, params: dict[str, Any]) -> Command | None:
        """
        Build a Command from action parameters.

        Args:
            params: Parameters from ActionNode._params

        Returns:
            Command object or None if action doesn't need Pi command
        """
        action = params.get("action", "")

        if action == "move":
            return Command(
                payload=MotorCommand(
                    direction=self._map_direction(params.get("direction", "stop")),
                    speed=params.get("speed", 0.5),
                    duration_ms=params.get("duration", 0) * 1000,  # seconds to ms
                )
            )

        elif action == "turn":
            return Command(
                payload=TurnCommand(
                    angle_degrees=params.get("angle", 0),
                    speed=params.get("speed", 0.5),
                )
            )

        elif action == "set_expression":
            return Command(
                payload=ExpressionCommand(
                    expression_name=params.get("expression_name", "neutral"),
                )
            )

        elif action == "play_sound":
            return Command(
                payload=SoundCommand(
                    sound_name=params.get("sound_name", ""),
                    volume=params.get("volume", 1.0),
                )
            )

        elif action == "scan":
            return Command(
                payload=ScanCommand(
                    scan_type=self._map_scan_type(params.get("scan_type", "partial")),
                )
            )

        elif action == "stop":
            return Command(payload=StopCommand())

        elif action == "wait":
            # Wait is handled locally by behavior tree, no Pi command needed
            return None

        else:
            logger.warning(f"Unknown action type: {action}")
            return None

    @staticmethod
    def _map_direction(direction: str) -> MotorDirection:
        """Map string direction to enum."""
        mapping = {
            "stop": MotorDirection.STOP,
            "forward": MotorDirection.FORWARD,
            "backward": MotorDirection.BACKWARD,
            "left": MotorDirection.LEFT,
            "right": MotorDirection.RIGHT,
        }
        return mapping.get(direction, MotorDirection.STOP)

    @staticmethod
    def _map_scan_type(scan_type: str) -> ScanType:
        """Map string scan type to enum."""
        mapping = {
            "quick": ScanType.QUICK,
            "partial": ScanType.PARTIAL,
            "full": ScanType.FULL,
        }
        return mapping.get(scan_type, ScanType.PARTIAL)

    def create_callback(self) -> callable:
        """
        Create a synchronous callback for BehaviorTreeExecutor.

        The executor expects a sync callback, so we wrap async dispatch
        in asyncio.create_task.

        Returns:
            Callback function with signature (str, dict) -> bool
        """
        def callback(action_name: str, params: dict[str, Any]) -> bool:
            """Synchronous wrapper that queues the async dispatch."""
            try:
                loop = asyncio.get_running_loop()
                # Queue the dispatch but don't wait for result
                # Return True optimistically; errors will be logged
                loop.create_task(self.dispatch(action_name, params))
                return True
            except RuntimeError:
                # No event loop running
                logger.warning("No event loop - cannot dispatch action")
                return False

        return callback
