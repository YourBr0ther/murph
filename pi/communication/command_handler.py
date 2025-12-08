"""
Murph - Pi Command Handler
Routes incoming commands to appropriate actuators.
"""

from __future__ import annotations

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
from pi.actuators import AudioController, DisplayController, MotorController

logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Routes commands from server to appropriate actuators.

    This is the bridge between WebSocket messages and hardware control.
    """

    def __init__(
        self,
        motors: MotorController | None = None,
        display: DisplayController | None = None,
        speaker: AudioController | None = None,
    ) -> None:
        """
        Initialize command handler with actuators.

        Args:
            motors: Motor controller for movement
            display: Display controller for expressions
            speaker: Audio controller for sounds
        """
        self._motors = motors
        self._display = display
        self._speaker = speaker

    def set_motors(self, motors: MotorController) -> None:
        """Set motor controller."""
        self._motors = motors

    def set_display(self, display: DisplayController) -> None:
        """Set display controller."""
        self._display = display

    def set_speaker(self, speaker: AudioController) -> None:
        """Set audio controller."""
        self._speaker = speaker

    async def handle_command(self, command: Command) -> bool:
        """
        Handle an incoming command.

        Args:
            command: The command to execute

        Returns:
            True if command executed successfully
        """
        payload = command.payload

        if payload is None:
            logger.warning(f"Command {command.sequence_id} has no payload")
            return False

        try:
            if isinstance(payload, MotorCommand):
                return await self._handle_motor(payload)

            elif isinstance(payload, TurnCommand):
                return await self._handle_turn(payload)

            elif isinstance(payload, ExpressionCommand):
                return await self._handle_expression(payload)

            elif isinstance(payload, SoundCommand):
                return await self._handle_sound(payload)

            elif isinstance(payload, ScanCommand):
                return await self._handle_scan(payload)

            elif isinstance(payload, StopCommand):
                return await self._handle_stop()

            else:
                logger.warning(f"Unknown command type: {type(payload)}")
                return False

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False

    async def _handle_motor(self, cmd: MotorCommand) -> bool:
        """Handle motor movement command."""
        if not self._motors:
            logger.warning("No motor controller available")
            return False

        direction_map = {
            MotorDirection.STOP: "stop",
            MotorDirection.FORWARD: "forward",
            MotorDirection.BACKWARD: "backward",
            MotorDirection.LEFT: "left",
            MotorDirection.RIGHT: "right",
        }

        direction = direction_map.get(cmd.direction, "stop")
        await self._motors.move(direction, cmd.speed, cmd.duration_ms)

        logger.debug(
            f"Motor command: {direction} at {cmd.speed:.2f} "
            f"for {cmd.duration_ms:.0f}ms"
        )
        return True

    async def _handle_turn(self, cmd: TurnCommand) -> bool:
        """Handle turn command."""
        if not self._motors:
            logger.warning("No motor controller available")
            return False

        await self._motors.turn(cmd.angle_degrees, cmd.speed)

        logger.debug(
            f"Turn command: {cmd.angle_degrees:.0f}deg at {cmd.speed:.2f}"
        )
        return True

    async def _handle_expression(self, cmd: ExpressionCommand) -> bool:
        """Handle expression display command."""
        if not self._display:
            logger.warning("No display controller available")
            return False

        await self._display.set_expression(cmd.expression_name)

        logger.debug(f"Expression command: {cmd.expression_name}")
        return True

    async def _handle_sound(self, cmd: SoundCommand) -> bool:
        """Handle sound playback command."""
        if not self._speaker:
            logger.warning("No audio controller available")
            return False

        await self._speaker.play_sound(cmd.sound_name, cmd.volume)

        logger.debug(f"Sound command: {cmd.sound_name} at {cmd.volume:.2f}")
        return True

    async def _handle_scan(self, cmd: ScanCommand) -> bool:
        """Handle scanning command (turn to look around)."""
        if not self._motors:
            logger.warning("No motor controller available")
            return False

        # Map scan types to turn angles
        scan_angles = {
            ScanType.QUICK: 90.0,
            ScanType.PARTIAL: 180.0,
            ScanType.FULL: 360.0,
        }

        angle = scan_angles.get(cmd.scan_type, 180.0)

        # Execute scan as a slow turn
        await self._motors.turn(angle, 0.3)

        logger.debug(f"Scan command: {cmd.scan_type.name} ({angle:.0f}deg)")
        return True

    async def _handle_stop(self) -> bool:
        """Handle emergency stop command."""
        if self._motors:
            await self._motors.stop()

        if self._speaker:
            await self._speaker.stop_sound()

        logger.debug("Stop command executed")
        return True

    def get_status(self) -> dict[str, Any]:
        """Get handler status."""
        return {
            "motors_available": self._motors is not None,
            "display_available": self._display is not None,
            "speaker_available": self._speaker is not None,
        }
