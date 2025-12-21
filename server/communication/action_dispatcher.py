"""
Murph - Action Dispatcher
Bridges behavior tree actions to Pi via WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from shared.messages import (
    Command,
    ExpressionCommand,
    MessageType,
    MotorCommand,
    MotorDirection,
    RobotMessage,
    ScanCommand,
    ScanType,
    SoundCommand,
    SpeechCommand,
    StopCommand,
    TurnCommand,
)
from .websocket_server import PiConnectionManager

if TYPE_CHECKING:
    from server.llm.services import SpeechService

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """
    Converts behavior tree action callbacks to Protocol Buffer commands
    and sends them to the Pi via WebSocket.

    This is the glue between BehaviorTreeExecutor and the Pi client.
    The executor calls dispatch() when an action node executes.
    """

    def __init__(
        self,
        connection: PiConnectionManager,
        speech_service: SpeechService | None = None,
    ) -> None:
        """
        Initialize the action dispatcher.

        Args:
            connection: WebSocket connection manager
            speech_service: Optional speech service for TTS
        """
        self._connection = connection
        self._speech_service = speech_service

    def set_speech_service(self, speech_service: SpeechService) -> None:
        """Set the speech service for TTS synthesis."""
        self._speech_service = speech_service

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
        action = params.get("action", "")

        # Handle speak action specially - requires async TTS synthesis
        if action == "speak":
            text = params.get("text", "")
            emotion = params.get("emotion", "neutral")
            asyncio.create_task(self._synthesize_and_send(text, emotion))
            return True  # Return immediately, synthesis happens async

        command = self._build_command(params)
        if command is None:
            logger.debug(f"Action '{action_name}' doesn't require Pi command")
            return True  # Some actions (like Wait) don't need Pi commands

        if not self._connection.is_connected:
            logger.warning(f"No Pi connection - action '{action_name}' dropped")
            return False

        return await self._connection.send_command(command)

    async def _synthesize_and_send(self, text: str, emotion: str) -> None:
        """
        Synthesize speech via TTS and send audio to Pi.

        Args:
            text: Text to speak
            emotion: Emotional tone for voice modulation
        """
        if self._speech_service is None:
            logger.warning("Speech service not available - falling back to sound")
            # Fall back to pre-recorded sound if available
            fallback_cmd = Command(
                payload=SoundCommand(sound_name="alert", volume=1.0)
            )
            if self._connection.is_connected:
                await self._connection.send_command(fallback_cmd)
            return

        # Synthesize speech
        audio_data = await self._speech_service.synthesize(text, emotion)
        if audio_data is None:
            logger.warning(f"TTS synthesis failed for: {text[:50]}")
            return

        # Encode and send to Pi
        audio_b64 = self._speech_service.encode_audio_base64(audio_data)

        speech_cmd = SpeechCommand(
            audio_data=audio_b64,
            audio_format="mp3",  # NanoGPT OpenAI-compatible TTS returns MP3
            sample_rate=24000,  # MP3 from tts-1 is 24kHz
            volume=1.0,
            emotion=emotion,
            text=text[:50],  # Truncate for logging
        )

        msg = RobotMessage(
            message_type=MessageType.SPEECH_COMMAND,
            payload=Command(payload=speech_cmd),
        )

        if self._connection.is_connected:
            await self._connection.send_message(msg)
            logger.debug(f"Speech sent to Pi: '{text[:30]}...'")
        else:
            logger.warning("No Pi connection - speech dropped")

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

        elif action == "speak":
            # Speech is handled asynchronously via _synthesize_and_send
            # Return None here; the dispatch method handles it specially
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
