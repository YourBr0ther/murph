"""Tests for ActionDispatcher."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.communication.action_dispatcher import ActionDispatcher
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


class TestActionDispatcherInit:
    """Tests for ActionDispatcher initialization."""

    def test_init_with_connection(self, mock_connection_manager):
        """Test initialization with connection manager."""
        dispatcher = ActionDispatcher(mock_connection_manager)

        assert dispatcher._connection is mock_connection_manager
        assert dispatcher._speech_service is None

    def test_init_with_speech_service(
        self, mock_connection_manager, mock_speech_service
    ):
        """Test initialization with speech service."""
        dispatcher = ActionDispatcher(
            mock_connection_manager, speech_service=mock_speech_service
        )

        assert dispatcher._speech_service is mock_speech_service


class TestActionDispatcherBuildCommand:
    """Tests for _build_command method."""

    def test_build_move_command(self, mock_connection_manager):
        """Test building move command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {
            "action": "move",
            "direction": "forward",
            "speed": 0.7,
            "duration": 2.0,  # seconds
        }

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, MotorCommand)
        assert cmd.payload.direction == MotorDirection.FORWARD
        assert cmd.payload.speed == 0.7
        assert cmd.payload.duration_ms == 2000  # converted to ms

    def test_build_turn_command(self, mock_connection_manager):
        """Test building turn command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {
            "action": "turn",
            "angle": 90,
            "speed": 0.5,
        }

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, TurnCommand)
        assert cmd.payload.angle_degrees == 90
        assert cmd.payload.speed == 0.5

    def test_build_expression_command(self, mock_connection_manager):
        """Test building set_expression command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {
            "action": "set_expression",
            "expression_name": "happy",
        }

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, ExpressionCommand)
        assert cmd.payload.expression_name == "happy"

    def test_build_sound_command(self, mock_connection_manager):
        """Test building play_sound command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {
            "action": "play_sound",
            "sound_name": "beep",
            "volume": 0.8,
        }

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, SoundCommand)
        assert cmd.payload.sound_name == "beep"
        assert cmd.payload.volume == 0.8

    def test_build_scan_command(self, mock_connection_manager):
        """Test building scan command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {
            "action": "scan",
            "scan_type": "full",
        }

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, ScanCommand)
        assert cmd.payload.scan_type == ScanType.FULL

    def test_build_stop_command(self, mock_connection_manager):
        """Test building stop command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "stop"}

        cmd = dispatcher._build_command(params)

        assert isinstance(cmd, Command)
        assert isinstance(cmd.payload, StopCommand)

    def test_build_wait_returns_none(self, mock_connection_manager):
        """Test wait action returns None (no Pi command)."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "wait", "duration": 1.0}

        cmd = dispatcher._build_command(params)

        assert cmd is None

    def test_build_speak_returns_none(self, mock_connection_manager):
        """Test speak action returns None (async path)."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "speak", "text": "hello"}

        cmd = dispatcher._build_command(params)

        assert cmd is None

    def test_build_unknown_action_returns_none(self, mock_connection_manager):
        """Test unknown action returns None with warning."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "unknown_action"}

        cmd = dispatcher._build_command(params)

        assert cmd is None


class TestActionDispatcherDirectionMapping:
    """Tests for _map_direction method."""

    def test_map_direction_forward(self):
        """Test forward direction mapping."""
        assert ActionDispatcher._map_direction("forward") == MotorDirection.FORWARD

    def test_map_direction_backward(self):
        """Test backward direction mapping."""
        assert ActionDispatcher._map_direction("backward") == MotorDirection.BACKWARD

    def test_map_direction_left(self):
        """Test left direction mapping."""
        assert ActionDispatcher._map_direction("left") == MotorDirection.LEFT

    def test_map_direction_right(self):
        """Test right direction mapping."""
        assert ActionDispatcher._map_direction("right") == MotorDirection.RIGHT

    def test_map_direction_stop(self):
        """Test stop direction mapping."""
        assert ActionDispatcher._map_direction("stop") == MotorDirection.STOP

    def test_map_direction_unknown_defaults_to_stop(self):
        """Test unknown direction defaults to STOP."""
        assert ActionDispatcher._map_direction("invalid") == MotorDirection.STOP


class TestActionDispatcherScanTypeMapping:
    """Tests for _map_scan_type method."""

    def test_map_scan_type_quick(self):
        """Test quick scan type mapping."""
        assert ActionDispatcher._map_scan_type("quick") == ScanType.QUICK

    def test_map_scan_type_partial(self):
        """Test partial scan type mapping."""
        assert ActionDispatcher._map_scan_type("partial") == ScanType.PARTIAL

    def test_map_scan_type_full(self):
        """Test full scan type mapping."""
        assert ActionDispatcher._map_scan_type("full") == ScanType.FULL

    def test_map_scan_type_unknown_defaults_to_partial(self):
        """Test unknown scan type defaults to PARTIAL."""
        assert ActionDispatcher._map_scan_type("invalid") == ScanType.PARTIAL


class TestActionDispatcherDispatch:
    """Tests for dispatch method."""

    @pytest.mark.asyncio
    async def test_dispatch_not_connected_returns_false(self, mock_connection_manager):
        """Test dispatch returns False when not connected."""
        mock_connection_manager.is_connected = False
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "forward", "speed": 0.5}

        result = await dispatcher.dispatch("Move(forward)", params)

        assert result is False

    @pytest.mark.asyncio
    async def test_dispatch_builds_and_sends_command(self, mock_connection_manager):
        """Test dispatch builds and sends command successfully."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "forward", "speed": 0.5}

        result = await dispatcher.dispatch("Move(forward)", params)

        assert result is True
        mock_connection_manager.send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_wait_returns_true(self, mock_connection_manager):
        """Test dispatch with wait action returns True without sending."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "wait", "duration": 1.0}

        result = await dispatcher.dispatch("Wait(1.0s)", params)

        assert result is True
        mock_connection_manager.send_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_action_returns_true(self, mock_connection_manager):
        """Test dispatch with unknown action returns True."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "unknown"}

        result = await dispatcher.dispatch("Unknown", params)

        assert result is True


class TestActionDispatcherSpeak:
    """Tests for speak action dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_speak_returns_immediately(
        self, mock_connection_manager, mock_speech_service
    ):
        """Test speak action returns True immediately."""
        dispatcher = ActionDispatcher(
            mock_connection_manager, speech_service=mock_speech_service
        )
        params = {"action": "speak", "text": "hello", "emotion": "happy"}

        result = await dispatcher.dispatch("Speak", params)

        assert result is True

    @pytest.mark.asyncio
    async def test_synthesize_and_send_with_speech_service(
        self, mock_connection_manager, mock_speech_service
    ):
        """Test _synthesize_and_send calls speech service."""
        dispatcher = ActionDispatcher(
            mock_connection_manager, speech_service=mock_speech_service
        )

        await dispatcher._synthesize_and_send("hello world", "happy")

        mock_speech_service.synthesize.assert_called_once_with("hello world", "happy")
        mock_speech_service.encode_audio_base64.assert_called_once()
        mock_connection_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_and_send_without_speech_service(
        self, mock_connection_manager
    ):
        """Test _synthesize_and_send falls back to sound without speech service."""
        dispatcher = ActionDispatcher(mock_connection_manager, speech_service=None)

        await dispatcher._synthesize_and_send("hello", "neutral")

        # Should fall back to sound command
        mock_connection_manager.send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_and_send_handles_synthesis_failure(
        self, mock_connection_manager, mock_speech_service
    ):
        """Test _synthesize_and_send handles synthesis failure gracefully."""
        mock_speech_service.synthesize = AsyncMock(return_value=None)
        dispatcher = ActionDispatcher(
            mock_connection_manager, speech_service=mock_speech_service
        )

        # Should not raise
        await dispatcher._synthesize_and_send("hello", "neutral")

        # Should not try to send message
        mock_connection_manager.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_and_send_not_connected(
        self, mock_connection_manager, mock_speech_service
    ):
        """Test _synthesize_and_send does nothing if not connected."""
        mock_connection_manager.is_connected = False
        dispatcher = ActionDispatcher(
            mock_connection_manager, speech_service=mock_speech_service
        )

        await dispatcher._synthesize_and_send("hello", "neutral")

        # Should not send
        mock_connection_manager.send_message.assert_not_called()


class TestActionDispatcherCallback:
    """Tests for create_callback method."""

    def test_create_callback_returns_callable(self, mock_connection_manager):
        """Test create_callback returns a callable."""
        dispatcher = ActionDispatcher(mock_connection_manager)

        callback = dispatcher.create_callback()

        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_queues_dispatch_task(self, mock_connection_manager):
        """Test callback queues dispatch task."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        callback = dispatcher.create_callback()

        # Run within event loop
        result = callback("Move(forward)", {"action": "move", "direction": "forward"})

        assert result is True
        # Give the task a moment to execute
        await asyncio.sleep(0.01)

    def test_callback_returns_false_without_event_loop(self, mock_connection_manager):
        """Test callback returns False when no event loop running."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        callback = dispatcher.create_callback()

        # Get a new event loop context that doesn't have a running loop
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            result = callback("Move", {"action": "move"})

        assert result is False


class TestActionDispatcherMoveDirections:
    """Tests for all move directions."""

    @pytest.mark.asyncio
    async def test_dispatch_move_forward(self, mock_connection_manager):
        """Test move forward command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "forward", "speed": 0.5, "duration": 1}

        await dispatcher.dispatch("Move", params)

        call_args = mock_connection_manager.send_command.call_args[0][0]
        assert call_args.payload.direction == MotorDirection.FORWARD

    @pytest.mark.asyncio
    async def test_dispatch_move_backward(self, mock_connection_manager):
        """Test move backward command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "backward", "speed": 0.5, "duration": 1}

        await dispatcher.dispatch("Move", params)

        call_args = mock_connection_manager.send_command.call_args[0][0]
        assert call_args.payload.direction == MotorDirection.BACKWARD

    @pytest.mark.asyncio
    async def test_dispatch_move_left(self, mock_connection_manager):
        """Test move left command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "left", "speed": 0.5, "duration": 1}

        await dispatcher.dispatch("Move", params)

        call_args = mock_connection_manager.send_command.call_args[0][0]
        assert call_args.payload.direction == MotorDirection.LEFT

    @pytest.mark.asyncio
    async def test_dispatch_move_right(self, mock_connection_manager):
        """Test move right command."""
        dispatcher = ActionDispatcher(mock_connection_manager)
        params = {"action": "move", "direction": "right", "speed": 0.5, "duration": 1}

        await dispatcher.dispatch("Move", params)

        call_args = mock_connection_manager.send_command.call_args[0][0]
        assert call_args.payload.direction == MotorDirection.RIGHT
