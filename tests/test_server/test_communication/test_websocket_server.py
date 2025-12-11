"""Tests for PiConnectionManager WebSocket server."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.communication.websocket_server import PiConnectionManager
from shared.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from shared.messages import (
    CommandAck,
    IMUData,
    LocalTrigger,
    MessageType,
    RobotMessage,
    SensorData,
    WebRTCIceCandidate,
    WebRTCOffer,
    create_command_ack,
    create_heartbeat,
)


class TestPiConnectionManagerInit:
    """Tests for PiConnectionManager initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        manager = PiConnectionManager()

        assert manager.host == DEFAULT_SERVER_HOST
        assert manager.port == DEFAULT_SERVER_PORT
        assert manager.is_connected is False
        assert manager._on_sensor_data is None
        assert manager._on_local_trigger is None
        assert manager._on_connection_change is None

    def test_init_custom_host_port(self):
        """Test initialization with custom host and port."""
        manager = PiConnectionManager(host="192.168.1.100", port=9999)

        assert manager.host == "192.168.1.100"
        assert manager.port == 9999

    def test_init_with_callbacks(self):
        """Test initialization with all callbacks."""
        sensor_cb = MagicMock()
        trigger_cb = MagicMock()
        conn_cb = MagicMock()
        offer_cb = MagicMock()
        ice_cb = MagicMock()

        manager = PiConnectionManager(
            on_sensor_data=sensor_cb,
            on_local_trigger=trigger_cb,
            on_connection_change=conn_cb,
            on_webrtc_offer=offer_cb,
            on_webrtc_ice_candidate=ice_cb,
        )

        assert manager._on_sensor_data is sensor_cb
        assert manager._on_local_trigger is trigger_cb
        assert manager._on_connection_change is conn_cb
        assert manager._on_webrtc_offer is offer_cb
        assert manager._on_webrtc_ice_candidate is ice_cb


class TestPiConnectionManagerProperties:
    """Tests for PiConnectionManager properties."""

    def test_is_connected_false_initially(self):
        """Test is_connected returns False when no connection."""
        manager = PiConnectionManager()
        assert manager.is_connected is False

    def test_is_connected_true_when_connected(self):
        """Test is_connected returns True when connection exists."""
        manager = PiConnectionManager()
        manager._connection = MagicMock()
        assert manager.is_connected is True

    def test_host_property(self):
        """Test host property returns configured host."""
        manager = PiConnectionManager(host="10.0.0.1")
        assert manager.host == "10.0.0.1"

    def test_port_property(self):
        """Test port property returns configured port."""
        manager = PiConnectionManager(port=8765)
        assert manager.port == 8765


class TestPiConnectionManagerServer:
    """Tests for server lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_server(self, mock_websocket_server):
        """Test start() creates WebSocket server."""
        import server.communication.websocket_server as ws_module

        manager = PiConnectionManager()

        # Create a mock websockets module
        mock_ws_module = MagicMock()
        mock_ws_module.serve = AsyncMock(return_value=mock_websocket_server)

        # Patch both HAS_WEBSOCKETS and the websockets module reference
        with patch.object(ws_module, "HAS_WEBSOCKETS", True):
            with patch.object(ws_module, "websockets", mock_ws_module, create=True):
                await manager.start()

                mock_ws_module.serve.assert_called_once()
                assert manager._running is True
                assert manager._server is mock_websocket_server
                assert manager._heartbeat_task is not None

                # Clean up
                await manager.stop()

    @pytest.mark.asyncio
    async def test_start_without_websockets_library(self):
        """Test start() handles missing websockets library."""
        import server.communication.websocket_server as ws_module

        manager = PiConnectionManager()

        with patch.object(ws_module, "HAS_WEBSOCKETS", False):
            await manager.start()
            assert manager._server is None

    @pytest.mark.asyncio
    async def test_stop_closes_connection(self, mock_websocket, mock_websocket_server):
        """Test stop() closes connection and server."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket
        manager._server = mock_websocket_server
        manager._running = True
        manager._heartbeat_task = asyncio.create_task(asyncio.sleep(100))

        await manager.stop()

        assert manager._running is False
        mock_websocket.close.assert_called_once()
        mock_websocket_server.close.assert_called_once()
        mock_websocket_server.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_heartbeat_task(self, mock_websocket_server):
        """Test stop() cancels heartbeat task gracefully."""
        manager = PiConnectionManager()
        manager._server = mock_websocket_server
        manager._running = True

        # Create a long-running heartbeat task
        manager._heartbeat_task = asyncio.create_task(asyncio.sleep(100))

        await manager.stop()

        assert manager._heartbeat_task.cancelled() or manager._heartbeat_task.done()


class TestPiConnectionManagerConnection:
    """Tests for connection handling."""

    @pytest.mark.asyncio
    async def test_handle_connection_accepts_first(self, mock_websocket):
        """Test first connection is accepted."""
        conn_callback = MagicMock()
        manager = PiConnectionManager(on_connection_change=conn_callback)

        # Make websocket async iteration end immediately (empty async generator)
        async def empty_async_iter():
            return
            yield  # Make it a generator

        mock_websocket.__aiter__ = lambda self: empty_async_iter()

        await manager._handle_connection(mock_websocket)

        conn_callback.assert_any_call(True)
        conn_callback.assert_any_call(False)

    @pytest.mark.asyncio
    async def test_handle_connection_replaces_existing(self, mock_websocket):
        """Test second connection replaces existing (closes old, accepts new)."""
        manager = PiConnectionManager()
        old_connection = MagicMock()
        old_connection.close = AsyncMock()
        manager._connection = old_connection

        # Make websocket async iteration end immediately
        async def empty_async_iter():
            return
            yield  # Make it a generator

        mock_websocket.__aiter__ = lambda self: empty_async_iter()

        await manager._handle_connection(mock_websocket)

        # Old connection should be closed
        old_connection.close.assert_called_once_with(1001, "Replaced by new connection")
        # New connection should NOT be closed
        mock_websocket.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_connection_cleans_up_on_close(self, mock_websocket):
        """Test connection is cleaned up when closed."""
        conn_callback = MagicMock()
        manager = PiConnectionManager(on_connection_change=conn_callback)

        # Make websocket async iteration end immediately
        async def empty_async_iter():
            return
            yield  # Make it a generator

        mock_websocket.__aiter__ = lambda self: empty_async_iter()

        await manager._handle_connection(mock_websocket)

        # Connection should be cleaned up
        assert manager._connection is None
        # Disconnect callback should fire
        conn_callback.assert_called_with(False)


class TestPiConnectionManagerMessages:
    """Tests for message handling."""

    @pytest.mark.asyncio
    async def test_handle_sensor_data_message(self, sensor_data_msg):
        """Test SENSOR_DATA callback is invoked."""
        sensor_callback = MagicMock()
        manager = PiConnectionManager(on_sensor_data=sensor_callback)

        await manager._handle_message(sensor_data_msg.serialize())

        sensor_callback.assert_called_once()
        call_arg = sensor_callback.call_args[0][0]
        assert isinstance(call_arg, SensorData)

    @pytest.mark.asyncio
    async def test_handle_local_trigger_message(self, local_trigger_msg):
        """Test LOCAL_TRIGGER callback is invoked."""
        trigger_callback = MagicMock()
        manager = PiConnectionManager(on_local_trigger=trigger_callback)

        await manager._handle_message(local_trigger_msg.serialize())

        trigger_callback.assert_called_once()
        call_arg = trigger_callback.call_args[0][0]
        assert isinstance(call_arg, LocalTrigger)
        assert call_arg.trigger_name == "bump"

    @pytest.mark.asyncio
    async def test_handle_heartbeat_updates_timestamp(self, heartbeat_msg):
        """Test HEARTBEAT updates last_heartbeat timestamp."""
        manager = PiConnectionManager()
        manager._last_heartbeat = 0

        await manager._handle_message(heartbeat_msg.serialize())

        assert manager._last_heartbeat > 0

    @pytest.mark.asyncio
    async def test_handle_webrtc_offer_callback(self, webrtc_offer_msg):
        """Test WEBRTC_OFFER callback is invoked."""
        offer_callback = MagicMock()
        manager = PiConnectionManager(on_webrtc_offer=offer_callback)

        await manager._handle_message(webrtc_offer_msg.serialize())

        offer_callback.assert_called_once()
        call_arg = offer_callback.call_args[0][0]
        assert isinstance(call_arg, WebRTCOffer)

    @pytest.mark.asyncio
    async def test_handle_webrtc_ice_callback(self, webrtc_ice_msg):
        """Test WEBRTC_ICE_CANDIDATE callback is invoked."""
        ice_callback = MagicMock()
        manager = PiConnectionManager(on_webrtc_ice_candidate=ice_callback)

        await manager._handle_message(webrtc_ice_msg.serialize())

        ice_callback.assert_called_once()
        call_arg = ice_callback.call_args[0][0]
        assert isinstance(call_arg, WebRTCIceCandidate)

    @pytest.mark.asyncio
    async def test_handle_malformed_message_logged(self):
        """Test malformed message is logged, not crashed."""
        manager = PiConnectionManager()

        # Should not raise
        await manager._handle_message(b"invalid json{{{")

    @pytest.mark.asyncio
    async def test_handle_message_no_callback(self, sensor_data_msg):
        """Test message handling without callback doesn't crash."""
        manager = PiConnectionManager()  # No callbacks

        # Should not raise
        await manager._handle_message(sensor_data_msg.serialize())


class TestPiConnectionManagerAck:
    """Tests for acknowledgment handling."""

    def test_handle_ack_sets_future_result(self):
        """Test ack sets future result for matching sequence ID."""
        manager = PiConnectionManager()
        future = asyncio.get_event_loop().create_future()
        manager._pending_acks[1] = future

        ack = CommandAck(sequence_id=1, success=True)
        manager._handle_ack(ack)

        assert future.done()
        assert future.result() == ack

    def test_handle_ack_unknown_sequence_ignored(self):
        """Test ack for unknown sequence ID is ignored."""
        manager = PiConnectionManager()

        # Should not raise
        ack = CommandAck(sequence_id=999, success=True)
        manager._handle_ack(ack)

    def test_handle_ack_already_done_future_ignored(self):
        """Test ack doesn't set result if future already done."""
        manager = PiConnectionManager()
        future = asyncio.get_event_loop().create_future()
        future.set_result(None)  # Already done
        manager._pending_acks[1] = future

        # Should not raise
        ack = CommandAck(sequence_id=1, success=True)
        manager._handle_ack(ack)


class TestPiConnectionManagerSendCommand:
    """Tests for send_command method."""

    @pytest.mark.asyncio
    async def test_send_command_not_connected_returns_false(self, motor_command):
        """Test send_command returns False when not connected."""
        manager = PiConnectionManager()

        result = await manager.send_command(motor_command)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_command_assigns_sequence_id(
        self, mock_websocket, motor_command
    ):
        """Test send_command assigns sequence ID and timestamp."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket
        manager._sequence_id = 5

        # Set up mock to return ack quickly
        async def send_and_ack(data):
            # Simulate receiving ack
            ack = CommandAck(sequence_id=6, success=True)
            manager._handle_ack(ack)

        mock_websocket.send = send_and_ack

        result = await manager.send_command(motor_command, timeout=0.5)

        assert motor_command.sequence_id == 6
        assert motor_command.timestamp_ms > 0

    @pytest.mark.asyncio
    async def test_send_command_returns_true_on_ack(
        self, mock_websocket, motor_command
    ):
        """Test send_command returns True on successful ack."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket

        async def send_and_ack(data):
            ack = CommandAck(sequence_id=1, success=True)
            manager._handle_ack(ack)

        mock_websocket.send = send_and_ack

        result = await manager.send_command(motor_command, timeout=0.5)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_command_returns_false_on_timeout(
        self, mock_websocket, motor_command
    ):
        """Test send_command returns False on timeout."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket

        # Don't send ack - will timeout
        result = await manager.send_command(motor_command, timeout=0.1)

        assert result is False
        # Pending ack should be cleaned up
        assert len(manager._pending_acks) == 0

    @pytest.mark.asyncio
    async def test_send_command_sequence_ids_increment(
        self, mock_websocket, motor_command
    ):
        """Test sequence IDs increment properly."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket

        # Send multiple commands (will timeout but that's ok)
        await manager.send_command(motor_command, timeout=0.05)
        first_seq = motor_command.sequence_id

        await manager.send_command(motor_command, timeout=0.05)
        second_seq = motor_command.sequence_id

        assert second_seq == first_seq + 1


class TestPiConnectionManagerSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_not_connected_returns_false(self):
        """Test send_message returns False when not connected."""
        manager = PiConnectionManager()

        msg = RobotMessage(message_type=MessageType.WEBRTC_ANSWER)
        result = await manager.send_message(msg)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_websocket):
        """Test send_message returns True on success."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket

        msg = RobotMessage(message_type=MessageType.WEBRTC_ANSWER)
        result = await manager.send_message(msg)

        assert result is True
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_exception_returns_false(self, mock_websocket):
        """Test send_message returns False on exception."""
        manager = PiConnectionManager()
        manager._connection = mock_websocket
        mock_websocket.send = AsyncMock(side_effect=Exception("Send failed"))

        msg = RobotMessage(message_type=MessageType.WEBRTC_ANSWER)
        result = await manager.send_message(msg)

        assert result is False


class TestPiConnectionManagerStatus:
    """Tests for get_status method."""

    def test_get_status_returns_dict(self):
        """Test get_status returns dict with expected keys."""
        manager = PiConnectionManager(host="10.0.0.1", port=8765)

        status = manager.get_status()

        assert isinstance(status, dict)
        assert "connected" in status
        assert "host" in status
        assert "port" in status
        assert "pending_commands" in status
        assert "last_heartbeat" in status

    def test_get_status_reflects_connection_state(self):
        """Test get_status reflects current connection state."""
        manager = PiConnectionManager()

        status1 = manager.get_status()
        assert status1["connected"] is False

        manager._connection = MagicMock()
        status2 = manager.get_status()
        assert status2["connected"] is True

    def test_get_status_shows_pending_commands(self):
        """Test get_status shows pending command count."""
        manager = PiConnectionManager()
        manager._pending_acks[1] = MagicMock()
        manager._pending_acks[2] = MagicMock()

        status = manager.get_status()

        assert status["pending_commands"] == 2
