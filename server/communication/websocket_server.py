"""
Murph - WebSocket Server
Manages WebSocket connection to the Pi client.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any

from shared.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from shared.messages import (
    Command,
    CommandAck,
    LocalTrigger,
    MessageType,
    RobotMessage,
    SensorData,
    WebRTCOffer,
    WebRTCIceCandidate,
    create_command_ack,
    create_heartbeat,
)

logger = logging.getLogger(__name__)


class PiConnectionManager:
    """
    Manages WebSocket connection to the Pi client.

    Handles:
    - Accepting connections from Pi
    - Sending commands and waiting for acknowledgments
    - Receiving sensor data and local triggers
    - Connection health monitoring via heartbeat
    """

    def __init__(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
        on_sensor_data: Callable[[SensorData], None] | None = None,
        on_local_trigger: Callable[[LocalTrigger], None] | None = None,
        on_connection_change: Callable[[bool], None] | None = None,
        on_webrtc_offer: Callable[[WebRTCOffer], Any] | None = None,
        on_webrtc_ice_candidate: Callable[[WebRTCIceCandidate], Any] | None = None,
    ) -> None:
        """
        Initialize the connection manager.

        Args:
            host: Host to listen on
            port: Port to listen on
            on_sensor_data: Callback for incoming sensor data
            on_local_trigger: Callback for local trigger events
            on_connection_change: Callback when connection state changes
            on_webrtc_offer: Callback for WebRTC SDP offer from Pi
            on_webrtc_ice_candidate: Callback for WebRTC ICE candidates from Pi
        """
        self._host = host
        self._port = port
        self._on_sensor_data = on_sensor_data
        self._on_local_trigger = on_local_trigger
        self._on_connection_change = on_connection_change
        self._on_webrtc_offer = on_webrtc_offer
        self._on_webrtc_ice_candidate = on_webrtc_ice_candidate

        self._server = None
        self._connection: WebSocketServerProtocol | None = None
        self._sequence_id = 0
        self._pending_acks: dict[int, asyncio.Future[CommandAck]] = {}
        self._running = False

        # Heartbeat tracking
        self._last_heartbeat: float = 0
        self._heartbeat_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        """Check if Pi is currently connected."""
        return self._connection is not None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.error("websockets library not available")
            return

        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")

        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._connection:
            await self._connection.close()
            self._connection = None

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped")

    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle an incoming Pi connection."""
        # Only allow one connection at a time
        if self._connection is not None:
            logger.warning("Rejecting connection - Pi already connected")
            await websocket.close(1008, "Already connected")
            return

        self._connection = websocket
        self._last_heartbeat = time.time()
        logger.info(f"Pi connected from {websocket.remote_address}")

        if self._on_connection_change:
            self._on_connection_change(True)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_message(message)
                else:
                    logger.warning(f"Unexpected message type: {type(message)}")

        except websockets.ConnectionClosed as e:
            logger.warning(f"Pi disconnected: {e}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self._connection = None
            if self._on_connection_change:
                self._on_connection_change(False)
            logger.info("Pi connection closed")

    async def _handle_message(self, raw: bytes) -> None:
        """Parse and dispatch incoming message."""
        try:
            msg = RobotMessage.deserialize(raw)

            if msg.message_type == MessageType.SENSOR_DATA:
                if isinstance(msg.payload, SensorData) and self._on_sensor_data:
                    self._on_sensor_data(msg.payload)

            elif msg.message_type == MessageType.LOCAL_TRIGGER:
                if isinstance(msg.payload, LocalTrigger) and self._on_local_trigger:
                    self._on_local_trigger(msg.payload)

            elif msg.message_type == MessageType.COMMAND_ACK:
                if isinstance(msg.payload, CommandAck):
                    self._handle_ack(msg.payload)

            elif msg.message_type == MessageType.HEARTBEAT:
                self._last_heartbeat = time.time()

            elif msg.message_type == MessageType.WEBRTC_OFFER:
                if isinstance(msg.payload, WebRTCOffer) and self._on_webrtc_offer:
                    self._on_webrtc_offer(msg.payload)

            elif msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE:
                if isinstance(msg.payload, WebRTCIceCandidate) and self._on_webrtc_ice_candidate:
                    self._on_webrtc_ice_candidate(msg.payload)

        except Exception as e:
            logger.error(f"Failed to parse message: {e}")

    def _handle_ack(self, ack: CommandAck) -> None:
        """Handle command acknowledgment."""
        future = self._pending_acks.pop(ack.sequence_id, None)
        if future and not future.done():
            future.set_result(ack)

    async def send_command(
        self,
        command: Command,
        timeout: float = 5.0,
    ) -> bool:
        """
        Send a command and wait for acknowledgment.

        Args:
            command: The command to send
            timeout: Seconds to wait for ack

        Returns:
            True if command was acknowledged successfully
        """
        if not self._connection:
            logger.warning("No Pi connection - command dropped")
            return False

        # Assign sequence ID
        self._sequence_id += 1
        command.sequence_id = self._sequence_id
        command.timestamp_ms = int(time.time() * 1000)

        # Create envelope
        msg = RobotMessage(
            message_type=MessageType.COMMAND,
            payload=command,
        )

        # Create future for ack
        ack_future: asyncio.Future[CommandAck] = asyncio.get_event_loop().create_future()
        self._pending_acks[self._sequence_id] = ack_future

        try:
            # Send command
            await self._connection.send(msg.serialize())

            # Wait for ack
            ack = await asyncio.wait_for(ack_future, timeout)
            return ack.success

        except asyncio.TimeoutError:
            logger.warning(f"Command {self._sequence_id} timed out")
            self._pending_acks.pop(self._sequence_id, None)
            return False

        except websockets.ConnectionClosed:
            logger.warning("Connection closed while sending command")
            self._pending_acks.pop(self._sequence_id, None)
            return False

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self._pending_acks.pop(self._sequence_id, None)
            return False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and monitor connection health."""
        heartbeat_interval = 30.0  # seconds
        heartbeat_timeout = 60.0  # seconds without heartbeat = disconnect

        while self._running:
            await asyncio.sleep(heartbeat_interval)

            if self._connection:
                # Check for timeout
                if time.time() - self._last_heartbeat > heartbeat_timeout:
                    logger.warning("Heartbeat timeout - closing connection")
                    await self._connection.close(1001, "Heartbeat timeout")
                    continue

                # Send heartbeat
                try:
                    msg = create_heartbeat(sequence=self._sequence_id)
                    await self._connection.send(msg.serialize())
                except Exception as e:
                    logger.warning(f"Failed to send heartbeat: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get connection status."""
        return {
            "connected": self.is_connected,
            "host": self._host,
            "port": self._port,
            "pending_commands": len(self._pending_acks),
            "last_heartbeat": self._last_heartbeat,
        }

    async def send_message(self, msg: RobotMessage) -> bool:
        """
        Send a RobotMessage to Pi (for WebRTC signaling).

        Args:
            msg: RobotMessage to send

        Returns:
            True if sent successfully
        """
        if not self._connection:
            logger.warning("Cannot send message - Pi not connected")
            return False

        try:
            await self._connection.send(msg.serialize())
            return True
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
            return False
