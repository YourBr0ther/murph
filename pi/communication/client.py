"""
Murph - Pi WebSocket Client
Connects Pi to server brain for command/sensor communication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any

from shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    MAX_RECONNECT_ATTEMPTS,
    RECONNECT_DELAY,
)
from shared.messages import (
    Command,
    Heartbeat,
    MessageType,
    RobotMessage,
    SensorData,
    LocalTrigger,
    WebRTCAnswer,
    WebRTCIceCandidate,
    create_command_ack,
    create_sensor_data,
    create_local_trigger,
    create_heartbeat,
)

logger = logging.getLogger(__name__)


class ServerConnection:
    """
    WebSocket client connecting Pi to server brain.

    Handles:
    - Connecting to server with automatic reconnection
    - Receiving commands and dispatching to handlers
    - Sending sensor data and local triggers
    - Heartbeat for connection health
    """

    def __init__(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
        on_command: Callable[[Command], None] | None = None,
        on_connection_change: Callable[[bool], None] | None = None,
        on_webrtc_answer: Callable[[WebRTCAnswer], Any] | None = None,
        on_webrtc_ice_candidate: Callable[[WebRTCIceCandidate], Any] | None = None,
    ) -> None:
        """
        Initialize the server connection.

        Args:
            host: Server host address
            port: Server port
            on_command: Callback for received commands
            on_connection_change: Callback when connection state changes
            on_webrtc_answer: Callback for WebRTC SDP answer from server
            on_webrtc_ice_candidate: Callback for WebRTC ICE candidates from server
        """
        self._host = host
        self._port = port
        self._on_command = on_command
        self._on_connection_change = on_connection_change
        self._on_webrtc_answer = on_webrtc_answer
        self._on_webrtc_ice_candidate = on_webrtc_ice_candidate

        self._websocket: WebSocketClientProtocol | None = None
        self._running = False
        self._connected = False
        self._connect_task: asyncio.Task[None] | None = None

        # Heartbeat tracking
        self._last_heartbeat_received: float = 0
        self._heartbeat_sequence: int = 0

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to server."""
        return self._connected and self._websocket is not None

    @property
    def uri(self) -> str:
        """Get WebSocket URI."""
        return f"ws://{self._host}:{self._port}"

    async def connect(self) -> None:
        """
        Connect to server with automatic reconnection.

        This runs indefinitely, reconnecting on disconnection.
        """
        if not HAS_WEBSOCKETS:
            logger.error("websockets library not available")
            return

        self._running = True
        reconnect_count = 0

        while self._running:
            try:
                logger.info(f"Connecting to server at {self.uri}")

                async with websockets.connect(
                    self.uri,
                    ping_interval=30,
                    ping_timeout=10,
                ) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    self._last_heartbeat_received = time.time()
                    reconnect_count = 0

                    logger.info("Connected to server")
                    if self._on_connection_change:
                        self._on_connection_change(True)

                    await self._message_loop()

            except websockets.ConnectionClosed as e:
                logger.warning(f"Disconnected from server: {e}")
            except ConnectionRefusedError:
                logger.warning("Connection refused - server not running?")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            finally:
                self._websocket = None
                self._connected = False
                if self._on_connection_change:
                    self._on_connection_change(False)

            # Reconnect with backoff
            if self._running:
                reconnect_count += 1
                if reconnect_count > MAX_RECONNECT_ATTEMPTS:
                    logger.error("Max reconnect attempts reached")
                    # Reset and try again
                    reconnect_count = 0

                delay = min(RECONNECT_DELAY * reconnect_count, 60)
                logger.info(f"Reconnecting in {delay}s...")
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._running = False

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        self._connected = False
        logger.info("Disconnected from server")

    async def _message_loop(self) -> None:
        """Process incoming messages from server."""
        if not self._websocket:
            return

        async for message in self._websocket:
            if isinstance(message, bytes):
                await self._handle_message(message)
            else:
                logger.warning(f"Unexpected message type: {type(message)}")

    async def _handle_message(self, raw: bytes) -> None:
        """Parse and dispatch incoming message."""
        try:
            msg = RobotMessage.deserialize(raw)

            if msg.message_type == MessageType.COMMAND:
                if isinstance(msg.payload, Command):
                    # Process command
                    if self._on_command:
                        self._on_command(msg.payload)

                    # Send acknowledgment
                    await self._send_ack(msg.payload.sequence_id, True)

            elif msg.message_type == MessageType.HEARTBEAT:
                self._last_heartbeat_received = time.time()
                # Echo heartbeat back
                await self._send_heartbeat()

            elif msg.message_type == MessageType.WEBRTC_ANSWER:
                if isinstance(msg.payload, WebRTCAnswer) and self._on_webrtc_answer:
                    self._on_webrtc_answer(msg.payload)

            elif msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE:
                if isinstance(msg.payload, WebRTCIceCandidate) and self._on_webrtc_ice_candidate:
                    self._on_webrtc_ice_candidate(msg.payload)

        except Exception as e:
            logger.error(f"Failed to parse message: {e}")

    async def _send_ack(
        self,
        sequence_id: int,
        success: bool,
        error: str = "",
    ) -> None:
        """Send command acknowledgment."""
        if not self._websocket:
            return

        try:
            msg = create_command_ack(sequence_id, success, error)
            await self._websocket.send(msg.serialize())
        except Exception as e:
            logger.warning(f"Failed to send ack: {e}")

    async def _send_heartbeat(self) -> None:
        """Send heartbeat response."""
        if not self._websocket:
            return

        try:
            self._heartbeat_sequence += 1
            msg = create_heartbeat(self._heartbeat_sequence)
            await self._websocket.send(msg.serialize())
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    async def send_sensor_data(self, sensor: SensorData) -> None:
        """
        Send sensor data to server.

        Args:
            sensor: Sensor data to send
        """
        if not self._websocket:
            return

        try:
            msg = RobotMessage(
                message_type=MessageType.SENSOR_DATA,
                payload=sensor,
            )
            await self._websocket.send(msg.serialize())
        except Exception as e:
            logger.warning(f"Failed to send sensor data: {e}")

    async def send_local_trigger(
        self,
        trigger_name: str,
        intensity: float = 1.0,
    ) -> None:
        """
        Notify server of a local trigger event.

        Args:
            trigger_name: Name of the trigger (e.g., "picked_up", "bump")
            intensity: Event intensity (0.0-1.0)
        """
        if not self._websocket:
            return

        try:
            msg = create_local_trigger(trigger_name, intensity)
            await self._websocket.send(msg.serialize())
            logger.debug(f"Sent local trigger: {trigger_name}")
        except Exception as e:
            logger.warning(f"Failed to send local trigger: {e}")

    async def send_message(self, msg: RobotMessage) -> None:
        """
        Send a RobotMessage directly (for WebRTC signaling).

        Args:
            msg: RobotMessage to send
        """
        if not self._websocket:
            logger.warning("Cannot send message - not connected")
            return

        try:
            await self._websocket.send(msg.serialize())
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get connection status."""
        return {
            "connected": self.is_connected,
            "uri": self.uri,
            "last_heartbeat": self._last_heartbeat_received,
        }
