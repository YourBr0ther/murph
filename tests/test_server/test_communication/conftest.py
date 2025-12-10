"""Shared fixtures for communication layer tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.messages import (
    Command,
    CommandAck,
    IMUData,
    LocalTrigger,
    MessageType,
    MotorCommand,
    MotorDirection,
    RobotMessage,
    SensorData,
    WebRTCIceCandidate,
    WebRTCOffer,
    create_command_ack,
    create_heartbeat,
    create_local_trigger,
    create_sensor_data,
)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.remote_address = ("127.0.0.1", 12345)
    ws.close = AsyncMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def mock_websocket_server():
    """Create a mock WebSocket server."""
    server = MagicMock()
    server.close = MagicMock()
    server.wait_closed = AsyncMock()
    return server


@pytest.fixture
def sensor_data_msg():
    """Create a sensor data message."""
    return RobotMessage(
        message_type=MessageType.SENSOR_DATA,
        payload=SensorData(
            payload=IMUData(accel_x=0.1, accel_y=0.2, accel_z=-1.0)
        ),
    )


@pytest.fixture
def local_trigger_msg():
    """Create a local trigger message."""
    return create_local_trigger("bump", intensity=0.8)


@pytest.fixture
def command_ack_msg():
    """Create a command ack message."""
    return create_command_ack(sequence_id=1, success=True)


@pytest.fixture
def heartbeat_msg():
    """Create a heartbeat message."""
    return create_heartbeat(sequence=1)


@pytest.fixture
def webrtc_offer_msg():
    """Create a WebRTC offer message."""
    return RobotMessage(
        message_type=MessageType.WEBRTC_OFFER,
        payload=WebRTCOffer(sdp="v=0\r\no=- 123 456 IN IP4 127.0.0.1\r\n"),
    )


@pytest.fixture
def webrtc_ice_msg():
    """Create a WebRTC ICE candidate message."""
    return RobotMessage(
        message_type=MessageType.WEBRTC_ICE_CANDIDATE,
        payload=WebRTCIceCandidate(
            candidate="candidate:1 1 UDP 2122252543 192.168.1.1 12345 typ host",
            sdp_mid="0",
            sdp_mline_index=0,
        ),
    )


@pytest.fixture
def motor_command():
    """Create a motor command."""
    return Command(
        payload=MotorCommand(
            direction=MotorDirection.FORWARD,
            speed=0.5,
            duration_ms=1000,
        )
    )


@pytest.fixture
def mock_connection_manager():
    """Create a mock PiConnectionManager."""
    manager = MagicMock()
    manager.is_connected = True
    manager.send_command = AsyncMock(return_value=True)
    manager.send_message = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_speech_service():
    """Create a mock SpeechService."""
    service = MagicMock()
    service.synthesize = AsyncMock(return_value=b"fake_audio_data")
    service.encode_audio_base64 = MagicMock(return_value="ZmFrZV9hdWRpb19kYXRh")
    return service
