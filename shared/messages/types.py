"""
Murph - Robot Message Types
Python dataclass definitions mirroring the Protocol Buffer schema.

These can be used directly without protobuf compilation.
Use serialize()/deserialize() methods for wire format.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


# =============================================================================
# ENUMS
# =============================================================================


class MotorDirection(IntEnum):
    """Motor movement direction."""

    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4


class ScanType(IntEnum):
    """Type of scanning motion."""

    QUICK = 0  # 90 degrees
    PARTIAL = 1  # 180 degrees
    FULL = 2  # 360 degrees


class MessageType(IntEnum):
    """Type of message in the envelope."""

    COMMAND = 10
    COMMAND_ACK = 11
    SENSOR_DATA = 20
    LOCAL_TRIGGER = 21
    HEARTBEAT = 30
    PI_STATUS = 31
    WEBRTC_OFFER = 40
    WEBRTC_ANSWER = 41
    WEBRTC_ICE_CANDIDATE = 42


# =============================================================================
# COMMAND MESSAGES (Server -> Pi)
# =============================================================================


@dataclass
class MotorCommand:
    """Motor control command."""

    direction: MotorDirection = MotorDirection.STOP
    speed: float = 0.0  # 0.0-1.0 normalized
    duration_ms: float = 0.0  # 0 = until stopped

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "motor",
            "direction": int(self.direction),
            "speed": self.speed,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorCommand:
        return cls(
            direction=MotorDirection(data.get("direction", 0)),
            speed=data.get("speed", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class TurnCommand:
    """Turn command."""

    angle_degrees: float = 0.0  # positive = clockwise
    speed: float = 0.5  # 0.0-1.0 normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "turn",
            "angle_degrees": self.angle_degrees,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnCommand:
        return cls(
            angle_degrees=data.get("angle_degrees", 0.0),
            speed=data.get("speed", 0.5),
        )


@dataclass
class ExpressionCommand:
    """Set facial expression."""

    expression_name: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "expression",
            "expression_name": self.expression_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpressionCommand:
        return cls(expression_name=data.get("expression_name", "neutral"))


@dataclass
class SoundCommand:
    """Play a sound."""

    sound_name: str = ""
    volume: float = 1.0  # 0.0-1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sound",
            "sound_name": self.sound_name,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SoundCommand:
        return cls(
            sound_name=data.get("sound_name", ""),
            volume=data.get("volume", 1.0),
        )


@dataclass
class ScanCommand:
    """Perform a scanning motion."""

    scan_type: ScanType = ScanType.PARTIAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "scan",
            "scan_type": int(self.scan_type),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScanCommand:
        return cls(scan_type=ScanType(data.get("scan_type", 1)))


@dataclass
class StopCommand:
    """Emergency stop."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "stop"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StopCommand:
        return cls()


# Union type for all commands
CommandPayload = MotorCommand | TurnCommand | ExpressionCommand | SoundCommand | ScanCommand | StopCommand


@dataclass
class Command:
    """Unified command wrapper with sequence ID."""

    sequence_id: int = 0
    timestamp_ms: int = 0
    payload: CommandPayload | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "sequence_id": self.sequence_id,
            "timestamp_ms": self.timestamp_ms,
        }
        if self.payload:
            result["payload"] = self.payload.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Command:
        payload_data = data.get("payload", {})
        payload_type = payload_data.get("type", "")

        payload: CommandPayload | None = None
        if payload_type == "motor":
            payload = MotorCommand.from_dict(payload_data)
        elif payload_type == "turn":
            payload = TurnCommand.from_dict(payload_data)
        elif payload_type == "expression":
            payload = ExpressionCommand.from_dict(payload_data)
        elif payload_type == "sound":
            payload = SoundCommand.from_dict(payload_data)
        elif payload_type == "scan":
            payload = ScanCommand.from_dict(payload_data)
        elif payload_type == "stop":
            payload = StopCommand.from_dict(payload_data)

        return cls(
            sequence_id=data.get("sequence_id", 0),
            timestamp_ms=data.get("timestamp_ms", 0),
            payload=payload,
        )


@dataclass
class CommandAck:
    """Command acknowledgment."""

    sequence_id: int = 0
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandAck:
        return cls(
            sequence_id=data.get("sequence_id", 0),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
        )


# =============================================================================
# SENSOR MESSAGES (Pi -> Server)
# =============================================================================


@dataclass
class IMUData:
    """IMU sensor data (accelerometer + gyroscope)."""

    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = -1.0  # Default: 1g downward at rest
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    temperature: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "imu",
            "accel_x": self.accel_x,
            "accel_y": self.accel_y,
            "accel_z": self.accel_z,
            "gyro_x": self.gyro_x,
            "gyro_y": self.gyro_y,
            "gyro_z": self.gyro_z,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IMUData:
        return cls(
            accel_x=data.get("accel_x", 0.0),
            accel_y=data.get("accel_y", 0.0),
            accel_z=data.get("accel_z", -1.0),
            gyro_x=data.get("gyro_x", 0.0),
            gyro_y=data.get("gyro_y", 0.0),
            gyro_z=data.get("gyro_z", 0.0),
            temperature=data.get("temperature", 25.0),
        )

    @property
    def acceleration_magnitude(self) -> float:
        """Calculate total acceleration magnitude in g."""
        return (self.accel_x**2 + self.accel_y**2 + self.accel_z**2) ** 0.5


@dataclass
class TouchData:
    """Touch sensor data (MPR121 capacitive)."""

    touched_electrodes: list[int] = field(default_factory=list)
    is_touched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "touch",
            "touched_electrodes": self.touched_electrodes,
            "is_touched": self.is_touched,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TouchData:
        return cls(
            touched_electrodes=data.get("touched_electrodes", []),
            is_touched=data.get("is_touched", False),
        )


@dataclass
class MotorState:
    """Motor encoder/state feedback."""

    left_speed: float = 0.0
    right_speed: float = 0.0
    is_moving: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "motor_state",
            "left_speed": self.left_speed,
            "right_speed": self.right_speed,
            "is_moving": self.is_moving,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorState:
        return cls(
            left_speed=data.get("left_speed", 0.0),
            right_speed=data.get("right_speed", 0.0),
            is_moving=data.get("is_moving", False),
        )


# Union type for all sensor data
SensorPayload = IMUData | TouchData | MotorState


@dataclass
class SensorData:
    """Unified sensor data wrapper."""

    timestamp_ms: int = 0
    payload: SensorPayload | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        result = {"timestamp_ms": self.timestamp_ms}
        if self.payload:
            result["payload"] = self.payload.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensorData:
        payload_data = data.get("payload", {})
        payload_type = payload_data.get("type", "")

        payload: SensorPayload | None = None
        if payload_type == "imu":
            payload = IMUData.from_dict(payload_data)
        elif payload_type == "touch":
            payload = TouchData.from_dict(payload_data)
        elif payload_type == "motor_state":
            payload = MotorState.from_dict(payload_data)

        return cls(
            timestamp_ms=data.get("timestamp_ms", 0),
            payload=payload,
        )


@dataclass
class LocalTrigger:
    """Local behavior triggered on Pi (e.g., picked_up, bump, falling)."""

    trigger_name: str = ""
    intensity: float = 0.0  # 0.0-1.0
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_name": self.trigger_name,
            "intensity": self.intensity,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalTrigger:
        return cls(
            trigger_name=data.get("trigger_name", ""),
            intensity=data.get("intensity", 0.0),
            timestamp_ms=data.get("timestamp_ms", 0),
        )


# =============================================================================
# CONNECTION/STATE MESSAGES
# =============================================================================


@dataclass
class Heartbeat:
    """Connection health heartbeat."""

    timestamp_ms: int = 0
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Heartbeat:
        return cls(
            timestamp_ms=data.get("timestamp_ms", 0),
            sequence=data.get("sequence", 0),
        )


@dataclass
class PiStatus:
    """Pi hardware status report."""

    cpu_temp: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    hardware_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_temp": self.cpu_temp,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "hardware_ok": self.hardware_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PiStatus:
        return cls(
            cpu_temp=data.get("cpu_temp", 0.0),
            cpu_usage=data.get("cpu_usage", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            hardware_ok=data.get("hardware_ok", True),
        )


# =============================================================================
# WEBRTC SIGNALING MESSAGES
# =============================================================================


@dataclass
class WebRTCOffer:
    """WebRTC SDP offer (Pi -> Server)."""

    sdp: str = ""
    type: str = "offer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sdp": self.sdp,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebRTCOffer:
        return cls(
            sdp=data.get("sdp", ""),
            type=data.get("type", "offer"),
        )


@dataclass
class WebRTCAnswer:
    """WebRTC SDP answer (Server -> Pi)."""

    sdp: str = ""
    type: str = "answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sdp": self.sdp,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebRTCAnswer:
        return cls(
            sdp=data.get("sdp", ""),
            type=data.get("type", "answer"),
        )


@dataclass
class WebRTCIceCandidate:
    """WebRTC ICE candidate (bidirectional)."""

    candidate: str = ""
    sdp_mid: str | None = None
    sdp_mline_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "sdp_mid": self.sdp_mid,
            "sdp_mline_index": self.sdp_mline_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebRTCIceCandidate:
        return cls(
            candidate=data.get("candidate", ""),
            sdp_mid=data.get("sdp_mid"),
            sdp_mline_index=data.get("sdp_mline_index"),
        )


# =============================================================================
# TOP-LEVEL MESSAGE ENVELOPE
# =============================================================================

# Union type for all message payloads
RobotMessagePayload = (
    Command
    | CommandAck
    | SensorData
    | LocalTrigger
    | Heartbeat
    | PiStatus
    | WebRTCOffer
    | WebRTCAnswer
    | WebRTCIceCandidate
)


@dataclass
class RobotMessage:
    """
    Top-level message envelope for all robot communication.

    This is the only message type sent over WebSocket.
    """

    timestamp_ms: int = 0
    message_type: MessageType = MessageType.COMMAND
    payload: RobotMessagePayload | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "timestamp_ms": self.timestamp_ms,
            "message_type": int(self.message_type),
        }
        if self.payload:
            result["payload"] = self.payload.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotMessage:
        msg_type = MessageType(data.get("message_type", MessageType.COMMAND))
        payload_data = data.get("payload", {})

        payload: RobotMessagePayload | None = None
        if msg_type == MessageType.COMMAND:
            payload = Command.from_dict(payload_data)
        elif msg_type == MessageType.COMMAND_ACK:
            payload = CommandAck.from_dict(payload_data)
        elif msg_type == MessageType.SENSOR_DATA:
            payload = SensorData.from_dict(payload_data)
        elif msg_type == MessageType.LOCAL_TRIGGER:
            payload = LocalTrigger.from_dict(payload_data)
        elif msg_type == MessageType.HEARTBEAT:
            payload = Heartbeat.from_dict(payload_data)
        elif msg_type == MessageType.PI_STATUS:
            payload = PiStatus.from_dict(payload_data)
        elif msg_type == MessageType.WEBRTC_OFFER:
            payload = WebRTCOffer.from_dict(payload_data)
        elif msg_type == MessageType.WEBRTC_ANSWER:
            payload = WebRTCAnswer.from_dict(payload_data)
        elif msg_type == MessageType.WEBRTC_ICE_CANDIDATE:
            payload = WebRTCIceCandidate.from_dict(payload_data)

        return cls(
            timestamp_ms=data.get("timestamp_ms", 0),
            message_type=msg_type,
            payload=payload,
        )

    def serialize(self) -> bytes:
        """Serialize message to bytes for wire transmission."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> RobotMessage:
        """Deserialize message from bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


# =============================================================================
# FACTORY HELPERS
# =============================================================================


def create_motor_command(
    direction: MotorDirection,
    speed: float,
    duration_ms: float = 0.0,
    sequence_id: int = 0,
) -> RobotMessage:
    """Create a motor command message."""
    return RobotMessage(
        message_type=MessageType.COMMAND,
        payload=Command(
            sequence_id=sequence_id,
            payload=MotorCommand(
                direction=direction,
                speed=speed,
                duration_ms=duration_ms,
            ),
        ),
    )


def create_turn_command(
    angle_degrees: float,
    speed: float = 0.5,
    sequence_id: int = 0,
) -> RobotMessage:
    """Create a turn command message."""
    return RobotMessage(
        message_type=MessageType.COMMAND,
        payload=Command(
            sequence_id=sequence_id,
            payload=TurnCommand(angle_degrees=angle_degrees, speed=speed),
        ),
    )


def create_expression_command(
    expression_name: str,
    sequence_id: int = 0,
) -> RobotMessage:
    """Create an expression command message."""
    return RobotMessage(
        message_type=MessageType.COMMAND,
        payload=Command(
            sequence_id=sequence_id,
            payload=ExpressionCommand(expression_name=expression_name),
        ),
    )


def create_sound_command(
    sound_name: str,
    volume: float = 1.0,
    sequence_id: int = 0,
) -> RobotMessage:
    """Create a sound command message."""
    return RobotMessage(
        message_type=MessageType.COMMAND,
        payload=Command(
            sequence_id=sequence_id,
            payload=SoundCommand(sound_name=sound_name, volume=volume),
        ),
    )


def create_stop_command(sequence_id: int = 0) -> RobotMessage:
    """Create an emergency stop command message."""
    return RobotMessage(
        message_type=MessageType.COMMAND,
        payload=Command(
            sequence_id=sequence_id,
            payload=StopCommand(),
        ),
    )


def create_sensor_data(payload: SensorPayload) -> RobotMessage:
    """Create a sensor data message."""
    return RobotMessage(
        message_type=MessageType.SENSOR_DATA,
        payload=SensorData(payload=payload),
    )


def create_command_ack(
    sequence_id: int,
    success: bool = True,
    error_message: str = "",
) -> RobotMessage:
    """Create a command acknowledgment message."""
    return RobotMessage(
        message_type=MessageType.COMMAND_ACK,
        payload=CommandAck(
            sequence_id=sequence_id,
            success=success,
            error_message=error_message,
        ),
    )


def create_local_trigger(
    trigger_name: str,
    intensity: float = 1.0,
) -> RobotMessage:
    """Create a local trigger message."""
    return RobotMessage(
        message_type=MessageType.LOCAL_TRIGGER,
        payload=LocalTrigger(trigger_name=trigger_name, intensity=intensity),
    )


def create_heartbeat(sequence: int = 0) -> RobotMessage:
    """Create a heartbeat message."""
    return RobotMessage(
        message_type=MessageType.HEARTBEAT,
        payload=Heartbeat(sequence=sequence),
    )


def create_webrtc_offer(sdp: str) -> RobotMessage:
    """Create a WebRTC offer message."""
    return RobotMessage(
        message_type=MessageType.WEBRTC_OFFER,
        payload=WebRTCOffer(sdp=sdp),
    )


def create_webrtc_answer(sdp: str) -> RobotMessage:
    """Create a WebRTC answer message."""
    return RobotMessage(
        message_type=MessageType.WEBRTC_ANSWER,
        payload=WebRTCAnswer(sdp=sdp),
    )


def create_webrtc_ice_candidate(
    candidate: str,
    sdp_mid: str | None = None,
    sdp_mline_index: int | None = None,
) -> RobotMessage:
    """Create a WebRTC ICE candidate message."""
    return RobotMessage(
        message_type=MessageType.WEBRTC_ICE_CANDIDATE,
        payload=WebRTCIceCandidate(
            candidate=candidate,
            sdp_mid=sdp_mid,
            sdp_mline_index=sdp_mline_index,
        ),
    )
