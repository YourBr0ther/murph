"""
Tests for shared message types.
"""

import pytest
import time

from shared.messages import (
    # Enums
    MessageType,
    MotorDirection,
    ScanType,
    # Command types
    Command,
    CommandAck,
    ExpressionCommand,
    MotorCommand,
    ScanCommand,
    SoundCommand,
    StopCommand,
    TurnCommand,
    # Sensor types
    IMUData,
    MotorState,
    SensorData,
    TouchData,
    # Other types
    Heartbeat,
    LocalTrigger,
    PiStatus,
    RobotMessage,
    # Factories
    create_command_ack,
    create_expression_command,
    create_heartbeat,
    create_local_trigger,
    create_motor_command,
    create_sensor_data,
    create_sound_command,
    create_stop_command,
    create_turn_command,
)


class TestMotorCommand:
    """Tests for MotorCommand message."""

    def test_create_motor_command(self):
        cmd = MotorCommand(
            direction=MotorDirection.FORWARD,
            speed=0.5,
            duration_ms=1000,
        )
        assert cmd.direction == MotorDirection.FORWARD
        assert cmd.speed == 0.5
        assert cmd.duration_ms == 1000

    def test_motor_command_to_dict(self):
        cmd = MotorCommand(
            direction=MotorDirection.LEFT,
            speed=0.8,
            duration_ms=500,
        )
        d = cmd.to_dict()
        assert d["type"] == "motor"
        assert d["direction"] == MotorDirection.LEFT
        assert d["speed"] == 0.8
        assert d["duration_ms"] == 500

    def test_motor_command_from_dict(self):
        d = {
            "type": "motor",
            "direction": 1,  # FORWARD
            "speed": 0.6,
            "duration_ms": 2000,
        }
        cmd = MotorCommand.from_dict(d)
        assert cmd.direction == MotorDirection.FORWARD
        assert cmd.speed == 0.6
        assert cmd.duration_ms == 2000


class TestTurnCommand:
    """Tests for TurnCommand message."""

    def test_create_turn_command(self):
        cmd = TurnCommand(angle_degrees=90, speed=0.5)
        assert cmd.angle_degrees == 90
        assert cmd.speed == 0.5

    def test_turn_command_roundtrip(self):
        cmd = TurnCommand(angle_degrees=-45, speed=0.3)
        d = cmd.to_dict()
        cmd2 = TurnCommand.from_dict(d)
        assert cmd2.angle_degrees == -45
        assert cmd2.speed == 0.3


class TestExpressionCommand:
    """Tests for ExpressionCommand message."""

    def test_create_expression_command(self):
        cmd = ExpressionCommand(expression_name="happy")
        assert cmd.expression_name == "happy"

    def test_expression_default(self):
        cmd = ExpressionCommand()
        assert cmd.expression_name == "neutral"


class TestSoundCommand:
    """Tests for SoundCommand message."""

    def test_create_sound_command(self):
        cmd = SoundCommand(sound_name="greeting", volume=0.8)
        assert cmd.sound_name == "greeting"
        assert cmd.volume == 0.8

    def test_sound_default_volume(self):
        cmd = SoundCommand(sound_name="chirp")
        assert cmd.volume == 1.0


class TestScanCommand:
    """Tests for ScanCommand message."""

    def test_create_scan_command(self):
        cmd = ScanCommand(scan_type=ScanType.FULL)
        assert cmd.scan_type == ScanType.FULL

    def test_scan_default(self):
        cmd = ScanCommand()
        assert cmd.scan_type == ScanType.PARTIAL


class TestCommand:
    """Tests for Command wrapper."""

    def test_command_with_motor_payload(self):
        motor = MotorCommand(direction=MotorDirection.FORWARD, speed=0.5)
        cmd = Command(sequence_id=1, payload=motor)
        assert cmd.sequence_id == 1
        assert isinstance(cmd.payload, MotorCommand)

    def test_command_auto_timestamp(self):
        cmd = Command()
        assert cmd.timestamp_ms > 0

    def test_command_roundtrip(self):
        motor = MotorCommand(direction=MotorDirection.BACKWARD, speed=0.7, duration_ms=500)
        cmd = Command(sequence_id=42, payload=motor)
        d = cmd.to_dict()
        cmd2 = Command.from_dict(d)
        assert cmd2.sequence_id == 42
        assert isinstance(cmd2.payload, MotorCommand)
        assert cmd2.payload.direction == MotorDirection.BACKWARD


class TestIMUData:
    """Tests for IMUData message."""

    def test_create_imu_data(self):
        imu = IMUData(
            accel_x=0.1,
            accel_y=0.0,
            accel_z=-1.0,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
        )
        assert imu.accel_z == -1.0

    def test_acceleration_magnitude(self):
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-1.0)
        assert abs(imu.acceleration_magnitude - 1.0) < 0.001

        imu2 = IMUData(accel_x=1, accel_y=1, accel_z=1)
        assert abs(imu2.acceleration_magnitude - 1.732) < 0.01


class TestTouchData:
    """Tests for TouchData message."""

    def test_create_touch_data(self):
        touch = TouchData(touched_electrodes=[1, 2, 3], is_touched=True)
        assert touch.is_touched
        assert 2 in touch.touched_electrodes

    def test_touch_data_default(self):
        touch = TouchData()
        assert not touch.is_touched
        assert touch.touched_electrodes == []


class TestSensorData:
    """Tests for SensorData wrapper."""

    def test_sensor_data_with_imu(self):
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-1.0)
        sensor = SensorData(payload=imu)
        assert isinstance(sensor.payload, IMUData)

    def test_sensor_data_roundtrip(self):
        touch = TouchData(touched_electrodes=[5, 6], is_touched=True)
        sensor = SensorData(payload=touch)
        d = sensor.to_dict()
        sensor2 = SensorData.from_dict(d)
        assert isinstance(sensor2.payload, TouchData)
        assert sensor2.payload.is_touched


class TestLocalTrigger:
    """Tests for LocalTrigger message."""

    def test_create_local_trigger(self):
        trigger = LocalTrigger(trigger_name="picked_up", intensity=0.8)
        assert trigger.trigger_name == "picked_up"
        assert trigger.intensity == 0.8

    def test_local_trigger_auto_timestamp(self):
        trigger = LocalTrigger(trigger_name="bump")
        assert trigger.timestamp_ms > 0


class TestHeartbeat:
    """Tests for Heartbeat message."""

    def test_create_heartbeat(self):
        hb = Heartbeat(sequence=5)
        assert hb.sequence == 5
        assert hb.timestamp_ms > 0


class TestRobotMessage:
    """Tests for RobotMessage envelope."""

    def test_robot_message_with_command(self):
        motor = MotorCommand(direction=MotorDirection.FORWARD, speed=0.5)
        cmd = Command(sequence_id=1, payload=motor)
        msg = RobotMessage(message_type=MessageType.COMMAND, payload=cmd)

        assert msg.message_type == MessageType.COMMAND
        assert isinstance(msg.payload, Command)

    def test_robot_message_serialize_deserialize(self):
        motor = MotorCommand(direction=MotorDirection.RIGHT, speed=0.3, duration_ms=100)
        cmd = Command(sequence_id=123, payload=motor)
        msg = RobotMessage(message_type=MessageType.COMMAND, payload=cmd)

        # Serialize to bytes
        data = msg.serialize()
        assert isinstance(data, bytes)

        # Deserialize
        msg2 = RobotMessage.deserialize(data)
        assert msg2.message_type == MessageType.COMMAND
        assert isinstance(msg2.payload, Command)
        assert msg2.payload.sequence_id == 123

    def test_robot_message_with_sensor_data(self):
        imu = IMUData(accel_x=0.5, accel_y=-0.2, accel_z=-0.9)
        sensor = SensorData(payload=imu)
        msg = RobotMessage(message_type=MessageType.SENSOR_DATA, payload=sensor)

        data = msg.serialize()
        msg2 = RobotMessage.deserialize(data)

        assert msg2.message_type == MessageType.SENSOR_DATA
        assert isinstance(msg2.payload.payload, IMUData)
        assert msg2.payload.payload.accel_x == 0.5


class TestFactoryFunctions:
    """Tests for factory helper functions."""

    def test_create_motor_command_factory(self):
        msg = create_motor_command(
            direction=MotorDirection.FORWARD,
            speed=0.7,
            duration_ms=500,
            sequence_id=10,
        )
        assert msg.message_type == MessageType.COMMAND
        assert msg.payload.sequence_id == 10

    def test_create_turn_command_factory(self):
        msg = create_turn_command(angle_degrees=180, speed=0.4, sequence_id=5)
        assert msg.message_type == MessageType.COMMAND
        assert isinstance(msg.payload.payload, TurnCommand)
        assert msg.payload.payload.angle_degrees == 180

    def test_create_expression_command_factory(self):
        msg = create_expression_command("curious", sequence_id=3)
        assert isinstance(msg.payload.payload, ExpressionCommand)
        assert msg.payload.payload.expression_name == "curious"

    def test_create_sound_command_factory(self):
        msg = create_sound_command("happy", volume=0.5, sequence_id=7)
        assert isinstance(msg.payload.payload, SoundCommand)
        assert msg.payload.payload.volume == 0.5

    def test_create_stop_command_factory(self):
        msg = create_stop_command(sequence_id=99)
        assert isinstance(msg.payload.payload, StopCommand)

    def test_create_sensor_data_factory(self):
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-1)
        msg = create_sensor_data(imu)
        assert msg.message_type == MessageType.SENSOR_DATA

    def test_create_command_ack_factory(self):
        msg = create_command_ack(sequence_id=42, success=True)
        assert msg.message_type == MessageType.COMMAND_ACK
        assert msg.payload.success

    def test_create_local_trigger_factory(self):
        msg = create_local_trigger("bump", intensity=0.9)
        assert msg.message_type == MessageType.LOCAL_TRIGGER
        assert msg.payload.trigger_name == "bump"

    def test_create_heartbeat_factory(self):
        msg = create_heartbeat(sequence=100)
        assert msg.message_type == MessageType.HEARTBEAT
        assert msg.payload.sequence == 100
