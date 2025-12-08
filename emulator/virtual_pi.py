"""
Murph - Virtual Pi
Simulates Raspberry Pi hardware for testing without physical robot.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from shared.messages import (
    Command,
    ExpressionCommand,
    MotorCommand,
    MotorDirection,
    ScanCommand,
    SoundCommand,
    StopCommand,
    TurnCommand,
    IMUData,
    TouchData,
    SensorData,
    RobotMessage,
    MessageType,
    create_sensor_data,
    create_command_ack,
    create_heartbeat,
)

logger = logging.getLogger(__name__)


@dataclass
class VirtualRobotState:
    """Current state of the virtual robot."""

    # Position in 2D space (units arbitrary, e.g., cm)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # degrees, 0 = facing right

    # Motor state
    is_moving: bool = False
    current_speed: float = 0.0
    move_direction: str = "stop"
    left_speed: float = 0.0
    right_speed: float = 0.0

    # Display state
    current_expression: str = "neutral"

    # Audio state
    playing_sound: str | None = None
    sound_end_time: float = 0.0

    # Sensor simulation state
    is_being_touched: bool = False
    touched_electrodes: list[int] = field(default_factory=list)

    # IMU simulation
    imu_noise: float = 0.02
    simulated_pickup: bool = False
    simulated_bump: bool = False
    simulated_shake: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "is_moving": self.is_moving,
            "current_speed": self.current_speed,
            "move_direction": self.move_direction,
            "left_speed": self.left_speed,
            "right_speed": self.right_speed,
            "current_expression": self.current_expression,
            "playing_sound": self.playing_sound,
            "is_being_touched": self.is_being_touched,
            "touched_electrodes": self.touched_electrodes,
        }


class VirtualPi:
    """
    Simulates Raspberry Pi hardware for testing without physical robot.

    Features:
    - Receives commands from server via WebSocket
    - Simulates motor movement with position tracking
    - Generates synthetic sensor data
    - Allows manual input for touch/IMU simulation
    - Provides state updates for UI visualization
    """

    def __init__(
        self,
        server_host: str = DEFAULT_SERVER_HOST,
        server_port: int = DEFAULT_SERVER_PORT,
        on_state_change: Callable[[VirtualRobotState], None] | None = None,
    ) -> None:
        """
        Initialize virtual Pi.

        Args:
            server_host: Server to connect to
            server_port: Server port
            on_state_change: Callback when robot state changes
        """
        self._host = server_host
        self._port = server_port
        self._state = VirtualRobotState()
        self._on_state_change = on_state_change
        self._running = False
        self._websocket = None

        # Movement simulation
        self._move_task: asyncio.Task[None] | None = None
        self._base_speed = 50.0  # units per second at speed=1.0

        # Sensor streaming
        self._sensor_task: asyncio.Task[None] | None = None
        self._sensor_interval_ms = 100  # 10Hz

    @property
    def state(self) -> VirtualRobotState:
        """Get current robot state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._websocket is not None

    async def start(self) -> None:
        """Start virtual Pi and connect to server."""
        self._running = True
        asyncio.create_task(self._connect_to_server())
        self._sensor_task = asyncio.create_task(self._sensor_loop())
        logger.info("Virtual Pi started")

    async def stop(self) -> None:
        """Stop virtual Pi."""
        self._running = False

        if self._move_task and not self._move_task.done():
            self._move_task.cancel()

        if self._sensor_task and not self._sensor_task.done():
            self._sensor_task.cancel()

        if self._websocket:
            await self._websocket.close()

        logger.info("Virtual Pi stopped")

    async def _connect_to_server(self) -> None:
        """Connect to server WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets library not available")
            return

        while self._running:
            try:
                uri = f"ws://{self._host}:{self._port}"
                logger.info(f"Virtual Pi connecting to {uri}")

                async with websockets.connect(uri) as ws:
                    self._websocket = ws
                    logger.info("Virtual Pi connected to server")
                    await self._message_loop()

            except Exception as e:
                logger.warning(f"Connection failed: {e}")
                await asyncio.sleep(2)

            finally:
                self._websocket = None

    async def _message_loop(self) -> None:
        """Process commands from server."""
        async for message in self._websocket:
            if isinstance(message, bytes):
                try:
                    msg = RobotMessage.deserialize(message)

                    if msg.message_type == MessageType.COMMAND:
                        if isinstance(msg.payload, Command):
                            await self._execute_command(msg.payload)

                    elif msg.message_type == MessageType.HEARTBEAT:
                        # Echo heartbeat
                        await self._send_heartbeat()

                except Exception as e:
                    logger.error(f"Failed to process message: {e}")

    async def _execute_command(self, cmd: Command) -> None:
        """Execute a command on virtual hardware."""
        payload = cmd.payload

        if isinstance(payload, MotorCommand):
            await self._handle_motor(payload)
        elif isinstance(payload, TurnCommand):
            await self._handle_turn(payload)
        elif isinstance(payload, ExpressionCommand):
            self._state.current_expression = payload.expression_name
        elif isinstance(payload, SoundCommand):
            self._state.playing_sound = payload.sound_name
            self._state.sound_end_time = time.time() + 0.5  # Assume 0.5s sounds
        elif isinstance(payload, StopCommand):
            await self._stop_movement()

        # Send acknowledgment
        await self._send_ack(cmd.sequence_id, True)
        self._notify_state_change()

    async def _handle_motor(self, motor: MotorCommand) -> None:
        """Simulate motor movement."""
        direction_map = {
            MotorDirection.STOP: "stop",
            MotorDirection.FORWARD: "forward",
            MotorDirection.BACKWARD: "backward",
            MotorDirection.LEFT: "left",
            MotorDirection.RIGHT: "right",
        }

        direction = direction_map.get(motor.direction, "stop")
        self._state.move_direction = direction
        self._state.current_speed = motor.speed
        self._state.is_moving = direction != "stop" and motor.speed > 0

        # Calculate wheel speeds
        if direction == "forward":
            self._state.left_speed = motor.speed
            self._state.right_speed = motor.speed
        elif direction == "backward":
            self._state.left_speed = -motor.speed
            self._state.right_speed = -motor.speed
        elif direction == "left":
            self._state.left_speed = -motor.speed
            self._state.right_speed = motor.speed
        elif direction == "right":
            self._state.left_speed = motor.speed
            self._state.right_speed = -motor.speed
        else:
            self._state.left_speed = 0.0
            self._state.right_speed = 0.0

        # Start movement simulation
        if motor.duration_ms > 0:
            if self._move_task and not self._move_task.done():
                self._move_task.cancel()
            self._move_task = asyncio.create_task(
                self._simulate_movement(motor.duration_ms)
            )

    async def _handle_turn(self, turn: TurnCommand) -> None:
        """Simulate turning."""
        # Update heading
        self._state.heading = (self._state.heading + turn.angle_degrees) % 360

        # Brief turn animation
        duration_ms = abs(turn.angle_degrees) / (180 * max(0.1, turn.speed)) * 1000
        self._state.is_moving = True
        await asyncio.sleep(duration_ms / 1000)
        self._state.is_moving = False
        self._notify_state_change()

    async def _simulate_movement(self, duration_ms: float) -> None:
        """Simulate position update during movement."""
        duration_s = duration_ms / 1000.0
        start_time = time.time()
        tick_interval = 0.05  # 50ms ticks

        while time.time() - start_time < duration_s:
            if not self._state.is_moving:
                break

            # Update position based on heading and speed
            distance = self._state.current_speed * self._base_speed * tick_interval
            heading_rad = math.radians(self._state.heading)

            if self._state.move_direction == "forward":
                self._state.x += distance * math.cos(heading_rad)
                self._state.y += distance * math.sin(heading_rad)
            elif self._state.move_direction == "backward":
                self._state.x -= distance * math.cos(heading_rad)
                self._state.y -= distance * math.sin(heading_rad)

            self._notify_state_change()
            await asyncio.sleep(tick_interval)

        await self._stop_movement()

    async def _stop_movement(self) -> None:
        """Stop all movement."""
        self._state.is_moving = False
        self._state.current_speed = 0.0
        self._state.move_direction = "stop"
        self._state.left_speed = 0.0
        self._state.right_speed = 0.0
        self._notify_state_change()

    async def _sensor_loop(self) -> None:
        """Generate synthetic sensor data periodically."""
        while self._running:
            if self._websocket:
                # Generate and send IMU data
                imu_data = self._generate_imu_data()
                await self._send_sensor(SensorData(payload=imu_data))

                # Generate and send touch data
                touch_data = self._generate_touch_data()
                await self._send_sensor(SensorData(payload=touch_data))

            await asyncio.sleep(self._sensor_interval_ms / 1000)

    def _generate_imu_data(self) -> IMUData:
        """Generate simulated IMU data."""
        noise = self._state.imu_noise

        # Base values at rest
        accel_x = random.gauss(0, noise)
        accel_y = random.gauss(0, noise)
        accel_z = random.gauss(-1.0, noise)  # -1g at rest

        # Add movement effects
        if self._state.is_moving:
            speed_factor = self._state.current_speed * 0.1
            if self._state.move_direction == "forward":
                accel_x += speed_factor
            elif self._state.move_direction == "backward":
                accel_x -= speed_factor

        # Add simulated events
        if self._state.simulated_pickup:
            accel_z += 1.0  # Upward acceleration
        if self._state.simulated_bump:
            accel_x += random.uniform(2.0, 3.0)
        if self._state.simulated_shake:
            accel_x += random.uniform(-2.0, 2.0)
            accel_y += random.uniform(-2.0, 2.0)

        return IMUData(
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            gyro_x=random.gauss(0, noise * 10),
            gyro_y=random.gauss(0, noise * 10),
            gyro_z=random.gauss(0, noise * 10),
            temperature=25.0,
        )

    def _generate_touch_data(self) -> TouchData:
        """Generate simulated touch data."""
        return TouchData(
            touched_electrodes=self._state.touched_electrodes.copy(),
            is_touched=self._state.is_being_touched,
        )

    async def _send_sensor(self, sensor: SensorData) -> None:
        """Send sensor data to server."""
        if self._websocket:
            try:
                msg = RobotMessage(
                    message_type=MessageType.SENSOR_DATA,
                    payload=sensor,
                )
                await self._websocket.send(msg.serialize())
            except Exception as e:
                logger.warning(f"Failed to send sensor data: {e}")

    async def _send_ack(self, sequence_id: int, success: bool) -> None:
        """Send command acknowledgment."""
        if self._websocket:
            try:
                msg = create_command_ack(sequence_id, success)
                await self._websocket.send(msg.serialize())
            except Exception as e:
                logger.warning(f"Failed to send ack: {e}")

    async def _send_heartbeat(self) -> None:
        """Send heartbeat."""
        if self._websocket:
            try:
                msg = create_heartbeat()
                await self._websocket.send(msg.serialize())
            except Exception as e:
                logger.warning(f"Failed to send heartbeat: {e}")

    def _notify_state_change(self) -> None:
        """Notify listener of state change."""
        if self._on_state_change:
            self._on_state_change(self._state)

    # === Manual simulation controls ===

    def simulate_touch(self, electrodes: list[int]) -> None:
        """Simulate touching specific electrodes."""
        self._state.is_being_touched = len(electrodes) > 0
        self._state.touched_electrodes = electrodes
        self._notify_state_change()

    def simulate_release(self) -> None:
        """Release all touch."""
        self._state.is_being_touched = False
        self._state.touched_electrodes = []
        self._notify_state_change()

    def simulate_pickup(self, duration: float = 0.5) -> None:
        """Simulate being picked up."""
        self._state.simulated_pickup = True
        asyncio.create_task(self._clear_event("pickup", duration))

    def simulate_bump(self, duration: float = 0.2) -> None:
        """Simulate a bump."""
        self._state.simulated_bump = True
        asyncio.create_task(self._clear_event("bump", duration))

    def simulate_shake(self, duration: float = 1.0) -> None:
        """Simulate being shaken."""
        self._state.simulated_shake = True
        asyncio.create_task(self._clear_event("shake", duration))

    async def _clear_event(self, event: str, duration: float) -> None:
        """Clear simulated event after duration."""
        await asyncio.sleep(duration)
        if event == "pickup":
            self._state.simulated_pickup = False
        elif event == "bump":
            self._state.simulated_bump = False
        elif event == "shake":
            self._state.simulated_shake = False
