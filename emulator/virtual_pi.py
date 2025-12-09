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
from typing import TYPE_CHECKING, Any, Callable

from shared.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
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
    IMUData,
    TouchData,
    SensorData,
    PiStatus,
    RobotMessage,
    MessageType,
    WebRTCAnswer,
    WebRTCIceCandidate,
    create_sensor_data,
    create_command_ack,
    create_heartbeat,
    create_local_trigger,
)

if TYPE_CHECKING:
    from .video import EmulatorVideoStreamer, MockWebcamCamera, WebcamCamera

logger = logging.getLogger(__name__)


class EmulatorReflexProcessor:
    """
    Simplified reflex detection for emulator.
    Auto-detects pickup, bump, shake from IMU data.
    """

    # Thresholds (from shared/constants.py)
    PICKUP_THRESHOLD = 1.5  # g
    BUMP_THRESHOLD = 2.0  # g (sudden spike)
    SHAKE_GYRO_THRESHOLD = 100.0  # deg/s
    FREEFALL_THRESHOLD = 0.3  # g

    def __init__(self) -> None:
        self._accel_history: list[float] = []
        self._gyro_history: list[float] = []
        self._cooldowns: dict[str, float] = {}
        self._was_held = False

    def process_imu(self, imu: IMUData) -> str | None:
        """
        Process IMU data and return trigger name if reflex detected.

        Returns:
            Trigger name ("picked_up_gentle", "bump", "shake", "set_down") or None
        """
        # Calculate magnitudes
        accel_mag = (imu.accel_x**2 + imu.accel_y**2 + imu.accel_z**2) ** 0.5
        gyro_mag = (imu.gyro_x**2 + imu.gyro_y**2 + imu.gyro_z**2) ** 0.5

        # Track history (last 10 samples)
        self._accel_history.append(accel_mag)
        self._gyro_history.append(gyro_mag)
        if len(self._accel_history) > 10:
            self._accel_history.pop(0)
            self._gyro_history.pop(0)

        now = time.time()

        # Check pickup (high upward acceleration)
        if not self._was_held and accel_mag > self.PICKUP_THRESHOLD:
            if now - self._cooldowns.get("pickup", 0) > 1.0:
                self._cooldowns["pickup"] = now
                self._was_held = True
                if accel_mag > 2.0:
                    return "picked_up_fast"
                return "picked_up_gentle"

        # Check bump (sudden acceleration spike)
        if len(self._accel_history) >= 3:
            prev_avg = sum(self._accel_history[:-1]) / len(self._accel_history[:-1])
            if accel_mag - prev_avg > self.BUMP_THRESHOLD:
                if now - self._cooldowns.get("bump", 0) > 0.5:
                    self._cooldowns["bump"] = now
                    return "bump"

        # Check shake (high gyro)
        if len(self._gyro_history) >= 5:
            avg_gyro = sum(self._gyro_history) / len(self._gyro_history)
            if avg_gyro > self.SHAKE_GYRO_THRESHOLD:
                if now - self._cooldowns.get("shake", 0) > 1.0:
                    self._cooldowns["shake"] = now
                    return "shake"

        # Check set_down (was held, now stable at ~1g)
        if self._was_held and 0.9 < accel_mag < 1.1:
            if len(self._accel_history) >= 5:
                variation = max(self._accel_history) - min(self._accel_history)
                if variation < 0.2:
                    if now - self._cooldowns.get("set_down", 0) > 1.0:
                        self._cooldowns["set_down"] = now
                        self._was_held = False
                        return "set_down"

        return None


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

    # Connection status
    server_connected: bool = False
    last_heartbeat_ms: int = 0

    # Latest IMU values for UI display
    imu_accel_x: float = 0.0
    imu_accel_y: float = 0.0
    imu_accel_z: float = -1.0

    # Video streaming state
    video_enabled: bool = False
    video_streaming: bool = False
    video_connected: bool = False
    webcam_available: bool = False

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
            "server_connected": self.server_connected,
            "last_heartbeat_ms": self.last_heartbeat_ms,
            "imu_accel_x": self.imu_accel_x,
            "imu_accel_y": self.imu_accel_y,
            "imu_accel_z": self.imu_accel_z,
            "video_enabled": self.video_enabled,
            "video_streaming": self.video_streaming,
            "video_connected": self.video_connected,
            "webcam_available": self.webcam_available,
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
        video_enabled: bool = True,
    ) -> None:
        """
        Initialize virtual Pi.

        Args:
            server_host: Server to connect to
            server_port: Server port
            on_state_change: Callback when robot state changes
            video_enabled: Enable webcam video streaming
        """
        self._host = server_host
        self._port = server_port
        self._state = VirtualRobotState()
        self._on_state_change = on_state_change
        self._running = False
        self._websocket = None
        self._video_enabled = video_enabled

        # Movement simulation
        self._move_task: asyncio.Task[None] | None = None
        self._base_speed = 50.0  # units per second at speed=1.0

        # Sensor streaming
        self._sensor_task: asyncio.Task[None] | None = None
        self._sensor_interval_ms = 100  # 10Hz
        self._status_counter = 0  # For periodic PI_STATUS

        # Reflex detection
        self._reflex_processor = EmulatorReflexProcessor()

        # Video streaming
        self._camera: WebcamCamera | MockWebcamCamera | None = None
        self._video_streamer: EmulatorVideoStreamer | None = None
        self._state.video_enabled = video_enabled

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

        # Initialize video if enabled
        if self._video_enabled:
            await self._init_video()

        asyncio.create_task(self._connect_to_server())
        self._sensor_task = asyncio.create_task(self._sensor_loop())
        logger.info("Virtual Pi started")

    async def stop(self) -> None:
        """Stop virtual Pi."""
        self._running = False

        # Shutdown video streaming
        await self._shutdown_video()

        if self._move_task and not self._move_task.done():
            self._move_task.cancel()

        if self._sensor_task and not self._sensor_task.done():
            self._sensor_task.cancel()

        if self._websocket:
            await self._websocket.close()

        logger.info("Virtual Pi stopped")

    async def _init_video(self) -> None:
        """Initialize webcam and video streamer."""
        from .video import EmulatorVideoStreamer, MockWebcamCamera, WebcamCamera

        logger.info("Initializing video streaming...")

        # Try real webcam first
        self._camera = WebcamCamera()
        if await self._camera.initialize():
            self._state.webcam_available = True
            logger.info("Webcam initialized successfully")
        else:
            # Fall back to mock
            logger.warning("Webcam not available, using mock frames")
            self._camera = MockWebcamCamera()
            await self._camera.initialize()
            self._state.webcam_available = False

        # Create video streamer
        self._video_streamer = EmulatorVideoStreamer(
            camera=self._camera,
            on_signaling=self._send_webrtc_signaling,
        )
        logger.info("Video streamer initialized")

    async def _shutdown_video(self) -> None:
        """Shutdown video streaming."""
        if self._video_streamer:
            await self._video_streamer.stop()
            self._video_streamer = None
            self._state.video_streaming = False
            self._state.video_connected = False

        if self._camera:
            await self._camera.shutdown()
            self._camera = None

        logger.info("Video streaming shutdown")

    async def _send_webrtc_signaling(self, msg: RobotMessage) -> None:
        """Send WebRTC signaling message via WebSocket."""
        if self._websocket:
            try:
                await self._websocket.send(msg.serialize())
                logger.debug(f"Sent WebRTC signaling: {msg.message_type.name}")
            except Exception as e:
                logger.warning(f"Failed to send WebRTC signaling: {e}")

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
                    self._state.server_connected = True
                    self._notify_state_change()
                    logger.info("Virtual Pi connected to server")

                    # Start video streaming when connected
                    if self._video_streamer and self._video_enabled:
                        await self._video_streamer.start()
                        self._state.video_streaming = True
                        self._notify_state_change()
                        logger.info("Video streaming started")

                    await self._message_loop()

            except Exception as e:
                logger.warning(f"Connection failed: {e}")
                await asyncio.sleep(2)

            finally:
                self._websocket = None
                self._state.server_connected = False

                # Stop video streaming on disconnect
                if self._video_streamer:
                    await self._video_streamer.stop()
                    self._state.video_streaming = False
                    self._state.video_connected = False

                self._notify_state_change()

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
                        # Echo heartbeat and record timestamp
                        self._state.last_heartbeat_ms = int(time.time() * 1000)
                        await self._send_heartbeat()

                    elif msg.message_type == MessageType.WEBRTC_ANSWER:
                        # Handle WebRTC answer from server
                        if isinstance(msg.payload, WebRTCAnswer) and self._video_streamer:
                            await self._video_streamer.handle_answer(msg.payload.sdp)
                            self._state.video_connected = True
                            self._notify_state_change()
                            logger.info("WebRTC answer received")

                    elif msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE:
                        # Handle ICE candidate from server
                        if isinstance(msg.payload, WebRTCIceCandidate) and self._video_streamer:
                            await self._video_streamer.add_ice_candidate(msg.payload)
                            logger.debug("WebRTC ICE candidate received")

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
        elif isinstance(payload, ScanCommand):
            await self._handle_scan(payload)
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

    async def _handle_scan(self, scan: ScanCommand) -> None:
        """Simulate scanning motion (rotating to look around)."""
        # Map scan types to rotation angles
        angle_map = {
            ScanType.QUICK: 90.0,
            ScanType.PARTIAL: 180.0,
            ScanType.FULL: 360.0,
        }
        angle = angle_map.get(scan.scan_type, 180.0)

        # Perform gradual rotation in steps
        steps = max(1, int(abs(angle) / 15))  # 15 degrees per step
        step_angle = angle / steps

        self._state.is_moving = True
        self._notify_state_change()

        for _ in range(steps):
            self._state.heading = (self._state.heading + step_angle) % 360
            self._notify_state_change()
            await asyncio.sleep(0.1)  # 100ms per step

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

                # Store IMU values for UI display
                self._state.imu_accel_x = imu_data.accel_x
                self._state.imu_accel_y = imu_data.accel_y
                self._state.imu_accel_z = imu_data.accel_z
                self._notify_state_change()

                # Auto-detect reflexes from IMU data
                trigger = self._reflex_processor.process_imu(imu_data)
                if trigger:
                    await self._send_local_trigger(trigger, 0.8)
                    logger.info(f"Auto-detected reflex: {trigger}")

                # Generate and send touch data
                touch_data = self._generate_touch_data()
                await self._send_sensor(SensorData(payload=touch_data))

                # Send PI_STATUS every 5 seconds (50 iterations at 100ms)
                self._status_counter = (self._status_counter + 1) % 50
                if self._status_counter == 0:
                    await self._send_status()

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

    async def _send_local_trigger(self, trigger_name: str, intensity: float = 1.0) -> None:
        """Send local trigger notification to server."""
        if self._websocket:
            try:
                msg = create_local_trigger(trigger_name, intensity)
                await self._websocket.send(msg.serialize())
                logger.debug(f"Sent local trigger: {trigger_name} (intensity={intensity})")
            except Exception as e:
                logger.warning(f"Failed to send local trigger: {e}")

    async def _send_status(self) -> None:
        """Send periodic Pi status to server."""
        if self._websocket:
            try:
                status = PiStatus(
                    cpu_temp=45.0 + random.uniform(-5, 5),
                    cpu_usage=random.uniform(10, 30),
                    memory_usage=random.uniform(20, 40),
                    hardware_ok=True,
                )
                msg = RobotMessage(
                    message_type=MessageType.PI_STATUS,
                    payload=status,
                )
                await self._websocket.send(msg.serialize())
            except Exception as e:
                logger.warning(f"Failed to send status: {e}")

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
        asyncio.create_task(self._send_local_trigger("picked_up_gentle", 0.7))

    def simulate_bump(self, duration: float = 0.2) -> None:
        """Simulate a bump."""
        self._state.simulated_bump = True
        asyncio.create_task(self._clear_event("bump", duration))
        asyncio.create_task(self._send_local_trigger("bump", 0.8))

    def simulate_shake(self, duration: float = 1.0) -> None:
        """Simulate being shaken."""
        self._state.simulated_shake = True
        asyncio.create_task(self._clear_event("shake", duration))
        asyncio.create_task(self._send_local_trigger("shake", 0.9))

    async def _clear_event(self, event: str, duration: float) -> None:
        """Clear simulated event after duration."""
        await asyncio.sleep(duration)
        if event == "pickup":
            self._state.simulated_pickup = False
        elif event == "bump":
            self._state.simulated_bump = False
        elif event == "shake":
            self._state.simulated_shake = False
