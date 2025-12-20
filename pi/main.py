"""
Murph - Raspberry Pi Client
Main entry point for the robot's body (sensors, actuators, local behaviors).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    PERCEPTION_CYCLE_MS,
)
from shared.messages import Command, RobotMessage, SensorData, WebRTCAnswer, WebRTCIceCandidate

# Import video components
from pi.video import CameraManager, MockCameraManager, OpenCVCameraManager, VideoStreamer

# Import audio components
from pi.audio import MicrophoneCapture, MockMicrophoneCapture, BaseMicrophoneCapture

# Import actuators (mock by default, real hardware if on Pi)
from pi.actuators import (
    MockMotorController,
    MockDisplayController,
    MockAudioController,
    MotorController,
    DisplayController,
    AudioController,
    PygameDisplayController,
)

# Import sensors (mock by default)
from pi.sensors import (
    MockIMUSensor,
    MockTouchSensor,
)

# Import communication
from pi.communication import ServerConnection, CommandHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("murph.pi")


class PiClient:
    """
    Main Pi client that coordinates all hardware and communication.

    Manages:
    - Hardware initialization (actuators and sensors)
    - WebSocket connection to server brain
    - Command handling from server
    - Sensor data streaming to server
    - Local behavior reflexes (optional)
    """

    def __init__(
        self,
        server_host: str = DEFAULT_SERVER_HOST,
        server_port: int = DEFAULT_SERVER_PORT,
        use_real_hardware: bool = False,
        camera_type: str = "auto",
        display_type: str = "auto",
        enable_microphone: bool = True,
    ) -> None:
        """
        Initialize Pi client.

        Args:
            server_host: Server brain host address
            server_port: Server brain port
            use_real_hardware: If True, attempt to use real hardware drivers
            camera_type: "auto", "picamera", "opencv", or "mock"
            display_type: "auto", "oled", "pygame", or "mock"
            enable_microphone: If True, enable microphone for audio input
        """
        self._server_host = server_host
        self._server_port = server_port
        self._use_real_hardware = use_real_hardware
        self._camera_type = camera_type
        self._display_type = display_type
        self._enable_microphone = enable_microphone
        self._running = False

        # Hardware components (initialized in start())
        self._motors: MotorController | None = None
        self._display: DisplayController | None = None
        self._speaker: AudioController | None = None
        self._imu = None
        self._touch = None

        # Video and audio streaming
        self._camera: CameraManager | MockCameraManager | OpenCVCameraManager | None = None
        self._microphone: BaseMicrophoneCapture | None = None
        self._video_streamer: VideoStreamer | None = None

        # Communication
        self._connection: ServerConnection | None = None
        self._command_handler: CommandHandler | None = None

        # Sensor streaming
        self._sensor_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the Pi client."""
        logger.info("Starting Murph Pi client...")

        # Initialize hardware
        await self._init_hardware()

        # Initialize video
        await self._init_video()

        # Setup command handler
        self._command_handler = CommandHandler(
            motors=self._motors,
            display=self._display,
            speaker=self._speaker,
        )

        # Setup server connection with WebRTC callbacks
        self._connection = ServerConnection(
            host=self._server_host,
            port=self._server_port,
            on_command=self._on_command,
            on_connection_change=self._on_connection_change,
            on_webrtc_answer=self._on_webrtc_answer,
            on_webrtc_ice_candidate=self._on_webrtc_ice_candidate,
        )

        self._running = True

        # Start sensor streaming
        self._sensor_task = asyncio.create_task(self._sensor_loop())

        # Connect to server (runs indefinitely with reconnection)
        await self._connection.connect()

    async def stop(self) -> None:
        """Stop the Pi client."""
        logger.info("Stopping Murph Pi client...")
        self._running = False

        # Stop video streaming
        await self._shutdown_video()

        # Stop sensor streaming
        if self._sensor_task and not self._sensor_task.done():
            self._sensor_task.cancel()
            try:
                await self._sensor_task
            except asyncio.CancelledError:
                pass

        # Disconnect from server
        if self._connection:
            await self._connection.disconnect()

        # Shutdown hardware
        await self._shutdown_hardware()

        logger.info("Murph Pi client stopped")

    async def _init_hardware(self) -> None:
        """Initialize all hardware components."""
        logger.info("Initializing hardware...")

        if self._use_real_hardware:
            await self._init_real_hardware()
        else:
            await self._init_mock_hardware()

        logger.info("Hardware initialized")

    async def _init_mock_hardware(self) -> None:
        """Initialize mock hardware for testing."""
        # Motors
        self._motors = MockMotorController()
        await self._motors.initialize()

        # Display
        self._display = MockDisplayController()
        await self._display.initialize()

        # Speaker
        self._speaker = MockAudioController()
        await self._speaker.initialize()

        # IMU
        self._imu = MockIMUSensor()
        await self._imu.initialize()

        # Touch
        self._touch = MockTouchSensor()
        await self._touch.initialize()

        logger.info("Mock hardware initialized")

    async def _init_real_hardware(self) -> None:
        """
        Initialize real hardware drivers.

        Falls back to mock if real hardware unavailable.
        """
        try:
            from pi.actuators import DRV8833MotorController
            self._motors = DRV8833MotorController()
            if not await self._motors.initialize():
                logger.warning("Real motors failed, using mock")
                self._motors = MockMotorController()
                await self._motors.initialize()
        except ImportError:
            logger.warning("DRV8833 driver not available, using mock")
            self._motors = MockMotorController()
            await self._motors.initialize()

        # Display: support OLED, pygame, or mock based on display_type
        if self._display_type == "pygame":
            # Use pygame for HDMI display
            self._display = PygameDisplayController()
            if not await self._display.initialize():
                logger.warning("Pygame display failed, using mock")
                self._display = MockDisplayController()
                await self._display.initialize()
        elif self._display_type == "oled" or self._display_type == "auto":
            # Try OLED first (auto or explicit oled)
            try:
                from pi.actuators import SSD1306DisplayController
                self._display = SSD1306DisplayController()
                if not await self._display.initialize():
                    logger.warning("Real display failed, trying pygame")
                    # Try pygame as fallback for auto mode
                    if self._display_type == "auto":
                        self._display = PygameDisplayController()
                        if not await self._display.initialize():
                            logger.warning("Pygame display also failed, using mock")
                            self._display = MockDisplayController()
                            await self._display.initialize()
                    else:
                        self._display = MockDisplayController()
                        await self._display.initialize()
            except ImportError:
                logger.warning("SSD1306 driver not available")
                if self._display_type == "auto":
                    # Try pygame as fallback
                    self._display = PygameDisplayController()
                    if not await self._display.initialize():
                        logger.warning("Pygame display also failed, using mock")
                        self._display = MockDisplayController()
                        await self._display.initialize()
                else:
                    self._display = MockDisplayController()
                    await self._display.initialize()
        else:
            # Mock display
            self._display = MockDisplayController()
            await self._display.initialize()

        try:
            from pi.actuators import MAX98357AudioController
            self._speaker = MAX98357AudioController()
            if not await self._speaker.initialize():
                logger.warning("Real speaker failed, using mock")
                self._speaker = MockAudioController()
                await self._speaker.initialize()
        except ImportError:
            logger.warning("MAX98357 driver not available, using mock")
            self._speaker = MockAudioController()
            await self._speaker.initialize()

        try:
            from pi.sensors import MPU6050IMUSensor
            self._imu = MPU6050IMUSensor()
            if not await self._imu.initialize():
                logger.warning("Real IMU failed, using mock")
                self._imu = MockIMUSensor()
                await self._imu.initialize()
        except ImportError:
            logger.warning("MPU6050 driver not available, using mock")
            self._imu = MockIMUSensor()
            await self._imu.initialize()

        try:
            from pi.sensors import MPR121TouchSensor
            self._touch = MPR121TouchSensor()
            if not await self._touch.initialize():
                logger.warning("Real touch sensor failed, using mock")
                self._touch = MockTouchSensor()
                await self._touch.initialize()
        except ImportError:
            logger.warning("MPR121 driver not available, using mock")
            self._touch = MockTouchSensor()
            await self._touch.initialize()

        logger.info("Hardware initialized (real where available)")

    async def _shutdown_hardware(self) -> None:
        """Shutdown all hardware safely."""
        if self._motors:
            await self._motors.shutdown()
        if self._display:
            await self._display.shutdown()
        if self._speaker:
            await self._speaker.shutdown()
        if self._imu:
            await self._imu.shutdown()
        if self._touch:
            await self._touch.shutdown()

    async def _init_video(self) -> None:
        """Initialize camera, microphone, and video streaming."""
        logger.info("Initializing video and audio...")

        # Initialize camera based on camera_type
        if not self._use_real_hardware:
            self._camera = MockCameraManager()
            await self._camera.initialize()
        elif self._camera_type == "opencv":
            # Explicitly use OpenCV (USB webcam)
            self._camera = OpenCVCameraManager()
            if not await self._camera.initialize():
                logger.warning("OpenCV camera failed, using mock")
                self._camera = MockCameraManager()
                await self._camera.initialize()
        elif self._camera_type == "picamera":
            # Explicitly use PiCamera2
            self._camera = CameraManager()
            if not await self._camera.initialize():
                logger.warning("PiCamera2 failed, using mock")
                self._camera = MockCameraManager()
                await self._camera.initialize()
        elif self._camera_type == "auto":
            # Auto-detect: try PiCamera2 first, then OpenCV, then mock
            self._camera = CameraManager()
            if not await self._camera.initialize():
                logger.warning("PiCamera2 not available, trying OpenCV...")
                self._camera = OpenCVCameraManager()
                if not await self._camera.initialize():
                    logger.warning("OpenCV camera also failed, using mock")
                    self._camera = MockCameraManager()
                    await self._camera.initialize()
        else:
            # Mock camera
            self._camera = MockCameraManager()
            await self._camera.initialize()

        # Initialize microphone if enabled
        if self._enable_microphone and self._use_real_hardware:
            self._microphone = MicrophoneCapture()
            # MicrophoneCapture initializes in constructor, start() is called by streamer
            logger.info("Microphone capture initialized")
        elif self._enable_microphone:
            self._microphone = MockMicrophoneCapture()
            logger.info("Mock microphone initialized")
        else:
            self._microphone = None
            logger.info("Microphone disabled")

        # Create video streamer with optional microphone
        self._video_streamer = VideoStreamer(
            camera=self._camera,
            microphone=self._microphone,
            on_signaling=self._send_webrtc_signaling,
        )

        logger.info("Video and audio initialized")

    async def _shutdown_video(self) -> None:
        """Shutdown video streaming."""
        if self._video_streamer:
            await self._video_streamer.stop()
            self._video_streamer = None

        if self._camera:
            await self._camera.shutdown()
            self._camera = None

    async def _send_webrtc_signaling(self, msg: RobotMessage) -> None:
        """Send WebRTC signaling message to server via WebSocket."""
        if self._connection and self._connection.is_connected:
            await self._connection.send_message(msg)

    def _on_webrtc_answer(self, answer: WebRTCAnswer) -> None:
        """Handle WebRTC answer from server."""
        if self._video_streamer:
            asyncio.create_task(self._video_streamer.handle_answer(answer.sdp))

    def _on_webrtc_ice_candidate(self, candidate: WebRTCIceCandidate) -> None:
        """Handle WebRTC ICE candidate from server."""
        if self._video_streamer:
            asyncio.create_task(self._video_streamer.add_ice_candidate(candidate))

    def _on_command(self, command: Command) -> None:
        """Handle incoming command from server."""
        if self._command_handler:
            asyncio.create_task(self._command_handler.handle_command(command))

    def _on_connection_change(self, connected: bool) -> None:
        """Handle connection state changes."""
        if connected:
            logger.info("Connected to server brain")
            # Show happy expression on connect
            if self._display:
                asyncio.create_task(self._display.set_expression("happy"))
            # Start video streaming
            if self._video_streamer:
                asyncio.create_task(self._video_streamer.start())
        else:
            logger.warning("Disconnected from server brain")
            # Show sad expression on disconnect
            if self._display:
                asyncio.create_task(self._display.set_expression("sad"))
            # Stop video streaming
            if self._video_streamer:
                asyncio.create_task(self._video_streamer.stop())

    async def _sensor_loop(self) -> None:
        """Stream sensor data to server."""
        interval = PERCEPTION_CYCLE_MS / 1000.0

        while self._running:
            if self._connection and self._connection.is_connected:
                # Read and send IMU data
                if self._imu:
                    imu_data = await self._imu.read()
                    await self._connection.send_sensor_data(
                        SensorData(payload=imu_data)
                    )

                # Read and send touch data
                if self._touch:
                    touch_data = await self._touch.read()
                    await self._connection.send_sensor_data(
                        SensorData(payload=touch_data)
                    )

            await asyncio.sleep(interval)


async def main(
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    use_real_hardware: bool = False,
    camera_type: str = "auto",
    display_type: str = "auto",
    enable_microphone: bool = True,
) -> None:
    """Main entry point for the Pi client."""
    client = PiClient(
        server_host=host,
        server_port=port,
        use_real_hardware=use_real_hardware,
        camera_type=camera_type,
        display_type=display_type,
        enable_microphone=enable_microphone,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(client.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await client.start()
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await client.stop()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Murph Raspberry Pi Client"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help=f"Server host (default: {DEFAULT_SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Server port (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="Attempt to use real hardware drivers",
    )
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera", "opencv", "mock"],
        default="auto",
        help="Camera type: auto (try PiCamera2 then OpenCV), picamera (Pi Camera Module), "
             "opencv (USB webcam), mock (test pattern) (default: auto)",
    )
    parser.add_argument(
        "--display",
        choices=["auto", "oled", "pygame", "mock"],
        default="auto",
        help="Display type: auto (try OLED then pygame), oled (I2C SSD1306), "
             "pygame (HDMI window), mock (log only) (default: auto)",
    )
    parser.add_argument(
        "--no-microphone",
        action="store_true",
        help="Disable microphone audio input",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main(
        host=args.host,
        port=args.port,
        use_real_hardware=args.real_hardware,
        camera_type=args.camera,
        display_type=args.display,
        enable_microphone=not args.no_microphone,
    ))
