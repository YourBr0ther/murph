"""
Murph - IMU Sensor
MPU6050 accelerometer/gyroscope for motion detection.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable

from .base import StreamingSensor
from shared.messages import IMUData

logger = logging.getLogger(__name__)


class MockIMUSensor(StreamingSensor):
    """
    Mock IMU sensor for testing without hardware.

    Generates simulated IMU data with configurable noise and events.
    Can simulate pickup, bump, and shake events for testing.
    """

    def __init__(self) -> None:
        self._ready = False
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._callback: Callable[[IMUData], None] | None = None
        self._interval_ms = 100.0

        # Simulation state
        self._noise_level = 0.02  # Baseline sensor noise
        self._simulated_accel = [0.0, 0.0, -1.0]  # At rest: 1g downward
        self._simulated_gyro = [0.0, 0.0, 0.0]
        self._temperature = 25.0

        # Event simulation
        self._simulate_pickup = False
        self._simulate_bump = False
        self._simulate_shake = False
        self._event_duration = 0.0

    @property
    def name(self) -> str:
        return "MockIMUSensor"

    async def initialize(self) -> bool:
        """Initialize mock IMU."""
        self._ready = True
        logger.info("MockIMUSensor initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown mock IMU."""
        await self.stop_streaming()
        self._ready = False
        logger.info("MockIMUSensor shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def read(self) -> IMUData:
        """Read current IMU data."""
        return self._generate_data()

    async def start_streaming(
        self,
        callback: Callable[[IMUData], None],
        interval_ms: float = 100,
    ) -> None:
        """Start streaming IMU data."""
        if not self._ready:
            logger.warning("IMU not ready")
            return

        self._callback = callback
        self._interval_ms = interval_ms
        self._streaming = True

        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.debug(f"IMU streaming started at {interval_ms}ms interval")

    async def stop_streaming(self) -> None:
        """Stop streaming."""
        self._streaming = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    def is_streaming(self) -> bool:
        return self._streaming

    async def _stream_loop(self) -> None:
        """Generate and send IMU data at regular intervals."""
        while self._streaming:
            data = self._generate_data()
            if self._callback:
                self._callback(data)
            await asyncio.sleep(self._interval_ms / 1000.0)

    def _generate_data(self) -> IMUData:
        """Generate simulated IMU data with noise and events."""
        # Base values (at rest)
        accel_x = self._simulated_accel[0]
        accel_y = self._simulated_accel[1]
        accel_z = self._simulated_accel[2]
        gyro_x = self._simulated_gyro[0]
        gyro_y = self._simulated_gyro[1]
        gyro_z = self._simulated_gyro[2]

        # Add events
        if self._simulate_pickup:
            # High upward acceleration
            accel_z += 1.0
            self._event_duration -= self._interval_ms / 1000.0
            if self._event_duration <= 0:
                self._simulate_pickup = False

        if self._simulate_bump:
            # High sudden deceleration
            accel_x += random.uniform(2.0, 3.0)
            self._event_duration -= self._interval_ms / 1000.0
            if self._event_duration <= 0:
                self._simulate_bump = False

        if self._simulate_shake:
            # High-frequency oscillation
            accel_x += random.uniform(-2.0, 2.0)
            accel_y += random.uniform(-2.0, 2.0)
            gyro_z += random.uniform(-100, 100)
            self._event_duration -= self._interval_ms / 1000.0
            if self._event_duration <= 0:
                self._simulate_shake = False

        # Add sensor noise
        accel_x += random.gauss(0, self._noise_level)
        accel_y += random.gauss(0, self._noise_level)
        accel_z += random.gauss(0, self._noise_level)
        gyro_x += random.gauss(0, self._noise_level * 10)
        gyro_y += random.gauss(0, self._noise_level * 10)
        gyro_z += random.gauss(0, self._noise_level * 10)

        return IMUData(
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            temperature=self._temperature,
        )

    # Simulation controls for testing/emulator

    def simulate_pickup(self, duration: float = 0.5) -> None:
        """Simulate being picked up."""
        self._simulate_pickup = True
        self._event_duration = duration
        logger.debug("Simulating pickup event")

    def simulate_bump(self, duration: float = 0.2) -> None:
        """Simulate a bump/collision."""
        self._simulate_bump = True
        self._event_duration = duration
        logger.debug("Simulating bump event")

    def simulate_shake(self, duration: float = 1.0) -> None:
        """Simulate being shaken."""
        self._simulate_shake = True
        self._event_duration = duration
        logger.debug("Simulating shake event")

    def set_motion_state(
        self,
        accel: tuple[float, float, float] | None = None,
        gyro: tuple[float, float, float] | None = None,
    ) -> None:
        """Set base motion state (for emulator position tracking)."""
        if accel:
            self._simulated_accel = list(accel)
        if gyro:
            self._simulated_gyro = list(gyro)


class MPU6050IMUSensor(StreamingSensor):
    """
    Real hardware implementation using MPU6050 IMU.

    Communicates via I2C.
    """

    I2C_ADDRESS = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    TEMP_OUT_H = 0x41

    ACCEL_SCALE = 16384.0  # +/- 2g
    GYRO_SCALE = 131.0  # +/- 250 deg/s
    TEMP_SCALE = 340.0
    TEMP_OFFSET = 36.53

    def __init__(self) -> None:
        self._ready = False
        self._bus = None
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._callback: Callable[[IMUData], None] | None = None

    @property
    def name(self) -> str:
        return "MPU6050IMUSensor"

    async def initialize(self) -> bool:
        """Initialize I2C communication with MPU6050."""
        try:
            import smbus2

            self._bus = smbus2.SMBus(1)

            # Wake up the MPU6050
            self._bus.write_byte_data(self.I2C_ADDRESS, self.PWR_MGMT_1, 0)

            self._ready = True
            logger.info("MPU6050IMUSensor initialized")
            return True

        except ImportError:
            logger.error("smbus2 not available")
            return False
        except Exception as e:
            logger.error(f"IMU init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown IMU."""
        await self.stop_streaming()
        if self._bus:
            self._bus.close()
        self._ready = False
        logger.info("MPU6050IMUSensor shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def read(self) -> IMUData:
        """Read current IMU values."""
        if not self._ready or not self._bus:
            return IMUData()

        try:
            # Read all data in one burst
            data = self._bus.read_i2c_block_data(
                self.I2C_ADDRESS, self.ACCEL_XOUT_H, 14
            )

            # Parse accelerometer (first 6 bytes)
            accel_x = self._convert_raw(data[0], data[1]) / self.ACCEL_SCALE
            accel_y = self._convert_raw(data[2], data[3]) / self.ACCEL_SCALE
            accel_z = self._convert_raw(data[4], data[5]) / self.ACCEL_SCALE

            # Parse temperature (bytes 6-7)
            temp_raw = self._convert_raw(data[6], data[7])
            temperature = temp_raw / self.TEMP_SCALE + self.TEMP_OFFSET

            # Parse gyroscope (bytes 8-13)
            gyro_x = self._convert_raw(data[8], data[9]) / self.GYRO_SCALE
            gyro_y = self._convert_raw(data[10], data[11]) / self.GYRO_SCALE
            gyro_z = self._convert_raw(data[12], data[13]) / self.GYRO_SCALE

            return IMUData(
                accel_x=accel_x,
                accel_y=accel_y,
                accel_z=accel_z,
                gyro_x=gyro_x,
                gyro_y=gyro_y,
                gyro_z=gyro_z,
                temperature=temperature,
            )

        except Exception as e:
            logger.error(f"IMU read failed: {e}")
            return IMUData()

    def _convert_raw(self, high: int, low: int) -> int:
        """Convert two bytes to signed 16-bit value."""
        value = (high << 8) | low
        if value >= 0x8000:
            value -= 0x10000
        return value

    async def start_streaming(
        self,
        callback: Callable[[IMUData], None],
        interval_ms: float = 100,
    ) -> None:
        """Start streaming IMU data."""
        if not self._ready:
            return

        self._callback = callback
        self._streaming = True
        self._stream_task = asyncio.create_task(
            self._stream_loop(interval_ms)
        )

    async def _stream_loop(self, interval_ms: float) -> None:
        """Read and send IMU data at regular intervals."""
        while self._streaming:
            data = await self.read()
            if self._callback:
                self._callback(data)
            await asyncio.sleep(interval_ms / 1000.0)

    async def stop_streaming(self) -> None:
        """Stop streaming."""
        self._streaming = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

    def is_streaming(self) -> bool:
        return self._streaming
