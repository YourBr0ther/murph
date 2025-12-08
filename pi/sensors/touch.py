"""
Murph - Touch Sensor
MPR121 capacitive touch sensor for detecting petting/interaction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .base import StreamingSensor
from shared.messages import TouchData

logger = logging.getLogger(__name__)


class MockTouchSensor(StreamingSensor):
    """
    Mock touch sensor for testing without hardware.

    Simulates 12-electrode capacitive touch sensor.
    """

    NUM_ELECTRODES = 12

    def __init__(self) -> None:
        self._ready = False
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._callback: Callable[[TouchData], None] | None = None
        self._interval_ms = 100.0

        # Simulation state
        self._touched_electrodes: set[int] = set()

    @property
    def name(self) -> str:
        return "MockTouchSensor"

    async def initialize(self) -> bool:
        """Initialize mock touch sensor."""
        self._ready = True
        logger.info("MockTouchSensor initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown mock touch sensor."""
        await self.stop_streaming()
        self._ready = False
        logger.info("MockTouchSensor shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def read(self) -> TouchData:
        """Read current touch state."""
        return TouchData(
            touched_electrodes=list(self._touched_electrodes),
            is_touched=len(self._touched_electrodes) > 0,
        )

    async def start_streaming(
        self,
        callback: Callable[[TouchData], None],
        interval_ms: float = 100,
    ) -> None:
        """Start streaming touch data."""
        if not self._ready:
            logger.warning("Touch sensor not ready")
            return

        self._callback = callback
        self._interval_ms = interval_ms
        self._streaming = True

        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.debug(f"Touch streaming started at {interval_ms}ms interval")

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
        """Send touch data at regular intervals."""
        while self._streaming:
            data = await self.read()
            if self._callback:
                self._callback(data)
            await asyncio.sleep(self._interval_ms / 1000.0)

    # Simulation controls for testing/emulator

    def simulate_touch(self, electrodes: list[int]) -> None:
        """
        Simulate touching specific electrodes.

        Args:
            electrodes: List of electrode indices (0-11) being touched
        """
        self._touched_electrodes = {
            e for e in electrodes if 0 <= e < self.NUM_ELECTRODES
        }
        logger.debug(f"Simulating touch on electrodes: {electrodes}")

    def simulate_release(self) -> None:
        """Simulate releasing all touch."""
        self._touched_electrodes.clear()
        logger.debug("Simulating touch release")

    def simulate_pet(self) -> None:
        """Simulate a petting gesture (multiple electrodes)."""
        # Simulate touching multiple adjacent electrodes
        self._touched_electrodes = {3, 4, 5, 6}
        logger.debug("Simulating pet gesture")


class MPR121TouchSensor(StreamingSensor):
    """
    Real hardware implementation using MPR121 capacitive touch sensor.

    Communicates via I2C.
    """

    I2C_ADDRESS = 0x5A
    TOUCH_STATUS_REG = 0x00
    NUM_ELECTRODES = 12

    def __init__(self) -> None:
        self._ready = False
        self._mpr121 = None
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._callback: Callable[[TouchData], None] | None = None

    @property
    def name(self) -> str:
        return "MPR121TouchSensor"

    async def initialize(self) -> bool:
        """Initialize I2C communication with MPR121."""
        try:
            import board
            import adafruit_mpr121

            i2c = board.I2C()
            self._mpr121 = adafruit_mpr121.MPR121(i2c, address=self.I2C_ADDRESS)

            self._ready = True
            logger.info("MPR121TouchSensor initialized")
            return True

        except ImportError:
            logger.error("adafruit_mpr121 not available")
            return False
        except Exception as e:
            logger.error(f"Touch sensor init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown touch sensor."""
        await self.stop_streaming()
        self._ready = False
        logger.info("MPR121TouchSensor shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def read(self) -> TouchData:
        """Read current touch state."""
        if not self._ready or not self._mpr121:
            return TouchData()

        try:
            touched = []
            for i in range(self.NUM_ELECTRODES):
                if self._mpr121[i].value:
                    touched.append(i)

            return TouchData(
                touched_electrodes=touched,
                is_touched=len(touched) > 0,
            )

        except Exception as e:
            logger.error(f"Touch read failed: {e}")
            return TouchData()

    async def start_streaming(
        self,
        callback: Callable[[TouchData], None],
        interval_ms: float = 100,
    ) -> None:
        """Start streaming touch data."""
        if not self._ready:
            return

        self._callback = callback
        self._streaming = True
        self._stream_task = asyncio.create_task(
            self._stream_loop(interval_ms)
        )

    async def _stream_loop(self, interval_ms: float) -> None:
        """Read and send touch data at regular intervals."""
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
