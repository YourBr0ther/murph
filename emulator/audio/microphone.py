"""
Murph - Emulator Microphone Capture
Audio input simulation for the emulator.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import sounddevice for real audio capture
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class AudioState:
    """Current state of audio capture."""

    is_running: bool = False
    audio_level: float = 0.0  # RMS amplitude 0.0-1.0
    is_voice_detected: bool = False
    samples_captured: int = 0
    last_update_ms: int = 0


class BaseMicrophoneCapture(ABC):
    """Base class for microphone capture."""

    @abstractmethod
    async def start(self) -> bool:
        """Start audio capture. Returns True if successful."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop audio capture."""
        pass

    @abstractmethod
    def get_audio_level(self) -> float:
        """Get current RMS audio level (0.0 to 1.0)."""
        pass

    @abstractmethod
    def is_voice_detected(self) -> bool:
        """Check if voice activity is detected."""
        pass

    @abstractmethod
    def get_state(self) -> AudioState:
        """Get current audio state."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if microphone is available."""
        pass


class MockMicrophoneCapture(BaseMicrophoneCapture):
    """
    Mock microphone that generates synthetic audio levels.
    Used when no real microphone is available or for testing.
    """

    def __init__(
        self,
        base_noise_level: float = 0.05,
        voice_probability: float = 0.1,
        voice_duration_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """
        Initialize mock microphone.

        Args:
            base_noise_level: Base ambient noise level (0.0-1.0)
            voice_probability: Probability of simulated voice activity per check
            voice_duration_range: (min, max) duration in seconds for voice events
        """
        self._base_noise = base_noise_level
        self._voice_probability = voice_probability
        self._voice_duration_range = voice_duration_range

        self._running = False
        self._audio_level = 0.0
        self._voice_detected = False
        self._samples_captured = 0
        self._last_update_ms = 0

        # Voice simulation state
        self._voice_active = False
        self._voice_end_time = 0.0
        self._update_task: asyncio.Task | None = None

    @property
    def is_available(self) -> bool:
        """Mock is always available."""
        return True

    async def start(self) -> bool:
        """Start mock audio generation."""
        if self._running:
            return True

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Mock microphone started")
        return True

    async def stop(self) -> None:
        """Stop mock audio generation."""
        self._running = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self._update_task = None
        logger.info("Mock microphone stopped")

    async def _update_loop(self) -> None:
        """Update audio levels periodically."""
        while self._running:
            self._update_audio_state()
            await asyncio.sleep(0.05)  # 20Hz update rate

    def _update_audio_state(self) -> None:
        """Update simulated audio levels."""
        now = time.time()

        # Check if current voice event should end
        if self._voice_active and now > self._voice_end_time:
            self._voice_active = False

        # Maybe start a new voice event
        if not self._voice_active and random.random() < self._voice_probability * 0.05:
            self._voice_active = True
            duration = random.uniform(*self._voice_duration_range)
            self._voice_end_time = now + duration

        # Calculate audio level
        if self._voice_active:
            # Voice: higher level with variation
            self._audio_level = min(1.0, self._base_noise + random.uniform(0.2, 0.7))
            self._voice_detected = True
        else:
            # Background noise
            self._audio_level = self._base_noise + random.gauss(0, 0.02)
            self._audio_level = max(0.0, min(1.0, self._audio_level))
            self._voice_detected = False

        self._samples_captured += 1
        self._last_update_ms = int(now * 1000)

    def get_audio_level(self) -> float:
        """Get current simulated audio level."""
        return self._audio_level

    def is_voice_detected(self) -> bool:
        """Check if simulated voice activity is occurring."""
        return self._voice_detected

    def get_state(self) -> AudioState:
        """Get current audio state."""
        return AudioState(
            is_running=self._running,
            audio_level=self._audio_level,
            is_voice_detected=self._voice_detected,
            samples_captured=self._samples_captured,
            last_update_ms=self._last_update_ms,
        )

    def simulate_voice(self, duration: float = 1.0) -> None:
        """Manually trigger a simulated voice event."""
        self._voice_active = True
        self._voice_end_time = time.time() + duration

    def simulate_silence(self) -> None:
        """Manually end any voice simulation."""
        self._voice_active = False
        self._voice_end_time = 0.0


class MicrophoneCapture(BaseMicrophoneCapture):
    """
    Real microphone capture using sounddevice/numpy.
    Falls back to MockMicrophoneCapture if hardware unavailable.
    """

    # Voice activity detection threshold
    VAD_THRESHOLD = 0.15  # Audio level above this is considered voice

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        device: int | str | None = None,
    ) -> None:
        """
        Initialize microphone capture.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Samples per chunk
            device: Audio device index or name (None for default)
        """
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._device = device

        self._running = False
        self._audio_level = 0.0
        self._voice_detected = False
        self._samples_captured = 0
        self._last_update_ms = 0

        self._stream = None
        self._initialized = False

        # Check if we can use real audio
        if not HAS_NUMPY:
            logger.warning("numpy not available, microphone will use mock data")
            self._mock = MockMicrophoneCapture()
        else:
            self._mock = None
            self._try_initialize()

    def _try_initialize(self) -> None:
        """Try to initialize sounddevice."""
        try:
            import sounddevice as sd

            # Check if we can access the device
            devices = sd.query_devices()
            if devices:
                self._initialized = True
                logger.info(f"Microphone initialized with {len(devices)} audio devices")
            else:
                logger.warning("No audio devices found, using mock")
                self._mock = MockMicrophoneCapture()
        except Exception as e:
            logger.warning(f"Failed to initialize audio: {e}, using mock")
            self._mock = MockMicrophoneCapture()

    @property
    def is_available(self) -> bool:
        """Check if real microphone is available."""
        return self._initialized and self._mock is None

    async def start(self) -> bool:
        """Start audio capture."""
        if self._mock:
            return await self._mock.start()

        if self._running:
            return True

        try:
            import sounddevice as sd

            def audio_callback(
                indata: "np.ndarray", frames: int, time_info: dict, status: int
            ) -> None:
                if status:
                    logger.warning(f"Audio status: {status}")
                # Calculate RMS level
                rms = float(np.sqrt(np.mean(indata**2)))
                self._audio_level = min(1.0, rms * 10)  # Scale for 0-1 range
                self._voice_detected = self._audio_level > self.VAD_THRESHOLD
                self._samples_captured += frames
                self._last_update_ms = int(time.time() * 1000)

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                device=self._device,
                channels=1,
                callback=audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info("Microphone capture started")
            return True
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            # Fall back to mock
            self._mock = MockMicrophoneCapture()
            return await self._mock.start()

    async def stop(self) -> None:
        """Stop audio capture."""
        if self._mock:
            return await self._mock.stop()

        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Microphone capture stopped")

    def get_audio_level(self) -> float:
        """Get current audio level."""
        if self._mock:
            return self._mock.get_audio_level()
        return self._audio_level

    def is_voice_detected(self) -> bool:
        """Check if voice activity is detected."""
        if self._mock:
            return self._mock.is_voice_detected()
        return self._voice_detected

    def get_state(self) -> AudioState:
        """Get current audio state."""
        if self._mock:
            return self._mock.get_state()
        return AudioState(
            is_running=self._running,
            audio_level=self._audio_level,
            is_voice_detected=self._voice_detected,
            samples_captured=self._samples_captured,
            last_update_ms=self._last_update_ms,
        )
