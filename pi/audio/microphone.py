"""
Murph - Pi Microphone Capture
Audio input for WebRTC streaming from Pi to server.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .track import MicrophoneAudioTrack

logger = logging.getLogger(__name__)

# Try to import numpy for audio processing
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def detect_active_microphone(
    sample_duration: float = 1.0,
    threshold: float = 0.02,
) -> int | None:
    """
    Auto-detect which microphone is actively picking up sound.

    Tests each available input device by recording briefly and checking
    if it picks up any audio above the noise threshold.

    Args:
        sample_duration: How long to sample each device (seconds)
        threshold: RMS threshold to consider as "hearing something"

    Returns:
        Device index that's hearing sound, or None if none found
    """
    try:
        import sounddevice as sd
        import numpy as np
        devices = sd.query_devices()
    except ImportError:
        logger.warning("sounddevice/numpy not available for auto-detection")
        return None
    except OSError as e:
        # PortAudio not installed (error happens on import or query)
        logger.warning(f"Audio system not available: {e}")
        logger.warning("Install PortAudio: sudo apt install libportaudio2 portaudio19-dev")
        return None

    input_devices = [
        (i, d) for i, d in enumerate(devices)
        if d['max_input_channels'] > 0
    ]

    if not input_devices:
        logger.warning("No input devices found")
        return None

    logger.info(f"Testing {len(input_devices)} input devices for audio...")

    best_device = None
    best_level = 0.0

    for idx, device in input_devices:
        try:
            # Record a short sample from this device
            sample_rate = int(device['default_samplerate'])
            frames = int(sample_rate * sample_duration)

            logger.debug(f"Testing device {idx}: {device['name']}")

            recording = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                device=idx,
                dtype=np.float32,
            )
            sd.wait()  # Wait for recording to complete

            # Calculate RMS level
            rms = float(np.sqrt(np.mean(recording**2)))
            logger.debug(f"  Device {idx} ({device['name']}): RMS={rms:.4f}")

            if rms > best_level:
                best_level = rms
                best_device = idx

        except Exception as e:
            logger.debug(f"  Device {idx} ({device['name']}): Failed - {e}")
            continue

    if best_device is not None and best_level > threshold:
        device_name = devices[best_device]['name']
        logger.info(f"Auto-detected microphone: {best_device} ({device_name}) with level {best_level:.4f}")
        return best_device
    elif best_device is not None:
        device_name = devices[best_device]['name']
        logger.warning(
            f"Best device {best_device} ({device_name}) has low level ({best_level:.4f}). "
            f"Try making noise during startup, or specify --mic-device manually."
        )
        return best_device
    else:
        logger.warning("No working microphone detected")
        return None


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

    @abstractmethod
    def get_audio_chunk(self) -> bytes | None:
        """Get next audio chunk for streaming (or None if empty)."""
        pass

    @abstractmethod
    def create_audio_track(self) -> MicrophoneAudioTrack:
        """Create an aiortc-compatible AudioStreamTrack."""
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

    def get_audio_chunk(self) -> bytes | None:
        """Generate synthetic audio chunk for mock."""
        if not self._running:
            return None

        # 20ms at 16kHz = 320 samples, 16-bit = 640 bytes
        samples = 320
        try:
            import numpy as np

            if self._voice_active:
                # Generate noise to simulate speech
                audio = np.random.randint(-1000, 1000, samples, dtype=np.int16)
            else:
                # Near-silent background noise
                audio = np.random.randint(-10, 10, samples, dtype=np.int16)
            return audio.tobytes()
        except ImportError:
            # Fallback without numpy - return silence
            return bytes(samples * 2)

    def create_audio_track(self) -> MicrophoneAudioTrack:
        """Create an aiortc-compatible AudioStreamTrack."""
        from .track import MicrophoneAudioTrack

        return MicrophoneAudioTrack(self)


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

        # Audio buffer for streaming (thread-safe for sounddevice callback)
        self._audio_buffer: collections.deque[bytes] = collections.deque(maxlen=100)
        self._buffer_lock = threading.Lock()

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
        except ImportError:
            logger.warning("sounddevice not available, using mock microphone")
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
            import numpy as np

            def audio_callback(
                indata: np.ndarray, frames: int, time_info: dict, status: int
            ) -> None:
                if status:
                    logger.warning(f"Audio status: {status}")
                # Calculate RMS level
                rms = float(np.sqrt(np.mean(indata**2)))
                self._audio_level = min(1.0, rms * 10)  # Scale for 0-1 range
                self._voice_detected = self._audio_level > self.VAD_THRESHOLD
                self._samples_captured += frames
                self._last_update_ms = int(time.time() * 1000)

                # Store raw audio for streaming
                with self._buffer_lock:
                    # Convert to 16-bit PCM and store
                    audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                    self._audio_buffer.append(audio_bytes)

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

    def get_audio_chunk(self) -> bytes | None:
        """Get next audio chunk for streaming (or None if empty)."""
        if self._mock:
            return self._mock.get_audio_chunk()
        with self._buffer_lock:
            if self._audio_buffer:
                return self._audio_buffer.popleft()
            return None

    def create_audio_track(self) -> MicrophoneAudioTrack:
        """Create an aiortc-compatible AudioStreamTrack."""
        from .track import MicrophoneAudioTrack

        return MicrophoneAudioTrack(self)
