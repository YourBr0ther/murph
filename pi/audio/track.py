"""
Murph - Pi Audio Track
aiortc-compatible audio streaming track for WebRTC.
"""

from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from shared.constants import AUDIO_SAMPLE_RATE_STT, VAD_SILENCE_DURATION_MS

if TYPE_CHECKING:
    from av import AudioFrame

    from .microphone import BaseMicrophoneCapture

logger = logging.getLogger(__name__)


class MicrophoneAudioTrack:
    """
    aiortc-compatible AudioStreamTrack wrapper for MicrophoneCapture.

    This class properly inherits from aiortc.AudioStreamTrack to ensure
    all required properties and methods are available for WebRTC.

    VAD-gated: sends actual audio frames when voice is detected,
    silence frames when not. This reduces bandwidth while maintaining
    a consistent frame rate for WebRTC.

    Audio format:
    - Sample rate: 16kHz (for Whisper STT compatibility)
    - Channels: 1 (mono)
    - Bit depth: 16-bit signed PCM (s16)
    - Frame duration: 20ms (320 samples)
    """

    kind = "audio"

    # Frame configuration
    FRAME_DURATION_MS = 20
    SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE_STT * FRAME_DURATION_MS / 1000)  # 320

    def __new__(cls, microphone: BaseMicrophoneCapture, sample_rate: int = AUDIO_SAMPLE_RATE_STT) -> Any:
        """Create audio track that properly inherits from aiortc.AudioStreamTrack."""
        try:
            from aiortc import AudioStreamTrack as BaseTrack
        except ImportError:
            # Fallback for testing without aiortc
            logger.warning("aiortc not available, using basic audio track")
            return object.__new__(cls)

        class _MicrophoneAudioTrack(BaseTrack):
            def __init__(self, mic: BaseMicrophoneCapture, rate: int) -> None:
                super().__init__()
                self._microphone = mic
                self._sample_rate = rate
                self._timestamp = 0
                self._start_time = time.time()
                self._frame_count = 0

                # VAD state tracking for utterance detection
                self._is_speaking = False
                self._silence_start: float | None = None

                logger.info(
                    f"MicrophoneAudioTrack initialized: {rate}Hz, "
                    f"{cls.FRAME_DURATION_MS}ms frames ({cls.SAMPLES_PER_FRAME} samples)"
                )

            async def recv(self) -> Any:
                """Receive the next audio frame."""
                try:
                    from av import AudioFrame as AVAudioFrame
                except ImportError:
                    raise RuntimeError("av (PyAV) package required for audio streaming")

                # Frame pacing - sleep to match target frame rate
                await asyncio.sleep(cls.FRAME_DURATION_MS / 1000)

                # Always send real audio - let server do its own VAD
                # This ensures the server hears everything for accurate STT
                audio_data = self._microphone.get_audio_chunk()
                if audio_data is None:
                    audio_data = self._create_silence()

                # Track VAD state for local reference
                is_voice = self._microphone.is_voice_detected()
                if is_voice:
                    self._is_speaking = True
                    self._silence_start = None
                elif self._is_speaking:
                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif (time.time() - self._silence_start) * 1000 > VAD_SILENCE_DURATION_MS:
                        self._is_speaking = False

                # Ensure audio_data is correct size
                expected_bytes = cls.SAMPLES_PER_FRAME * 2
                if len(audio_data) < expected_bytes:
                    audio_data = audio_data + bytes(expected_bytes - len(audio_data))
                elif len(audio_data) > expected_bytes:
                    audio_data = audio_data[:expected_bytes]

                # Create av.AudioFrame
                frame = AVAudioFrame(format="s16", layout="mono", samples=cls.SAMPLES_PER_FRAME)
                frame.planes[0].update(audio_data)
                frame.pts = self._timestamp
                frame.sample_rate = self._sample_rate
                frame.time_base = Fraction(1, self._sample_rate)

                self._timestamp += cls.SAMPLES_PER_FRAME
                self._frame_count += 1

                # Log periodically
                if self._frame_count % 500 == 0:
                    logger.debug(f"Audio track: {self._frame_count} frames sent")

                return frame

            def _create_silence(self) -> bytes:
                """Create a silence frame (all zeros)."""
                return bytes(cls.SAMPLES_PER_FRAME * 2)

            @property
            def is_speaking(self) -> bool:
                """Check if currently in a speech segment."""
                return self._is_speaking

            def __repr__(self) -> str:
                return f"MicrophoneAudioTrack(rate={self._sample_rate}, speaking={self._is_speaking})"

        return _MicrophoneAudioTrack(microphone, sample_rate)

    def __init__(
        self,
        microphone: BaseMicrophoneCapture,
        sample_rate: int = AUDIO_SAMPLE_RATE_STT,
    ) -> None:
        """Fallback init when aiortc not available."""
        self._microphone = microphone
        self._sample_rate = sample_rate
        self._timestamp = 0
        self._start_time = time.time()
        self._is_speaking = False
        self._silence_start: float | None = None

        logger.info(
            f"MicrophoneAudioTrack initialized: {sample_rate}Hz, "
            f"{self.FRAME_DURATION_MS}ms frames ({self.SAMPLES_PER_FRAME} samples)"
        )

    async def recv(self) -> AudioFrame:
        """
        Receive the next audio frame.

        Returns audio data when VAD is active, silence otherwise.
        This maintains a consistent frame rate for WebRTC.

        Returns:
            av.AudioFrame with audio data
        """
        try:
            from av import AudioFrame as AVAudioFrame
        except ImportError:
            raise RuntimeError("av (PyAV) package required for audio streaming")

        # Frame pacing - sleep to match target frame rate
        await asyncio.sleep(self.FRAME_DURATION_MS / 1000)

        # Check VAD state from microphone
        is_voice = self._microphone.is_voice_detected()

        if is_voice:
            # Get actual audio from microphone buffer
            audio_data = self._microphone.get_audio_chunk()
            if audio_data is None:
                # No data available yet, send silence
                audio_data = self._create_silence()
            self._is_speaking = True
            self._silence_start = None
        else:
            # Send silence frame
            audio_data = self._create_silence()

            # Track silence duration for end-of-utterance detection
            if self._is_speaking:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif (time.time() - self._silence_start) * 1000 > VAD_SILENCE_DURATION_MS:
                    self._is_speaking = False

        # Ensure audio_data is correct size (320 samples * 2 bytes = 640)
        expected_bytes = self.SAMPLES_PER_FRAME * 2
        if len(audio_data) < expected_bytes:
            audio_data = audio_data + bytes(expected_bytes - len(audio_data))
        elif len(audio_data) > expected_bytes:
            audio_data = audio_data[:expected_bytes]

        # Create av.AudioFrame (s16 = signed 16-bit, mono layout)
        frame = AVAudioFrame(format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME)
        frame.planes[0].update(audio_data)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)

        self._timestamp += self.SAMPLES_PER_FRAME

        return frame

    def _create_silence(self) -> bytes:
        """Create a silence frame (all zeros)."""
        return bytes(self.SAMPLES_PER_FRAME * 2)

    def stop(self) -> None:
        """Stop the audio track."""
        # Microphone lifecycle managed by MicrophoneCapture
        logger.debug("MicrophoneAudioTrack stopped")

    @property
    def is_speaking(self) -> bool:
        """Check if currently in a speech segment."""
        return self._is_speaking

    @property
    def readyState(self) -> str:
        """aiortc compatibility - track state."""
        return "live"

    def __repr__(self) -> str:
        return (
            f"MicrophoneAudioTrack(rate={self._sample_rate}, "
            f"speaking={self._is_speaking})"
        )
