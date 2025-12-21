"""
Murph - Audio Receiver
WebRTC audio receiving and STT processing on server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from aiortc.mediastreams import MediaStreamTrack

    from ..llm.services.speech_service import SpeechService

from shared.constants import AUDIO_SAMPLE_RATE_STT, VAD_SILENCE_DURATION_MS

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Buffer for accumulating audio chunks during an utterance.

    Handles VAD-based utterance segmentation:
    - Accumulates audio chunks while voice is detected
    - Returns complete utterance when silence threshold exceeded
    - Tracks speaking state for UI feedback
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE_STT,
        silence_threshold_ms: int = VAD_SILENCE_DURATION_MS,
    ) -> None:
        """
        Initialize audio buffer.

        Args:
            sample_rate: Audio sample rate (Hz)
            silence_threshold_ms: Silence duration (ms) before end-of-utterance
        """
        self._sample_rate = sample_rate
        self._silence_threshold_ms = silence_threshold_ms

        self._chunks: list[bytes] = []
        self._is_speaking = False
        self._silence_start: float | None = None
        self._last_audio_time: float = 0

    def add_chunk(self, audio_data: bytes, is_voice: bool) -> bytes | None:
        """
        Add audio chunk to buffer.

        Args:
            audio_data: Raw PCM audio bytes (16-bit mono)
            is_voice: True if voice activity detected in this chunk

        Returns:
            Complete utterance audio when silence detected after speech,
            or None if still accumulating.
        """
        now = time.time()
        self._last_audio_time = now

        if is_voice:
            self._chunks.append(audio_data)
            self._is_speaking = True
            self._silence_start = None
            return None

        # Silence received
        if self._is_speaking:
            # Also add silence chunks so STT gets complete audio
            self._chunks.append(audio_data)

            if self._silence_start is None:
                self._silence_start = now

            silence_duration_ms = (now - self._silence_start) * 1000

            if silence_duration_ms >= self._silence_threshold_ms:
                # Utterance complete - return accumulated audio
                utterance = b"".join(self._chunks)
                self._chunks.clear()
                self._is_speaking = False
                self._silence_start = None
                return utterance

        return None

    def get_audio_duration_ms(self) -> float:
        """Get duration of buffered audio in milliseconds."""
        total_bytes = sum(len(c) for c in self._chunks)
        samples = total_bytes // 2  # 16-bit = 2 bytes/sample
        return (samples / self._sample_rate) * 1000

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        self._chunks.clear()
        self._is_speaking = False
        self._silence_start = None

    @property
    def is_speaking(self) -> bool:
        """Check if currently in a speech segment."""
        return self._is_speaking


class AudioReceiver:
    """
    WebRTC audio receiving and STT processing on server.

    Handles:
    - WebRTC audio track reception
    - Audio buffering for utterance segmentation
    - STT transcription via SpeechService
    - Callback with transcribed text

    Usage:
        receiver = AudioReceiver(
            on_transcription=handle_speech,
            speech_service=speech_service,
        )
        await receiver.start()

        # When video receiver gets audio track
        receiver.handle_track(audio_track)
    """

    # RMS threshold for server-side VAD confirmation
    # Lowered from 0.01 to detect quieter microphones (webcam mics often have low output)
    VOICE_RMS_THRESHOLD = 0.002

    # Minimum utterance duration to process (ms)
    MIN_UTTERANCE_MS = 200

    # Maximum utterance duration (prevent runaway buffers)
    MAX_UTTERANCE_MS = 30000

    def __init__(
        self,
        on_transcription: Callable[[str], None] | None = None,
        speech_service: SpeechService | None = None,
    ) -> None:
        """
        Initialize audio receiver.

        Args:
            on_transcription: Callback for transcribed text
            speech_service: SpeechService instance for STT
        """
        self._on_transcription = on_transcription
        self._speech_service = speech_service

        self._audio_track: MediaStreamTrack | None = None
        self._running = False
        self._process_task: asyncio.Task[None] | None = None

        self._buffer = AudioBuffer()

        # Stats
        self._utterances_processed = 0
        self._total_audio_ms = 0.0
        self._frames_received = 0

        logger.info("AudioReceiver initialized")

    def set_speech_service(self, service: SpeechService) -> None:
        """Set the speech service for STT."""
        self._speech_service = service
        logger.info("Speech service set for AudioReceiver")

    def set_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """Set the transcription callback."""
        self._on_transcription = callback

    async def start(self) -> None:
        """Start the audio receiver."""
        self._running = True
        logger.info("AudioReceiver started, waiting for audio track")

    async def stop(self) -> None:
        """Stop the audio receiver."""
        logger.info("Stopping AudioReceiver...")
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        self._audio_track = None
        self._buffer.clear()
        logger.info(
            f"AudioReceiver stopped "
            f"(processed {self._utterances_processed} utterances, "
            f"{self._frames_received} frames)"
        )

    def handle_track(self, track: MediaStreamTrack) -> None:
        """
        Handle incoming audio track from WebRTC.

        Called from VideoReceiver's @pc.on("track") handler.

        Args:
            track: aiortc MediaStreamTrack (audio)
        """
        if track.kind != "audio":
            return

        logger.info("Received audio track from Pi")
        self._audio_track = track

        # Start processing if not already running
        if self._process_task is None or self._process_task.done():
            self._process_task = asyncio.create_task(
                self._process_audio(track),
                name="audio_processing",
            )

    async def _process_audio(self, track: MediaStreamTrack) -> None:
        """
        Process audio frames from track.

        Args:
            track: aiortc audio track
        """
        logger.info("Starting audio processing from track")

        try:
            while self._running:
                try:
                    # Receive frame from track
                    frame = await track.recv()
                    self._frames_received += 1

                    # Convert to bytes (s16 format)
                    audio_bytes = bytes(frame.planes[0])

                    # Server-side VAD check (double-check Pi's VAD)
                    samples = np.frombuffer(audio_bytes, dtype=np.int16)
                    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0
                    is_voice = rms > self.VOICE_RMS_THRESHOLD

                    # Log every 50 frames (~1 second at 20ms/frame)
                    if self._frames_received % 50 == 0:
                        buffer_ms = self._buffer.get_audio_duration_ms()
                        logger.info(
                            f"Audio: frames={self._frames_received}, RMS={rms:.4f}, "
                            f"voice={is_voice}, buffer={buffer_ms:.0f}ms"
                        )

                    # Add to buffer, check for complete utterance
                    utterance = self._buffer.add_chunk(audio_bytes, is_voice)

                    # Check for runaway buffer
                    if self._buffer.get_audio_duration_ms() > self.MAX_UTTERANCE_MS:
                        logger.warning("Audio buffer exceeded max duration, forcing flush")
                        utterance = b"".join(self._buffer._chunks)
                        self._buffer.clear()

                    if utterance and len(utterance) > 0:
                        await self._process_utterance(utterance)

                except Exception as e:
                    if "MediaStreamError" in str(type(e).__name__):
                        logger.info("Audio track ended")
                        break
                    logger.error(f"Audio processing error: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Audio processing cancelled")
        finally:
            logger.info(
                f"Audio processing ended "
                f"(received {self._frames_received} frames)"
            )

    async def _process_utterance(self, audio_data: bytes) -> None:
        """
        Process a complete utterance through STT.

        Args:
            audio_data: Complete utterance audio (16-bit PCM)
        """
        audio_ms = len(audio_data) / 2 / AUDIO_SAMPLE_RATE_STT * 1000
        self._total_audio_ms += audio_ms

        # Skip very short utterances (likely noise)
        if audio_ms < self.MIN_UTTERANCE_MS:
            logger.debug(f"Skipping short utterance ({audio_ms:.0f}ms)")
            return

        logger.info(f"Processing utterance ({audio_ms:.0f}ms)")

        if self._speech_service:
            try:
                text = await self._speech_service.transcribe(
                    audio_data,
                    sample_rate=AUDIO_SAMPLE_RATE_STT,
                )

                if text and text.strip():
                    self._utterances_processed += 1
                    logger.info(f"Transcribed: '{text}'")

                    if self._on_transcription:
                        self._on_transcription(text)
                else:
                    logger.debug("Empty transcription result")

            except Exception as e:
                logger.error(f"STT error: {e}")
        else:
            logger.warning("No speech service configured - transcription skipped")

    @property
    def is_receiving(self) -> bool:
        """Check if actively receiving audio."""
        return self._audio_track is not None and self._running

    @property
    def is_speaking(self) -> bool:
        """Check if voice activity detected."""
        return self._buffer.is_speaking

    def get_stats(self) -> dict[str, Any]:
        """Get receiver statistics."""
        return {
            "running": self._running,
            "has_audio_track": self._audio_track is not None,
            "frames_received": self._frames_received,
            "utterances_processed": self._utterances_processed,
            "total_audio_ms": self._total_audio_ms,
            "buffer_duration_ms": self._buffer.get_audio_duration_ms(),
            "is_speaking": self._buffer.is_speaking,
        }

    def __repr__(self) -> str:
        return (
            f"AudioReceiver(running={self._running}, "
            f"utterances={self._utterances_processed}, "
            f"speaking={self._buffer.is_speaking})"
        )
