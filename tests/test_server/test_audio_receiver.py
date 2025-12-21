"""Tests for server audio receiver and buffer."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.audio.receiver import AudioBuffer, AudioReceiver


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_initialization(self) -> None:
        """Test buffer initializes correctly."""
        buffer = AudioBuffer()
        assert buffer.is_speaking is False
        assert buffer.get_audio_duration_ms() == 0

    def test_add_voice_chunk(self) -> None:
        """Test adding voice chunk accumulates."""
        buffer = AudioBuffer()
        chunk = b"\xff\x00" * 320  # 640 bytes = 20ms

        result = buffer.add_chunk(chunk, is_voice=True)

        assert result is None  # Still accumulating
        assert buffer.is_speaking is True
        assert buffer.get_audio_duration_ms() == pytest.approx(20, abs=1)

    def test_add_multiple_voice_chunks(self) -> None:
        """Test accumulating multiple voice chunks."""
        buffer = AudioBuffer()
        chunk = b"\xff\x00" * 320

        for _ in range(5):
            buffer.add_chunk(chunk, is_voice=True)

        # 5 chunks * 20ms = 100ms
        assert buffer.get_audio_duration_ms() == pytest.approx(100, abs=5)

    def test_returns_utterance_after_silence(self) -> None:
        """Test returns complete utterance after silence threshold."""
        buffer = AudioBuffer(silence_threshold_ms=100)
        chunk = b"\xff\x00" * 320

        # Add voice chunk
        buffer.add_chunk(chunk, is_voice=True)
        assert buffer.is_speaking is True

        # Start silence timer with first silent chunk
        buffer.add_chunk(b"\x00\x00" * 320, is_voice=False)

        # Wait past the silence threshold
        time.sleep(0.15)  # 150ms > 100ms threshold

        # Second silent chunk should trigger utterance completion
        result = buffer.add_chunk(b"\x00\x00" * 320, is_voice=False)

        assert result is not None
        # 640 bytes voice + 2x 640 bytes silence chunks
        assert len(result) == 640 + 640 + 640
        assert buffer.is_speaking is False

    def test_no_utterance_during_voice(self) -> None:
        """Test no utterance returned while voice active."""
        buffer = AudioBuffer(silence_threshold_ms=100)
        chunk = b"\xff\x00" * 320

        for _ in range(10):
            result = buffer.add_chunk(chunk, is_voice=True)
            assert result is None

    def test_clear_resets_buffer(self) -> None:
        """Test clear() resets all state."""
        buffer = AudioBuffer()
        buffer.add_chunk(b"\xff\x00" * 320, is_voice=True)

        buffer.clear()

        assert buffer.get_audio_duration_ms() == 0
        assert buffer.is_speaking is False

    def test_silence_before_voice_ignored(self) -> None:
        """Test silence before any voice is ignored."""
        buffer = AudioBuffer(silence_threshold_ms=100)

        # Add silence - should not trigger utterance
        time.sleep(0.15)
        result = buffer.add_chunk(b"\x00\x00" * 320, is_voice=False)

        assert result is None
        assert buffer.is_speaking is False


class TestAudioReceiver:
    """Tests for AudioReceiver class."""

    def test_initialization(self) -> None:
        """Test receiver initializes correctly."""
        receiver = AudioReceiver()

        stats = receiver.get_stats()
        assert stats["running"] is False
        assert stats["has_audio_track"] is False
        assert stats["utterances_processed"] == 0

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test receiver start and stop."""
        receiver = AudioReceiver()

        await receiver.start()
        assert receiver.get_stats()["running"] is True

        await receiver.stop()
        assert receiver.get_stats()["running"] is False

    def test_set_speech_service(self) -> None:
        """Test setting speech service."""
        receiver = AudioReceiver()
        mock_service = MagicMock()

        receiver.set_speech_service(mock_service)

        # Service should be set
        assert receiver._speech_service is mock_service

    def test_set_transcription_callback(self) -> None:
        """Test setting transcription callback."""
        receiver = AudioReceiver()
        callback = MagicMock()

        receiver.set_transcription_callback(callback)

        assert receiver._on_transcription is callback

    @pytest.mark.asyncio
    async def test_handle_non_audio_track_ignored(self) -> None:
        """Test non-audio tracks are ignored."""
        receiver = AudioReceiver()
        await receiver.start()

        mock_track = MagicMock()
        mock_track.kind = "video"

        receiver.handle_track(mock_track)

        assert receiver._audio_track is None

        await receiver.stop()

    def test_is_receiving_property(self) -> None:
        """Test is_receiving property."""
        receiver = AudioReceiver()
        assert receiver.is_receiving is False

    def test_is_speaking_property(self) -> None:
        """Test is_speaking property."""
        receiver = AudioReceiver()
        assert receiver.is_speaking is False

    def test_repr(self) -> None:
        """Test string representation."""
        receiver = AudioReceiver()
        repr_str = repr(receiver)
        assert "AudioReceiver" in repr_str
        assert "running=False" in repr_str


class TestAudioReceiverTranscription:
    """Tests for AudioReceiver transcription processing."""

    @pytest.mark.asyncio
    async def test_process_utterance_calls_speech_service(self) -> None:
        """Test utterance processing calls speech service."""
        mock_service = AsyncMock()
        mock_service.transcribe = AsyncMock(return_value="hello world")

        callback = MagicMock()

        receiver = AudioReceiver(
            on_transcription=callback,
            speech_service=mock_service,
        )

        # Process a sufficiently long utterance (>200ms)
        # 16kHz * 0.3s = 4800 samples * 2 bytes = 9600 bytes
        audio_data = b"\xff\x00" * 4800

        await receiver._process_utterance(audio_data)

        mock_service.transcribe.assert_called_once()
        callback.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_process_short_utterance_skipped(self) -> None:
        """Test very short utterances are skipped."""
        mock_service = AsyncMock()
        callback = MagicMock()

        receiver = AudioReceiver(
            on_transcription=callback,
            speech_service=mock_service,
        )

        # Very short utterance (< 200ms)
        short_audio = b"\xff\x00" * 100

        await receiver._process_utterance(short_audio)

        mock_service.transcribe.assert_not_called()
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_empty_transcription_ignored(self) -> None:
        """Test empty transcription results are not passed to callback."""
        mock_service = AsyncMock()
        mock_service.transcribe = AsyncMock(return_value="")

        callback = MagicMock()

        receiver = AudioReceiver(
            on_transcription=callback,
            speech_service=mock_service,
        )

        audio_data = b"\xff\x00" * 4800

        await receiver._process_utterance(audio_data)

        mock_service.transcribe.assert_called_once()
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_without_speech_service(self) -> None:
        """Test processing without speech service logs warning."""
        callback = MagicMock()

        receiver = AudioReceiver(on_transcription=callback)
        # No speech service set

        audio_data = b"\xff\x00" * 4800

        # Should not raise
        await receiver._process_utterance(audio_data)

        callback.assert_not_called()


class TestAudioReceiverStats:
    """Tests for AudioReceiver statistics."""

    @pytest.mark.asyncio
    async def test_utterances_processed_count(self) -> None:
        """Test utterances_processed counter increments."""
        mock_service = AsyncMock()
        mock_service.transcribe = AsyncMock(return_value="hello")

        receiver = AudioReceiver(speech_service=mock_service)

        assert receiver.get_stats()["utterances_processed"] == 0

        audio_data = b"\xff\x00" * 4800
        await receiver._process_utterance(audio_data)

        assert receiver.get_stats()["utterances_processed"] == 1

    @pytest.mark.asyncio
    async def test_total_audio_ms_accumulates(self) -> None:
        """Test total_audio_ms accumulates."""
        mock_service = AsyncMock()
        mock_service.transcribe = AsyncMock(return_value="hello")

        receiver = AudioReceiver(speech_service=mock_service)

        assert receiver.get_stats()["total_audio_ms"] == 0

        # 4800 samples = 300ms at 16kHz
        audio_data = b"\xff\x00" * 4800
        await receiver._process_utterance(audio_data)

        assert receiver.get_stats()["total_audio_ms"] == pytest.approx(300, abs=10)
