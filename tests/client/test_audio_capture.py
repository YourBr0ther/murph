# tests/client/test_audio_capture.py
import pytest
import numpy as np
from unittest.mock import MagicMock


def test_audio_capture_init(mock_pyaudio):
    mock_pa = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa

    from murph_client.audio.capture import AudioCapture
    capture = AudioCapture(sample_rate=16000, chunk_size=1024)

    mock_pa.open.assert_called_once()
    assert capture.sample_rate == 16000
    assert capture.chunk_size == 1024


def test_audio_capture_read_returns_numpy(mock_pyaudio):
    mock_pa = MagicMock()
    mock_stream = MagicMock()
    fake_audio = np.zeros(1024, dtype=np.int16).tobytes()
    mock_stream.read.return_value = fake_audio
    mock_pa.open.return_value = mock_stream
    mock_pyaudio.PyAudio.return_value = mock_pa

    from murph_client.audio.capture import AudioCapture
    capture = AudioCapture(sample_rate=16000, chunk_size=1024)
    audio = capture.read()

    assert isinstance(audio, np.ndarray)
    assert len(audio) == 1024
