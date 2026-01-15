# tests/client/test_playback.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_pyaudio():
    with patch("murph_client.audio.playback.pyaudio") as mock:
        mock.paInt16 = 8
        yield mock


def test_audio_playback_init(mock_pyaudio):
    from murph_client.audio.playback import AudioPlayback
    playback = AudioPlayback(sample_rate=22050)

    mock_pyaudio.PyAudio.assert_called_once()


def test_audio_playback_play(mock_pyaudio):
    mock_pa = MagicMock()
    mock_stream = MagicMock()
    mock_pa.open.return_value = mock_stream
    mock_pyaudio.PyAudio.return_value = mock_pa

    from murph_client.audio.playback import AudioPlayback
    playback = AudioPlayback(sample_rate=22050)

    # Create fake audio bytes
    audio_bytes = np.zeros(2048, dtype=np.int16).tobytes()
    playback.play(audio_bytes)

    # Should have opened a stream and written to it
    mock_pa.open.assert_called_once()
    assert mock_stream.write.called
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()


def test_audio_playback_cleanup(mock_pyaudio):
    mock_pa = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa

    from murph_client.audio.playback import AudioPlayback
    playback = AudioPlayback()
    playback.cleanup()

    mock_pa.terminate.assert_called_once()
