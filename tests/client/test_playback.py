# tests/client/test_playback.py
import pytest
import numpy as np
from unittest.mock import MagicMock


def test_audio_playback_init(mock_gpio):
    mock_pwm = MagicMock()
    mock_gpio.PWM.return_value = mock_pwm

    from murph_client.audio.playback import AudioPlayback
    playback = AudioPlayback(pin=18, sample_rate=22050)

    mock_gpio.setup.assert_called_with(18, mock_gpio.OUT)
    mock_gpio.PWM.assert_called_once()


def test_audio_playback_play(mock_gpio):
    mock_pwm = MagicMock()
    mock_gpio.PWM.return_value = mock_pwm

    from murph_client.audio.playback import AudioPlayback
    playback = AudioPlayback(pin=18, sample_rate=22050)

    # Create fake audio bytes (small for speed)
    audio_bytes = np.zeros(10, dtype=np.int16).tobytes()
    playback.play(audio_bytes)

    # Should have called ChangeDutyCycle during playback
    assert mock_pwm.ChangeDutyCycle.called
