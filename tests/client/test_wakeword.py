# tests/client/test_wakeword.py
import pytest
import numpy as np
from unittest.mock import MagicMock


def test_wakeword_detector_init(mock_openwakeword):
    mock_model = MagicMock()
    mock_openwakeword.Model.return_value = mock_model

    from murph_client.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector(wake_word="hey_jarvis")

    mock_openwakeword.Model.assert_called_once()


def test_wakeword_detector_process(mock_openwakeword):
    mock_model = MagicMock()
    mock_model.predict.return_value = None
    mock_model.get_prediction.return_value = {"hey_jarvis": 0.9}
    mock_openwakeword.Model.return_value = mock_model

    from murph_client.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector(wake_word="hey_jarvis", threshold=0.5)

    audio = np.zeros(1280, dtype=np.int16)
    result = detector.process(audio)

    assert result is True
