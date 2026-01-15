# tests/client/test_wakeword.py
import pytest
import numpy as np
from unittest.mock import MagicMock


def test_wakeword_detector_init(mock_openwakeword):
    mock_model = MagicMock()
    mock_openwakeword.Model.return_value = mock_model

    from murph_client.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector(model_path="hey_jarvis")

    mock_openwakeword.Model.assert_called_once()
    assert detector.wake_word == "hey_jarvis"


def test_wakeword_detector_process(mock_openwakeword):
    mock_model = MagicMock()
    # predict returns a dict with wake word scores
    mock_model.predict.return_value = {"hey_jarvis": 0.9}
    mock_openwakeword.Model.return_value = mock_model

    from murph_client.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector(model_path="hey_jarvis", threshold=0.5)

    audio = np.zeros(1280, dtype=np.int16)
    result = detector.process(audio)

    assert result is True


def test_wakeword_detector_below_threshold(mock_openwakeword):
    mock_model = MagicMock()
    # Score below threshold
    mock_model.predict.return_value = {"hey_jarvis": 0.3}
    mock_openwakeword.Model.return_value = mock_model

    from murph_client.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector(model_path="hey_jarvis", threshold=0.5)

    audio = np.zeros(1280, dtype=np.int16)
    result = detector.process(audio)

    assert result is False
