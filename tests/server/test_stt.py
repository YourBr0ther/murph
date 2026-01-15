# tests/server/test_stt.py
import sys
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Mock faster_whisper before importing the module
mock_whisper_module = MagicMock()
sys.modules["faster_whisper"] = mock_whisper_module


def test_stt_init_with_model():
    # Reset mock for this test
    mock_model_class = Mock()
    mock_whisper_module.WhisperModel = mock_model_class

    # Re-import to get fresh module with mock
    if "murph_server.audio.stt" in sys.modules:
        del sys.modules["murph_server.audio.stt"]

    from murph_server.audio.stt import SpeechToText
    stt = SpeechToText(model_size="base")
    mock_model_class.assert_called_once()


def test_transcribe_returns_text():
    mock_model = Mock()
    mock_segment = Mock()
    mock_segment.text = "Hello Murph"
    mock_model.transcribe.return_value = ([mock_segment], None)

    mock_model_class = Mock(return_value=mock_model)
    mock_whisper_module.WhisperModel = mock_model_class

    # Re-import to get fresh module with mock
    if "murph_server.audio.stt" in sys.modules:
        del sys.modules["murph_server.audio.stt"]

    from murph_server.audio.stt import SpeechToText
    stt = SpeechToText(model_size="base")
    audio_data = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(audio_data)

    assert result == "Hello Murph"
