# tests/server/test_tts.py
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, Mock

# Mock piper before importing the module
mock_piper_module = MagicMock()
sys.modules["piper"] = mock_piper_module


def test_tts_synthesize_returns_audio():
    mock_voice = MagicMock()
    mock_audio = np.random.rand(16000).astype(np.float32)
    mock_voice.synthesize.return_value = iter([mock_audio.tobytes()])

    mock_voice_class = MagicMock()
    mock_voice_class.load.return_value = mock_voice
    mock_piper_module.PiperVoice = mock_voice_class

    # Re-import to get fresh module with mock
    if "murph_server.audio.tts" in sys.modules:
        del sys.modules["murph_server.audio.tts"]

    from murph_server.audio.tts import TextToSpeech

    # Create TTS with mocked voice finder
    tts = TextToSpeech.__new__(TextToSpeech)
    tts.voice_name = "en_US-lessac-medium"
    tts.voice = mock_voice

    audio = tts.synthesize("Hello there")

    assert isinstance(audio, bytes)
    assert len(audio) > 0
