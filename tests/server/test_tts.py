# tests/server/test_tts.py
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, Mock

# Mock scipy and piper before importing the module
mock_scipy = MagicMock()
mock_signal = MagicMock()
mock_scipy.signal = mock_signal
sys.modules["scipy"] = mock_scipy
sys.modules["scipy.signal"] = mock_signal

mock_piper_module = MagicMock()
sys.modules["piper"] = mock_piper_module


def test_tts_synthesize_returns_audio():
    # Create a mock audio chunk with the expected attributes
    mock_chunk = MagicMock()
    mock_audio_int16 = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    mock_chunk.audio_int16_bytes = mock_audio_int16.tobytes()
    mock_chunk.sample_rate = 16000

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = iter([mock_chunk])

    mock_voice_class = MagicMock()
    mock_voice_class.load.return_value = mock_voice
    mock_piper_module.PiperVoice = mock_voice_class

    # Re-import to get fresh module with mock
    if "murph_server.audio.tts" in sys.modules:
        del sys.modules["murph_server.audio.tts"]

    from murph_server.audio.tts import TextToSpeech

    # Create TTS with mocked voice finder
    tts = TextToSpeech.__new__(TextToSpeech)
    tts.voice_name = "en_US-danny-low"
    tts.voice = mock_voice

    # Test with radio effect disabled to simplify
    audio = tts.synthesize("Hello there", apply_radio_effect=False)

    assert isinstance(audio, bytes)
    assert len(audio) > 0
