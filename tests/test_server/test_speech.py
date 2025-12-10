"""
Unit tests for Murph's Speech system.
Tests SpeakAction, SpeechService, and speech message types.
"""

import base64
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.cognition.behavior.actions import SpeakAction
from server.llm.services.speech_service import SpeechService, PHRASES, EMOTION_PARAMS
from shared.messages import (
    SpeechCommand,
    VoiceActivityMessage,
    AudioDataMessage,
    MessageType,
    create_speech_command,
    create_voice_activity,
    create_audio_data,
)


class TestSpeakAction:
    """Tests for SpeakAction behavior tree node."""

    def test_speak_action_with_phrase_key(self):
        """Test SpeakAction resolves phrase keys."""
        action = SpeakAction("greeting")
        assert "Speak" in action.name
        assert action._text == "beep boop, hello!"
        assert action._phrase_key == "greeting"
        assert action._duration == 1.5  # greeting duration

    def test_speak_action_with_raw_text(self):
        """Test SpeakAction with custom text."""
        action = SpeakAction("Hello world!")
        assert action._text == "Hello world!"
        assert action._phrase_key is None
        # Duration estimated from text length: ~100ms per char
        assert action._duration >= 0.5

    def test_speak_action_with_emotion(self):
        """Test SpeakAction passes emotion parameter."""
        action = SpeakAction("greeting", emotion="happy")
        assert action._emotion == "happy"
        assert action._params["emotion"] == "happy"

    def test_speak_action_params(self):
        """Test SpeakAction sets correct params dict."""
        action = SpeakAction("curious", emotion="curious")
        params = action._params
        assert params["action"] == "speak"
        assert params["text"] == "hmm?"  # resolved phrase
        assert params["emotion"] == "curious"
        assert params["phrase_key"] == "curious"

    def test_speak_action_all_phrases(self):
        """Test all phrase keys are recognized."""
        for key in SpeakAction.PHRASES:
            action = SpeakAction(key)
            assert action._phrase_key == key
            assert action._text == SpeakAction.PHRASES[key]

    def test_speak_action_long_text_truncates_name(self):
        """Test long text is truncated in action name."""
        long_text = "This is a very long sentence that exceeds twenty characters"
        action = SpeakAction(long_text)
        assert len(action.name) < 40  # Name should be truncated
        assert "..." in action.name

    def test_speak_action_completes_after_duration(self):
        """Test SpeakAction returns SUCCESS after duration."""
        action = SpeakAction("alert")  # short duration 0.4s
        action.initialise()
        time.sleep(0.5)
        status = action.update()
        assert status.name == "SUCCESS"

    def test_speak_action_running_before_duration(self):
        """Test SpeakAction returns RUNNING before duration."""
        action = SpeakAction("sleepy")  # long duration 1.5s
        action.initialise()
        status = action.update()
        assert status.name == "RUNNING"

    def test_speak_action_no_wait(self):
        """Test SpeakAction with wait_for_completion=False."""
        action = SpeakAction("sleepy", wait_for_completion=False)
        action.initialise()
        status = action.update()
        assert status.name == "SUCCESS"  # Returns immediately


class TestSpeechMessages:
    """Tests for speech-related message types."""

    def test_speech_command_serialization(self):
        """Test SpeechCommand to_dict/from_dict."""
        audio_b64 = base64.b64encode(b"fake audio data").decode()
        cmd = SpeechCommand(
            audio_data=audio_b64,
            audio_format="wav",
            sample_rate=22050,
            volume=0.8,
            emotion="happy",
            text="hello",
        )

        d = cmd.to_dict()
        assert d["type"] == "speech"
        assert d["audio_data"] == audio_b64
        assert d["audio_format"] == "wav"
        assert d["sample_rate"] == 22050
        assert d["volume"] == 0.8
        assert d["emotion"] == "happy"
        assert d["text"] == "hello"

        restored = SpeechCommand.from_dict(d)
        assert restored.audio_data == audio_b64
        assert restored.audio_format == "wav"
        assert restored.sample_rate == 22050
        assert restored.volume == 0.8
        assert restored.emotion == "happy"
        assert restored.text == "hello"

    def test_voice_activity_message_serialization(self):
        """Test VoiceActivityMessage to_dict/from_dict."""
        msg = VoiceActivityMessage(
            is_speaking=True,
            audio_level=0.75,
        )

        d = msg.to_dict()
        assert d["is_speaking"] is True
        assert d["audio_level"] == 0.75
        assert "timestamp_ms" in d

        restored = VoiceActivityMessage.from_dict(d)
        assert restored.is_speaking is True
        assert restored.audio_level == 0.75

    def test_audio_data_message_serialization(self):
        """Test AudioDataMessage to_dict/from_dict."""
        audio_b64 = base64.b64encode(b"pcm audio").decode()
        msg = AudioDataMessage(
            audio_data=audio_b64,
            sample_rate=16000,
            channels=1,
            chunk_index=5,
            is_final=True,
        )

        d = msg.to_dict()
        assert d["audio_data"] == audio_b64
        assert d["sample_rate"] == 16000
        assert d["channels"] == 1
        assert d["chunk_index"] == 5
        assert d["is_final"] is True

        restored = AudioDataMessage.from_dict(d)
        assert restored.audio_data == audio_b64
        assert restored.sample_rate == 16000
        assert restored.is_final is True

    def test_message_types_exist(self):
        """Test speech message types are defined."""
        assert MessageType.SPEECH_COMMAND == 51
        assert MessageType.VOICE_ACTIVITY == 52
        assert MessageType.AUDIO_DATA == 53


class TestSpeechMessageFactories:
    """Tests for speech message factory functions."""

    def test_create_speech_command(self):
        """Test create_speech_command factory."""
        audio_b64 = base64.b64encode(b"audio").decode()
        msg = create_speech_command(
            audio_data=audio_b64,
            emotion="happy",
            text="hello",
        )

        assert msg.message_type == MessageType.SPEECH_COMMAND
        payload = msg.payload.payload
        assert payload.audio_data == audio_b64
        assert payload.emotion == "happy"

    def test_create_voice_activity(self):
        """Test create_voice_activity factory."""
        msg = create_voice_activity(is_speaking=True, audio_level=0.5)

        assert msg.message_type == MessageType.VOICE_ACTIVITY
        assert msg.payload.is_speaking is True
        assert msg.payload.audio_level == 0.5

    def test_create_audio_data(self):
        """Test create_audio_data factory."""
        audio_b64 = base64.b64encode(b"audio").decode()
        msg = create_audio_data(audio_data=audio_b64, chunk_index=3, is_final=True)

        assert msg.message_type == MessageType.AUDIO_DATA
        assert msg.payload.chunk_index == 3
        assert msg.payload.is_final is True


class TestSpeechService:
    """Tests for SpeechService."""

    def test_phrase_resolution(self):
        """Test phrase key resolution."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)

        # Known phrase
        assert service.resolve_phrase("greeting") == "beep boop, hello!"
        assert service.resolve_phrase("happy") == "wheee!"

        # Unknown key returns as-is
        assert service.resolve_phrase("custom text") == "custom text"

    def test_emotion_params(self):
        """Test emotion-to-voice parameter mapping."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)

        happy_params = service.get_emotion_params("happy")
        assert happy_params["pitch"] == 1.2
        assert happy_params["speed"] == 1.1

        sad_params = service.get_emotion_params("sad")
        assert sad_params["pitch"] == 0.8
        assert sad_params["speed"] == 0.85

        # Unknown emotion returns neutral
        unknown_params = service.get_emotion_params("unknown")
        assert unknown_params["pitch"] == 1.0
        assert unknown_params["speed"] == 1.0

    def test_audio_encoding(self):
        """Test audio base64 encoding/decoding."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)

        original = b"test audio data"
        encoded = service.encode_audio_base64(original)
        decoded = service.decode_audio_base64(encoded)

        assert decoded == original

    def test_is_available_without_api_key(self):
        """Test service is unavailable without API key."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)
        assert service.is_available is False

    def test_is_available_with_api_key(self):
        """Test service is available with API key."""
        config = MagicMock()
        config.nanogpt_api_key = "test-key"
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)
        assert service.is_available is True

    def test_stats(self):
        """Test service stats tracking."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)
        stats = service.get_stats()

        assert "initialized" in stats
        assert "tts_calls" in stats
        assert "stt_calls" in stats
        assert "cache_hits" in stats
        assert "errors" in stats

    @pytest.mark.asyncio
    async def test_synthesize_without_api_key_returns_none(self):
        """Test synthesize returns None without API key."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)
        result = await service.synthesize("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_without_api_key_returns_none(self):
        """Test transcribe returns None without API key."""
        config = MagicMock()
        config.nanogpt_api_key = None
        config.nanogpt_base_url = "https://test.com"
        config.request_timeout_seconds = 10
        config.tts_model = "kokoro"
        config.stt_model = "whisper"

        service = SpeechService(config)
        result = await service.transcribe(b"audio")
        assert result is None


class TestSpeechPhrases:
    """Tests for speech phrase definitions."""

    def test_all_service_phrases_exist(self):
        """Test all phrases defined in service module."""
        assert "greeting" in PHRASES
        assert "happy" in PHRASES
        assert "sad" in PHRASES
        assert "curious" in PHRASES
        assert "scared" in PHRASES

    def test_all_emotion_params_exist(self):
        """Test all emotion params defined."""
        assert "happy" in EMOTION_PARAMS
        assert "sad" in EMOTION_PARAMS
        assert "neutral" in EMOTION_PARAMS
        assert "scared" in EMOTION_PARAMS
        assert "sleepy" in EMOTION_PARAMS

    def test_speak_action_and_service_phrases_match(self):
        """Test SpeakAction and SpeechService have consistent phrases."""
        for key in SpeakAction.PHRASES:
            assert key in PHRASES
            assert SpeakAction.PHRASES[key] == PHRASES[key]
