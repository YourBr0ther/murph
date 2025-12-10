"""
Murph - Speech Service
Speech synthesis (TTS) and recognition (STT) using NanoGPT APIs.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import LLMConfig

logger = logging.getLogger(__name__)


# Voice personality phrases (Wall-E/BMO/BB-8 style)
PHRASES: dict[str, str] = {
    "greeting": "beep boop, hello!",
    "happy": "wheee!",
    "sad": "aww...",
    "curious": "hmm?",
    "scared": "eep!",
    "affection": "I like you",
    "playful": "boop boop boop!",
    "sleepy": "zzz...",
    "alert": "oh!",
    "confused": "buh?",
    "excited": "ooh ooh!",
    "goodbye": "bye bye!",
    "yes": "uh huh!",
    "no": "nuh uh",
    "thanks": "beep!",
}

# Emotion-to-voice parameter mapping (for future use if API supports it)
EMOTION_PARAMS: dict[str, dict[str, float]] = {
    "happy": {"pitch": 1.2, "speed": 1.1},
    "sad": {"pitch": 0.8, "speed": 0.85},
    "curious": {"pitch": 1.0, "speed": 1.0},
    "scared": {"pitch": 1.3, "speed": 1.3},
    "sleepy": {"pitch": 0.7, "speed": 0.7},
    "playful": {"pitch": 1.1, "speed": 1.2},
    "love": {"pitch": 1.0, "speed": 0.9},
    "neutral": {"pitch": 1.0, "speed": 1.0},
}


class SpeechService:
    """
    Speech synthesis and recognition service using NanoGPT APIs.

    Features:
    - TTS via NanoGPT /api/text-to-speech (Kokoro 82M default)
    - STT via NanoGPT Whisper Large V3
    - Emotional voice modulation
    - Audio caching for common phrases
    - Rate limiting and error handling

    Usage:
        service = SpeechService(config)
        audio = await service.synthesize("Hello!", emotion="happy")
        text = await service.transcribe(audio_buffer)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initialize speech service.

        Args:
            config: LLM configuration (uses env if None)
        """
        from ..config import LLMConfig

        self._config = config or LLMConfig.from_env()
        self._base_url = self._config.nanogpt_base_url.rstrip("/")
        self._api_key = self._config.nanogpt_api_key
        self._timeout = self._config.request_timeout_seconds
        self._session: Any = None

        # State
        self._initialized = False
        self._available = False

        # Cache for common phrases
        self._phrase_cache: dict[str, bytes] = {}
        self._cache_max = 50

        # Stats
        self._tts_calls = 0
        self._stt_calls = 0
        self._cache_hits = 0
        self._errors = 0

        logger.info("SpeechService created")

    async def _ensure_session(self) -> Any:
        """Lazy-initialize HTTP session."""
        if self._session is None or self._session.closed:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._timeout)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
            )
            self._initialized = True
            self._available = self._api_key is not None and len(self._api_key) > 0
        return self._session

    def resolve_phrase(self, text_or_key: str) -> str:
        """
        Resolve a phrase key to its text, or return the text as-is.

        Args:
            text_or_key: Either a phrase key (e.g., "greeting") or raw text

        Returns:
            The resolved text
        """
        return PHRASES.get(text_or_key, text_or_key)

    async def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        voice_style: str = "robot",
    ) -> bytes | None:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize (or phrase key like "greeting")
            emotion: Current emotional state for voice modulation
            voice_style: "robot" (beepy), "words" (clear speech)

        Returns:
            Audio bytes (WAV format) or None on error
        """
        if not self._api_key:
            logger.warning("Speech synthesis unavailable: no API key")
            return None

        # Resolve phrase key to actual text
        actual_text = self.resolve_phrase(text)

        # Check cache
        cache_key = f"{actual_text}:{emotion}:{voice_style}"
        if cache_key in self._phrase_cache:
            self._cache_hits += 1
            logger.debug(f"TTS cache hit: {text[:20]}...")
            return self._phrase_cache[cache_key]

        self._tts_calls += 1

        try:
            session = await self._ensure_session()

            # Build request payload
            payload = {
                "model": self._config.tts_model,
                "input": actual_text,
                "response_format": "wav",
            }

            logger.debug(f"TTS request: '{actual_text[:30]}...' (emotion={emotion})")

            async with session.post(
                f"{self._base_url}/text-to-speech",
                json=payload,
            ) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()

                    # Cache the result
                    if len(self._phrase_cache) < self._cache_max:
                        self._phrase_cache[cache_key] = audio_data

                    logger.debug(
                        f"TTS success: {len(audio_data)} bytes for '{actual_text[:20]}...'"
                    )
                    return audio_data
                else:
                    self._errors += 1
                    error_text = await resp.text()
                    logger.error(f"TTS API error {resp.status}: {error_text[:100]}")
                    return None

        except Exception as e:
            self._errors += 1
            logger.error(f"TTS synthesis error: {e}")
            return None

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str | None:
        """
        Transcribe speech to text using Whisper.

        Args:
            audio_data: Raw PCM audio bytes (16-bit mono)
            sample_rate: Audio sample rate (default 16kHz for Whisper)

        Returns:
            Transcribed text or None on error
        """
        if not self._api_key:
            logger.warning("Speech transcription unavailable: no API key")
            return None

        self._stt_calls += 1

        try:
            session = await self._ensure_session()

            # Create WAV file in memory from raw PCM
            wav_buffer = self._create_wav(audio_data, sample_rate)

            import aiohttp

            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer,
                filename="audio.wav",
                content_type="audio/wav",
            )
            form.add_field("model", self._config.stt_model)

            logger.debug(f"STT request: {len(audio_data)} bytes of audio")

            async with session.post(
                f"{self._base_url}/audio/transcriptions",
                data=form,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("text", "")
                    logger.debug(f"STT success: '{text[:50]}...'")
                    return text
                else:
                    self._errors += 1
                    error_text = await resp.text()
                    logger.error(f"STT API error {resp.status}: {error_text[:100]}")
                    return None

        except Exception as e:
            self._errors += 1
            logger.error(f"STT transcription error: {e}")
            return None

    def _create_wav(self, pcm_data: bytes, sample_rate: int) -> io.BytesIO:
        """
        Create WAV file in memory from raw PCM data.

        Args:
            pcm_data: Raw 16-bit mono PCM audio
            sample_rate: Sample rate in Hz

        Returns:
            BytesIO containing WAV file data
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        buffer.seek(0)
        return buffer

    def get_emotion_params(self, emotion: str) -> dict[str, float]:
        """
        Get voice parameters for an emotional state.

        Args:
            emotion: Emotional state name

        Returns:
            Dict with pitch and speed parameters
        """
        return EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])

    def encode_audio_base64(self, audio_data: bytes) -> str:
        """
        Encode audio bytes to base64 string for wire transmission.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(audio_data).decode("utf-8")

    def decode_audio_base64(self, audio_b64: str) -> bytes:
        """
        Decode base64 string to audio bytes.

        Args:
            audio_b64: Base64-encoded audio string

        Returns:
            Raw audio bytes
        """
        return base64.b64decode(audio_b64)

    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self._api_key is not None and len(self._api_key) > 0

    @property
    def is_initialized(self) -> bool:
        """Check if service has been initialized."""
        return self._initialized

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "initialized": self._initialized,
            "available": self._available,
            "tts_calls": self._tts_calls,
            "stt_calls": self._stt_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._phrase_cache),
            "errors": self._errors,
        }

    def clear_cache(self) -> None:
        """Clear the phrase cache."""
        self._phrase_cache.clear()
        logger.debug("Speech cache cleared")

    async def close(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._phrase_cache.clear()
        logger.info("SpeechService closed")

    def __repr__(self) -> str:
        key_display = "***" if self._api_key else "None"
        return (
            f"SpeechService(api_key={key_display}, "
            f"tts_model={self._config.tts_model}, "
            f"initialized={self._initialized})"
        )
