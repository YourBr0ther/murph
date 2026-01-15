# server/src/murph_server/audio/tts.py
from pathlib import Path
from piper import PiperVoice


class TextToSpeech:
    def __init__(self, voice: str = "en_US-lessac-medium"):
        self.voice_name = voice
        voice_path = self._find_voice_model(voice)
        self.voice = PiperVoice.load(voice_path)

    def _find_voice_model(self, voice: str) -> Path:
        search_paths = [
            Path.home() / ".local/share/piper/voices",
            Path("/usr/share/piper/voices"),
            Path("./voices"),
        ]
        for base in search_paths:
            model_path = base / f"{voice}.onnx"
            if model_path.exists():
                return model_path
        raise FileNotFoundError(f"Voice model not found: {voice}")

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            audio_chunks.append(chunk.audio_int16_bytes)
        return b"".join(audio_chunks)
