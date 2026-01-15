# server/src/murph_server/audio/stt.py
import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        segments, _ = self.model.transcribe(audio_data, language="en")
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()
