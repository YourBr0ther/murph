# server/src/murph_server/audio/stt.py
import os
import sys
import glob

# Add CUDA Toolkit DLL path on Windows before importing faster_whisper
if sys.platform == "win32":
    cuda_paths = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin")
    for cuda_bin in cuda_paths:
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)
            os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")

import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        segments, _ = self.model.transcribe(audio_data, language="en")
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()
