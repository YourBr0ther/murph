# server/src/murph_server/audio/stt.py
import os
import sys

# Add NVIDIA DLL paths on Windows before importing faster_whisper
if sys.platform == "win32":
    # Find site-packages in the current environment (works with venv)
    import importlib.util
    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec and nvidia_spec.submodule_search_locations:
        nvidia_path = nvidia_spec.submodule_search_locations[0]
        for lib in ["cublas", "cudnn", "cuda_runtime", "cufft", "curand"]:
            bin_path = os.path.join(nvidia_path, lib, "bin")
            if os.path.exists(bin_path):
                os.add_dll_directory(bin_path)

import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        segments, _ = self.model.transcribe(audio_data, language="en")
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()
