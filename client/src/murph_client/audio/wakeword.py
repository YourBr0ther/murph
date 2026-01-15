# client/src/murph_client/audio/wakeword.py
from pathlib import Path
import numpy as np
import openwakeword


class WakeWordDetector:
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        # Use custom model or fall back to default
        if model_path is None:
            # Look for hey_murph.onnx in models directory
            default_model = Path(__file__).parent.parent / "models" / "hey_murph.onnx"
            if default_model.exists():
                model_path = str(default_model)
            else:
                model_path = "hey_jarvis"  # Fall back to built-in

        self.model_path = model_path
        self.threshold = threshold

        # Get wake word name from model filename
        if model_path.endswith(".onnx"):
            self.wake_word = Path(model_path).stem
        else:
            self.wake_word = model_path

        self.model = openwakeword.Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )

    def process(self, audio: np.ndarray) -> bool:
        scores = self.model.predict(audio)
        score = scores.get(self.wake_word, 0.0)
        return score >= self.threshold

    def reset(self):
        self.model.reset()
