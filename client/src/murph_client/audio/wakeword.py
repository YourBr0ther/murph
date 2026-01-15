# client/src/murph_client/audio/wakeword.py
import numpy as np
import openwakeword


class WakeWordDetector:
    def __init__(self, wake_word: str = "hey_jarvis", threshold: float = 0.5):
        self.wake_word = wake_word
        self.threshold = threshold
        self.model = openwakeword.Model(
            wakeword_models=[wake_word],
            inference_framework="onnx",
        )

    def process(self, audio: np.ndarray) -> bool:
        self.model.predict(audio)
        scores = self.model.get_prediction()
        score = scores.get(self.wake_word, 0.0)
        return score >= self.threshold

    def reset(self):
        self.model.reset()
