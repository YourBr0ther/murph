# client/src/murph_client/audio/capture.py
import numpy as np
import pyaudio


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

    def read(self) -> np.ndarray:
        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def read_seconds(self, seconds: float) -> np.ndarray:
        frames = []
        num_chunks = int(self.sample_rate * seconds / self.chunk_size)
        for _ in range(num_chunks):
            frames.append(self.read())
        return np.concatenate(frames)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
