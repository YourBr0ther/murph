# client/src/murph_client/audio/capture.py
import os
import sys
import numpy as np
from scipy import signal


import pyaudio


def _create_pyaudio():
    """Create PyAudio instance with stderr suppressed to hide JACK warnings."""
    if sys.platform != "linux":
        return pyaudio.PyAudio()

    # Suppress JACK errors during PyAudio init
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(sys.stderr.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    try:
        pa = pyaudio.PyAudio()
    finally:
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(devnull)
        os.close(old_stderr)
    return pa


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        device_index: int = None,
        device_sample_rate: int = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index

        # If device has different native rate, we'll capture at that rate and resample
        self.device_sample_rate = device_sample_rate or sample_rate
        self.needs_resampling = self.device_sample_rate != sample_rate

        # Calculate device chunk size to maintain timing
        if self.needs_resampling:
            self.device_chunk_size = int(chunk_size * self.device_sample_rate / sample_rate)
        else:
            self.device_chunk_size = chunk_size

        self.pa = _create_pyaudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.device_sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.device_chunk_size,
        )

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from device rate to target rate."""
        if not self.needs_resampling:
            return audio
        # Use polyphase resampling for efficiency
        # 44100 -> 16000 is 441:160 ratio (simplified from GCD)
        from math import gcd
        g = gcd(self.device_sample_rate, self.sample_rate)
        up = self.sample_rate // g
        down = self.device_sample_rate // g
        return signal.resample_poly(audio, up, down).astype(np.int16)

    def read(self) -> np.ndarray:
        data = self.stream.read(self.device_chunk_size, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        return self._resample(audio)

    def read_seconds(self, seconds: float) -> np.ndarray:
        frames = []
        num_chunks = int(self.device_sample_rate * seconds / self.device_chunk_size)
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
