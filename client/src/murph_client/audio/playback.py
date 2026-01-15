# client/src/murph_client/audio/playback.py
import wave
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None


class AudioPlayback:
    """Audio playback using PyAudio (3.5mm jack or USB audio)."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, device_index: int = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index  # None = default device

        if pyaudio is None:
            raise RuntimeError("PyAudio not available")

        self.pa = pyaudio.PyAudio()

    def play(self, audio_bytes: bytes):
        """Play raw audio bytes (16-bit PCM)."""
        # Use larger buffer to prevent underruns
        frames_per_buffer = 2048
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=frames_per_buffer,
        )

        # Write all audio at once - PyAudio handles buffering
        stream.write(audio_bytes)

        stream.stop_stream()
        stream.close()

    def play_file(self, filepath: str):
        """Play a WAV file."""
        with wave.open(filepath, 'rb') as wf:
            stream = self.pa.open(
                format=self.pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=self.device_index,
            )

            chunk = 1024
            data = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)

            stream.stop_stream()
            stream.close()

    def beep(self, frequency: int = 800, duration: float = 0.15):
        """Play a short beep tone to signal recording start."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Generate sine wave with fade in/out to avoid clicks
        tone = np.sin(2 * np.pi * frequency * t)
        fade_samples = int(self.sample_rate * 0.01)  # 10ms fade
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        audio = (tone * 32767 * 0.5).astype(np.int16)
        self.play(audio.tobytes())

    def stop(self):
        """Stop playback (no-op for this implementation)."""
        pass

    def cleanup(self):
        """Release PyAudio resources."""
        self.pa.terminate()
