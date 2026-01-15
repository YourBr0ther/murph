# client/src/murph_client/audio/playback.py
import wave
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None


class AudioPlayback:
    """Audio playback using PyAudio (3.5mm jack or USB audio)."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1, device_index: int = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index  # None = default device

        if pyaudio is None:
            raise RuntimeError("PyAudio not available")

        self.pa = pyaudio.PyAudio()

    def play(self, audio_bytes: bytes):
        """Play raw audio bytes (16-bit PCM)."""
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
        )

        # Write audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio_bytes), chunk_size):
            stream.write(audio_bytes[i:i + chunk_size])

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

    def stop(self):
        """Stop playback (no-op for this implementation)."""
        pass

    def cleanup(self):
        """Release PyAudio resources."""
        self.pa.terminate()
