# client/src/murph_client/audio/playback.py
import random
import wave
from pathlib import Path
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

# Path to sounds directory
SOUNDS_DIR = Path(__file__).parent.parent / "sounds"


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

        # Add silence padding to ensure short sounds don't get cut off
        padding = bytes(frames_per_buffer * 2)  # One buffer of silence
        stream.write(audio_bytes + padding)

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

    def beep(self, frequency: int = 800, duration: float = 0.2):
        """Play a short beep tone to signal recording start."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Generate sine wave with fade in/out to avoid clicks
        tone = np.sin(2 * np.pi * frequency * t)
        fade_samples = int(self.sample_rate * 0.015)  # 15ms fade
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        audio = (tone * 32767 * 0.7).astype(np.int16)  # Louder beep
        self.play(audio.tobytes())

    def say_yes_sir(self):
        """Play the 'Yes sir?' response."""
        yes_sir_path = SOUNDS_DIR / "yes_sir.wav"
        if yes_sir_path.exists():
            self.play_file(str(yes_sir_path))
        else:
            # Fallback to beep if sound file missing
            self.beep(frequency=600, duration=0.15)
            self.beep(frequency=800, duration=0.2)

    def chirp(self):
        """Play a random R2-D2 style chirp sound."""
        chirp_files = list(SOUNDS_DIR.glob("chirp*.wav"))
        if chirp_files:
            self.play_file(str(random.choice(chirp_files)))
        else:
            # Fallback to generated chirp
            freqs = random.choice([
                [600, 900, 1200],
                [800, 600, 800],
                [500, 700, 900],
            ])
            for freq in freqs:
                self.beep(frequency=freq, duration=0.08)

    def stop(self):
        """Stop playback (no-op for this implementation)."""
        pass

    def cleanup(self):
        """Release PyAudio resources."""
        self.pa.terminate()
