# client/src/murph_client/audio/playback.py
import random
import wave
from pathlib import Path
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

# Path to sounds directory
SOUNDS_DIR = Path(__file__).parent.parent / "sounds"


class AudioPlayback:
    """Audio playback using PyAudio (3.5mm jack or USB audio)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int = None,
        output_sample_rate: int = 48000,
        output_channels: int = 2,
    ):
        self.sample_rate = sample_rate  # Input sample rate (from TTS)
        self.channels = channels  # Input channels
        self.device_index = device_index  # None = default device
        self.output_sample_rate = output_sample_rate  # Output device native rate
        self.output_channels = output_channels  # Output device channels

        if pyaudio is None:
            raise RuntimeError("PyAudio not available")

        self.pa = pyaudio.PyAudio()

    def _resample_to_output(self, audio: np.ndarray, input_rate: int) -> np.ndarray:
        """Resample audio to output device sample rate and convert to stereo if needed."""
        # Resample if rates differ
        if input_rate != self.output_sample_rate:
            if scipy_signal is not None:
                num_samples = int(len(audio) * self.output_sample_rate / input_rate)
                audio = scipy_signal.resample(audio, num_samples).astype(np.int16)
            else:
                # Simple linear interpolation fallback
                indices = np.linspace(0, len(audio) - 1, int(len(audio) * self.output_sample_rate / input_rate))
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)

        # Convert mono to stereo if needed
        if self.output_channels == 2 and len(audio.shape) == 1:
            audio = np.column_stack((audio, audio)).flatten().astype(np.int16)

        return audio

    def play(self, audio_bytes: bytes):
        """Play raw audio bytes (16-bit PCM)."""
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Resample and convert to output format
        audio = self._resample_to_output(audio, self.sample_rate)

        # Use larger buffer to prevent underruns
        frames_per_buffer = 2048
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.output_channels,
            rate=self.output_sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=frames_per_buffer,
        )

        # Add silence padding to ensure short sounds don't get cut off
        padding_samples = frames_per_buffer * self.output_channels
        padding = np.zeros(padding_samples, dtype=np.int16)
        output_audio = np.concatenate([audio, padding])
        stream.write(output_audio.tobytes())

        stream.stop_stream()
        stream.close()

    def play_file(self, filepath: str):
        """Play a WAV file."""
        with wave.open(filepath, 'rb') as wf:
            # Read entire file
            audio_data = wf.readframes(wf.getnframes())
            file_rate = wf.getframerate()
            file_channels = wf.getnchannels()

            # Convert to numpy array
            audio = np.frombuffer(audio_data, dtype=np.int16)

            # If stereo, convert to mono first for resampling
            if file_channels == 2:
                audio = audio[::2]  # Take left channel

            # Resample and convert to output format
            audio = self._resample_to_output(audio, file_rate)

            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.output_channels,
                rate=self.output_sample_rate,
                output=True,
                output_device_index=self.device_index,
            )

            stream.write(audio.tobytes())
            stream.stop_stream()
            stream.close()

    def beep(self, frequency: int = 800, duration: float = 0.2):
        """Play a short beep tone to signal recording start."""
        # Generate at output sample rate directly for efficiency
        t = np.linspace(0, duration, int(self.output_sample_rate * duration), False)
        # Generate sine wave with fade in/out to avoid clicks
        tone = np.sin(2 * np.pi * frequency * t)
        fade_samples = int(self.output_sample_rate * 0.015)  # 15ms fade
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        audio = (tone * 32767 * 0.7).astype(np.int16)  # Louder beep

        # Convert to stereo if needed
        if self.output_channels == 2:
            audio = np.column_stack((audio, audio)).flatten().astype(np.int16)

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.output_channels,
            rate=self.output_sample_rate,
            output=True,
            output_device_index=self.device_index,
        )
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()

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
