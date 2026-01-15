# server/src/murph_server/audio/tts.py
from pathlib import Path
import numpy as np
from scipy import signal
from piper import PiperVoice


def radio_effect(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply vintage radio effect like Alastor from Hazbin Hotel."""
    audio = audio.astype(np.float32) / 32768.0

    # Pad audio to avoid filter edge artifacts
    pad_len = sample_rate // 4  # 250ms padding
    audio = np.pad(audio, (pad_len, pad_len), mode='reflect')

    # 1. Bandpass filter (300-3400 Hz) - classic telephone/radio range
    low_cut = 300 / (sample_rate / 2)
    high_cut = 3400 / (sample_rate / 2)
    sos = signal.butter(4, [low_cut, high_cut], btype='band', output='sos')
    audio = signal.sosfiltfilt(sos, audio)

    # 2. Soft clipping / tube saturation
    drive = 1.5  # Amount of drive/distortion
    audio = np.tanh(audio * drive) / np.tanh(drive)

    # 3. Add subtle harmonics (warmth)
    harmonics = np.tanh(audio * 2) * 0.1
    audio = audio + harmonics

    # 4. Slight compression (reduce dynamic range)
    threshold = 0.4
    ratio = 3.0
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )

    # 5. Add very subtle crackle/noise
    noise = np.random.randn(len(audio)) * 0.005
    audio = audio + noise

    # 6. Final bandpass to clean up
    audio = signal.sosfiltfilt(sos, audio)

    # Remove padding
    audio = audio[pad_len:-pad_len]

    # 7. Fade in/out to prevent clicks (20ms)
    fade_samples = int(sample_rate * 0.02)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Normalize and convert back to int16
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.75
    return (audio * 32767).astype(np.int16)


class TextToSpeech:
    def __init__(self, voice: str = "en_US-danny-low"):
        self.voice_name = voice
        voice_path = self._find_voice_model(voice)
        self.voice = PiperVoice.load(voice_path)

    def _find_voice_model(self, voice: str) -> Path:
        search_paths = [
            Path.home() / ".local/share/piper/voices",
            Path("/usr/share/piper/voices"),
            Path("./voices"),
        ]
        for base in search_paths:
            model_path = base / f"{voice}.onnx"
            if model_path.exists():
                return model_path
        raise FileNotFoundError(f"Voice model not found: {voice}")

    def synthesize(self, text: str, apply_radio_effect: bool = True) -> bytes:
        if not text.strip():
            return b""
        audio_chunks = []
        sample_rate = None
        for chunk in self.voice.synthesize(text):
            audio_chunks.append(chunk.audio_int16_bytes)
            sample_rate = chunk.sample_rate

        raw_audio = b"".join(audio_chunks)

        if apply_radio_effect and raw_audio:
            # Convert to numpy, apply effect, convert back
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)
            processed = radio_effect(audio_array, sample_rate or 16000)
            return processed.tobytes()

        return raw_audio
