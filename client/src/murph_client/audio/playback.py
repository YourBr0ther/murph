# client/src/murph_client/audio/playback.py
import time
import numpy as np

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class AudioPlayback:
    def __init__(self, pin: int = 18, sample_rate: int = 22050, pwm_freq: int = 44100):
        self.pin = pin
        self.sample_rate = sample_rate
        self.pwm_freq = pwm_freq

        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available")

        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, pwm_freq)
        self.pwm.start(50)

    def play(self, audio_bytes: bytes):
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Normalize to 0-100 duty cycle range
        audio_normalized = ((audio.astype(np.float32) + 32768) / 65536 * 100).astype(int)

        # Calculate sample interval
        interval = 1.0 / self.sample_rate

        for sample in audio_normalized:
            self.pwm.ChangeDutyCycle(max(0, min(100, sample)))
            time.sleep(interval)

        # Return to 50% duty cycle (silence)
        self.pwm.ChangeDutyCycle(50)

    def stop(self):
        self.pwm.stop()

    def cleanup(self):
        self.stop()
