# client/src/murph_client/sensors/ultrasonic.py
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class UltrasonicSensor:
    def __init__(self, trig_pin: int = 4, echo_pin: int = 5):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available")

        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)

    def get_distance(self) -> float:
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.1:
                return -1

        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end - pulse_start > 0.1:
                return -1

        pulse_duration = pulse_end - pulse_start
        return (pulse_duration * 34300) / 2

    def obstacle_detected(self, threshold_cm: float = 20.0) -> bool:
        distance = self.get_distance()
        return 0 < distance < threshold_cm
