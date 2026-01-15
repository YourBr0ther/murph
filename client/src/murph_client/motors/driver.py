# client/src/murph_client/motors/driver.py
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

from murph_client.config import GPIOPins


class MotorDriver:
    def __init__(self, max_duty_cycle: int = 25, pwm_freq: int = 1000):
        self.pins = GPIOPins()
        self.max_duty_cycle = max_duty_cycle
        self.pwm_freq = pwm_freq

        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available")

        GPIO.setmode(GPIO.BCM)
        self._setup_pins()
        self._setup_pwm()
        self.stop()

    def _setup_pins(self):
        motor_pins = [
            self.pins.NSLEEP1, self.pins.NSLEEP2,
            self.pins.M1_A, self.pins.M1_B, self.pins.M2_A, self.pins.M2_B,
            self.pins.M3_A, self.pins.M3_B, self.pins.M4_A, self.pins.M4_B,
        ]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

    def _setup_pwm(self):
        self.pwm1 = GPIO.PWM(self.pins.NSLEEP1, self.pwm_freq)
        self.pwm2 = GPIO.PWM(self.pins.NSLEEP2, self.pwm_freq)
        self.pwm1.start(0)
        self.pwm2.start(0)

    def set_speed(self, duty_cycle: int):
        capped = min(duty_cycle, self.max_duty_cycle)
        self.pwm1.ChangeDutyCycle(capped)
        self.pwm2.ChangeDutyCycle(capped)

    def stop(self):
        for pin in [self.pins.M1_A, self.pins.M1_B, self.pins.M2_A, self.pins.M2_B,
                    self.pins.M3_A, self.pins.M3_B, self.pins.M4_A, self.pins.M4_B]:
            GPIO.output(pin, GPIO.LOW)

    def forward(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.LOW)
        GPIO.output(self.pins.M1_B, GPIO.HIGH)
        GPIO.output(self.pins.M2_A, GPIO.LOW)
        GPIO.output(self.pins.M2_B, GPIO.HIGH)
        GPIO.output(self.pins.M3_A, GPIO.LOW)
        GPIO.output(self.pins.M3_B, GPIO.HIGH)
        GPIO.output(self.pins.M4_A, GPIO.LOW)
        GPIO.output(self.pins.M4_B, GPIO.HIGH)

    def backward(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.HIGH)
        GPIO.output(self.pins.M1_B, GPIO.LOW)
        GPIO.output(self.pins.M2_A, GPIO.HIGH)
        GPIO.output(self.pins.M2_B, GPIO.LOW)
        GPIO.output(self.pins.M3_A, GPIO.HIGH)
        GPIO.output(self.pins.M3_B, GPIO.LOW)
        GPIO.output(self.pins.M4_A, GPIO.HIGH)
        GPIO.output(self.pins.M4_B, GPIO.LOW)

    def rotate_left(self):
        # Left side (M1=FL, M4=BL) backward, Right side (M2=FR, M3=BR) forward
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.HIGH)  # FL backward
        GPIO.output(self.pins.M1_B, GPIO.LOW)
        GPIO.output(self.pins.M4_A, GPIO.HIGH)  # BL backward
        GPIO.output(self.pins.M4_B, GPIO.LOW)
        GPIO.output(self.pins.M2_A, GPIO.LOW)   # FR forward
        GPIO.output(self.pins.M2_B, GPIO.HIGH)
        GPIO.output(self.pins.M3_A, GPIO.LOW)   # BR forward
        GPIO.output(self.pins.M3_B, GPIO.HIGH)

    def rotate_right(self):
        # Left side (M1=FL, M4=BL) forward, Right side (M2=FR, M3=BR) backward
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.LOW)   # FL forward
        GPIO.output(self.pins.M1_B, GPIO.HIGH)
        GPIO.output(self.pins.M4_A, GPIO.LOW)   # BL forward
        GPIO.output(self.pins.M4_B, GPIO.HIGH)
        GPIO.output(self.pins.M2_A, GPIO.HIGH)  # FR backward
        GPIO.output(self.pins.M2_B, GPIO.LOW)
        GPIO.output(self.pins.M3_A, GPIO.HIGH)  # BR backward
        GPIO.output(self.pins.M3_B, GPIO.LOW)

    def move_for_distance(self, direction: str, inches: float, ips: float = 1.0):
        duration = inches / ips
        getattr(self, direction, self.stop)()
        time.sleep(duration)
        self.stop()

    def cleanup(self):
        self.pwm1.stop()
        self.pwm2.stop()
        GPIO.cleanup()
