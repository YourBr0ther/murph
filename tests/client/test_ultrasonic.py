# tests/client/test_ultrasonic.py
import pytest


def test_ultrasonic_init_configures_pins(mock_gpio):
    from murph_client.sensors.ultrasonic import UltrasonicSensor
    sensor = UltrasonicSensor(trig_pin=4, echo_pin=5)
    mock_gpio.setup.assert_any_call(4, mock_gpio.OUT)
    mock_gpio.setup.assert_any_call(5, mock_gpio.IN)
