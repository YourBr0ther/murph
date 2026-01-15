# tests/client/test_motor_driver.py
import pytest
from unittest.mock import MagicMock


def test_motor_driver_init_sets_gpio_mode(mock_gpio):
    from murph_client.motors.driver import MotorDriver
    driver = MotorDriver()
    mock_gpio.setmode.assert_called_with(mock_gpio.BCM)


def test_duty_cycle_capped_at_25(mock_gpio):
    mock_pwm = MagicMock()
    mock_gpio.PWM.return_value = mock_pwm

    from murph_client.motors.driver import MotorDriver

    driver = MotorDriver(max_duty_cycle=25)
    driver.set_speed(100)
    mock_pwm.ChangeDutyCycle.assert_called_with(25)
