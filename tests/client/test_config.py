# tests/client/test_config.py
import pytest
from murph_client.config import ClientConfig, GPIOPins


def test_config_has_server_settings():
    config = ClientConfig()
    assert config.server_host == "10.0.2.192"
    assert config.server_port == 8765


def test_gpio_pins_motor_mapping():
    pins = GPIOPins()
    assert pins.NSLEEP1 == 12
    assert pins.NSLEEP2 == 13
    assert pins.M1_A == 17
    assert pins.M1_B == 27
