# tests/client/conftest.py
import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def reset_client_modules():
    """Reset all murph_client module imports before each test."""
    # Store modules to delete
    to_delete = [mod for mod in sys.modules.keys()
                 if mod.startswith("murph_client") and mod != "murph_client"]
    for mod in to_delete:
        del sys.modules[mod]

    yield

    # Clean up after test
    to_delete = [mod for mod in sys.modules.keys()
                 if mod.startswith("murph_client") and mod != "murph_client"]
    for mod in to_delete:
        del sys.modules[mod]


@pytest.fixture
def mock_gpio():
    """Provide a properly configured GPIO mock."""
    mock = MagicMock()
    mock.BCM = 11
    mock.OUT = 0
    mock.IN = 1
    mock.HIGH = 1
    mock.LOW = 0

    mock_rpi = MagicMock()
    mock_rpi.GPIO = mock

    sys.modules["RPi"] = mock_rpi
    sys.modules["RPi.GPIO"] = mock

    return mock


@pytest.fixture
def mock_pyaudio():
    """Provide a pyaudio mock."""
    mock = MagicMock()
    mock.paInt16 = 8
    sys.modules["pyaudio"] = mock
    return mock


@pytest.fixture
def mock_openwakeword():
    """Provide an openwakeword mock."""
    mock = MagicMock()
    sys.modules["openwakeword"] = mock
    sys.modules["openwakeword.model"] = MagicMock()
    return mock
