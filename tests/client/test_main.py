# tests/client/test_main.py
import sys
import pytest
from unittest.mock import MagicMock


def test_client_can_import(mock_gpio, mock_pyaudio, mock_openwakeword):
    # Also mock websockets
    sys.modules["websockets"] = MagicMock()

    from murph_client.main import MurphClient
    assert MurphClient is not None


def test_client_config():
    from murph_client.config import ClientConfig
    config = ClientConfig()
    assert config.server_uri == "ws://10.0.2.192:8765/ws"
