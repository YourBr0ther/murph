# tests/client/test_main.py
import sys
import pytest
from unittest.mock import MagicMock


def test_client_can_import(mock_gpio, mock_pyaudio, mock_openwakeword):
    # Mock websockets with its submodules
    mock_websockets = MagicMock()
    mock_exceptions = MagicMock()
    mock_exceptions.ConnectionClosed = Exception
    mock_exceptions.ConnectionClosedError = Exception
    mock_websockets.exceptions = mock_exceptions

    sys.modules["websockets"] = mock_websockets
    sys.modules["websockets.exceptions"] = mock_exceptions

    from murph_client.main import MurphClient
    assert MurphClient is not None


def test_client_config():
    from murph_client.config import ClientConfig
    config = ClientConfig()
    assert config.server_uri == "ws://10.0.2.192:8765/ws"
