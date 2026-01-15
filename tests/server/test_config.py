# tests/server/test_config.py
import pytest
from murph_server.config import ServerConfig


def test_config_has_required_fields():
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8765
    assert config.whisper_model == "base"
    assert config.ollama_model == "llama3.2"
    assert config.piper_voice == "en_US-lessac-medium"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MURPH_PORT", "9000")
    monkeypatch.setenv("MURPH_OLLAMA_MODEL", "mistral")
    config = ServerConfig()
    assert config.port == 9000
    assert config.ollama_model == "mistral"
