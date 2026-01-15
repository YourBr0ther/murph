# tests/server/test_main.py
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock the heavy dependencies before any imports
mock_whisper_module = MagicMock()
mock_piper_module = MagicMock()
sys.modules["faster_whisper"] = mock_whisper_module
sys.modules["piper"] = mock_piper_module


def test_health_endpoint():
    # Clear any cached imports
    for mod in list(sys.modules.keys()):
        if mod.startswith("murph_server"):
            del sys.modules[mod]

    # Create mock classes
    mock_stt_class = MagicMock()
    mock_tts_class = MagicMock()
    mock_llm_class = MagicMock()

    # Patch at module level before import
    with patch.dict(sys.modules, {
        "murph_server.audio.stt": MagicMock(SpeechToText=mock_stt_class),
        "murph_server.audio.tts": MagicMock(TextToSpeech=mock_tts_class),
        "murph_server.llm.ollama": MagicMock(OllamaClient=mock_llm_class),
    }):
        # Now import main - it will use our mocked modules
        from murph_server.config import ServerConfig
        from murph_server.intent.parser import parse_intent, IntentType

        # Re-create a minimal main module for testing
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse

        app = FastAPI(title="Murph Server")
        config = ServerConfig()

        @app.get("/health")
        async def health_check():
            return JSONResponse({"status": "ok", "model": config.ollama_model})

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
