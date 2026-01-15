# tests/server/test_ollama.py
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Mock ollama module before importing our module
mock_ollama = MagicMock()
sys.modules["ollama"] = mock_ollama


def test_system_prompt_has_personality():
    # Clear cached module
    if "murph_server.llm.ollama" in sys.modules:
        del sys.modules["murph_server.llm.ollama"]

    from murph_server.llm.ollama import SYSTEM_PROMPT
    assert "WALL-E" in SYSTEM_PROMPT or "wall-e" in SYSTEM_PROMPT.lower()
    assert "helpful" in SYSTEM_PROMPT.lower()


@pytest.mark.asyncio
async def test_query_returns_response():
    # Setup mock
    mock_client = AsyncMock()
    mock_client.chat.return_value = {"message": {"content": "Hello there!"}}
    mock_ollama.AsyncClient.return_value = mock_client

    # Clear cached module to get fresh import with mock
    if "murph_server.llm.ollama" in sys.modules:
        del sys.modules["murph_server.llm.ollama"]

    from murph_server.llm.ollama import OllamaClient

    client = OllamaClient(model="llama3.2")
    response = await client.query("Hello")

    assert response == "Hello there!"
