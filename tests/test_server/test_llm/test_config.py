"""Tests for LLM configuration."""

import os
from unittest.mock import patch

import pytest

from server.llm.config import LLMConfig


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == "ollama"
        assert config.nanogpt_api_key is None
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.vision_enabled is True
        assert config.reasoning_enabled is True
        assert config.max_requests_per_minute == 20

    def test_from_env_defaults(self) -> None:
        """Test loading config from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()

        assert config.provider == "ollama"
        assert config.vision_enabled is True

    def test_from_env_custom_values(self) -> None:
        """Test loading config from environment with custom values."""
        env = {
            "MURPH_LLM_PROVIDER": "nanogpt",
            "NANOGPT_API_KEY": "test-key-123",
            "NANOGPT_MODEL": "gpt-4",
            "MURPH_LLM_VISION_ENABLED": "false",
            "MURPH_LLM_REASONING_ENABLED": "true",
            "MURPH_LLM_MAX_RPM": "10",
            "MURPH_LLM_VISION_INTERVAL": "5.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()

        assert config.provider == "nanogpt"
        assert config.nanogpt_api_key == "test-key-123"
        assert config.nanogpt_model == "gpt-4"
        assert config.vision_enabled is False
        assert config.reasoning_enabled is True
        assert config.max_requests_per_minute == 10
        assert config.vision_interval_seconds == 5.0

    def test_validate_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = LLMConfig(
            provider="ollama",
            vision_interval_seconds=3.0,
            max_requests_per_minute=20,
        )

        issues = config.validate()
        assert issues == []
        assert config.is_valid()

    def test_validate_nanogpt_without_key(self) -> None:
        """Test validation fails for NanoGPT without API key."""
        config = LLMConfig(
            provider="nanogpt",
            nanogpt_api_key=None,
        )

        issues = config.validate()
        assert len(issues) == 1
        assert "API_KEY" in issues[0]
        assert not config.is_valid()

    def test_validate_vision_interval_too_small(self) -> None:
        """Test validation fails for vision interval too small."""
        config = LLMConfig(
            vision_interval_seconds=0.1,
        )

        issues = config.validate()
        assert any("interval" in issue.lower() for issue in issues)

    def test_validate_invalid_rate_limit(self) -> None:
        """Test validation fails for invalid rate limit."""
        config = LLMConfig(
            max_requests_per_minute=0,
        )

        issues = config.validate()
        assert any("requests" in issue.lower() for issue in issues)

    def test_repr_hides_api_key(self) -> None:
        """Test repr hides API key for security."""
        config = LLMConfig(
            nanogpt_api_key="secret-key-12345",
        )

        repr_str = repr(config)
        assert "secret-key" not in repr_str
        assert "***" in repr_str
