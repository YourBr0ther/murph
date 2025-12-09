"""
Murph - LLM Configuration
Configuration for LLM integration with environment variable support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""

    # Provider selection
    provider: Literal["nanogpt", "ollama", "mock"] = "ollama"

    # NanoGPT settings
    nanogpt_api_key: str | None = None
    nanogpt_base_url: str = "https://nano-gpt.com/api/v1"
    nanogpt_model: str = "gpt-4o-mini"
    nanogpt_vision_model: str = "gpt-4o-mini"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_vision_model: str = "llama3.2-vision"

    # Feature toggles
    vision_enabled: bool = True
    reasoning_enabled: bool = True

    # Vision analysis settings
    vision_interval_seconds: float = 3.0

    # Rate limiting
    max_requests_per_minute: int = 20

    # Caching
    cache_ttl_seconds: float = 30.0
    cache_max_entries: int = 100

    # Timeouts
    request_timeout_seconds: float = 10.0

    # Behavior reasoning
    reasoning_score_threshold: float = 0.3  # Consult LLM if score diff < this

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            MURPH_LLM_PROVIDER: Provider ("nanogpt", "ollama", "mock")
            NANOGPT_API_KEY: API key for NanoGPT
            NANOGPT_BASE_URL: Base URL for NanoGPT API
            NANOGPT_MODEL: Default model for NanoGPT
            NANOGPT_VISION_MODEL: Vision model for NanoGPT
            OLLAMA_BASE_URL: Base URL for Ollama
            OLLAMA_MODEL: Default text model for Ollama
            OLLAMA_VISION_MODEL: Vision model for Ollama
            MURPH_LLM_VISION_ENABLED: Enable vision analysis
            MURPH_LLM_REASONING_ENABLED: Enable behavior reasoning
            MURPH_LLM_VISION_INTERVAL: Vision analysis interval (seconds)
            MURPH_LLM_MAX_RPM: Max requests per minute
            MURPH_LLM_CACHE_TTL: Cache TTL (seconds)
            MURPH_LLM_TIMEOUT: Request timeout (seconds)
            MURPH_LLM_REASONING_THRESHOLD: Score threshold for reasoning
        """
        return cls(
            provider=os.getenv("MURPH_LLM_PROVIDER", "ollama"),  # type: ignore
            nanogpt_api_key=os.getenv("NANOGPT_API_KEY"),
            nanogpt_base_url=os.getenv("NANOGPT_BASE_URL", "https://nano-gpt.com/api/v1"),
            nanogpt_model=os.getenv("NANOGPT_MODEL", "gpt-4o-mini"),
            nanogpt_vision_model=os.getenv("NANOGPT_VISION_MODEL", "gpt-4o-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision"),
            vision_enabled=os.getenv("MURPH_LLM_VISION_ENABLED", "true").lower() == "true",
            reasoning_enabled=os.getenv("MURPH_LLM_REASONING_ENABLED", "true").lower() == "true",
            vision_interval_seconds=float(os.getenv("MURPH_LLM_VISION_INTERVAL", "3.0")),
            max_requests_per_minute=int(os.getenv("MURPH_LLM_MAX_RPM", "20")),
            cache_ttl_seconds=float(os.getenv("MURPH_LLM_CACHE_TTL", "30.0")),
            request_timeout_seconds=float(os.getenv("MURPH_LLM_TIMEOUT", "10.0")),
            reasoning_score_threshold=float(os.getenv("MURPH_LLM_REASONING_THRESHOLD", "0.3")),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        if self.provider == "nanogpt" and not self.nanogpt_api_key:
            issues.append("NanoGPT provider selected but NANOGPT_API_KEY not set")

        if self.vision_interval_seconds < 0.5:
            issues.append("Vision interval too small (min 0.5 seconds)")

        if self.max_requests_per_minute < 1:
            issues.append("Max requests per minute must be >= 1")

        if self.cache_ttl_seconds < 0:
            issues.append("Cache TTL cannot be negative")

        if self.request_timeout_seconds < 1:
            issues.append("Request timeout must be >= 1 second")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def __repr__(self) -> str:
        api_key_display = "***" if self.nanogpt_api_key else "None"
        return (
            f"LLMConfig(provider={self.provider}, "
            f"vision={self.vision_enabled}, reasoning={self.reasoning_enabled}, "
            f"nanogpt_key={api_key_display})"
        )
