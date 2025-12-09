"""
Murph - LLM Types
Data types for LLM integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMMessage:
    """A single message in a chat completion."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to API-compatible dict."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMRequest:
    """Request to an LLM provider."""

    messages: list[LLMMessage]
    model: str | None = None  # Override default model
    temperature: float = 0.7
    max_tokens: int = 500

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dict."""
        d: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            d["model"] = self.model
        return d


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for debugging."""
        return {
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "provider": self.provider,
        }


@dataclass
class SceneAnalysis:
    """Result of LLM vision scene analysis."""

    description: str
    detected_objects: list[str] = field(default_factory=list)
    detected_activities: list[str] = field(default_factory=list)
    mood_indicators: list[str] = field(default_factory=list)
    suggested_triggers: list[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for debugging."""
        return {
            "description": self.description,
            "detected_objects": self.detected_objects,
            "detected_activities": self.detected_activities,
            "mood_indicators": self.mood_indicators,
            "suggested_triggers": self.suggested_triggers,
            "confidence": self.confidence,
            "age_seconds": time.time() - self.timestamp,
        }


@dataclass
class BehaviorRecommendation:
    """LLM recommendation for behavior selection."""

    recommended_behavior: str
    reasoning: str
    confidence: float = 0.5
    alternative_behaviors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for debugging."""
        return {
            "recommended_behavior": self.recommended_behavior,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternative_behaviors": self.alternative_behaviors,
        }
