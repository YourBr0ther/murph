"""Tests for LLM vision analyzer."""

import json
import time

import numpy as np
import pytest

from server.llm.config import LLMConfig
from server.llm.providers.mock import MockProvider
from server.llm.services.llm_service import LLMService
from server.llm.services.vision_analyzer import VisionAnalyzer
from server.cognition.behavior.context import WorldContext


class TestVisionAnalyzer:
    """Tests for VisionAnalyzer."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        """Create test configuration."""
        return LLMConfig(
            provider="mock",
            vision_enabled=True,
            vision_interval_seconds=0.1,  # Short interval for testing
        )

    @pytest.fixture
    def service(self, config: LLMConfig) -> LLMService:
        """Create LLM service."""
        return LLMService(config)

    @pytest.fixture
    def analyzer(self, service: LLMService, config: LLMConfig) -> VisionAnalyzer:
        """Create vision analyzer."""
        return VisionAnalyzer(service, config)

    @pytest.fixture
    def frame(self) -> np.ndarray:
        """Create test frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_init(self, analyzer: VisionAnalyzer) -> None:
        """Test analyzer initialization."""
        assert analyzer._interval == 0.1
        assert analyzer._last_result is None
        assert analyzer._analyses_completed == 0

    @pytest.mark.asyncio
    async def test_analyze_if_ready_first_call(
        self, analyzer: VisionAnalyzer, frame: np.ndarray
    ) -> None:
        """Test first analysis call succeeds."""
        result = await analyzer.analyze_if_ready(frame)

        assert result is not None
        assert analyzer._analyses_completed == 1

    @pytest.mark.asyncio
    async def test_analyze_if_ready_throttling(
        self, analyzer: VisionAnalyzer, frame: np.ndarray
    ) -> None:
        """Test analysis is throttled by interval."""
        # First call should succeed
        r1 = await analyzer.analyze_if_ready(frame)
        assert r1 is not None

        # Immediate second call should return cached result (same object reference)
        r2 = await analyzer.analyze_if_ready(frame)
        # Check it's the same cached analysis (same timestamp)
        assert r2.timestamp == r1.timestamp
        assert analyzer._analyses_completed == 1
        assert analyzer._analyses_skipped == 1

    @pytest.mark.asyncio
    async def test_analyze_if_ready_after_interval(
        self, analyzer: VisionAnalyzer, frame: np.ndarray
    ) -> None:
        """Test new analysis after interval passes."""
        r1 = await analyzer.analyze_if_ready(frame)
        assert r1 is not None

        # Wait for interval
        time.sleep(0.15)

        r2 = await analyzer.analyze_if_ready(frame)
        assert analyzer._analyses_completed == 2

    @pytest.mark.asyncio
    async def test_analyze_now_bypasses_throttle(
        self, analyzer: VisionAnalyzer, frame: np.ndarray
    ) -> None:
        """Test analyze_now bypasses throttling."""
        r1 = await analyzer.analyze_now(frame)
        r2 = await analyzer.analyze_now(frame)

        # Both should complete (no throttling)
        assert r1 is not None
        assert r2 is not None

    @pytest.mark.asyncio
    async def test_parse_json_response(
        self, service: LLMService, config: LLMConfig, frame: np.ndarray
    ) -> None:
        """Test parsing valid JSON response."""
        # Set up mock response
        mock_response = json.dumps({
            "description": "A person sitting at a desk",
            "objects": ["desk", "chair", "computer"],
            "activities": ["working"],
            "mood": ["focused"],
            "triggers": ["llm_person_engaged", "llm_activity_working"],
            "confidence": 0.9,
        })

        analyzer = VisionAnalyzer(service, config)
        # Access internal provider to set response
        await service._ensure_initialized()
        service._provider.set_responses([mock_response])

        result = await analyzer.analyze_now(frame)

        assert result is not None
        assert "person" in result.description.lower()
        assert "desk" in result.detected_objects
        assert "llm_person_engaged" in result.suggested_triggers
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_parse_invalid_json_fallback(
        self, service: LLMService, config: LLMConfig, frame: np.ndarray
    ) -> None:
        """Test fallback when JSON parsing fails."""
        analyzer = VisionAnalyzer(service, config)
        await service._ensure_initialized()
        service._provider.set_responses(["This is not valid JSON at all."])

        result = await analyzer.analyze_now(frame)

        assert result is not None
        assert result.description == "This is not valid JSON at all."
        assert result.confidence == 0.3  # Low confidence fallback
        assert analyzer._parse_errors == 1

    @pytest.mark.asyncio
    async def test_filters_invalid_triggers(
        self, service: LLMService, config: LLMConfig, frame: np.ndarray
    ) -> None:
        """Test that invalid triggers are filtered out."""
        mock_response = json.dumps({
            "description": "Test",
            "objects": [],
            "activities": [],
            "mood": [],
            "triggers": ["llm_person_engaged", "invalid_trigger", "llm_mood_happy"],
            "confidence": 0.8,
        })

        analyzer = VisionAnalyzer(service, config)
        await service._ensure_initialized()
        service._provider.set_responses([mock_response])

        result = await analyzer.analyze_now(frame)

        assert "llm_person_engaged" in result.suggested_triggers
        assert "llm_mood_happy" in result.suggested_triggers
        assert "invalid_trigger" not in result.suggested_triggers

    @pytest.mark.asyncio
    async def test_update_world_context(
        self, analyzer: VisionAnalyzer, frame: np.ndarray
    ) -> None:
        """Test updating WorldContext from analysis."""
        context = WorldContext()

        result = await analyzer.analyze_if_ready(frame)
        analyzer.update_world_context(context, result)

        # Should have some state (exact triggers depend on mock response)
        # At minimum, the method should not raise

    @pytest.mark.asyncio
    async def test_update_world_context_clears_old_triggers(
        self, analyzer: VisionAnalyzer
    ) -> None:
        """Test that updating context clears old LLM triggers."""
        context = WorldContext()
        context.add_llm_trigger("llm_person_engaged")

        # Update with None should clear triggers
        analyzer.update_world_context(context, None)

        assert "llm_person_engaged" not in context.llm_triggers

    def test_stats(self, analyzer: VisionAnalyzer) -> None:
        """Test statistics tracking."""
        stats = analyzer.get_stats()

        assert "analyses_completed" in stats
        assert "analyses_skipped" in stats
        assert "parse_errors" in stats
        assert "interval_seconds" in stats

    def test_repr(self, analyzer: VisionAnalyzer) -> None:
        """Test string representation."""
        repr_str = repr(analyzer)
        assert "VisionAnalyzer" in repr_str
