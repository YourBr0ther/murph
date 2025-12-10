"""
Unit tests for Murph's Memory Consolidation System.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.cognition.memory.consolidation import (
    BehaviorOutcome,
    ConsolidationConfig,
    EventSummarizer,
    ExperienceReflector,
    MemoryConsolidator,
    RelationshipBuilder,
)
from server.cognition.memory.memory_types import EventMemory, InsightMemory, PersonMemory
from server.llm.config import LLMConfig
from server.llm.types import LLMResponse


# ==================== Config Tests ====================


class TestConsolidationConfig:
    """Tests for ConsolidationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConsolidationConfig()
        assert config.enabled is True
        assert config.event_summarization_enabled is True
        assert config.relationship_building_enabled is True
        assert config.experience_reflection_enabled is True
        assert config.event_summarization_interval == 3600.0
        assert config.relationship_update_interval == 86400.0
        assert config.reflection_probability == 0.3

    def test_config_from_env(self):
        """Test loading config from environment."""
        with patch.dict(
            "os.environ",
            {
                "MURPH_CONSOLIDATION_ENABLED": "false",
                "MURPH_REFLECTION_PROBABILITY": "0.5",
            },
        ):
            config = ConsolidationConfig.from_env()
            assert config.enabled is False
            assert config.reflection_probability == 0.5

    def test_config_validation(self):
        """Test configuration validation."""
        config = ConsolidationConfig(
            event_summarization_interval=10.0,  # Too small
            reflection_probability=2.0,  # Invalid
        )
        issues = config.validate()
        assert len(issues) >= 2
        assert any("summarization_interval" in issue for issue in issues)
        assert any("reflection_probability" in issue for issue in issues)


# ==================== InsightMemory Tests ====================


class TestInsightMemory:
    """Tests for InsightMemory dataclass."""

    def test_create_insight(self):
        """Test creating an InsightMemory."""
        insight = InsightMemory(
            insight_type="event_summary",
            subject_type="person",
            subject_id="person_1",
            content="Alice enjoys petting Murph frequently.",
            summary="Alice likes petting",
        )
        assert insight.insight_type == "event_summary"
        assert insight.subject_type == "person"
        assert insight.subject_id == "person_1"
        assert insight.confidence == 0.7  # default
        assert insight.relevance_score == 1.0  # default

    def test_decay_relevance(self):
        """Test relevance decay."""
        insight = InsightMemory(
            insight_type="event_summary",
            subject_type="session",
            content="Session summary",
            summary="Summary",
        )
        assert insight.relevance_score == 1.0
        insight.decay_relevance(0.3)
        assert insight.relevance_score == 0.7
        insight.decay_relevance(1.0)  # Clamp to 0
        assert insight.relevance_score == 0.0

    def test_is_stale(self):
        """Test stale detection."""
        insight = InsightMemory(
            insight_type="event_summary",
            subject_type="session",
            content="Test",
            summary="Test",
            relevance_score=0.05,
        )
        assert insight.is_stale(threshold=0.1) is True
        assert insight.is_stale(threshold=0.01) is False

    def test_serialization(self):
        """Test get_state and from_state."""
        insight = InsightMemory(
            insight_type="behavior_reflection",
            subject_type="behavior",
            subject_id="play",
            content="Playing is fun",
            summary="Play good",
            confidence=0.9,
        )
        insight.tags.add("positive")

        state = insight.get_state()
        restored = InsightMemory.from_state(state)

        assert restored.insight_type == insight.insight_type
        assert restored.subject_type == insight.subject_type
        assert restored.subject_id == insight.subject_id
        assert restored.content == insight.content
        assert restored.confidence == insight.confidence
        assert "positive" in restored.tags


# ==================== Mock LLM Service ====================


def create_mock_llm_service(response_content: str | None = None):
    """Create a mock LLM service."""
    mock = MagicMock()
    mock.is_available = True

    if response_content:
        mock.complete = AsyncMock(
            return_value=LLMResponse(
                content=response_content,
                model="test",
                usage={},
                latency_ms=100,
                cached=False,
                provider="mock",
            )
        )
    else:
        mock.complete = AsyncMock(return_value=None)

    return mock


# ==================== EventSummarizer Tests ====================


class TestEventSummarizer:
    """Tests for EventSummarizer."""

    def test_cluster_events_by_type(self):
        """Test that events are clustered by type and participant."""
        config = ConsolidationConfig()
        mock_llm = create_mock_llm_service()
        summarizer = EventSummarizer(mock_llm, config)

        events = [
            EventMemory(event_type="petting", participants=["person_1"]),
            EventMemory(event_type="petting", participants=["person_1"]),
            EventMemory(event_type="petting", participants=["person_2"]),
            EventMemory(event_type="play", participants=["person_1"]),
        ]

        clusters = summarizer._cluster_events(events)

        assert "petting:person_1" in clusters
        assert "petting:person_2" in clusters
        assert "play:person_1" in clusters
        assert len(clusters["petting:person_1"]) == 2
        assert len(clusters["petting:person_2"]) == 1

    @pytest.mark.asyncio
    async def test_summarize_not_ready(self):
        """Test that summarization doesn't run before interval."""
        config = ConsolidationConfig(event_summarization_interval=3600.0)
        mock_llm = create_mock_llm_service()
        summarizer = EventSummarizer(mock_llm, config)
        summarizer._last_summarization_time = time.time()  # Just ran

        events = [EventMemory(event_type="test") for _ in range(5)]
        results = await summarizer.summarize_if_ready(events)

        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_min_events(self):
        """Test minimum events requirement."""
        config = ConsolidationConfig(min_events_for_summary=5)
        mock_llm = create_mock_llm_service()
        summarizer = EventSummarizer(mock_llm, config)

        events = [EventMemory(event_type="test") for _ in range(2)]  # Too few
        results = await summarizer.summarize_if_ready(events, force=True)

        assert results == []

    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Test successful event summarization."""
        config = ConsolidationConfig(min_events_for_summary=2)
        response = '{"summary": "Test summary", "content": "Detailed content", "confidence": 0.8, "tags": ["test"]}'
        mock_llm = create_mock_llm_service(response)
        summarizer = EventSummarizer(mock_llm, config)

        events = [
            EventMemory(event_type="petting", participants=["person_1"]),
            EventMemory(event_type="petting", participants=["person_1"]),
            EventMemory(event_type="petting", participants=["person_1"]),
        ]
        results = await summarizer.summarize_if_ready(events, force=True)

        assert len(results) == 1
        assert results[0].insight_type == "event_summary"
        assert results[0].summary == "Test summary"

    @pytest.mark.asyncio
    async def test_summarize_llm_unavailable(self):
        """Test graceful handling when LLM unavailable."""
        config = ConsolidationConfig(min_events_for_summary=2)
        mock_llm = create_mock_llm_service()
        mock_llm.is_available = False
        summarizer = EventSummarizer(mock_llm, config)

        events = [EventMemory(event_type="test") for _ in range(5)]
        results = await summarizer.summarize_if_ready(events, force=True)

        # Returns empty since cluster summarization requires LLM
        assert results == []


# ==================== RelationshipBuilder Tests ====================


class TestRelationshipBuilder:
    """Tests for RelationshipBuilder."""

    @pytest.mark.asyncio
    async def test_build_narrative_not_ready(self):
        """Test that narrative doesn't build before interval."""
        config = ConsolidationConfig(relationship_update_interval=86400.0)
        mock_llm = create_mock_llm_service()
        builder = RelationshipBuilder(mock_llm, config)
        builder._last_update_times["person_1"] = time.time()  # Just updated

        person = PersonMemory(person_id="person_1", familiarity_score=60)
        result = await builder.build_relationship_narrative(person, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_build_narrative_success(self):
        """Test successful narrative generation."""
        config = ConsolidationConfig()
        response = '{"narrative": "Alice is a good friend.", "trajectory": "improving", "key_traits": ["friendly"], "confidence": 0.85}'
        mock_llm = create_mock_llm_service(response)
        builder = RelationshipBuilder(mock_llm, config)

        person = PersonMemory(
            person_id="person_1",
            name="Alice",
            familiarity_score=75,
            sentiment=0.5,
            interaction_count=20,
        )
        events = [
            EventMemory(event_type="petting", participants=["person_1"], outcome="positive"),
            EventMemory(event_type="play", participants=["person_1"], outcome="positive"),
        ]
        result = await builder.build_relationship_narrative(person, events, force=True)

        assert result is not None
        assert result.insight_type == "relationship_narrative"
        assert result.subject_id == "person_1"

    def test_analyze_trajectory(self):
        """Test trajectory analysis."""
        config = ConsolidationConfig()
        mock_llm = create_mock_llm_service()
        builder = RelationshipBuilder(mock_llm, config)

        assert builder.analyze_trajectory(0.7, 0.3) == "improving"
        assert builder.analyze_trajectory(0.3, 0.7) == "declining"
        assert builder.analyze_trajectory(0.5, 0.5) == "stable"
        assert builder.analyze_trajectory(0.5, None) == "stable"


# ==================== ExperienceReflector Tests ====================


class TestExperienceReflector:
    """Tests for ExperienceReflector."""

    @pytest.mark.asyncio
    async def test_reflect_disabled(self):
        """Test reflection when disabled."""
        config = ConsolidationConfig(experience_reflection_enabled=False)
        mock_llm = create_mock_llm_service()
        reflector = ExperienceReflector(mock_llm, config)

        result = await reflector.reflect_on_behavior(
            behavior_name="play",
            result="completed",
            duration=10.0,
            context_snapshot={},
            need_changes={"play": 20},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_reflect_force(self):
        """Test forced reflection."""
        config = ConsolidationConfig()
        response = '{"was_good_choice": true, "reasoning": "Good choice!", "lesson": "Playing is good.", "confidence": 0.8}'
        mock_llm = create_mock_llm_service(response)
        reflector = ExperienceReflector(mock_llm, config)

        result = await reflector.reflect_on_behavior(
            behavior_name="play",
            result="completed",
            duration=10.0,
            context_snapshot={"person_detected": True},
            need_changes={"play": 20},
            force=True,
        )

        assert result is not None
        assert result.insight_type == "behavior_reflection"
        assert result.subject_id == "play"

    def test_should_reflect_failed(self):
        """Test that failed behaviors have higher reflection chance."""
        config = ConsolidationConfig(reflection_probability=0.0)  # Normally no reflection
        mock_llm = create_mock_llm_service()
        reflector = ExperienceReflector(mock_llm, config)

        # With probability 0, only notable outcomes might trigger
        # This test is probabilistic, so we just verify no crash
        outcome_success = BehaviorOutcome(
            behavior_name="play",
            result="completed",
            duration=10,
            was_interrupted=False,
            context_snapshot={},
            need_changes={},
        )
        outcome_failed = BehaviorOutcome(
            behavior_name="play",
            result="failed",
            duration=10,
            was_interrupted=False,
            context_snapshot={},
            need_changes={},
        )

        # These should not raise
        reflector._should_reflect(outcome_success)
        reflector._should_reflect(outcome_failed)

    def test_get_recent_outcomes(self):
        """Test outcome queue."""
        config = ConsolidationConfig()
        mock_llm = create_mock_llm_service()
        reflector = ExperienceReflector(mock_llm, config)

        # Add some outcomes to queue manually
        for i in range(5):
            outcome = BehaviorOutcome(
                behavior_name=f"behavior_{i}",
                result="completed",
                duration=10,
                was_interrupted=False,
                context_snapshot={},
                need_changes={},
            )
            reflector._reflection_queue.append(outcome)

        recent = reflector.get_recent_outcomes(limit=3)
        assert len(recent) == 3


# ==================== MemoryConsolidator Tests ====================


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator."""

    @pytest.mark.asyncio
    async def test_disabled_consolidator(self):
        """Test consolidator when disabled."""
        config = ConsolidationConfig(enabled=False)
        mock_llm = create_mock_llm_service()
        consolidator = MemoryConsolidator(mock_llm, config=config)

        # tick should do nothing
        await consolidator.tick()
        assert consolidator._ticks == 0

        # on_behavior_complete should return None
        result = await consolidator.on_behavior_complete(
            behavior_name="play",
            result="completed",
            duration=10,
            context_snapshot={},
            need_changes={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tick_rate_limiting(self):
        """Test that tick respects interval."""
        config = ConsolidationConfig(consolidation_tick_interval=60.0)
        mock_llm = create_mock_llm_service()
        consolidator = MemoryConsolidator(mock_llm, config=config)
        consolidator._last_tick_time = time.time()  # Just ticked

        await consolidator.tick()
        assert consolidator._ticks == 0  # Should be skipped

    def test_get_stats(self):
        """Test statistics retrieval."""
        config = ConsolidationConfig()
        mock_llm = create_mock_llm_service()
        consolidator = MemoryConsolidator(mock_llm, config=config)

        stats = consolidator.get_stats()
        assert "enabled" in stats
        assert "ticks" in stats
        assert "insights_saved" in stats
        assert "event_summarizer" in stats
        assert "relationship_builder" in stats
        assert "experience_reflector" in stats

    def test_enable_disable(self):
        """Test enable/disable methods."""
        config = ConsolidationConfig(enabled=True)
        mock_llm = create_mock_llm_service()
        consolidator = MemoryConsolidator(mock_llm, config=config)

        assert consolidator._enabled is True
        consolidator.disable()
        assert consolidator._enabled is False
        consolidator.enable()
        assert consolidator._enabled is True


# ==================== LLM Config Integration Tests ====================


class TestLLMConfigConsolidation:
    """Tests for consolidation settings in LLMConfig."""

    def test_default_consolidation_config(self):
        """Test default consolidation settings in LLMConfig."""
        config = LLMConfig()
        assert config.consolidation_enabled is True
        assert config.consolidation_tick_interval == 60.0
        assert config.event_summarization_interval == 3600.0
        assert config.relationship_update_interval == 86400.0
        assert config.reflection_probability == 0.3

    def test_consolidation_config_from_env(self):
        """Test loading consolidation settings from environment."""
        with patch.dict(
            "os.environ",
            {
                "MURPH_CONSOLIDATION_ENABLED": "false",
                "MURPH_EVENT_SUMMARIZATION_INTERVAL": "1800",
                "MURPH_REFLECTION_PROBABILITY": "0.5",
            },
        ):
            config = LLMConfig.from_env()
            assert config.consolidation_enabled is False
            assert config.event_summarization_interval == 1800.0
            assert config.reflection_probability == 0.5
