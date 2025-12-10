"""
Murph - Memory Consolidation Configuration
Configuration options for memory consolidation services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ConsolidationConfig:
    """
    Configuration for memory consolidation services.

    Controls timing, thresholds, and feature toggles for:
    - Event summarization
    - Relationship building
    - Experience reflection
    """

    # Feature toggles
    enabled: bool = True
    event_summarization_enabled: bool = True
    relationship_building_enabled: bool = True
    experience_reflection_enabled: bool = True

    # Timing intervals (seconds)
    event_summarization_interval: float = 3600.0  # 1 hour
    relationship_update_interval: float = 86400.0  # 24 hours
    consolidation_tick_interval: float = 60.0  # Check every minute

    # Event summarization settings
    min_events_for_summary: int = 3
    event_cluster_window_seconds: float = 3600.0  # Group events within 1 hour

    # Reflection settings
    reflection_probability: float = 0.3  # Only reflect 30% of the time
    notable_outcome_threshold: float = 0.7  # Score threshold for "notable"

    # Insight retention
    max_insights_per_session: int = 50
    max_relationship_narratives: int = 20
    insight_retention_days: int = 30
    insight_decay_rate: float = 0.01  # Per hour

    # Relevance thresholds
    min_relevance_score: float = 0.1  # Below this, insight is pruned

    @classmethod
    def from_env(cls) -> ConsolidationConfig:
        """
        Create configuration from environment variables.

        Environment variables:
        - MURPH_CONSOLIDATION_ENABLED: "true"/"false"
        - MURPH_EVENT_SUMMARIZATION_INTERVAL: seconds (float)
        - MURPH_RELATIONSHIP_UPDATE_INTERVAL: seconds (float)
        - MURPH_REFLECTION_PROBABILITY: 0.0-1.0
        """
        return cls(
            enabled=os.getenv("MURPH_CONSOLIDATION_ENABLED", "true").lower() == "true",
            event_summarization_enabled=os.getenv(
                "MURPH_EVENT_SUMMARIZATION_ENABLED", "true"
            ).lower() == "true",
            relationship_building_enabled=os.getenv(
                "MURPH_RELATIONSHIP_BUILDING_ENABLED", "true"
            ).lower() == "true",
            experience_reflection_enabled=os.getenv(
                "MURPH_EXPERIENCE_REFLECTION_ENABLED", "true"
            ).lower() == "true",
            event_summarization_interval=float(
                os.getenv("MURPH_EVENT_SUMMARIZATION_INTERVAL", "3600")
            ),
            relationship_update_interval=float(
                os.getenv("MURPH_RELATIONSHIP_UPDATE_INTERVAL", "86400")
            ),
            reflection_probability=float(
                os.getenv("MURPH_REFLECTION_PROBABILITY", "0.3")
            ),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        if self.event_summarization_interval < 60:
            issues.append("event_summarization_interval should be >= 60 seconds")

        if self.relationship_update_interval < 3600:
            issues.append("relationship_update_interval should be >= 3600 seconds")

        if not 0.0 <= self.reflection_probability <= 1.0:
            issues.append("reflection_probability must be between 0.0 and 1.0")

        if self.min_events_for_summary < 1:
            issues.append("min_events_for_summary must be >= 1")

        if self.insight_retention_days < 1:
            issues.append("insight_retention_days must be >= 1")

        return issues
