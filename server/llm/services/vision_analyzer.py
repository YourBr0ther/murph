"""
Murph - Vision Analyzer
LLM-based scene analysis from video frames.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..prompts.vision_prompts import (
    SCENE_ANALYSIS_SYSTEM_PROMPT,
    SCENE_ANALYSIS_USER_PROMPT,
    VALID_LLM_TRIGGERS,
)
from ..types import LLMMessage, SceneAnalysis

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..services.llm_service import LLMService
    from server.cognition.behavior.context import WorldContext

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    Analyzes video frames using LLM vision capabilities.

    Features:
    - Throttled analysis (configurable interval)
    - Skip-if-busy semantics (non-blocking)
    - Structured JSON response parsing
    - WorldContext trigger updates

    Usage:
        analyzer = VisionAnalyzer(llm_service)
        analysis = await analyzer.analyze_if_ready(frame)
        if analysis:
            analyzer.update_world_context(context, analysis)
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize vision analyzer.

        Args:
            llm_service: LLM service for API calls
            config: LLM configuration (uses service config if None)
        """
        self._llm = llm_service
        self._config = config or llm_service._config
        self._interval = self._config.vision_interval_seconds

        # State
        self._last_analysis_time: float = 0.0
        self._last_result: SceneAnalysis | None = None
        self._analyzing = False

        # Stats
        self._analyses_completed = 0
        self._analyses_skipped = 0
        self._parse_errors = 0

        logger.info(f"VisionAnalyzer created (interval={self._interval}s)")

    async def analyze_if_ready(
        self,
        frame: np.ndarray,
    ) -> SceneAnalysis | None:
        """
        Analyze frame if enough time has passed.

        Implements throttling to avoid overwhelming the LLM.
        Returns cached result if not ready for new analysis.

        Args:
            frame: RGB numpy array (H, W, 3)

        Returns:
            SceneAnalysis result, or cached result if not ready
        """
        now = time.time()

        # Check interval
        if now - self._last_analysis_time < self._interval:
            self._analyses_skipped += 1
            return self._last_result

        # Skip if already analyzing (non-blocking)
        if self._analyzing:
            self._analyses_skipped += 1
            return self._last_result

        self._analyzing = True
        try:
            result = await self._analyze_frame(frame)
            if result:
                self._last_result = result
                self._last_analysis_time = time.time()  # Time when analysis FINISHED
                self._analyses_completed += 1
                logger.debug(f"Scene analysis: {result.description[:50]}...")
            return result
        finally:
            self._analyzing = False

    async def analyze_now(
        self,
        frame: np.ndarray,
    ) -> SceneAnalysis | None:
        """
        Analyze frame immediately (ignores throttling).

        Use sparingly - prefer analyze_if_ready() for normal operation.

        Args:
            frame: RGB numpy array (H, W, 3)

        Returns:
            SceneAnalysis result, or None on error
        """
        return await self._analyze_frame(frame)

    async def _analyze_frame(
        self,
        frame: np.ndarray,
    ) -> SceneAnalysis | None:
        """
        Perform actual frame analysis.

        Args:
            frame: RGB numpy array

        Returns:
            Parsed SceneAnalysis or None on error
        """
        messages = [
            LLMMessage(role="system", content=SCENE_ANALYSIS_SYSTEM_PROMPT),
            LLMMessage(role="user", content=SCENE_ANALYSIS_USER_PROMPT),
        ]

        response = await self._llm.complete_with_vision(messages, frame)
        if response is None:
            return None

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> SceneAnalysis:
        """
        Parse LLM response into SceneAnalysis.

        Handles both clean JSON and markdown-wrapped JSON.

        Args:
            content: Raw LLM response text

        Returns:
            Parsed SceneAnalysis
        """
        # Try to extract JSON from response
        json_str = content.strip()

        # Handle markdown code blocks
        if json_str.startswith("```"):
            # Find JSON content between code blocks
            lines = json_str.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        try:
            data = json.loads(json_str)

            # Filter triggers to only valid ones
            triggers = [
                t for t in data.get("triggers", [])
                if t in VALID_LLM_TRIGGERS
            ]

            return SceneAnalysis(
                description=data.get("description", ""),
                detected_objects=data.get("objects", []),
                detected_activities=data.get("activities", []),
                mood_indicators=data.get("mood", []),
                suggested_triggers=triggers,
                confidence=float(data.get("confidence", 0.5)),
                timestamp=time.time(),
            )

        except json.JSONDecodeError as e:
            self._parse_errors += 1
            preview = json_str[:100] if json_str else "(empty)"
            logger.warning(f"Failed to parse scene analysis JSON: {e}. Content: {preview}")
            # Return basic analysis from raw text
            return SceneAnalysis(
                description=content[:200],
                detected_objects=[],
                detected_activities=[],
                mood_indicators=[],
                suggested_triggers=[],
                confidence=0.3,
                timestamp=time.time(),
            )

    def update_world_context(
        self,
        context: WorldContext,
        analysis: SceneAnalysis | None,
    ) -> None:
        """
        Update WorldContext from scene analysis.

        Sets LLM-derived triggers and updates objects_in_view.

        Args:
            context: WorldContext to update
            analysis: SceneAnalysis result (or None to clear)
        """
        # Clear previous LLM triggers
        self._clear_llm_triggers(context)

        if analysis is None:
            return

        # Set suggested triggers
        for trigger in analysis.suggested_triggers:
            if trigger in VALID_LLM_TRIGGERS:
                context.add_llm_trigger(trigger)

        # Update objects if we detected any
        if analysis.detected_objects:
            # Merge with existing objects (face detection may have added some)
            existing = set(context.objects_in_view or [])
            existing.update(analysis.detected_objects)
            context.objects_in_view = list(existing)

    def _clear_llm_triggers(self, context: WorldContext) -> None:
        """Clear all LLM-derived triggers from context."""
        for trigger in VALID_LLM_TRIGGERS:
            context.remove_llm_trigger(trigger)

    @property
    def last_analysis(self) -> SceneAnalysis | None:
        """Get most recent analysis result."""
        return self._last_result

    @property
    def time_since_analysis(self) -> float:
        """Seconds since last analysis."""
        if self._last_analysis_time == 0:
            return float("inf")
        return time.time() - self._last_analysis_time

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "analyses_completed": self._analyses_completed,
            "analyses_skipped": self._analyses_skipped,
            "parse_errors": self._parse_errors,
            "time_since_analysis": self.time_since_analysis,
            "has_result": self._last_result is not None,
            "interval_seconds": self._interval,
        }

    def __repr__(self) -> str:
        return (
            f"VisionAnalyzer(interval={self._interval}s, "
            f"completed={self._analyses_completed})"
        )
