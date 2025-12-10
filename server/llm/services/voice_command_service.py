"""
Murph - Voice Command Service
Voice command parsing and conversation response generation.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import LLMConfig
    from .llm_service import LLMService
    from ...cognition.behavior.context import WorldContext
    from ...cognition.needs.needs_system import NeedsSystem

logger = logging.getLogger(__name__)


class CommandType(IntEnum):
    """Types of voice commands."""

    BEHAVIOR_TRIGGER = 1  # Trigger a behavior (approach, play, rest)
    DIRECT_ACTION = 2  # Direct action (speak, stop)
    FEEDBACK = 3  # Positive/negative feedback
    QUERY = 4  # Question about state ("are you tired?")


@dataclass
class VoiceCommand:
    """Parsed voice command."""

    command_type: CommandType
    action: str  # "approach", "rest", "speak", "stop", "positive_feedback", etc.
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0-1, how confident we are in the parse


@dataclass
class VoiceCommandResult:
    """Result of processing a voice command."""

    command: VoiceCommand | None
    response_text: str  # What Murph should say
    need_adjustments: dict[str, float] = field(default_factory=dict)
    execute_immediately: bool = False  # True for direct actions


# Command patterns (regex -> action mapping)
COMMAND_PATTERNS: dict[str, list[str]] = {
    # Movement commands
    "approach": [
        r"come\s+here",
        r"come\s+over",
        r"come\s+to\s+me",
        r"come\s+closer",
        r"over\s+here",
    ],
    # Behavior commands
    "rest": [r"\bsleep\b", r"\brest\b", r"go\s+to\s+sleep", r"take\s+a\s+nap"],
    "play": [r"\bplay\b", r"let's\s+play", r"wanna\s+play", r"want\s+to\s+play"],
    "explore": [r"\bexplore\b", r"go\s+explore", r"look\s+around"],
    # Control commands
    "stop": [r"\bstop\b", r"\bhalt\b", r"\bfreeze\b", r"don't\s+move"],
    # Speech commands
    "speak": [r"\bspeak\b", r"say\s+something", r"\btalk\b", r"tell\s+me\s+something"],
    # Feedback
    "positive_feedback": [
        r"good\s+(boy|girl|robot|murph|murphy)",
        r"well\s+done",
        r"good\s+job",
        r"that's\s+a\s+good",
        r"you're\s+a\s+good",
        r"i\s+love\s+you",
        r"love\s+you",
    ],
    "negative_feedback": [
        r"bad\s+(boy|girl|robot|murph|murphy)",
        r"\bno\s+murph",
        r"\bno\s+murphy",
        r"stop\s+that",
        r"don't\s+do\s+that",
    ],
}

# Response templates organized by action and mood
RESPONSE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "approach": {
        "happy": ["wheee!", "coming!", "boop boop!"],
        "tired": ["...okay", "zzz... coming"],
        "curious": ["hmm? okay!", "ooh, coming!"],
        "playful": ["boop boop boop!", "wheee!"],
        "neutral": ["beep!", "on my way", "okay!"],
    },
    "rest": {
        "tired": ["zzz...", "finally...", "sleepy time..."],
        "playful": ["but... play?", "aww..."],
        "happy": ["okay...", "rest time!"],
        "neutral": ["okay...", "sleepy time"],
    },
    "play": {
        "playful": ["yes yes yes!", "boop boop boop!", "wheee!"],
        "tired": ["...too tired", "zzz... maybe later"],
        "happy": ["ooh!", "play time!", "yay!"],
        "neutral": ["okay!", "let's play!"],
    },
    "explore": {
        "curious": ["ooh! adventure!", "hmm hmm hmm!"],
        "happy": ["explore time!", "let's go!"],
        "tired": ["...okay", "exploring..."],
        "neutral": ["beep! exploring!"],
    },
    "stop": {
        "any": ["beep!", "stopped!", "okay okay!"],
    },
    "speak": {
        "happy": ["wheee!", "boop boop!", "hello hello!"],
        "tired": ["zzz...", "...hi"],
        "playful": ["boop boop boop!", "beep beep!"],
        "curious": ["hmm?", "ooh!"],
        "neutral": ["beep!", "hello!"],
    },
    "positive_feedback": {
        "any": ["wheee!", "beep beep!", "I like you!", "happy beeps!"],
    },
    "negative_feedback": {
        "any": ["aww...", "sorry...", "buh?", "...okay"],
    },
    "unknown": {
        "any": ["hmm?", "buh?", "beep?"],
    },
    "not_addressed": {
        "any": [],  # No response when not addressed
    },
}

# Default wake words (longer ones first to avoid partial matches)
DEFAULT_WAKE_WORDS = ["hey murphy", "hey murph", "murphy", "murph"]


class VoiceCommandService:
    """
    Voice command parsing and conversation response generation.

    Features:
    - Wake word detection ("Murph")
    - Fast keyword-based command parsing
    - Optional LLM fallback for natural language
    - Mood-based response generation
    - Need adjustment on social interaction
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize voice command service.

        Args:
            llm_service: LLM service for fallback parsing (optional)
            config: LLM configuration (uses env if None)
        """
        from ..config import LLMConfig

        self._llm_service = llm_service
        self._config = config or LLMConfig.from_env()

        # Configuration
        self._wake_words = DEFAULT_WAKE_WORDS.copy()
        self._llm_fallback_enabled = True

        # Compile regex patterns for efficiency
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        for action, patterns in COMMAND_PATTERNS.items():
            self._compiled_patterns[action] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Stats
        self._commands_processed = 0
        self._keyword_matches = 0
        self._llm_fallbacks = 0
        self._commands_ignored = 0

        logger.info("VoiceCommandService created")

    def extract_wake_word(self, text: str) -> tuple[bool, str]:
        """
        Check for wake word and extract command text.

        Args:
            text: Full transcribed text

        Returns:
            Tuple of (wake_word_found, command_text_after_wake_word)
        """
        text_lower = text.lower().strip()

        for wake_word in self._wake_words:
            # Check for wake word at start
            if text_lower.startswith(wake_word):
                # Extract text after wake word
                command_text = text_lower[len(wake_word) :].strip()
                # Remove common punctuation after wake word
                command_text = command_text.lstrip(",!.? ")
                return True, command_text

        return False, text_lower

    def parse_command_fast(self, text: str) -> VoiceCommand | None:
        """
        Fast keyword-based command parsing.

        Args:
            text: Command text (after wake word extraction)

        Returns:
            Parsed command or None if no match
        """
        text_lower = text.lower()

        for action, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    self._keyword_matches += 1

                    # Determine command type
                    if action in ("stop", "speak"):
                        cmd_type = CommandType.DIRECT_ACTION
                    elif action in ("positive_feedback", "negative_feedback"):
                        cmd_type = CommandType.FEEDBACK
                    else:
                        cmd_type = CommandType.BEHAVIOR_TRIGGER

                    return VoiceCommand(
                        command_type=cmd_type,
                        action=action,
                        confidence=1.0,
                    )

        return None

    async def parse_command_llm(
        self,
        text: str,
        world_context: WorldContext | None = None,
    ) -> VoiceCommand | None:
        """
        LLM-based intent extraction for natural language.

        Args:
            text: Command text
            world_context: Current world state for context

        Returns:
            Parsed command or None if no match
        """
        if not self._llm_service or not self._llm_fallback_enabled:
            return None

        self._llm_fallbacks += 1

        try:
            # Build prompt for intent extraction
            available_actions = list(COMMAND_PATTERNS.keys())
            prompt = f"""You are parsing voice commands for a pet robot named Murph.

The user said: "{text}"

Available actions: {', '.join(available_actions)}

What action does the user want? Respond with ONLY the action name (one of the available actions), or "unknown" if unclear.
Do not include any explanation, just the single word action name."""

            response = await self._llm_service.generate(prompt)
            if response:
                action = response.strip().lower().replace('"', "").replace("'", "")

                # Validate action
                if action in available_actions:
                    # Determine command type
                    if action in ("stop", "speak"):
                        cmd_type = CommandType.DIRECT_ACTION
                    elif action in ("positive_feedback", "negative_feedback"):
                        cmd_type = CommandType.FEEDBACK
                    else:
                        cmd_type = CommandType.BEHAVIOR_TRIGGER

                    return VoiceCommand(
                        command_type=cmd_type,
                        action=action,
                        confidence=0.8,  # Lower confidence for LLM
                    )

        except Exception as e:
            logger.error(f"LLM command parsing error: {e}")

        return None

    def _get_mood(self, needs_system: NeedsSystem | None) -> str:
        """
        Determine current mood from needs system.

        Args:
            needs_system: Needs system to query

        Returns:
            Mood string (happy, tired, playful, curious, neutral)
        """
        if not needs_system:
            return "neutral"

        # Get the most urgent need
        most_urgent = needs_system.get_most_urgent_need()
        if not most_urgent:
            return "happy"

        # Map needs to moods
        need_to_mood = {
            "energy": "tired",
            "play": "playful",
            "curiosity": "curious",
            "social": "happy",
            "affection": "happy",
            "comfort": "neutral",
            "safety": "neutral",
        }

        # If the need is critical, use that mood
        if most_urgent.is_critical():
            return need_to_mood.get(most_urgent.name, "neutral")

        # Otherwise check overall happiness
        happiness = needs_system.get_happiness()
        if happiness > 70:
            return "happy"
        elif happiness > 50:
            return "playful"
        elif happiness > 30:
            return "neutral"
        else:
            return "tired"

    def generate_response(
        self,
        command: VoiceCommand | None,
        needs_system: NeedsSystem | None = None,
    ) -> str:
        """
        Generate mood-appropriate response text.

        Args:
            command: Parsed command (None if unknown)
            needs_system: Needs system for mood detection

        Returns:
            Response text for TTS
        """
        action = command.action if command else "unknown"
        mood = self._get_mood(needs_system)

        # Get templates for this action
        templates = RESPONSE_TEMPLATES.get(action, RESPONSE_TEMPLATES["unknown"])

        # Try mood-specific first, fall back to "any", then "neutral"
        response_options = (
            templates.get(mood)
            or templates.get("any")
            or templates.get("neutral")
            or ["beep!"]
        )

        if not response_options:
            return ""

        return random.choice(response_options)

    def _calculate_need_adjustments(
        self, command: VoiceCommand | None
    ) -> dict[str, float]:
        """
        Calculate need adjustments from voice interaction.

        Args:
            command: Parsed command

        Returns:
            Dict of need name -> adjustment value
        """
        adjustments: dict[str, float] = {}

        # Any voice interaction satisfies social need somewhat
        adjustments["social"] = 5.0

        if not command:
            return adjustments

        if command.action == "positive_feedback":
            adjustments["affection"] = 15.0
            adjustments["social"] = 10.0  # Extra social
        elif command.action == "negative_feedback":
            adjustments["affection"] = -10.0
            adjustments["social"] = 5.0  # Still social interaction
        elif command.action == "play":
            adjustments["social"] = 10.0
        elif command.action == "approach":
            adjustments["social"] = 10.0

        return adjustments

    async def process_speech(
        self,
        text: str,
        world_context: WorldContext | None = None,
        needs_system: NeedsSystem | None = None,
    ) -> VoiceCommandResult | None:
        """
        Main entry point for speech processing.

        Args:
            text: Transcribed speech text
            world_context: Current world state
            needs_system: Needs system for mood detection

        Returns:
            VoiceCommandResult or None if not a command for Murph
        """
        self._commands_processed += 1

        # Check for wake word
        wake_word_found, command_text = self.extract_wake_word(text)

        if not wake_word_found:
            self._commands_ignored += 1
            logger.debug(f"No wake word in: '{text[:30]}...'")
            return None

        logger.info(f"Wake word detected, command: '{command_text}'")

        # Try fast keyword parsing first
        command = self.parse_command_fast(command_text)

        # Fall back to LLM if enabled and no keyword match
        if command is None and self._llm_fallback_enabled and self._llm_service:
            command = await self.parse_command_llm(command_text, world_context)

        # Generate response
        response_text = self.generate_response(command, needs_system)

        # Calculate need adjustments
        need_adjustments = self._calculate_need_adjustments(command)

        # Determine if immediate execution
        execute_immediately = (
            command is not None and command.command_type == CommandType.DIRECT_ACTION
        )

        return VoiceCommandResult(
            command=command,
            response_text=response_text,
            need_adjustments=need_adjustments,
            execute_immediately=execute_immediately,
        )

    def set_wake_words(self, wake_words: list[str]) -> None:
        """Set custom wake words."""
        self._wake_words = [w.lower() for w in wake_words]

    def set_llm_fallback(self, enabled: bool) -> None:
        """Enable/disable LLM fallback."""
        self._llm_fallback_enabled = enabled

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "commands_processed": self._commands_processed,
            "keyword_matches": self._keyword_matches,
            "llm_fallbacks": self._llm_fallbacks,
            "commands_ignored": self._commands_ignored,
            "wake_words": self._wake_words,
            "llm_fallback_enabled": self._llm_fallback_enabled,
        }

    def __repr__(self) -> str:
        return (
            f"VoiceCommandService("
            f"wake_words={self._wake_words}, "
            f"llm_fallback={self._llm_fallback_enabled})"
        )
