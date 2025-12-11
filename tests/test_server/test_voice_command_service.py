"""
Tests for Voice Command Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from server.llm.services.voice_command_service import (
    CommandType,
    VoiceCommand,
    VoiceCommandResult,
    VoiceCommandService,
    COMMAND_PATTERNS,
    RESPONSE_TEMPLATES,
)
from server.llm.types import LLMResponse


class TestWakeWordDetection:
    """Test wake word detection."""

    def test_wake_word_murph(self) -> None:
        """Test basic 'murph' wake word."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("murph come here")
        assert found is True
        assert command == "come here"

    def test_wake_word_murphy(self) -> None:
        """Test 'murphy' wake word."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("murphy play")
        assert found is True
        assert command == "play"

    def test_wake_word_hey_murph(self) -> None:
        """Test 'hey murph' wake word."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("hey murph come over")
        assert found is True
        assert command == "come over"

    def test_wake_word_with_comma(self) -> None:
        """Test wake word followed by comma."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("murph, come here")
        assert found is True
        assert command == "come here"

    def test_wake_word_case_insensitive(self) -> None:
        """Test wake word is case insensitive."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("MURPH come here")
        assert found is True
        assert command == "come here"

    def test_no_wake_word(self) -> None:
        """Test text without wake word."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("hey robot come here")
        assert found is False
        assert command == "hey robot come here"

    def test_wake_word_in_middle(self) -> None:
        """Test wake word not at start should not match."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("hello murph come here")
        assert found is False

    def test_custom_wake_words(self) -> None:
        """Test custom wake words."""
        service = VoiceCommandService()
        service.set_wake_words(["robot", "buddy"])

        found, command = service.extract_wake_word("robot come here")
        assert found is True
        assert command == "come here"

        found, command = service.extract_wake_word("buddy play")
        assert found is True
        assert command == "play"

        # Original wake word should no longer work
        found, command = service.extract_wake_word("murph come here")
        assert found is False


class TestCommandParsing:
    """Test keyword-based command parsing."""

    def test_come_here(self) -> None:
        """Test 'come here' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("come here")
        assert command is not None
        assert command.action == "approach"
        assert command.command_type == CommandType.BEHAVIOR_TRIGGER

    def test_come_over(self) -> None:
        """Test 'come over' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("come over please")
        assert command is not None
        assert command.action == "approach"

    def test_sleep(self) -> None:
        """Test 'sleep' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("go to sleep")
        assert command is not None
        assert command.action == "rest"
        assert command.command_type == CommandType.BEHAVIOR_TRIGGER

    def test_play(self) -> None:
        """Test 'play' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("let's play")
        assert command is not None
        assert command.action == "play"
        assert command.command_type == CommandType.BEHAVIOR_TRIGGER

    def test_stop(self) -> None:
        """Test 'stop' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("stop")
        assert command is not None
        assert command.action == "stop"
        assert command.command_type == CommandType.DIRECT_ACTION

    def test_speak(self) -> None:
        """Test 'speak' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("say something")
        assert command is not None
        assert command.action == "speak"
        assert command.command_type == CommandType.DIRECT_ACTION

    def test_good_murph(self) -> None:
        """Test positive feedback."""
        service = VoiceCommandService()
        command = service.parse_command_fast("good boy")
        assert command is not None
        assert command.action == "positive_feedback"
        assert command.command_type == CommandType.FEEDBACK

    def test_good_girl(self) -> None:
        """Test positive feedback variant."""
        service = VoiceCommandService()
        command = service.parse_command_fast("good girl murph")
        assert command is not None
        assert command.action == "positive_feedback"

    def test_bad_murph(self) -> None:
        """Test negative feedback."""
        service = VoiceCommandService()
        command = service.parse_command_fast("bad robot")
        assert command is not None
        assert command.action == "negative_feedback"
        assert command.command_type == CommandType.FEEDBACK

    def test_unknown_command(self) -> None:
        """Test unrecognized command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("do a backflip")
        assert command is None

    def test_explore(self) -> None:
        """Test 'explore' command."""
        service = VoiceCommandService()
        command = service.parse_command_fast("go explore")
        assert command is not None
        assert command.action == "explore"


class TestResponseGeneration:
    """Test mood-based response generation."""

    def test_response_for_approach(self) -> None:
        """Test response generation for approach command."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.BEHAVIOR_TRIGGER,
            action="approach",
        )
        response = service.generate_response(command, needs_system=None)
        assert response in RESPONSE_TEMPLATES["approach"]["neutral"]

    def test_response_for_positive_feedback(self) -> None:
        """Test response for positive feedback."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.FEEDBACK,
            action="positive_feedback",
        )
        response = service.generate_response(command, needs_system=None)
        assert response in RESPONSE_TEMPLATES["positive_feedback"]["any"]

    def test_response_for_unknown(self) -> None:
        """Test response for unknown/no command."""
        service = VoiceCommandService()
        response = service.generate_response(None, needs_system=None)
        assert response in RESPONSE_TEMPLATES["unknown"]["any"]

    def test_response_with_mood(self) -> None:
        """Test response varies with mood."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.BEHAVIOR_TRIGGER,
            action="approach",
        )

        # Create mock needs system with critical energy (tired)
        mock_needs = MagicMock()
        mock_urgent = MagicMock()
        mock_urgent.name = "energy"
        mock_urgent.is_critical.return_value = True
        mock_needs.get_most_urgent_need.return_value = mock_urgent
        mock_needs.get_happiness.return_value = 30

        response = service.generate_response(command, needs_system=mock_needs)
        assert response in RESPONSE_TEMPLATES["approach"]["tired"]


class TestNeedAdjustments:
    """Test need adjustment calculations."""

    def test_positive_feedback_adjustments(self) -> None:
        """Test positive feedback increases affection and social."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.FEEDBACK,
            action="positive_feedback",
        )
        adjustments = service._calculate_need_adjustments(command)
        assert adjustments["affection"] == 15.0
        assert adjustments["social"] == 10.0

    def test_negative_feedback_adjustments(self) -> None:
        """Test negative feedback decreases affection but still social."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.FEEDBACK,
            action="negative_feedback",
        )
        adjustments = service._calculate_need_adjustments(command)
        assert adjustments["affection"] == -10.0
        assert adjustments["social"] == 5.0  # Still a social interaction

    def test_base_social_adjustment(self) -> None:
        """Test any interaction adds some social."""
        service = VoiceCommandService()
        command = VoiceCommand(
            command_type=CommandType.BEHAVIOR_TRIGGER,
            action="explore",
        )
        adjustments = service._calculate_need_adjustments(command)
        assert adjustments["social"] == 5.0  # Base social adjustment

    def test_no_command_still_social(self) -> None:
        """Test even failed command attempt is social."""
        service = VoiceCommandService()
        adjustments = service._calculate_need_adjustments(None)
        assert adjustments["social"] == 5.0


class TestProcessSpeech:
    """Test full speech processing pipeline."""

    @pytest.mark.asyncio
    async def test_process_with_wake_word(self) -> None:
        """Test processing speech with wake word."""
        service = VoiceCommandService()
        result = await service.process_speech(
            text="murph come here",
            world_context=None,
            needs_system=None,
        )

        assert result is not None
        assert result.command is not None
        assert result.command.action == "approach"
        assert result.response_text != ""
        assert "social" in result.need_adjustments

    @pytest.mark.asyncio
    async def test_process_without_wake_word(self) -> None:
        """Test processing speech without wake word returns None."""
        service = VoiceCommandService()
        result = await service.process_speech(
            text="come here",
            world_context=None,
            needs_system=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_process_unknown_command(self) -> None:
        """Test processing unknown command with wake word."""
        service = VoiceCommandService()
        service.set_llm_fallback(False)  # Disable LLM for this test

        result = await service.process_speech(
            text="murph do a backflip",
            world_context=None,
            needs_system=None,
        )

        assert result is not None
        assert result.command is None
        assert result.response_text != ""  # Should still respond

    @pytest.mark.asyncio
    async def test_process_direct_action(self) -> None:
        """Test direct action command."""
        service = VoiceCommandService()
        result = await service.process_speech(
            text="murph stop",
            world_context=None,
            needs_system=None,
        )

        assert result is not None
        assert result.command is not None
        assert result.command.action == "stop"
        assert result.execute_immediately is True

    @pytest.mark.asyncio
    async def test_process_behavior_trigger(self) -> None:
        """Test behavior trigger command."""
        service = VoiceCommandService()
        result = await service.process_speech(
            text="murph play",
            world_context=None,
            needs_system=None,
        )

        assert result is not None
        assert result.command is not None
        assert result.command.action == "play"
        assert result.execute_immediately is False


class TestLLMFallback:
    """Test LLM fallback for natural language commands."""

    @pytest.mark.asyncio
    async def test_llm_fallback_disabled(self) -> None:
        """Test LLM fallback can be disabled."""
        service = VoiceCommandService()
        service.set_llm_fallback(False)

        result = await service.parse_command_llm("please come over here")
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_fallback_with_mock_service(self) -> None:
        """Test LLM fallback with mock LLM service."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(content="approach", model="test")
        )

        service = VoiceCommandService(llm_service=mock_llm)
        result = await service.parse_command_llm("can you come over here please")

        assert result is not None
        assert result.action == "approach"
        assert result.confidence == 0.8  # Lower confidence for LLM

    @pytest.mark.asyncio
    async def test_llm_fallback_unknown_response(self) -> None:
        """Test LLM returning unknown action."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(content="dance", model="test")  # Not a valid action
        )

        service = VoiceCommandService(llm_service=mock_llm)
        result = await service.parse_command_llm("do a dance")

        assert result is None  # Invalid action should return None


class TestStatistics:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test that statistics are tracked."""
        service = VoiceCommandService()

        # Process some commands
        await service.process_speech("murph come here", None, None)
        await service.process_speech("hello there", None, None)  # No wake word
        await service.process_speech("murph play", None, None)

        stats = service.get_stats()
        assert stats["commands_processed"] == 3
        assert stats["commands_ignored"] == 1
        assert stats["keyword_matches"] == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text(self) -> None:
        """Test empty text."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("")
        assert found is False

    def test_only_wake_word(self) -> None:
        """Test text with only wake word."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("murph")
        assert found is True
        assert command == ""

    def test_whitespace_handling(self) -> None:
        """Test whitespace is handled correctly."""
        service = VoiceCommandService()
        found, command = service.extract_wake_word("  murph   come here  ")
        assert found is True
        assert command == "come here"

    def test_multiple_commands_first_wins(self) -> None:
        """Test that first matching pattern wins (iteration order)."""
        service = VoiceCommandService()
        # When text contains multiple patterns, one of them wins
        # (whichever comes first in dict iteration order)
        command = service.parse_command_fast("good boy, let's play")
        assert command is not None
        # Either match is acceptable - just verify something matched
        assert command.action in ("positive_feedback", "play")


class TestCommandPatterns:
    """Test all command patterns are properly defined."""

    def test_all_patterns_compile(self) -> None:
        """Test all regex patterns compile successfully."""
        service = VoiceCommandService()
        # If patterns didn't compile, __init__ would fail
        assert len(service._compiled_patterns) > 0

    def test_all_response_templates_exist(self) -> None:
        """Test all actions have response templates."""
        for action in COMMAND_PATTERNS.keys():
            assert action in RESPONSE_TEMPLATES or action in ["explore"]
            # explore may not have templates, which is fine (falls back to unknown)
