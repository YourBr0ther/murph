# tests/server/test_intent_parser.py
import pytest
from murph_server.intent.parser import parse_intent, Intent, IntentType


def test_parse_forward_movement():
    intent = parse_intent("go forward 6 inches")
    assert intent.type == IntentType.MOVEMENT
    assert intent.command == "forward"
    assert intent.distance == 6.0


def test_parse_backward_movement():
    intent = parse_intent("move backward")
    assert intent.type == IntentType.MOVEMENT
    assert intent.command == "backward"
    assert intent.distance == 3.0  # default


def test_parse_rotation():
    intent = parse_intent("turn left")
    assert intent.type == IntentType.MOVEMENT
    assert intent.command == "rotate_left"


def test_parse_question():
    intent = parse_intent("what is the weather today")
    assert intent.type == IntentType.QUESTION
    assert intent.text == "what is the weather today"


def test_parse_greeting():
    intent = parse_intent("hello murph")
    assert intent.type == IntentType.GREETING


def test_parse_stop():
    intent = parse_intent("stop")
    assert intent.type == IntentType.STOP
