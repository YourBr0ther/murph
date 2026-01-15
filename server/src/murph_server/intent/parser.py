# server/src/murph_server/intent/parser.py
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IntentType(Enum):
    MOVEMENT = "movement"
    QUESTION = "question"
    GREETING = "greeting"
    STOP = "stop"


@dataclass
class Intent:
    type: IntentType
    command: Optional[str] = None
    distance: Optional[float] = None
    text: Optional[str] = None


MOVEMENT_PATTERNS = {
    r"\b(go\s+)?forward\b": "forward",
    r"\b(go\s+)?backward\b|back\s*up": "backward",
    r"\bturn\s+left\b|rotate\s+left\b": "rotate_left",
    r"\bturn\s+right\b|rotate\s+right\b": "rotate_right",
    r"\bstrafe\s+left\b|slide\s+left\b": "strafe_left",
    r"\bstrafe\s+right\b|slide\s+right\b": "strafe_right",
}

GREETING_PATTERNS = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhow\s+are\s+you\b"]
STOP_PATTERNS = [r"\bstop\b", r"\bhalt\b", r"\bfreeze\b", r"\bquiet\b"]


def extract_distance(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in)\b", text.lower())
    return float(match.group(1)) if match else 3.0


def parse_intent(text: str) -> Intent:
    text_lower = text.lower().strip()

    for pattern in STOP_PATTERNS:
        if re.search(pattern, text_lower):
            return Intent(type=IntentType.STOP)

    for pattern, command in MOVEMENT_PATTERNS.items():
        if re.search(pattern, text_lower):
            distance = extract_distance(text_lower)
            return Intent(type=IntentType.MOVEMENT, command=command, distance=distance)

    for pattern in GREETING_PATTERNS:
        if re.search(pattern, text_lower):
            return Intent(type=IntentType.GREETING, text=text)

    return Intent(type=IntentType.QUESTION, text=text)
