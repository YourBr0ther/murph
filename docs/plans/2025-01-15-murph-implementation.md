# Murph Robot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a voice-interactive desk robot with WALL-E personality using Raspberry Pi client + Windows server architecture.

**Architecture:** Client-server over WebSocket. Pi handles wake word detection, audio I/O, motors, and sensors. Windows server runs GPU-accelerated STT (faster-whisper), LLM (Ollama), and TTS (Piper). All communication is real-time bidirectional.

**Tech Stack:** Python 3.10+, FastAPI, WebSockets, OpenWakeWord, faster-whisper, Ollama, Piper TTS, RPi.GPIO

---

## Phase 1: Server Foundation

### Task 1: Server Configuration Module

**Files:**
- Create: `server/src/murph_server/config.py`
- Test: `tests/server/test_config.py`

**Step 1: Write the failing test**

```python
# tests/server/test_config.py
import pytest
from murph_server.config import ServerConfig


def test_config_has_required_fields():
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8765
    assert config.whisper_model == "base"
    assert config.ollama_model == "llama3.2"
    assert config.piper_voice == "en_US-lessac-medium"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MURPH_PORT", "9000")
    monkeypatch.setenv("MURPH_OLLAMA_MODEL", "mistral")
    config = ServerConfig()
    assert config.port == 9000
    assert config.ollama_model == "mistral"
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# server/src/murph_server/config.py
import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    whisper_model: str = "base"
    ollama_model: str = "llama3.2"
    piper_voice: str = "en_US-lessac-medium"

    def __post_init__(self):
        self.port = int(os.getenv("MURPH_PORT", self.port))
        self.whisper_model = os.getenv("MURPH_WHISPER_MODEL", self.whisper_model)
        self.ollama_model = os.getenv("MURPH_OLLAMA_MODEL", self.ollama_model)
        self.piper_voice = os.getenv("MURPH_PIPER_VOICE", self.piper_voice)
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/config.py tests/server/test_config.py
git commit -m "feat(server): add configuration module with env overrides"
```

---

### Task 2: Intent Parser - Movement Detection

**Files:**
- Create: `server/src/murph_server/intent/parser.py`
- Test: `tests/server/test_intent_parser.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_intent_parser.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_intent_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/intent/parser.py tests/server/test_intent_parser.py
git commit -m "feat(server): add intent parser with movement/question detection"
```

---

### Task 3: Ollama LLM Client

**Files:**
- Create: `server/src/murph_server/llm/ollama.py`
- Test: `tests/server/test_ollama.py`

**Step 1: Write the failing test**

```python
# tests/server/test_ollama.py
import pytest
from unittest.mock import patch, AsyncMock
from murph_server.llm.ollama import OllamaClient, SYSTEM_PROMPT


def test_system_prompt_has_personality():
    assert "WALL-E" in SYSTEM_PROMPT or "wall-e" in SYSTEM_PROMPT.lower()
    assert "helpful" in SYSTEM_PROMPT.lower()


@pytest.mark.asyncio
async def test_query_returns_response():
    with patch("murph_server.llm.ollama.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.return_value = {"message": {"content": "Hello there!"}}
        mock_client_class.return_value = mock_client

        client = OllamaClient(model="llama3.2")
        response = await client.query("Hello")

        assert response == "Hello there!"
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_ollama.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# server/src/murph_server/llm/ollama.py
from ollama import AsyncClient


SYSTEM_PROMPT = """You are Murph, a friendly desk robot with a WALL-E-inspired personality.

Personality traits:
- Bright, bubbly, and enthusiastic
- Caring and makes users feel valued
- Genuinely helpful with a human-like warmth
- Express yourself with short, cheerful responses

Keep responses brief (1-2 sentences) unless asked for more detail.
Never mention being an AI or language model - you're Murph the robot!
"""


class OllamaClient:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.client = AsyncClient()

    async def query(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = await self.client.chat(model=self.model, messages=messages)
        return response["message"]["content"]
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_ollama.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/llm/ollama.py tests/server/test_ollama.py
git commit -m "feat(server): add Ollama LLM client with WALL-E personality"
```

---

### Task 4: Speech-to-Text with faster-whisper

**Files:**
- Create: `server/src/murph_server/audio/stt.py`
- Test: `tests/server/test_stt.py`

**Step 1: Write the failing test**

```python
# tests/server/test_stt.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from murph_server.audio.stt import SpeechToText


def test_stt_init_with_model():
    with patch("murph_server.audio.stt.WhisperModel") as mock_model:
        stt = SpeechToText(model_size="base")
        mock_model.assert_called_once()


def test_transcribe_returns_text():
    with patch("murph_server.audio.stt.WhisperModel") as mock_model_class:
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Hello Murph"
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_model_class.return_value = mock_model

        stt = SpeechToText(model_size="base")
        audio_data = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(audio_data)

        assert result == "Hello Murph"
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_stt.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# server/src/murph_server/audio/stt.py
import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        segments, _ = self.model.transcribe(audio_data, language="en")
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_stt.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/audio/stt.py tests/server/test_stt.py
git commit -m "feat(server): add faster-whisper speech-to-text integration"
```

---

### Task 5: Text-to-Speech with Piper

**Files:**
- Create: `server/src/murph_server/audio/tts.py`
- Test: `tests/server/test_tts.py`

**Step 1: Write the failing test**

```python
# tests/server/test_tts.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from murph_server.audio.tts import TextToSpeech


def test_tts_synthesize_returns_audio():
    with patch("murph_server.audio.tts.PiperVoice") as mock_voice_class:
        mock_voice = MagicMock()
        mock_audio = np.random.rand(16000).astype(np.float32)
        mock_voice.synthesize.return_value = iter([mock_audio.tobytes()])
        mock_voice_class.load.return_value = mock_voice

        tts = TextToSpeech(voice="en_US-lessac-medium")
        audio = tts.synthesize("Hello there")

        assert isinstance(audio, bytes)
        assert len(audio) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_tts.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# server/src/murph_server/audio/tts.py
from pathlib import Path
from piper import PiperVoice


class TextToSpeech:
    def __init__(self, voice: str = "en_US-lessac-medium"):
        self.voice_name = voice
        voice_path = self._find_voice_model(voice)
        self.voice = PiperVoice.load(voice_path)

    def _find_voice_model(self, voice: str) -> Path:
        search_paths = [
            Path.home() / ".local/share/piper/voices",
            Path("/usr/share/piper/voices"),
            Path("./voices"),
        ]
        for base in search_paths:
            model_path = base / f"{voice}.onnx"
            if model_path.exists():
                return model_path
        raise FileNotFoundError(f"Voice model not found: {voice}")

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        audio_chunks = []
        for audio_bytes in self.voice.synthesize(text):
            audio_chunks.append(audio_bytes)
        return b"".join(audio_chunks)
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_tts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/audio/tts.py tests/server/test_tts.py
git commit -m "feat(server): add Piper text-to-speech integration"
```

---

### Task 6: WebSocket Server Main Entry

**Files:**
- Create: `server/src/murph_server/main.py`
- Test: `tests/server/test_main.py`

**Step 1: Write the failing test**

```python
# tests/server/test_main.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


def test_health_endpoint():
    with patch("murph_server.main.SpeechToText"), \
         patch("murph_server.main.TextToSpeech"), \
         patch("murph_server.main.OllamaClient"):
        from murph_server.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

**Step 2: Run test to verify it fails**

Run: `cd server && pytest tests/test_main.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# server/src/murph_server/main.py
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from murph_server.config import ServerConfig
from murph_server.audio.stt import SpeechToText
from murph_server.audio.tts import TextToSpeech
from murph_server.llm.ollama import OllamaClient
from murph_server.intent.parser import parse_intent, IntentType


app = FastAPI(title="Murph Server")
config = ServerConfig()

stt = SpeechToText(model_size=config.whisper_model)
tts = TextToSpeech(voice=config.piper_voice)
llm = OllamaClient(model=config.ollama_model)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "model": config.ollama_model})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            text = stt.transcribe(audio_array)
            if not text.strip():
                await websocket.send_json({"type": "no_speech"})
                continue

            intent = parse_intent(text)

            if intent.type == IntentType.MOVEMENT:
                await websocket.send_json({
                    "type": "command",
                    "command": intent.command,
                    "distance": intent.distance,
                })
            elif intent.type == IntentType.STOP:
                await websocket.send_json({"type": "command", "command": "stop"})
            else:
                response_text = await llm.query(text)
                audio_bytes = tts.synthesize(response_text)
                await websocket.send_json({"type": "response", "text": response_text})
                await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        print("Client disconnected")


def main():
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd server && pytest tests/test_main.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/src/murph_server/main.py tests/server/test_main.py
git commit -m "feat(server): add FastAPI WebSocket server with full pipeline"
```

---

## Phase 2: Client Foundation

### Task 7: Client Configuration Module

**Files:**
- Create: `client/src/murph_client/config.py`
- Test: `tests/client/test_config.py`

**Step 1: Write the failing test**

```python
# tests/client/test_config.py
import pytest
from murph_client.config import ClientConfig, GPIOPins


def test_config_has_server_settings():
    config = ClientConfig()
    assert config.server_host == "10.0.2.192"
    assert config.server_port == 8765


def test_gpio_pins_motor_mapping():
    pins = GPIOPins()
    assert pins.NSLEEP1 == 12
    assert pins.NSLEEP2 == 13
    assert pins.M1_A == 17
    assert pins.M1_B == 27
```

**Step 2: Run test to verify it fails**

Run: `cd client && pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# client/src/murph_client/config.py
import os
from dataclasses import dataclass


@dataclass
class GPIOPins:
    NSLEEP1: int = 12
    M1_A: int = 17
    M1_B: int = 27
    M2_A: int = 22
    M2_B: int = 23
    NSLEEP2: int = 13
    M3_A: int = 24
    M3_B: int = 25
    M4_A: int = 26
    M4_B: int = 16
    PWM_AUDIO: int = 18
    ULTRASONIC_TRIG: int = 4
    ULTRASONIC_ECHO: int = 5


@dataclass
class ClientConfig:
    server_host: str = "10.0.2.192"
    server_port: int = 8765
    sample_rate: int = 16000
    chunk_size: int = 1024
    max_duty_cycle: int = 25
    pwm_frequency: int = 1000
    obstacle_threshold_cm: float = 20.0
    backup_distance_inches: float = 2.0

    def __post_init__(self):
        self.server_host = os.getenv("MURPH_SERVER_HOST", self.server_host)
        self.server_port = int(os.getenv("MURPH_SERVER_PORT", self.server_port))

    @property
    def server_uri(self) -> str:
        return f"ws://{self.server_host}:{self.server_port}/ws"
```

**Step 4: Run test to verify it passes**

Run: `cd client && pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/murph_client/config.py tests/client/test_config.py
git commit -m "feat(client): add configuration with GPIO pin mapping"
```

---

### Task 8: Motor Driver Module

**Files:**
- Create: `client/src/murph_client/motors/driver.py`
- Test: `tests/client/test_motor_driver.py`

**Step 1: Write the failing test**

```python
# tests/client/test_motor_driver.py
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_gpio():
    with patch("murph_client.motors.driver.GPIO") as mock:
        mock.BCM = 11
        mock.OUT = 0
        mock.HIGH = 1
        mock.LOW = 0
        yield mock


def test_motor_driver_init_sets_gpio_mode(mock_gpio):
    from murph_client.motors.driver import MotorDriver
    driver = MotorDriver()
    mock_gpio.setmode.assert_called_once_with(mock_gpio.BCM)


def test_duty_cycle_capped_at_25(mock_gpio):
    from murph_client.motors.driver import MotorDriver
    mock_pwm = MagicMock()
    mock_gpio.PWM.return_value = mock_pwm

    driver = MotorDriver(max_duty_cycle=25)
    driver.set_speed(100)
    mock_pwm.ChangeDutyCycle.assert_called_with(25)
```

**Step 2: Run test to verify it fails**

Run: `cd client && pytest tests/test_motor_driver.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# client/src/murph_client/motors/driver.py
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

from murph_client.config import GPIOPins


class MotorDriver:
    def __init__(self, max_duty_cycle: int = 25, pwm_freq: int = 1000):
        self.pins = GPIOPins()
        self.max_duty_cycle = max_duty_cycle
        self.pwm_freq = pwm_freq

        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available")

        GPIO.setmode(GPIO.BCM)
        self._setup_pins()
        self._setup_pwm()
        self.stop()

    def _setup_pins(self):
        motor_pins = [
            self.pins.NSLEEP1, self.pins.NSLEEP2,
            self.pins.M1_A, self.pins.M1_B, self.pins.M2_A, self.pins.M2_B,
            self.pins.M3_A, self.pins.M3_B, self.pins.M4_A, self.pins.M4_B,
        ]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

    def _setup_pwm(self):
        self.pwm1 = GPIO.PWM(self.pins.NSLEEP1, self.pwm_freq)
        self.pwm2 = GPIO.PWM(self.pins.NSLEEP2, self.pwm_freq)
        self.pwm1.start(0)
        self.pwm2.start(0)

    def set_speed(self, duty_cycle: int):
        capped = min(duty_cycle, self.max_duty_cycle)
        self.pwm1.ChangeDutyCycle(capped)
        self.pwm2.ChangeDutyCycle(capped)

    def stop(self):
        for pin in [self.pins.M1_A, self.pins.M1_B, self.pins.M2_A, self.pins.M2_B,
                    self.pins.M3_A, self.pins.M3_B, self.pins.M4_A, self.pins.M4_B]:
            GPIO.output(pin, GPIO.LOW)

    def forward(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.LOW)
        GPIO.output(self.pins.M1_B, GPIO.HIGH)
        GPIO.output(self.pins.M2_A, GPIO.LOW)
        GPIO.output(self.pins.M2_B, GPIO.HIGH)
        GPIO.output(self.pins.M3_A, GPIO.LOW)
        GPIO.output(self.pins.M3_B, GPIO.HIGH)
        GPIO.output(self.pins.M4_A, GPIO.LOW)
        GPIO.output(self.pins.M4_B, GPIO.HIGH)

    def backward(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.HIGH)
        GPIO.output(self.pins.M1_B, GPIO.LOW)
        GPIO.output(self.pins.M2_A, GPIO.HIGH)
        GPIO.output(self.pins.M2_B, GPIO.LOW)
        GPIO.output(self.pins.M3_A, GPIO.HIGH)
        GPIO.output(self.pins.M3_B, GPIO.LOW)
        GPIO.output(self.pins.M4_A, GPIO.HIGH)
        GPIO.output(self.pins.M4_B, GPIO.LOW)

    def rotate_left(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.HIGH)
        GPIO.output(self.pins.M1_B, GPIO.LOW)
        GPIO.output(self.pins.M2_A, GPIO.LOW)
        GPIO.output(self.pins.M2_B, GPIO.HIGH)
        GPIO.output(self.pins.M3_A, GPIO.HIGH)
        GPIO.output(self.pins.M3_B, GPIO.LOW)
        GPIO.output(self.pins.M4_A, GPIO.LOW)
        GPIO.output(self.pins.M4_B, GPIO.HIGH)

    def rotate_right(self):
        self.set_speed(self.max_duty_cycle)
        GPIO.output(self.pins.M1_A, GPIO.LOW)
        GPIO.output(self.pins.M1_B, GPIO.HIGH)
        GPIO.output(self.pins.M2_A, GPIO.HIGH)
        GPIO.output(self.pins.M2_B, GPIO.LOW)
        GPIO.output(self.pins.M3_A, GPIO.LOW)
        GPIO.output(self.pins.M3_B, GPIO.HIGH)
        GPIO.output(self.pins.M4_A, GPIO.HIGH)
        GPIO.output(self.pins.M4_B, GPIO.LOW)

    def move_for_distance(self, direction: str, inches: float, ips: float = 1.0):
        duration = inches / ips
        getattr(self, direction, self.stop)()
        time.sleep(duration)
        self.stop()

    def cleanup(self):
        self.pwm1.stop()
        self.pwm2.stop()
        GPIO.cleanup()
```

**Step 4: Run test to verify it passes**

Run: `cd client && pytest tests/test_motor_driver.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/murph_client/motors/driver.py tests/client/test_motor_driver.py
git commit -m "feat(client): add motor driver with safety-capped duty cycle"
```

---

### Task 9: Ultrasonic Sensor Module

**Files:**
- Create: `client/src/murph_client/sensors/ultrasonic.py`
- Test: `tests/client/test_ultrasonic.py`

**Step 1: Write the failing test**

```python
# tests/client/test_ultrasonic.py
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_gpio():
    with patch("murph_client.sensors.ultrasonic.GPIO") as mock:
        mock.BCM = 11
        mock.OUT = 0
        mock.IN = 1
        mock.HIGH = 1
        mock.LOW = 0
        yield mock


def test_ultrasonic_init_configures_pins(mock_gpio):
    from murph_client.sensors.ultrasonic import UltrasonicSensor
    sensor = UltrasonicSensor(trig_pin=4, echo_pin=5)
    mock_gpio.setup.assert_any_call(4, mock_gpio.OUT)
    mock_gpio.setup.assert_any_call(5, mock_gpio.IN)
```

**Step 2: Run test to verify it fails**

Run: `cd client && pytest tests/test_ultrasonic.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# client/src/murph_client/sensors/ultrasonic.py
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class UltrasonicSensor:
    def __init__(self, trig_pin: int = 4, echo_pin: int = 5):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

        if GPIO is None:
            raise RuntimeError("RPi.GPIO not available")

        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)

    def get_distance(self) -> float:
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.1:
                return -1

        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end - pulse_start > 0.1:
                return -1

        pulse_duration = pulse_end - pulse_start
        return (pulse_duration * 34300) / 2

    def obstacle_detected(self, threshold_cm: float = 20.0) -> bool:
        distance = self.get_distance()
        return 0 < distance < threshold_cm
```

**Step 4: Run test to verify it passes**

Run: `cd client && pytest tests/test_ultrasonic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add client/src/murph_client/sensors/ultrasonic.py tests/client/test_ultrasonic.py
git commit -m "feat(client): add HC-SR04 ultrasonic sensor driver"
```

---

### Task 10-13: Audio Modules and Client Main

Tasks 10-13 follow the same TDD pattern for:
- **Task 10:** Audio Capture (`client/src/murph_client/audio/capture.py`)
- **Task 11:** Wake Word Detection (`client/src/murph_client/audio/wakeword.py`)
- **Task 12:** PWM Audio Playback (`client/src/murph_client/audio/playback.py`)
- **Task 13:** Client Main Loop (`client/src/murph_client/main.py`)

Each task: write failing test, implement, verify, commit.

---

## Phase 3: Utility Scripts

### Task 14: Motor Calibration Script

Create `scripts/calibrate_motors.py` - interactive script to measure inches/second at 25% duty cycle.

### Task 15: Audio Test Script

Create `scripts/test_audio.py` - plays test tones through PWM to verify speaker circuit.

---

## Phase 4: Final Integration

### Task 16: Create README

Create `README.md` with quick start instructions for server and client.

### Task 17: Package Exports

Update `__init__.py` files with version and public exports.

---

## Summary

| Phase | Tasks | Components |
|-------|-------|------------|
| 1 | 1-6 | Server: Config, Intent, Ollama, STT, TTS, WebSocket |
| 2 | 7-13 | Client: Config, Motors, Ultrasonic, Audio, Main |
| 3 | 14-15 | Calibration and test scripts |
| 4 | 16-17 | README and package exports |

**Total: 17 tasks with TDD approach**
