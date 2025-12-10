# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Voice Command System - 2024-12-10

## Next Feature Options (from PROGRESS.md)
1. Additional behavior sets
2. Memory consolidation / LLM context building

## Notes
Completed voice command system implementation:

### 1. Core Service
- `server/llm/services/voice_command_service.py` - New voice command service
  - Wake word detection ("murph", "murphy", "hey murph", "hey murphy")
  - Keyword-based command parsing (approach, rest, play, stop, speak, feedback)
  - LLM fallback for unmatched commands (via existing LLMService)
  - Mood-based response generation (happy/tired/playful/neutral variants)
  - Need adjustment calculations (social, affection)

### 2. Orchestrator Integration
- `server/orchestrator.py` - Extended with voice command processing
  - `_process_voice_command()` - Processes transcriptions through VoiceCommandService
  - `_execute_direct_command()` - Handles immediate actions (stop, speak)
  - `_speak_response()` - TTS synthesis with mood-based emotion
  - Automatic need adjustments on voice interaction

### 3. Context Triggers
- `server/cognition/behavior/context.py` - Added computed triggers
  - `addressed_by_name` - True when "murph" or "murphy" in speech
  - `voice_command_pending` - True when recent speech detected

### 4. Configuration
- `server/llm/config.py` - Added voice command settings
  - `voice_commands_enabled` (default: True)
  - `voice_wake_words` (customizable)
  - `voice_llm_fallback` (default: True)

### 5. Emulator Support
- `emulator/static/index.html` - Voice input text field + "Say" button
- `emulator/static/emulator.js` - `sendVoiceCommand()` function
- `emulator/app.py` - WebSocket handler for `voice_input` messages
- `emulator/virtual_pi.py` - `inject_voice_text()` method
- `shared/messages/types.py` - `SimulatedTranscription` message type
- `server/communication/websocket_server.py` - Handler for simulated transcriptions

### Supported Commands
| Command | Keywords | Action |
|---------|----------|--------|
| Come here | "come here", "come over" | Triggers approach behavior |
| Rest/Sleep | "sleep", "rest" | Triggers rest behavior |
| Play | "play", "let's play" | Triggers play behavior |
| Stop | "stop", "halt" | Stops current behavior |
| Speak | "say something" | TTS random phrase |
| Good Murph | "good boy/girl/murph" | Positive feedback (+affection, +social) |
| Bad Murph | "bad boy/girl/murph" | Negative feedback (-affection, +social) |

### Test Coverage
- 42 new tests in `tests/test_server/test_voice_command_service.py`
- 1027 total tests passing (1 flaky VAD timing test unrelated to feature)
