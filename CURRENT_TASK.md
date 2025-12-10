# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Speech Synthesis (TTS) Implementation - 2024-12-10

## Next Feature Options (from PROGRESS.md)
1. Speech recognition streaming (Pi -> Server STT)
2. Additional behavior sets
3. Dashboard/web UI for monitoring

## Notes
Speech synthesis (TTS) implementation is complete:

### Files Created
- `server/llm/services/speech_service.py` - TTS/STT service using NanoGPT APIs
- `tests/test_server/test_speech.py` - 27 tests for speech functionality

### Files Modified
- `shared/messages/types.py` - Added SpeechCommand, VoiceActivityMessage, AudioDataMessage
- `shared/messages/__init__.py` - Exported new message types
- `shared/constants.py` - Added speech-related constants
- `server/llm/config.py` - Added speech_enabled, tts_model, stt_model settings
- `server/llm/services/__init__.py` - Exported SpeechService
- `server/cognition/behavior/actions.py` - Added SpeakAction class
- `server/communication/action_dispatcher.py` - Added speak action handling with TTS
- `pi/actuators/base.py` - Added play_audio_data() method
- `pi/actuators/speaker.py` - Implemented play_audio_data() in both controllers
- `pi/communication/command_handler.py` - Added SpeechCommand handling

### Features Added
1. **TTS via NanoGPT**: SpeechService.synthesize() calls Kokoro 82M model
2. **STT via NanoGPT**: SpeechService.transcribe() calls Whisper Large V3
3. **SpeakAction**: Behavior tree action for TTS speech
4. **Voice Personality**: Wall-E/BMO style phrases (beeps, boops, simple words)
5. **Phrase Library**: 15 pre-defined emotional phrases
6. **Emotion Mapping**: Voice parameters vary by emotion (pitch, speed)
7. **Audio Caching**: Common phrases cached for faster playback
8. **Message Protocol**: SpeechCommand, VoiceActivityMessage, AudioDataMessage

### SpeakAction Phrases
- greeting: "beep boop, hello!"
- happy: "wheee!"
- sad: "aww..."
- curious: "hmm?"
- scared: "eep!"
- affection: "I like you"
- playful: "boop boop boop!"
- sleepy: "zzz..."
- alert: "oh!"
- And 6 more...

### Test Coverage
- 27 new tests for speech functionality
- 770 total tests passing (up from 743)

### Environment Variables
- `MURPH_SPEECH_ENABLED` - Enable/disable speech (default: true)
- `MURPH_TTS_MODEL` - TTS model (default: kokoro-82m)
- `MURPH_STT_MODEL` - STT model (default: whisper-large-v3)
