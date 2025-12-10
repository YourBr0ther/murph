# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Speech Recognition Streaming (Pi -> Server STT) - 2024-12-10

## Next Feature Options (from PROGRESS.md)
1. Additional behavior sets
2. Dashboard/web UI for monitoring

## Notes
Completed speech recognition streaming implementation:

### 1. Audio Capture and Streaming (Emulator Side)
- `emulator/audio/microphone.py` - Added `get_audio_chunk()` and `create_audio_track()` methods
- `emulator/audio/track.py` - New `MicrophoneAudioTrack` class (aiortc-compatible)
- `emulator/video/streamer.py` - Extended to support optional audio track alongside video

### 2. Audio Reception (Server Side)
- `server/audio/receiver.py` - New `AudioBuffer` and `AudioReceiver` classes
- `server/video/receiver.py` - Added audio track handling in `@pc.on("track")` callback
- `server/orchestrator.py` - Wired up AudioReceiver with SpeechService for STT

### 3. WorldContext Integration
- `server/cognition/behavior/context.py` - Added `last_heard_text` and `time_since_last_speech` fields
- New triggers: `heard_speech`, `heard_speech_recent`

### 4. Virtual Pi Integration
- `emulator/virtual_pi.py` - Now passes microphone to media streamer when audio enabled

### Test Coverage
- 955 tests passing (+33 new tests)
- 12 skipped (require cv2/aiohttp/PyAV not in test env)
