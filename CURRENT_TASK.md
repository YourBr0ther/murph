# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Emulator Webcam Video Streaming - 2024-12-09

## Next Feature Options (from PROGRESS.md)
1. Speech recognition and synthesis
2. Additional behavior sets
3. Dashboard/web UI for monitoring

## Notes
Emulator webcam implementation is complete with:

### New Files Created
- `emulator/video/__init__.py` - Package initialization
- `emulator/video/webcam.py` - USB webcam capture using OpenCV
  - `WebcamCamera` - Real webcam capture via cv2.VideoCapture
  - `WebcamVideoTrack` - aiortc-compatible video track
  - `MockWebcamCamera` - Synthetic test frames fallback
- `emulator/video/streamer.py` - WebRTC streaming
  - `EmulatorVideoStreamer` - Same signaling flow as Pi
- `tests/test_emulator/__init__.py` - Test package
- `tests/test_emulator/test_webcam.py` - 19 unit tests

### Files Modified
- `emulator/virtual_pi.py` - Video streaming integration
  - WebRTC message handling (WEBRTC_ANSWER, WEBRTC_ICE_CANDIDATE)
  - Auto-start video on server connection
  - Graceful fallback to mock if no webcam
- `emulator/app.py` - CLI and API updates
  - `--video/--no-video` flag for enabling/disabling webcam
  - Video stats in `/api/status` endpoint
- `emulator/__init__.py` - Export video classes

### Features
- **USB Webcam Support**: OpenCV-based capture at 640x480 @ 10fps
- **WebRTC Streaming**: Same protocol as real Pi (offer/answer/ICE)
- **Automatic Fallback**: If no webcam available, uses mock frames
- **CLI Control**: `--no-video` flag to disable video streaming
- **Video State**: Tracks webcam_available, video_streaming, video_connected

### Test Coverage
- 19 new tests for webcam and streamer
- 697 total tests passing

### Dependencies Required
- `opencv-python` - For webcam capture
- `aiortc` - For WebRTC (already used by Pi)

### Usage
```bash
# Start emulator with webcam (default)
python -m emulator.app

# Start emulator without video
python -m emulator.app --no-video

# Custom server connection
python -m emulator.app --server-host 192.168.1.100 --server-port 8765
```
