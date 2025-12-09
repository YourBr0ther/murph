# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
WebRTC video streaming integration - 2024-12-09

## Next Feature Options (from PROGRESS.md)
1. Spatial navigation behaviors (use spatial map for exploration)
2. Basic behaviors (idle, seek_attention, explore)

## Notes
WebRTC video streaming integration is complete with:

### Message Types (shared/messages/types.py)
Three new WebRTC signaling message types:
- **WebRTCOffer**: SDP offer message (type=40)
- **WebRTCAnswer**: SDP answer message (type=41)
- **WebRTCIceCandidate**: ICE candidate exchange (type=42)

### Video Constants (shared/constants.py)
- VIDEO_WIDTH = 640, VIDEO_HEIGHT = 480, VIDEO_FPS = 10
- VIDEO_BITRATE_KBPS = 1500, VIDEO_CODEC = "h264"
- VISION_RESULT_TTL_MS = 500, VISION_FRAME_STALE_MS = 2000

### Server Video Components (server/video/)
- **FrameBuffer**: Thread-safe single-frame buffer with latest-frame semantics
- **VideoReceiver**: aiortc RTCPeerConnection for receiving video
- **VisionProcessor**: Wraps FaceRecognizer + PersonTracker with skip-if-busy semantics

### Pi Video Components (pi/video/)
- **CameraManager**: picamera2 wrapper with H.264 hardware encoding
- **VideoStreamer**: aiortc RTCPeerConnection for streaming to server

### Integration Points
- Server orchestrator perception loop processes frames from FrameBuffer
- Vision results update WorldContext fields (person_detected, person_is_familiar, etc.)
- WebSocket signaling extended for SDP/ICE exchange on both Pi and server
- Pi main.py initializes video components and auto-reconnects on failure

### Architecture
```
Pi Camera (picamera2) → VideoStreamer (aiortc) → WebRTC → VideoReceiver (aiortc) → FrameBuffer → VisionProcessor → WorldContext
                                    ↓                              ↓
                          WebSocket signaling (SDP/ICE)
```

### Test Coverage
- 54 new tests for WebRTC messages, FrameBuffer, and VisionProcessor
- 534 total tests passing
