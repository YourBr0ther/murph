# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Person recognition with FaceNet - 2024-12-08

## Next Feature Options (from PROGRESS.md)
1. Core behaviors - hardware integration (trees are implemented)
2. Spatial map storage (environment awareness)
3. WebRTC video streaming integration (connect vision to Pi camera)

## Notes
Person recognition system is complete with:
- FaceDetector: MTCNN wrapper for face detection in video frames
- FaceEncoder: InceptionResnetV1 with 128-dim projection layer
- FaceRecognizer: Full pipeline orchestrating detection, encoding, and memory lookup
- PersonTracker: Cross-frame tracking with IoU and embedding similarity
- Data types: DetectedFace, FaceEncoding, RecognitionResult, TrackedPerson
- Integration with existing MemorySystem.lookup_person_by_face()
- Match confirmation logic (3 consecutive matches required)
- Quality-based filtering for embedding storage
- Face recognition constants added to shared/constants.py
- Comprehensive test suite (73 new tests, 314 total tests passing)
