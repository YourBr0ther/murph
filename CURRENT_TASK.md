# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Quality Improvements (Tests + Expression System) - 2024-12-10

## Next Feature Options (from PROGRESS.md)
1. Speech recognition streaming (Pi -> Server STT)
2. Additional behavior sets
3. Dashboard/web UI for monitoring

## Notes
Completed three quality improvements:

### 1. Communication Layer Tests
- `tests/test_server/test_communication/` - 72 tests
- Covers: PiConnectionManager, ActionDispatcher
- Full coverage of WebSocket handling, command dispatch, ack handling

### 2. Expression System
- `server/expression/types.py` - ExpressionType enum, ExpressionMetadata
- `server/expression/registry.py` - Central catalog with valence/arousal metadata
- `server/expression/selector.py` - Needs-based expression selection
- `tests/test_server/test_expression.py` - 38 tests

### 3. LLM Provider Tests
- `tests/test_server/test_llm/test_ollama_provider.py` - ~25 tests
- `tests/test_server/test_llm/test_nanogpt_provider.py` - ~23 tests
- Covers: complete(), complete_with_vision(), health_check(), list_models()

### Test Coverage
- 922 tests passing (+150 new tests)
- 8 skipped (require cv2/aiohttp not in test env)
