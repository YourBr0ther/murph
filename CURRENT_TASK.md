# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
LLM Integration (NanoGPT + Ollama) - 2024-12-09

## Next Feature Options (from PROGRESS.md)
1. Speech recognition and synthesis
2. Additional behavior sets
3. Dashboard/web UI for monitoring

## Notes
LLM integration implementation is complete with:

### Multi-Provider Architecture
- **NanoGPT**: Cloud API provider (OpenAI-compatible)
- **Ollama**: Local model provider (vision + text models)
- **Mock**: Testing provider with configurable responses

### Core Infrastructure
- `LLMConfig`: Configuration with environment variable loading
- `ResponseCache`: TTL-based LRU cache for response deduplication
- `RateLimiter`: Token bucket rate limiting (configurable RPM)
- `LLMService`: Main service orchestrator with lazy initialization

### Vision Integration
- `VisionAnalyzer`: Throttled scene analysis from video frames
- Structured JSON parsing with fallback for plain text
- WorldContext trigger updates (llm_person_*, llm_mood_*, etc.)
- Integration with perception loop (100ms cycle)

### Behavior Reasoning Integration
- `BehaviorReasoner`: LLM-assisted behavior selection
- Consultation when top behaviors have similar scores
- Context summarization for prompts
- Integration with BehaviorEvaluator (`select_best_async`)

### Files Created (17 new)
- `server/llm/__init__.py`
- `server/llm/types.py` - LLMMessage, LLMResponse, SceneAnalysis, BehaviorRecommendation
- `server/llm/config.py` - LLMConfig with env loading
- `server/llm/cache.py` - ResponseCache
- `server/llm/rate_limiter.py` - RateLimiter
- `server/llm/providers/__init__.py`
- `server/llm/providers/base.py` - Abstract LLMProvider
- `server/llm/providers/mock.py` - MockProvider for testing
- `server/llm/providers/ollama.py` - OllamaProvider
- `server/llm/providers/nanogpt.py` - NanoGPTProvider
- `server/llm/services/__init__.py`
- `server/llm/services/llm_service.py` - Main LLMService
- `server/llm/services/vision_analyzer.py` - VisionAnalyzer
- `server/llm/services/behavior_reasoner.py` - BehaviorReasoner
- `server/llm/prompts/__init__.py`
- `server/llm/prompts/vision_prompts.py` - Scene analysis prompts
- `server/llm/prompts/reasoning_prompts.py` - Behavior reasoning prompts

### Files Modified
- `server/cognition/behavior/context.py` - Added llm_triggers support
- `server/cognition/behavior/evaluator.py` - Added select_best_async method
- `server/orchestrator.py` - LLM component initialization and integration
- `shared/constants.py` - Added LLM constants

### Test Coverage
- 60 new tests in `tests/test_server/test_llm/`
- 659 total tests passing

### Configuration
Environment variables:
- MURPH_LLM_PROVIDER (ollama/nanogpt/mock)
- NANOGPT_API_KEY
- NANOGPT_MODEL
- OLLAMA_BASE_URL
- OLLAMA_MODEL / OLLAMA_TEXT_MODEL
- MURPH_LLM_VISION_ENABLED
- MURPH_LLM_REASONING_ENABLED
- MURPH_LLM_MAX_RPM
- MURPH_LLM_VISION_INTERVAL
