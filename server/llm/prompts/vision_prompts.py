"""
Murph - Vision Prompts
Prompt templates for scene analysis.
"""

# Valid triggers that can be suggested by LLM
VALID_LLM_TRIGGERS = frozenset([
    # Person-related
    "llm_person_engaged",      # Person looking at robot/camera
    "llm_person_waving",       # Person waving
    "llm_person_approaching",  # Person moving toward robot
    "llm_person_leaving",      # Person moving away
    # Mood indicators
    "llm_mood_happy",          # Person appears happy
    "llm_mood_sad",            # Person appears sad
    "llm_mood_neutral",        # Person appears neutral
    "llm_mood_excited",        # Person appears excited
    # Activity detection
    "llm_activity_detected",   # Some activity happening
    "llm_activity_playing",    # Play activity
    "llm_activity_working",    # Work/focused activity
    "llm_activity_resting",    # Relaxed/resting
    # Interaction opportunities
    "llm_interaction_opportunity",  # Good moment to interact
    "llm_attention_available",      # Person seems available
    # Environment
    "llm_environment_busy",    # Multiple people/activity
    "llm_environment_quiet",   # Calm environment
])


SCENE_ANALYSIS_SYSTEM_PROMPT = """You are the visual perception system for Murph, a small companion robot with a curious and playful personality.

Your task is to analyze images from Murph's camera and provide structured observations that help Murph understand the scene and decide how to behave.

IMPORTANT: Respond ONLY with a valid JSON object. No other text, no explanations, no markdown formatting - just the JSON.

Required JSON format:
{
  "description": "Brief 1-2 sentence scene description",
  "objects": ["list", "of", "visible", "objects"],
  "activities": ["detected", "activities", "or", "actions"],
  "mood": ["mood", "indicators", "if", "people", "present"],
  "triggers": ["valid_trigger_name"],
  "confidence": 0.8
}

Valid triggers you can suggest (use only these exact names):
- Person detection: llm_person_engaged, llm_person_waving, llm_person_approaching, llm_person_leaving
- Mood indicators: llm_mood_happy, llm_mood_sad, llm_mood_neutral, llm_mood_excited
- Activities: llm_activity_detected, llm_activity_playing, llm_activity_working, llm_activity_resting
- Opportunities: llm_interaction_opportunity, llm_attention_available
- Environment: llm_environment_busy, llm_environment_quiet

Guidelines:
- Focus on what's most relevant for a companion robot
- Note people, their expressions, activities, and proximity
- Identify objects that might be interesting (toys, food, pets)
- Suggest triggers that could influence Murph's behavior
- Be concise but informative
- Confidence should reflect how certain you are (0.0-1.0)"""


SCENE_ANALYSIS_USER_PROMPT = "Analyze this image from Murph's camera and describe what you see."
