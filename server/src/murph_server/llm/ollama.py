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
