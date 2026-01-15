# server/src/murph_server/config.py
import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    whisper_model: str = "base"
    ollama_model: str = "llama3.2"
    piper_voice: str = "en_US-lessac-medium"

    def __post_init__(self):
        self.port = int(os.getenv("MURPH_PORT", self.port))
        self.whisper_model = os.getenv("MURPH_WHISPER_MODEL", self.whisper_model)
        self.ollama_model = os.getenv("MURPH_OLLAMA_MODEL", self.ollama_model)
        self.piper_voice = os.getenv("MURPH_PIPER_VOICE", self.piper_voice)
