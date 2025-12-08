"""
Murph - Audio Controller
MAX98357A I2S speaker for sound playback.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .base import AudioController

logger = logging.getLogger(__name__)


# Sound durations (for mock timing)
SOUND_DURATIONS: dict[str, float] = {
    "greeting": 1.0,
    "happy": 0.5,
    "sad": 0.8,
    "curious": 0.6,
    "surprised": 0.4,
    "sleepy": 1.2,
    "playful": 0.7,
    "affection": 0.8,
    "alert": 0.3,
    "startled": 0.3,
    "chirp": 0.3,
    "oof": 0.2,
    "scream": 0.5,
    "sigh": 0.6,
}


class MockAudioController(AudioController):
    """
    Mock audio controller for testing without hardware.

    Logs sound playback and tracks state.
    """

    def __init__(self) -> None:
        self._ready = False
        self._current_sound: str | None = None
        self._is_playing = False
        self._play_task: asyncio.Task[None] | None = None
        self._volume = 1.0

    @property
    def name(self) -> str:
        return "MockAudioController"

    async def initialize(self) -> bool:
        """Initialize mock audio."""
        self._ready = True
        logger.info("MockAudioController initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown mock audio."""
        await self.stop_sound()
        self._ready = False
        logger.info("MockAudioController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def play_sound(self, name: str, volume: float = 1.0) -> None:
        """Simulate playing a sound."""
        if not self._ready:
            logger.warning("Audio controller not ready")
            return

        # Stop any current sound
        await self.stop_sound()

        self._current_sound = name
        self._is_playing = True
        self._volume = max(0.0, min(1.0, volume))

        duration = SOUND_DURATIONS.get(name, 0.5)
        logger.debug(f"Playing sound: {name} at volume {volume:.2f}")

        # Schedule sound end
        self._play_task = asyncio.create_task(
            self._finish_sound(duration)
        )

    async def _finish_sound(self, duration: float) -> None:
        """Mark sound as finished after duration."""
        await asyncio.sleep(duration)
        self._current_sound = None
        self._is_playing = False

    async def stop_sound(self) -> None:
        """Stop current sound."""
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
            try:
                await self._play_task
            except asyncio.CancelledError:
                pass
            self._play_task = None

        self._current_sound = None
        self._is_playing = False

    def is_playing(self) -> bool:
        return self._is_playing

    def get_current_sound(self) -> str | None:
        return self._current_sound

    def get_volume(self) -> float:
        """Get current volume level."""
        return self._volume


class MAX98357AudioController(AudioController):
    """
    Real hardware implementation using MAX98357A I2S amplifier.

    Requires pygame or simpleaudio for playback.
    """

    def __init__(self, sounds_dir: str | Path = "assets/sounds") -> None:
        self._ready = False
        self._sounds_dir = Path(sounds_dir)
        self._current_sound: str | None = None
        self._is_playing = False
        self._sound_cache: dict[str, Any] = {}
        self._mixer = None

    @property
    def name(self) -> str:
        return "MAX98357AudioController"

    async def initialize(self) -> bool:
        """Initialize pygame mixer for audio playback."""
        try:
            import pygame

            pygame.mixer.init(
                frequency=44100,
                size=-16,
                channels=1,
                buffer=512,
            )

            self._mixer = pygame.mixer

            # Pre-load sounds
            await self._load_sounds()

            self._ready = True
            logger.info("MAX98357AudioController initialized")
            return True

        except ImportError:
            logger.error("pygame not available")
            return False
        except Exception as e:
            logger.error(f"Audio init failed: {e}")
            return False

    async def _load_sounds(self) -> None:
        """Pre-load sound files into cache."""
        import pygame

        if not self._sounds_dir.exists():
            logger.warning(f"Sounds directory not found: {self._sounds_dir}")
            return

        for name in SOUND_DURATIONS.keys():
            # Try common audio formats
            for ext in [".wav", ".mp3", ".ogg"]:
                path = self._sounds_dir / f"{name}{ext}"
                if path.exists():
                    try:
                        self._sound_cache[name] = pygame.mixer.Sound(str(path))
                        logger.debug(f"Loaded sound: {name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")

    async def shutdown(self) -> None:
        """Shutdown audio system."""
        await self.stop_sound()
        if self._mixer:
            self._mixer.quit()
        self._ready = False
        logger.info("MAX98357AudioController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def play_sound(self, name: str, volume: float = 1.0) -> None:
        """Play a sound file."""
        if not self._ready:
            return

        await self.stop_sound()

        sound = self._sound_cache.get(name)
        if sound:
            sound.set_volume(max(0.0, min(1.0, volume)))
            sound.play()
            self._current_sound = name
            self._is_playing = True

            # Schedule state update when sound finishes
            duration = SOUND_DURATIONS.get(name, 0.5)
            asyncio.create_task(self._finish_sound(duration))
        else:
            logger.warning(f"Sound not found: {name}")

    async def _finish_sound(self, duration: float) -> None:
        """Update state when sound finishes."""
        await asyncio.sleep(duration)
        self._current_sound = None
        self._is_playing = False

    async def stop_sound(self) -> None:
        """Stop any playing sound."""
        if self._mixer:
            self._mixer.stop()
        self._current_sound = None
        self._is_playing = False

    def is_playing(self) -> bool:
        return self._is_playing

    def get_current_sound(self) -> str | None:
        return self._current_sound
