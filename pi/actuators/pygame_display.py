"""
Murph - Pygame Display Controller
Shows facial expressions on HDMI display using pygame.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from .base import DisplayController
from .display import EXPRESSIONS

logger = logging.getLogger(__name__)


class PygameDisplayController(DisplayController):
    """
    Display controller using pygame for HDMI output.

    Shows facial expressions in a window on the HDMI screen.
    Useful for testing without the I2C OLED display.
    """

    # Window configuration
    WINDOW_WIDTH = 400
    WINDOW_HEIGHT = 300
    BACKGROUND_COLOR = (0, 0, 0)  # Black
    TEXT_COLOR = (0, 255, 0)  # Green (terminal-like)
    FONT_SIZE = 24

    def __init__(
        self,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
    ) -> None:
        """
        Initialize pygame display controller.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self._width = width
        self._height = height
        self._ready = False
        self._current_expression = "neutral"

        # Pygame resources (initialized in initialize())
        self._screen = None
        self._font = None
        self._running = False

        # Thread for pygame event loop
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "PygameDisplayController"

    async def initialize(self) -> bool:
        """Initialize pygame and create display window."""
        try:
            import pygame
        except ImportError:
            logger.error("pygame not available - display disabled")
            return False

        try:
            # Initialize pygame in a separate thread for event handling
            self._running = True
            self._thread = threading.Thread(target=self._pygame_thread, daemon=True)
            self._thread.start()

            # Wait a moment for pygame to initialize
            await asyncio.sleep(0.5)

            self._ready = True
            logger.info(
                f"PygameDisplayController initialized ({self._width}x{self._height})"
            )
            return True

        except Exception as e:
            logger.error(f"Pygame display initialization failed: {e}")
            return False

    def _pygame_thread(self) -> None:
        """Background thread for pygame event loop."""
        try:
            import pygame

            pygame.init()
            pygame.display.set_caption("Murph - Expressions")

            self._screen = pygame.display.set_mode((self._width, self._height))
            self._font = pygame.font.Font(None, self.FONT_SIZE)

            # Initial render
            self._render_expression()

            # Event loop
            clock = pygame.time.Clock()
            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False

                clock.tick(30)  # 30 FPS

            pygame.quit()

        except Exception as e:
            logger.error(f"Pygame thread error: {e}")
            self._running = False

    def _render_expression(self) -> None:
        """Render the current expression to the screen."""
        if not self._screen or not self._font:
            return

        try:
            import pygame

            with self._lock:
                expression = self._current_expression

            # Clear screen
            self._screen.fill(self.BACKGROUND_COLOR)

            # Get expression art
            art = EXPRESSIONS.get(expression, EXPRESSIONS.get("neutral", []))

            # Calculate starting position to center the art
            total_height = len(art) * self.FONT_SIZE
            y_start = (self._height - total_height) // 2

            # Render each line
            for i, line in enumerate(art):
                text_surface = self._font.render(line, True, self.TEXT_COLOR)
                text_rect = text_surface.get_rect()
                text_rect.centerx = self._width // 2
                text_rect.y = y_start + i * self.FONT_SIZE
                self._screen.blit(text_surface, text_rect)

            # Show expression name at bottom
            name_text = self._font.render(
                f"Expression: {expression}",
                True,
                (100, 100, 100),  # Gray
            )
            name_rect = name_text.get_rect()
            name_rect.centerx = self._width // 2
            name_rect.bottom = self._height - 10
            self._screen.blit(name_text, name_rect)

            pygame.display.flip()

        except Exception as e:
            logger.error(f"Render error: {e}")

    async def shutdown(self) -> None:
        """Shutdown pygame display."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._ready = False
        logger.info("PygameDisplayController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def set_expression(self, name: str) -> None:
        """Set the facial expression."""
        if not self._ready:
            logger.warning("Display controller not ready")
            return

        if name not in EXPRESSIONS:
            logger.warning(f"Unknown expression: {name}, using neutral")
            name = "neutral"

        with self._lock:
            self._current_expression = name

        # Trigger re-render in pygame thread
        self._render_expression()

        logger.debug(f"Display expression: {name}")

    async def clear(self) -> None:
        """Clear the display."""
        with self._lock:
            self._current_expression = "neutral"
        self._render_expression()
        logger.debug("Display cleared")

    def get_current_expression(self) -> str:
        """Get current expression."""
        with self._lock:
            return self._current_expression

    def get_expression_art(self) -> list[str]:
        """Get ASCII art for current expression."""
        with self._lock:
            return EXPRESSIONS.get(self._current_expression, EXPRESSIONS.get("neutral", []))
