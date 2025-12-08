"""
Murph - Display Controller
SSD1306 OLED display for facial expressions.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import DisplayController

logger = logging.getLogger(__name__)


# Expression pixel art could be defined here or loaded from files
EXPRESSIONS: dict[str, list[str]] = {
    "neutral": [
        "  ****  ****  ",
        " *    **    * ",
        "*      *     *",
        "*      *     *",
        " *    **    * ",
        "  ****  ****  ",
    ],
    "happy": [
        "  ****  ****  ",
        " *    **    * ",
        "*  **  *  ** *",
        "*      *     *",
        " *    **    * ",
        "  ****  ****  ",
    ],
    "sad": [
        "  ****  ****  ",
        " *    **    * ",
        "*      *     *",
        "*  **  *  ** *",
        " *    **    * ",
        "  ****  ****  ",
    ],
    "curious": [
        "  ****  ****  ",
        " *    * *    *",
        "*      **     *",
        "*      **     *",
        " *    * *    *",
        "  ****  ****  ",
    ],
    "surprised": [
        "  ****  ****  ",
        " *    **    * ",
        "*  **  *  ** *",
        "*  **  *  ** *",
        " *    **    * ",
        "  ****  ****  ",
    ],
    "sleepy": [
        "             ",
        "  ****  ****  ",
        " *----**----* ",
        "  ****  ****  ",
        "             ",
        "             ",
    ],
    "playful": [
        "  ****   ****  ",
        " *    * *    * ",
        "*  ^^  * ^^   *",
        "*      *      *",
        " *    **    *  ",
        "  ****  ****   ",
    ],
    "love": [
        "  ****  ****  ",
        " * ** ** ** * ",
        "*  ****  ****  *",
        "*   ******   *",
        " *   ****   * ",
        "  ****  ****  ",
    ],
    "scared": [
        "  ****  ****  ",
        " *OOOO**OOOO* ",
        "*  OO  *  OO *",
        "*      *     *",
        " *    **    * ",
        "  ****  ****  ",
    ],
    "alert": [
        "  ****  ****  ",
        " *!!!!**!!!!* ",
        "*  !!  *  !! *",
        "*      *     *",
        " *    **    * ",
        "  ****  ****  ",
    ],
}


class MockDisplayController(DisplayController):
    """
    Mock display controller for testing without hardware.

    Logs expression changes and tracks current state.
    """

    def __init__(self) -> None:
        self._ready = False
        self._current_expression = "neutral"

    @property
    def name(self) -> str:
        return "MockDisplayController"

    async def initialize(self) -> bool:
        """Initialize mock display."""
        self._ready = True
        self._current_expression = "neutral"
        logger.info("MockDisplayController initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown mock display."""
        await self.clear()
        self._ready = False
        logger.info("MockDisplayController shut down")

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

        self._current_expression = name
        logger.debug(f"Display expression: {name}")

    async def clear(self) -> None:
        """Clear the display."""
        self._current_expression = "neutral"
        logger.debug("Display cleared")

    def get_current_expression(self) -> str:
        """Get current expression."""
        return self._current_expression

    def get_expression_art(self) -> list[str]:
        """Get ASCII art for current expression (for emulator)."""
        return EXPRESSIONS.get(self._current_expression, EXPRESSIONS["neutral"])


class SSD1306DisplayController(DisplayController):
    """
    Real hardware implementation using SSD1306 OLED display.

    Requires luma.oled library and I2C access.
    """

    I2C_ADDRESS = 0x3C
    WIDTH = 128
    HEIGHT = 64

    def __init__(self) -> None:
        self._ready = False
        self._device = None
        self._current_expression = "neutral"
        self._image = None
        self._draw = None

    @property
    def name(self) -> str:
        return "SSD1306DisplayController"

    async def initialize(self) -> bool:
        """Initialize I2C display."""
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            from PIL import Image, ImageDraw

            # Initialize I2C
            serial = i2c(port=1, address=self.I2C_ADDRESS)
            self._device = ssd1306(serial, width=self.WIDTH, height=self.HEIGHT)

            # Create drawing context
            self._image = Image.new("1", (self.WIDTH, self.HEIGHT))
            self._draw = ImageDraw.Draw(self._image)

            # Show neutral expression
            await self.set_expression("neutral")

            self._ready = True
            logger.info("SSD1306DisplayController initialized")
            return True

        except ImportError:
            logger.error("luma.oled not available")
            return False
        except Exception as e:
            logger.error(f"Display init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Clear and shutdown display."""
        if self._device:
            self._device.clear()
            self._device.hide()
        self._ready = False
        logger.info("SSD1306DisplayController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def set_expression(self, name: str) -> None:
        """Render expression on display."""
        if not self._ready or not self._device:
            return

        if name not in EXPRESSIONS:
            name = "neutral"

        self._current_expression = name

        # Clear and redraw
        self._draw.rectangle(
            (0, 0, self.WIDTH, self.HEIGHT),
            fill=0
        )

        # Draw expression (simplified - real impl would use proper graphics)
        art = EXPRESSIONS.get(name, EXPRESSIONS["neutral"])
        y_offset = (self.HEIGHT - len(art) * 8) // 2

        for i, line in enumerate(art):
            x_offset = (self.WIDTH - len(line) * 6) // 2
            self._draw.text(
                (x_offset, y_offset + i * 8),
                line,
                fill=1
            )

        self._device.display(self._image)

    async def clear(self) -> None:
        """Clear the display."""
        if self._device:
            self._device.clear()
        self._current_expression = "neutral"

    def get_current_expression(self) -> str:
        return self._current_expression
