"""
Murph - Server Brain
Main entry point for the robot's brain (perception, cognition, expression).
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("murph.server")


async def main() -> None:
    """Main entry point for the server brain."""
    logger.info("Starting Murph server brain...")
    # TODO: Initialize perception, cognition, and expression systems
    # TODO: Start WebSocket/WebRTC servers
    # TODO: Start main cognitive loop
    logger.info("Murph server brain started")


if __name__ == "__main__":
    asyncio.run(main())
