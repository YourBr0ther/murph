"""
Murph - Raspberry Pi Client
Main entry point for the robot's body (sensors, actuators, local behaviors).
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("murph.pi")


async def main() -> None:
    """Main entry point for the Pi client."""
    logger.info("Starting Murph Pi client...")
    # TODO: Initialize sensors, actuators, and communication
    # TODO: Start main loop
    logger.info("Murph Pi client started")


if __name__ == "__main__":
    asyncio.run(main())
