"""
Murph - Server Brain
Main entry point for the robot's brain (perception, cognition, expression).
"""

import asyncio
import logging
import signal

from .orchestrator import CognitionOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("murph.server")


async def main() -> None:
    """Main entry point for the server brain."""
    logger.info("Starting Murph server brain...")

    orchestrator = CognitionOrchestrator()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await orchestrator.start()
        logger.info("Murph server brain started - waiting for shutdown signal")
        await shutdown_event.wait()
    finally:
        await orchestrator.stop()
        logger.info("Murph server brain stopped")


if __name__ == "__main__":
    asyncio.run(main())
