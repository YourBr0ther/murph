"""
Murph - Server Brain
Main entry point for the robot's brain (perception, cognition, expression).
"""

import asyncio
import logging
import signal
import sys
import warnings
from pathlib import Path

# Suppress PyTorch FutureWarnings from facenet_pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

from dotenv import load_dotenv
import uvicorn

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from .orchestrator import CognitionOrchestrator
from .monitoring import MonitoringServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Quiet down noisy libraries
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger("murph.server")

# Dashboard server configuration
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 6081


async def main() -> None:
    """Main entry point for the server brain."""
    logger.info("Starting Murph server brain...")

    orchestrator = CognitionOrchestrator()

    # Create monitoring server for dashboard
    monitoring = MonitoringServer(orchestrator)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    if sys.platform == "win32":
        # Windows: use signal.signal() with call_soon_threadsafe
        def windows_handler(signum: int, frame: object) -> None:
            loop.call_soon_threadsafe(shutdown_event.set)

        signal.signal(signal.SIGINT, windows_handler)
        # Note: SIGTERM doesn't exist on Windows
    else:
        # Unix: use asyncio's signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    # Configure uvicorn for the dashboard server
    config = uvicorn.Config(
        monitoring.app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",  # Reduce uvicorn log noise
    )
    server = uvicorn.Server(config)

    try:
        # Start orchestrator
        await orchestrator.start()
        logger.info("Murph server brain started")

        # Start monitoring server and broadcast loop as background tasks
        dashboard_task = asyncio.create_task(
            server.serve(),
            name="dashboard_server"
        )
        broadcast_task = asyncio.create_task(
            monitoring.start_broadcast_loop(),
            name="broadcast_loop"
        )

        logger.info(f"Dashboard available at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        logger.info("Waiting for shutdown signal...")

        # Wait for shutdown signal
        await shutdown_event.wait()

    finally:
        logger.info("Shutting down...")

        # Stop broadcast loop
        await monitoring.stop_broadcast_loop()

        # Signal uvicorn to shutdown
        server.should_exit = True

        # Cancel tasks
        if 'dashboard_task' in dir():
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass

        if 'broadcast_task' in dir():
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

        # Stop orchestrator
        await orchestrator.stop()
        logger.info("Murph server brain stopped")


if __name__ == "__main__":
    asyncio.run(main())
