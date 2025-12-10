"""
Murph - Monitoring Dashboard Server
FastAPI app providing real-time monitoring and control of the robot brain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from .orchestrator import CognitionOrchestrator

logger = logging.getLogger(__name__)

# Dashboard static files directory
STATIC_DIR = Path(__file__).parent / "static" / "dashboard"


class NeedAdjustRequest(BaseModel):
    """Request to adjust a need value."""

    name: str
    delta: float


class TriggerBehaviorRequest(BaseModel):
    """Request to trigger a specific behavior."""

    behavior_name: str


class MonitoringServer:
    """
    Monitoring server for the Murph robot brain.

    Provides:
    - REST endpoints for status and control
    - WebSocket for real-time state broadcast
    - Static file serving for dashboard UI
    """

    def __init__(self, orchestrator: CognitionOrchestrator) -> None:
        """
        Initialize the monitoring server.

        Args:
            orchestrator: The CognitionOrchestrator instance to monitor
        """
        self._orchestrator = orchestrator
        self._app = self._create_app()
        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="Murph Monitoring Dashboard",
            description="Real-time monitoring and control for Murph robot brain",
            version="1.0.0",
        )

        # REST endpoints
        @app.get("/api/status")
        async def get_status() -> dict[str, Any]:
            """Get full orchestrator status."""
            return self._orchestrator.get_extended_status()

        @app.get("/api/behaviors")
        async def get_behaviors() -> dict[str, Any]:
            """Get top behavior suggestions with scores."""
            return {"behaviors": self._orchestrator.get_behavior_suggestions(count=10)}

        @app.post("/api/control/need")
        async def adjust_need(request: NeedAdjustRequest) -> JSONResponse:
            """Adjust a need value by delta."""
            try:
                self._orchestrator.adjust_need(request.name, request.delta)
                return JSONResponse(
                    {"success": True, "message": f"Adjusted {request.name} by {request.delta}"}
                )
            except ValueError as e:
                return JSONResponse(
                    {"success": False, "message": str(e)},
                    status_code=400,
                )

        @app.post("/api/control/trigger")
        async def trigger_behavior(request: TriggerBehaviorRequest) -> JSONResponse:
            """Request to trigger a specific behavior."""
            try:
                await self._orchestrator.request_behavior(request.behavior_name)
                return JSONResponse(
                    {"success": True, "message": f"Requested behavior: {request.behavior_name}"}
                )
            except ValueError as e:
                return JSONResponse(
                    {"success": False, "message": str(e)},
                    status_code=400,
                )

        # WebSocket endpoint
        @app.websocket("/ws/monitor")
        async def websocket_monitor(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time state broadcast."""
            await websocket.accept()
            self._clients.add(websocket)
            logger.info(f"Dashboard client connected ({len(self._clients)} total)")

            try:
                # Keep connection alive, handle any incoming messages
                while True:
                    try:
                        # Wait for messages (mainly for ping/pong)
                        await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)
                logger.info(f"Dashboard client disconnected ({len(self._clients)} total)")

        # Serve dashboard static files
        @app.get("/")
        async def serve_dashboard() -> FileResponse:
            """Serve the dashboard HTML."""
            return FileResponse(STATIC_DIR / "index.html")

        # Mount static files (CSS, JS)
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=STATIC_DIR), name="dashboard_static")

        return app

    async def start_broadcast_loop(self) -> None:
        """Start the state broadcast loop."""
        self._running = True
        logger.info("Starting monitoring broadcast loop")

        while self._running:
            if self._clients:
                try:
                    state = {
                        "type": "state",
                        "timestamp": time.time(),
                        "data": self._orchestrator.get_extended_status(),
                    }

                    # Broadcast to all connected clients
                    disconnected = []
                    for client in self._clients:
                        try:
                            await client.send_json(state)
                        except Exception:
                            disconnected.append(client)

                    # Clean up disconnected clients
                    for client in disconnected:
                        self._clients.discard(client)

                except Exception as e:
                    logger.error(f"Error in broadcast loop: {e}")

            # Broadcast at 2Hz (500ms interval)
            await asyncio.sleep(0.5)

    async def stop_broadcast_loop(self) -> None:
        """Stop the state broadcast loop."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

    async def broadcast_behavior_change(
        self, previous: str | None, current: str | None, reason: str
    ) -> None:
        """
        Broadcast a behavior change event to all clients.

        Args:
            previous: Previous behavior name
            current: Current behavior name
            reason: Reason for the change
        """
        if not self._clients:
            return

        message = {
            "type": "behavior_change",
            "timestamp": time.time(),
            "data": {
                "previous": previous,
                "current": current,
                "reason": reason,
            },
        }

        disconnected = []
        for client in self._clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self._clients.discard(client)
