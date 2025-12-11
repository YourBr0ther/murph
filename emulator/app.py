"""
Murph - Emulator Web Application
FastAPI-based web interface for robot simulation.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

from .config import EmulatorConfig
from .virtual_pi import VirtualPi, VirtualRobotState

logger = logging.getLogger(__name__)

# Global state
virtual_pi: VirtualPi | None = None
ui_clients: list[WebSocket] = []


def create_app(
    server_host: str = "localhost",
    server_port: int = 8765,
    video_enabled: bool = True,
    camera_device: int | None = None,
) -> FastAPI:
    """
    Create the emulator FastAPI application.

    Args:
        server_host: Server brain host
        server_port: Server brain port
        video_enabled: Enable webcam video streaming
        camera_device: Camera device index (None = auto-detect)

    Returns:
        Configured FastAPI application
    """
    global virtual_pi

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application startup and shutdown."""
        global virtual_pi
        config = EmulatorConfig(camera_device=camera_device)
        virtual_pi = VirtualPi(
            server_host=server_host,
            server_port=server_port,
            on_state_change=on_state_change,
            video_enabled=video_enabled,
            config=config,
        )
        await virtual_pi.start()
        video_status = "enabled" if video_enabled else "disabled"
        logger.info(f"Emulator started (video: {video_status})")

        yield

        if virtual_pi:
            await virtual_pi.stop()
        logger.info("Emulator stopped")

    app = FastAPI(
        title="Murph Emulator",
        description="Web-based robot simulator for testing",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store video_enabled for use in other handlers
    app.state.video_enabled = video_enabled

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve main emulator UI."""
        html_path = static_dir / "index.html"
        if html_path.exists():
            return FileResponse(html_path)
        return HTMLResponse(content=get_fallback_html(), status_code=200)

    @app.get("/api/status")
    async def get_status():
        """Get current robot status."""
        if virtual_pi:
            # Get video stats if streamer is available
            video_stats = {}
            if virtual_pi._video_streamer:
                video_stats = virtual_pi._video_streamer.get_stats()

            return {
                "connected": virtual_pi.is_connected,
                "state": virtual_pi.state.to_dict(),
                "video": video_stats,
            }
        return {"connected": False, "state": None, "video": {}}

    @app.get("/api/video/stream")
    async def video_stream():
        """Stream video frames as MJPEG from the robot's camera."""
        if not virtual_pi or not virtual_pi._camera:
            return HTMLResponse(
                content="Video not available",
                status_code=503,
            )

        async def generate_frames():
            """Generate MJPEG frames from camera."""
            import cv2

            while True:
                frame = virtual_pi._camera.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # Convert RGB to BGR for OpenCV encoding
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Encode as JPEG
                _, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Yield as MJPEG frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )

                # Target ~10 FPS
                await asyncio.sleep(0.1)

        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/touch")
    async def simulate_touch(electrodes: list[int]):
        """Simulate touch on electrodes."""
        if virtual_pi:
            virtual_pi.simulate_touch(electrodes)
        return {"ok": True}

    @app.post("/api/release")
    async def simulate_release():
        """Release touch."""
        if virtual_pi:
            virtual_pi.simulate_release()
        return {"ok": True}

    @app.post("/api/pickup")
    async def simulate_pickup():
        """Simulate being picked up."""
        if virtual_pi:
            virtual_pi.simulate_pickup()
        return {"ok": True}

    @app.post("/api/bump")
    async def simulate_bump():
        """Simulate a bump."""
        if virtual_pi:
            virtual_pi.simulate_bump()
        return {"ok": True}

    @app.post("/api/shake")
    async def simulate_shake():
        """Simulate being shaken."""
        if virtual_pi:
            virtual_pi.simulate_shake()
        return {"ok": True}

    @app.post("/api/falling")
    async def simulate_falling():
        """Simulate falling/being dropped."""
        if virtual_pi:
            virtual_pi.simulate_falling()
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time state updates."""
        await websocket.accept()
        ui_clients.append(websocket)

        try:
            # Send initial state
            if virtual_pi:
                await websocket.send_json({
                    "type": "state",
                    "data": virtual_pi.state.to_dict(),
                })

            while True:
                # Receive commands from UI
                data = await websocket.receive_json()
                await handle_ui_command(data)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            ui_clients.remove(websocket)

    return app


def on_state_change(state: VirtualRobotState) -> None:
    """Callback when robot state changes - broadcast to UI clients."""
    asyncio.create_task(broadcast_state(state))


async def broadcast_state(state: VirtualRobotState) -> None:
    """Send state update to all UI clients."""
    message = {
        "type": "state",
        "data": state.to_dict(),
    }

    for client in ui_clients[:]:  # Copy list to avoid modification during iteration
        try:
            await client.send_json(message)
        except Exception:
            # Client disconnected
            if client in ui_clients:
                ui_clients.remove(client)


async def handle_ui_command(data: dict[str, Any]) -> None:
    """Handle command from UI."""
    global virtual_pi

    if not virtual_pi:
        return

    cmd_type = data.get("type", "")

    if cmd_type == "touch":
        electrodes = data.get("electrodes", [])
        virtual_pi.simulate_touch(electrodes)

    elif cmd_type == "release":
        virtual_pi.simulate_release()

    elif cmd_type == "pickup":
        virtual_pi.simulate_pickup()

    elif cmd_type == "bump":
        virtual_pi.simulate_bump()

    elif cmd_type == "shake":
        virtual_pi.simulate_shake()

    elif cmd_type == "falling":
        virtual_pi.simulate_falling()

    elif cmd_type == "voice_input":
        text = data.get("text", "")
        if text:
            await virtual_pi.inject_voice_text(text)


def get_fallback_html() -> str:
    """Return fallback HTML if static files not found."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Murph Emulator</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #00d9ff; }
        .panel { background: #16213e; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .robot { width: 100px; height: 100px; background: #00d9ff; border-radius: 50%; margin: 20px auto; position: relative; }
        .eyes { display: flex; justify-content: center; gap: 20px; padding-top: 30px; }
        .eye { width: 15px; height: 15px; background: #1a1a2e; border-radius: 50%; }
        button { background: #00d9ff; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #00b8d9; }
        .status { color: #888; }
        .connected { color: #0f0; }
        .disconnected { color: #f00; }
        #position { font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Murph Emulator</h1>

        <div class="panel">
            <h2>Robot Visualization</h2>
            <div class="robot" id="robot">
                <div class="eyes">
                    <div class="eye"></div>
                    <div class="eye"></div>
                </div>
            </div>
            <p id="position">Position: (0, 0) | Heading: 0°</p>
            <p id="expression">Expression: neutral</p>
            <p id="connection" class="status disconnected">Disconnected</p>
        </div>

        <div class="panel">
            <h2>Sensor Simulation</h2>
            <button onclick="simulateTouch()">Touch (Pet)</button>
            <button onclick="simulateRelease()">Release</button>
            <button onclick="simulatePickup()">Pick Up</button>
            <button onclick="simulateBump()">Bump</button>
            <button onclick="simulateShake()">Shake</button>
            <button onclick="simulateFalling()">Fall</button>
        </div>

        <div class="panel">
            <h2>Voice Command</h2>
            <input type="text" id="voiceInput" placeholder="Murph, come here..." style="padding: 8px; width: 200px;">
            <button onclick="sendVoice()">Send</button>
        </div>

        <div class="panel">
            <h2>State</h2>
            <pre id="state">Loading...</pre>
        </div>
    </div>

    <script>
        let ws;

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connection').textContent = 'Connected';
                document.getElementById('connection').className = 'status connected';
            };

            ws.onclose = () => {
                document.getElementById('connection').textContent = 'Disconnected';
                document.getElementById('connection').className = 'status disconnected';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'state') {
                    updateUI(msg.data);
                }
            };
        }

        function updateUI(state) {
            document.getElementById('position').textContent =
                `Position: (${state.x.toFixed(1)}, ${state.y.toFixed(1)}) | Heading: ${state.heading.toFixed(0)}°`;
            document.getElementById('expression').textContent =
                `Expression: ${state.current_expression}` +
                (state.playing_sound ? ` | Sound: ${state.playing_sound}` : '');
            document.getElementById('state').textContent = JSON.stringify(state, null, 2);

            // Rotate robot based on heading
            const robot = document.getElementById('robot');
            robot.style.transform = `rotate(${state.heading}deg)`;
        }

        function simulateTouch() {
            ws.send(JSON.stringify({ type: 'touch', electrodes: [3, 4, 5, 6] }));
        }

        function simulateRelease() {
            ws.send(JSON.stringify({ type: 'release' }));
        }

        function simulatePickup() {
            ws.send(JSON.stringify({ type: 'pickup' }));
        }

        function simulateBump() {
            ws.send(JSON.stringify({ type: 'bump' }));
        }

        function simulateShake() {
            ws.send(JSON.stringify({ type: 'shake' }));
        }

        function simulateFalling() {
            ws.send(JSON.stringify({ type: 'falling' }));
        }

        function sendVoice() {
            const text = document.getElementById('voiceInput').value;
            if (text) {
                ws.send(JSON.stringify({ type: 'voice_input', text: text }));
                document.getElementById('voiceInput').value = '';
            }
        }

        connect();
    </script>
</body>
</html>
"""


def main() -> None:
    """Entry point for running the emulator."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Murph Robot Emulator")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--server-host", default="localhost", help="Server brain host (default: localhost)"
    )
    parser.add_argument(
        "--server-port", type=int, default=8765, help="Server brain port (default: 8765)"
    )
    parser.add_argument(
        "--video/--no-video",
        dest="video",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable webcam video streaming (default: enabled)",
    )
    parser.add_argument(
        "--camera-device",
        type=int,
        default=None,
        help="Camera device index (default: auto-detect)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # Suppress aioice binding errors for link-local addresses (169.254.x.x)
    # These are non-fatal on Windows - connection succeeds via valid addresses
    logging.getLogger("aioice.ice").setLevel(logging.WARNING)
    app = create_app(
        server_host=args.server_host,
        server_port=args.server_port,
        video_enabled=args.video,
        camera_device=args.camera_device,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
