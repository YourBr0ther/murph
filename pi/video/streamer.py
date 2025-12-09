"""
Murph - Video Streamer
WebRTC video streaming from Pi to server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from shared.constants import VIDEO_BITRATE_KBPS
from shared.messages import (
    MessageType,
    RobotMessage,
    WebRTCIceCandidate,
    WebRTCOffer,
)

if TYPE_CHECKING:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    from .camera import CameraManager

logger = logging.getLogger(__name__)


class VideoStreamer:
    """
    WebRTC video streaming from Pi to server.

    Manages:
    - WebRTC peer connection lifecycle
    - SDP offer/answer exchange via signaling callback
    - ICE candidate exchange
    - Video track from camera
    - Automatic reconnection on failure

    Usage:
        streamer = VideoStreamer(
            camera=camera_manager,
            on_signaling=send_via_websocket,
        )
        await streamer.start()

        # When server sends answer
        await streamer.handle_answer(answer_sdp)
    """

    def __init__(
        self,
        camera: CameraManager,
        on_signaling: Callable[[RobotMessage], Any] | None = None,
    ) -> None:
        """
        Initialize video streamer.

        Args:
            camera: CameraManager instance for video capture
            on_signaling: Callback for sending signaling messages to server
        """
        self._camera = camera
        self._on_signaling = on_signaling

        self._pc: RTCPeerConnection | None = None
        self._running = False
        self._connected = False
        self._connection_state = "new"

        # Reconnection settings
        self._reconnect_delay = 5.0
        self._max_reconnect_delay = 60.0
        self._reconnect_task: asyncio.Task[None] | None = None

        logger.info("VideoStreamer initialized")

    async def start(self) -> None:
        """
        Start the video streamer.

        Creates peer connection and initiates offer/answer exchange.
        """
        self._running = True

        # Start camera if not already running
        if not self._camera.is_running:
            await self._camera.start()

        # Create peer connection and send offer
        await self._create_connection()

        logger.info("VideoStreamer started")

    async def stop(self) -> None:
        """Stop the video streamer and close connection."""
        logger.info("Stopping VideoStreamer...")
        self._running = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        self._connected = False
        logger.info("VideoStreamer stopped")

    async def _create_connection(self) -> None:
        """Create WebRTC peer connection and send offer."""
        try:
            from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
        except ImportError:
            logger.error("aiortc not installed - video streaming disabled")
            return

        # Close existing connection
        if self._pc:
            await self._pc.close()

        # Create peer connection with ICE servers (STUN for NAT traversal)
        # For local network, STUN isn't strictly needed but doesn't hurt
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        self._pc = RTCPeerConnection(configuration=config)
        self._setup_pc_handlers()

        # Add video track from camera
        video_track = self._camera.create_video_track()
        self._pc.addTrack(video_track)

        # Create and send offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # Send offer via signaling callback
        if self._on_signaling:
            msg = RobotMessage(
                message_type=MessageType.WEBRTC_OFFER,
                payload=WebRTCOffer(sdp=self._pc.localDescription.sdp),
            )
            await self._on_signaling(msg)
            logger.info("WebRTC offer sent to server")

    def _setup_pc_handlers(self) -> None:
        """Set up peer connection event handlers."""
        if not self._pc:
            return

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = self._pc.connectionState
            self._connection_state = state
            logger.info(f"WebRTC connection state: {state}")

            if state == "connected":
                self._connected = True
                # Reset reconnect delay on successful connection
                self._reconnect_delay = 5.0

            elif state == "disconnected":
                self._connected = False
                logger.warning("WebRTC disconnected")

            elif state == "failed":
                self._connected = False
                logger.error("WebRTC connection failed")
                # Schedule reconnection
                if self._running:
                    self._schedule_reconnect()

            elif state == "closed":
                self._connected = False

        @self._pc.on("icecandidate")
        async def on_ice_candidate(candidate: Any) -> None:
            if candidate and self._on_signaling:
                # Send ICE candidate to server
                msg = RobotMessage(
                    message_type=MessageType.WEBRTC_ICE_CANDIDATE,
                    payload=WebRTCIceCandidate(
                        candidate=candidate.candidate,
                        sdp_mid=candidate.sdpMid,
                        sdp_mline_index=candidate.sdpMLineIndex,
                    ),
                )
                await self._on_signaling(msg)

        @self._pc.on("icegatheringstatechange")
        async def on_ice_gathering_state_change() -> None:
            logger.debug(f"ICE gathering state: {self._pc.iceGatheringState}")

    async def handle_answer(self, sdp: str) -> None:
        """
        Handle SDP answer from server.

        Args:
            sdp: SDP answer string
        """
        if not self._pc:
            logger.warning("Received answer but no peer connection exists")
            return

        try:
            from aiortc import RTCSessionDescription
        except ImportError:
            return

        answer = RTCSessionDescription(sdp=sdp, type="answer")
        await self._pc.setRemoteDescription(answer)
        logger.info("WebRTC answer received and set")

    async def add_ice_candidate(self, candidate_data: WebRTCIceCandidate | dict) -> None:
        """
        Add ICE candidate from server.

        Args:
            candidate_data: ICE candidate data
        """
        if not self._pc:
            logger.warning("Received ICE candidate but no peer connection exists")
            return

        try:
            from aiortc import RTCIceCandidate
        except ImportError:
            return

        # Handle both dataclass and dict
        if isinstance(candidate_data, dict):
            candidate = candidate_data.get("candidate", "")
            sdp_mid = candidate_data.get("sdp_mid")
            sdp_mline_index = candidate_data.get("sdp_mline_index")
        else:
            candidate = candidate_data.candidate
            sdp_mid = candidate_data.sdp_mid
            sdp_mline_index = candidate_data.sdp_mline_index

        if not candidate:
            logger.debug("Received empty ICE candidate (end of candidates)")
            return

        try:
            ice_candidate = RTCIceCandidate(
                candidate=candidate,
                sdpMid=sdp_mid,
                sdpMLineIndex=sdp_mline_index,
            )
            await self._pc.addIceCandidate(ice_candidate)
            logger.debug(f"Added ICE candidate from server")
        except Exception as e:
            logger.warning(f"Failed to add ICE candidate: {e}")

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        if self._reconnect_task and not self._reconnect_task.done():
            return  # Already scheduled

        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        while self._running and not self._connected:
            logger.info(f"Reconnecting in {self._reconnect_delay:.1f}s...")
            await asyncio.sleep(self._reconnect_delay)

            if not self._running:
                break

            try:
                await self._create_connection()
                # Wait a bit to see if connection succeeds
                await asyncio.sleep(5.0)

                if self._connected:
                    logger.info("Reconnection successful")
                    break

            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

            # Exponential backoff
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    @property
    def is_connected(self) -> bool:
        """Check if WebRTC connection is established."""
        return self._connected

    @property
    def connection_state(self) -> str:
        """Get current connection state."""
        return self._connection_state

    def get_stats(self) -> dict[str, Any]:
        """Get streamer statistics."""
        return {
            "running": self._running,
            "connected": self._connected,
            "connection_state": self._connection_state,
            "camera_running": self._camera.is_running if self._camera else False,
        }

    def __repr__(self) -> str:
        return (
            f"VideoStreamer(connected={self._connected}, "
            f"state={self._connection_state})"
        )
