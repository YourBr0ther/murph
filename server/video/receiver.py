"""
Murph - Video Receiver
WebRTC video receiving and frame extraction on server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack

    from ..audio import AudioReceiver

from shared.messages import (
    MessageType,
    RobotMessage,
    WebRTCAnswer,
    WebRTCIceCandidate,
)

logger = logging.getLogger(__name__)


class VideoReceiver:
    """
    WebRTC video receiving and frame extraction on server.

    Handles:
    - WebRTC peer connection lifecycle
    - SDP offer/answer exchange
    - ICE candidate handling
    - Video frame extraction and conversion
    - Connection state monitoring

    Usage:
        receiver = VideoReceiver(
            on_frame=frame_buffer.put,
            on_signaling=send_to_pi,
        )
        await receiver.start()

        # When Pi sends offer
        answer = await receiver.handle_offer(offer_sdp)

        # When Pi sends ICE candidate
        await receiver.add_ice_candidate(candidate_data)
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
        on_signaling: Callable[[RobotMessage], Any] | None = None,
        audio_receiver: AudioReceiver | None = None,
    ) -> None:
        """
        Initialize video receiver.

        Args:
            on_frame: Callback for received video frames (numpy RGB array)
            on_signaling: Callback for sending signaling messages to Pi
            audio_receiver: AudioReceiver for handling audio tracks
        """
        self._on_frame = on_frame
        self._on_signaling = on_signaling
        self._audio_receiver = audio_receiver

        self._pc: RTCPeerConnection | None = None
        self._video_track: MediaStreamTrack | None = None
        self._running = False
        self._connected = False
        self._frame_task: asyncio.Task[None] | None = None

        # Stats
        self._frames_received = 0
        self._connection_state = "new"

        logger.info("VideoReceiver initialized")

    async def start(self) -> None:
        """Start the video receiver (prepares for incoming connection)."""
        self._running = True
        logger.info("VideoReceiver started, waiting for WebRTC offer")

    async def stop(self) -> None:
        """Stop the video receiver and close connection."""
        logger.info("Stopping VideoReceiver...")
        self._running = False

        if self._frame_task:
            self._frame_task.cancel()
            try:
                await self._frame_task
            except asyncio.CancelledError:
                pass
            self._frame_task = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        self._video_track = None
        self._connected = False
        logger.info("VideoReceiver stopped")

    async def handle_offer(self, sdp: str) -> str:
        """
        Handle WebRTC SDP offer from Pi.

        Creates peer connection, sets remote description, creates answer.

        Args:
            sdp: SDP offer string from Pi

        Returns:
            SDP answer string to send back to Pi
        """
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ImportError:
            logger.error("aiortc not installed - video streaming disabled")
            return ""

        logger.info("Received WebRTC offer, creating answer...")

        # Close existing connection if any
        if self._pc:
            await self._pc.close()

        # Create new peer connection
        self._pc = RTCPeerConnection()
        self._setup_pc_handlers()

        # Set remote description (the offer)
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self._pc.setRemoteDescription(offer)

        # Create and set local description (the answer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        logger.info("WebRTC answer created")
        return self._pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_data: WebRTCIceCandidate | dict) -> None:
        """
        Add ICE candidate received from Pi.

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
            # Parse and add the candidate
            ice_candidate = RTCIceCandidate(
                candidate=candidate,
                sdpMid=sdp_mid,
                sdpMLineIndex=sdp_mline_index,
            )
            await self._pc.addIceCandidate(ice_candidate)
            logger.debug(f"Added ICE candidate: {candidate[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to add ICE candidate: {e}")

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
            elif state in ("disconnected", "failed", "closed"):
                self._connected = False
                if state == "failed":
                    logger.error("WebRTC connection failed")

        @self._pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            logger.info(f"Received track: {track.kind} (id={getattr(track, 'id', 'unknown')})")

            if track.kind == "video":
                self._video_track = track
                logger.info("Video track received - starting frame extraction")
                # Start frame extraction task
                if self._frame_task is None or self._frame_task.done():
                    self._frame_task = asyncio.create_task(
                        self._extract_frames(track),
                        name="frame_extraction",
                    )
                    logger.info("Frame extraction task created")
                else:
                    logger.warning("Frame extraction task already running")

            elif track.kind == "audio":
                # Forward to audio receiver for STT processing
                if self._audio_receiver:
                    self._audio_receiver.handle_track(track)
                    logger.info("Audio track forwarded to audio receiver")
                else:
                    logger.warning("Audio track received but no audio receiver configured")

        @self._pc.on("icecandidate")
        async def on_ice_candidate(candidate: Any) -> None:
            if candidate and self._on_signaling:
                # Send ICE candidate to Pi
                msg = RobotMessage(
                    message_type=MessageType.WEBRTC_ICE_CANDIDATE,
                    payload=WebRTCIceCandidate(
                        candidate=candidate.candidate,
                        sdp_mid=candidate.sdpMid,
                        sdp_mline_index=candidate.sdpMLineIndex,
                    ),
                )
                await self._on_signaling(msg)

    async def _extract_frames(self, track: MediaStreamTrack) -> None:
        """
        Extract frames from video track and push to callback.

        Args:
            track: aiortc video track
        """
        logger.info("Starting frame extraction from video track")

        try:
            while self._running:
                try:
                    # Receive frame from track
                    frame = await track.recv()

                    # Convert to numpy RGB array
                    img = frame.to_ndarray(format="rgb24")

                    # Push to callback
                    if self._on_frame:
                        self._on_frame(img)

                    self._frames_received += 1

                    # Log first frame and periodically after
                    if self._frames_received == 1:
                        logger.info(f"First frame received! Shape: {img.shape}")
                    elif self._frames_received % 100 == 0:
                        logger.debug(f"Frames received: {self._frames_received}")

                except Exception as e:
                    if "MediaStreamError" in str(type(e).__name__):
                        logger.info("Video track ended")
                        break
                    logger.error(f"Frame extraction error: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Frame extraction cancelled")
        finally:
            logger.info(f"Frame extraction ended (received {self._frames_received} frames)")

    @property
    def is_connected(self) -> bool:
        """Check if WebRTC connection is established."""
        return self._connected

    @property
    def connection_state(self) -> str:
        """Get current connection state."""
        return self._connection_state

    def get_stats(self) -> dict[str, Any]:
        """Get receiver statistics."""
        return {
            "running": self._running,
            "connected": self._connected,
            "connection_state": self._connection_state,
            "frames_received": self._frames_received,
            "has_video_track": self._video_track is not None,
        }

    def __repr__(self) -> str:
        return (
            f"VideoReceiver(connected={self._connected}, "
            f"state={self._connection_state}, "
            f"frames={self._frames_received})"
        )


class MockVideoReceiver:
    """
    Mock video receiver for testing without WebRTC.

    Generates synthetic frames at a configurable rate.
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
    ) -> None:
        """
        Initialize mock receiver.

        Args:
            on_frame: Callback for generated frames
            width: Frame width
            height: Frame height
            fps: Frames per second to generate
        """
        self._on_frame = on_frame
        self._width = width
        self._height = height
        self._fps = fps
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._frames_generated = 0

    async def start(self) -> None:
        """Start generating mock frames."""
        self._running = True
        self._task = asyncio.create_task(self._generate_frames())
        logger.info(f"MockVideoReceiver started ({self._width}x{self._height} @ {self._fps}fps)")

    async def stop(self) -> None:
        """Stop generating frames."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MockVideoReceiver stopped")

    async def _generate_frames(self) -> None:
        """Generate synthetic test frames."""
        interval = 1.0 / self._fps

        while self._running:
            # Generate a simple test pattern
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)

            # Add some variation based on frame count
            color = (self._frames_generated * 10) % 256
            frame[:, :, self._frames_generated % 3] = color

            # Add a moving rectangle to simulate motion
            x = (self._frames_generated * 5) % (self._width - 100)
            y = (self._frames_generated * 3) % (self._height - 100)
            frame[y : y + 100, x : x + 100] = [255, 255, 255]

            if self._on_frame:
                self._on_frame(frame)

            self._frames_generated += 1
            await asyncio.sleep(interval)

    async def handle_offer(self, sdp: str) -> str:
        """Mock handle offer - returns empty string."""
        return ""

    async def add_ice_candidate(self, candidate_data: Any) -> None:
        """Mock add ICE candidate - no-op."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._running

    @property
    def connection_state(self) -> str:
        return "connected" if self._running else "closed"

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "connected": self._running,
            "connection_state": self.connection_state,
            "frames_generated": self._frames_generated,
            "mock": True,
        }
