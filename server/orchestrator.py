"""
Murph - Cognition Orchestrator
Main orchestration class that coordinates perception, cognition, and execution loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from shared.constants import (
    ACTION_CYCLE_MS,
    COGNITION_CYCLE_MS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    PERCEPTION_CYCLE_MS,
    VISION_FRAME_STALE_MS,
)
from shared.messages import (
    LocalTrigger,
    MessageType,
    RobotMessage,
    SensorData,
    WebRTCAnswer,
    WebRTCIceCandidate,
    WebRTCOffer,
)

from .cognition.behavior.context import WorldContext
from .cognition.behavior.evaluator import BehaviorEvaluator
from .cognition.behavior.tree_executor import BehaviorTreeExecutor, ExecutionResult
from .cognition.needs import NeedsSystem
from .communication.action_dispatcher import ActionDispatcher
from .communication.websocket_server import PiConnectionManager
from .perception.sensor_processor import SensorProcessor
from .video import FrameBuffer, VideoReceiver, VisionProcessor

logger = logging.getLogger(__name__)


class CognitionOrchestrator:
    """
    Central orchestrator for Murph's cognitive processes.

    Manages three concurrent async loops:
    - Perception (100ms): Process sensor data, update WorldContext
    - Cognition (200ms): Evaluate behaviors, select next action
    - Execution (50ms): Tick behavior trees, dispatch actions

    Handles:
    - Graceful startup and shutdown
    - Pi disconnection recovery
    - Behavior interruption for high-priority situations
    - Need effects after behavior completion
    """

    def __init__(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        self._host = host
        self._port = port

        # State
        self._running = False
        self._pi_connected = False
        self._tasks: list[asyncio.Task[None]] = []

        # World context (shared between loops)
        self._world_context = WorldContext()

        # Sensor processing (perception)
        self._sensor_processor = SensorProcessor()

        # WebSocket connection (with callbacks including WebRTC signaling)
        self._connection = PiConnectionManager(
            host=host,
            port=port,
            on_sensor_data=self._on_sensor_data,
            on_local_trigger=self._on_local_trigger,
            on_connection_change=self._on_connection_change,
            on_webrtc_offer=self._on_webrtc_offer,
            on_webrtc_ice_candidate=self._on_webrtc_ice_candidate,
        )

        # Action dispatch (bridges executor to Pi)
        self._action_dispatcher = ActionDispatcher(self._connection)

        # Video streaming and vision processing
        self._frame_buffer = FrameBuffer()
        self._video_receiver = VideoReceiver(
            on_frame=self._frame_buffer.put,
            on_signaling=self._send_webrtc_signaling,
        )
        self._vision_processor = VisionProcessor()

        # Needs system (cognition)
        self._needs_system = NeedsSystem()

        # Behavior evaluation and selection (cognition)
        self._evaluator = BehaviorEvaluator(self._needs_system)

        # Behavior tree execution
        self._executor = BehaviorTreeExecutor(
            action_callback=self._action_dispatcher.create_callback()
        )

        # Timing stats
        self._last_perception_time: float = 0
        self._last_cognition_time: float = 0
        self._last_execution_time: float = 0

    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._running

    @property
    def is_pi_connected(self) -> bool:
        """Check if Pi is connected."""
        return self._pi_connected

    async def start(self) -> None:
        """
        Start the orchestrator.

        Initializes all subsystems and starts the three concurrent loops.
        """
        logger.info("Starting CognitionOrchestrator...")

        self._running = True

        # Start WebSocket server
        await self._connection.start()
        logger.info(f"WebSocket server started on ws://{self._host}:{self._port}")

        # Start video receiver (waits for WebRTC offer from Pi)
        await self._video_receiver.start()
        logger.info("Video receiver started")

        # Start the three loops concurrently
        self._tasks = [
            asyncio.create_task(self._perception_loop(), name="perception"),
            asyncio.create_task(self._cognition_loop(), name="cognition"),
            asyncio.create_task(self._execution_loop(), name="execution"),
        ]

        logger.info("All cognitive loops started")

    async def stop(self) -> None:
        """
        Stop the orchestrator gracefully.

        Cancels all loops and cleans up resources.
        """
        logger.info("Stopping CognitionOrchestrator...")

        self._running = False

        # Stop current behavior
        if self._executor.is_running:
            self._executor.force_stop()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop video receiver
        await self._video_receiver.stop()

        # Stop WebSocket server
        await self._connection.stop()

        logger.info("CognitionOrchestrator stopped")

    async def _perception_loop(self) -> None:
        """
        Perception loop - runs at 10Hz (100ms cycle).

        Updates WorldContext from sensor data and vision.
        """
        cycle_time = PERCEPTION_CYCLE_MS / 1000.0
        stale_threshold = VISION_FRAME_STALE_MS / 1000.0

        logger.info(f"Perception loop started ({PERCEPTION_CYCLE_MS}ms cycle)")

        while self._running:
            start = time.time()

            try:
                # Update world context from sensor state
                self._sensor_processor.update_world_context(self._world_context)

                # Process vision if frame available
                frame, frame_time = self._frame_buffer.get_latest()
                if frame is not None and (time.time() - frame_time) < stale_threshold:
                    vision_result = await self._vision_processor.process_if_available(frame)
                    if vision_result:
                        self._vision_processor.update_world_context(
                            self._world_context, vision_result
                        )
                else:
                    # Clear vision state if no fresh frame
                    self._vision_processor.update_world_context(self._world_context, None)

                # Update executor's context for condition nodes
                self._executor.update_context(self._world_context)

                # Clear one-shot states
                self._sensor_processor.clear_transient_state()

                self._last_perception_time = time.time()

            except Exception as e:
                logger.error(f"Perception loop error: {e}", exc_info=True)

            # Maintain consistent cycle time
            elapsed = time.time() - start
            await asyncio.sleep(max(0, cycle_time - elapsed))

    async def _cognition_loop(self) -> None:
        """
        Cognition loop - runs at 5Hz (200ms cycle).

        Updates needs, evaluates behaviors, and starts new behaviors.
        """
        cycle_time = COGNITION_CYCLE_MS / 1000.0

        logger.info(f"Cognition loop started ({COGNITION_CYCLE_MS}ms cycle)")

        while self._running:
            start = time.time()

            try:
                # Update needs (time-based decay)
                self._needs_system.update(cycle_time)

                # Update time since last interaction
                if not self._world_context.is_being_petted and \
                   not self._world_context.is_being_held and \
                   not self._world_context.person_detected:
                    self._world_context.time_since_last_interaction += cycle_time
                else:
                    self._world_context.time_since_last_interaction = 0

                # If no behavior running, select and start one
                if not self._executor.is_running:
                    best = self._evaluator.select_best(self._world_context)
                    if best:
                        self._start_behavior(best)

                self._last_cognition_time = time.time()

            except Exception as e:
                logger.error(f"Cognition loop error: {e}", exc_info=True)

            # Maintain consistent cycle time
            elapsed = time.time() - start
            await asyncio.sleep(max(0, cycle_time - elapsed))

    async def _execution_loop(self) -> None:
        """
        Execution loop - runs at 20Hz (50ms cycle).

        Ticks behavior trees and handles completion.
        """
        cycle_time = ACTION_CYCLE_MS / 1000.0

        logger.info(f"Execution loop started ({ACTION_CYCLE_MS}ms cycle)")

        while self._running:
            start = time.time()

            try:
                if self._executor.is_running:
                    result = self._executor.tick()
                    if result is not None:
                        self._handle_behavior_completion(result)

                self._last_execution_time = time.time()

            except Exception as e:
                logger.error(f"Execution loop error: {e}", exc_info=True)

            # Maintain consistent cycle time
            elapsed = time.time() - start
            await asyncio.sleep(max(0, cycle_time - elapsed))

    def _start_behavior(self, scored: Any) -> None:
        """
        Start executing a scored behavior.

        Args:
            scored: ScoredBehavior from evaluator
        """
        behavior = scored.behavior
        success = self._executor.start_behavior(
            behavior,
            self._world_context,
            self._needs_system,
        )

        if success:
            self._evaluator.mark_behavior_used(behavior.name)
            self._world_context.current_behavior = behavior.name
            logger.info(
                f"Started behavior '{behavior.name}' "
                f"(score: {scored.total_score:.2f})"
            )
        else:
            logger.warning(f"Failed to start behavior '{behavior.name}'")

    def _handle_behavior_completion(self, result: ExecutionResult) -> None:
        """
        Handle behavior completion - apply need effects.

        Args:
            result: Execution result from the executor
        """
        behavior = self._evaluator.registry.get(result.behavior_name)

        if behavior and result.succeeded():
            # Apply need effects
            for need_name, effect in behavior.need_effects.items():
                if effect > 0:
                    self._needs_system.satisfy_need(need_name, effect)
                else:
                    self._needs_system.deplete_need(need_name, abs(effect))

            logger.info(
                f"Behavior '{result.behavior_name}' completed successfully "
                f"(duration: {result.duration:.1f}s, ticks: {result.ticks})"
            )
        elif result.failed():
            logger.warning(
                f"Behavior '{result.behavior_name}' failed "
                f"(duration: {result.duration:.1f}s)"
            )
        elif result.was_interrupted():
            logger.info(
                f"Behavior '{result.behavior_name}' was interrupted "
                f"(duration: {result.duration:.1f}s)"
            )

        # Clear current behavior from context
        self._world_context.current_behavior = None

    def _on_sensor_data(self, data: SensorData) -> None:
        """
        Callback for incoming sensor data from Pi.

        Args:
            data: Sensor data message
        """
        self._sensor_processor.process_sensor_data(data)

    def _on_local_trigger(self, trigger: LocalTrigger) -> None:
        """
        Callback for local trigger events from Pi.

        Args:
            trigger: Local trigger event
        """
        self._sensor_processor.handle_local_trigger(trigger)

        # Handle high-priority triggers that might need behavior interruption
        if trigger.trigger_name == "falling" and self._executor.is_running:
            current = self._executor.current_behavior
            if current and current.interruptible:
                logger.warning("Interrupting behavior due to falling trigger")
                self._executor.interrupt()

    def _on_connection_change(self, connected: bool) -> None:
        """
        Callback for Pi connection state changes.

        Args:
            connected: True if Pi connected, False if disconnected
        """
        self._pi_connected = connected

        if not connected:
            logger.warning("Pi disconnected - interrupting current behavior")

            # Force stop any running behavior
            if self._executor.is_running:
                self._executor.force_stop()

            # Reset world context since sensor data is now stale
            self._world_context = WorldContext()
            self._sensor_processor = SensorProcessor()

            # Clear frame buffer
            self._frame_buffer.clear()
        else:
            logger.info("Pi connected - resuming normal operation")

    def _on_webrtc_offer(self, offer: WebRTCOffer) -> None:
        """
        Handle WebRTC SDP offer from Pi.

        Args:
            offer: WebRTC offer containing SDP
        """
        asyncio.create_task(self._handle_webrtc_offer(offer))

    async def _handle_webrtc_offer(self, offer: WebRTCOffer) -> None:
        """Process WebRTC offer and send answer."""
        logger.info("Processing WebRTC offer from Pi")
        answer_sdp = await self._video_receiver.handle_offer(offer.sdp)
        if answer_sdp:
            answer_msg = RobotMessage(
                message_type=MessageType.WEBRTC_ANSWER,
                payload=WebRTCAnswer(sdp=answer_sdp),
            )
            await self._connection.send_message(answer_msg)
            logger.info("Sent WebRTC answer to Pi")

    def _on_webrtc_ice_candidate(self, candidate: WebRTCIceCandidate) -> None:
        """
        Handle WebRTC ICE candidate from Pi.

        Args:
            candidate: ICE candidate data
        """
        asyncio.create_task(self._video_receiver.add_ice_candidate(candidate))

    async def _send_webrtc_signaling(self, msg: RobotMessage) -> None:
        """
        Send WebRTC signaling message to Pi.

        Args:
            msg: RobotMessage containing WebRTC signaling payload
        """
        await self._connection.send_message(msg)

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for debugging."""
        return {
            "running": self._running,
            "pi_connected": self._pi_connected,
            "current_behavior": self._executor.current_behavior_name,
            "executor_state": self._executor.execution_state.name,
            "needs": self._needs_system.summary(),
            "world_context": self._world_context.summary(),
            "connection": self._connection.get_status(),
            "timing": {
                "last_perception": self._last_perception_time,
                "last_cognition": self._last_cognition_time,
                "last_execution": self._last_execution_time,
            },
        }

    def __str__(self) -> str:
        return (
            f"CognitionOrchestrator("
            f"running={self._running}, "
            f"pi_connected={self._pi_connected}, "
            f"behavior={self._executor.current_behavior_name})"
        )
