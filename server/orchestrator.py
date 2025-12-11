"""
Murph - Cognition Orchestrator
Main orchestration class that coordinates perception, cognition, and execution loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

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
from .audio import AudioReceiver
from .perception.sensor_processor import SensorProcessor
from .video import FrameBuffer, VideoReceiver, VisionProcessor

if TYPE_CHECKING:
    from .cognition.memory.consolidation import ConsolidationConfig, MemoryConsolidator
    from .cognition.memory.long_term_memory import LongTermMemory
    from .cognition.memory.memory_system import MemorySystem
    from .llm.services.context_builder import ContextBuilder
    from .llm.services.speech_service import SpeechService
    from .llm.services.voice_command_service import VoiceCommandService
    from .llm import BehaviorReasoner, LLMConfig, LLMService, VisionAnalyzer

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
        llm_config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            host: WebSocket server host
            port: WebSocket server port
            llm_config: Optional LLM configuration (uses env if None)
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
            on_simulated_transcription=self._on_transcription,  # Reuse transcription callback
        )

        # Action dispatch (bridges executor to Pi)
        self._action_dispatcher = ActionDispatcher(self._connection)

        # Audio receiving and STT
        self._audio_receiver = AudioReceiver(
            on_transcription=self._on_transcription,
        )
        self._speech_service: SpeechService | None = None

        # Video streaming and vision processing
        self._frame_buffer = FrameBuffer()
        self._video_receiver = VideoReceiver(
            on_frame=self._frame_buffer.put,
            on_signaling=self._send_webrtc_signaling,
            audio_receiver=self._audio_receiver,
        )
        self._vision_processor = VisionProcessor()

        # LLM integration (optional)
        self._llm_config = llm_config
        self._llm_service: LLMService | None = None
        self._vision_analyzer: VisionAnalyzer | None = None
        self._behavior_reasoner: BehaviorReasoner | None = None
        self._voice_command_service: VoiceCommandService | None = None

        # Memory integration (optional)
        self._memory_system: MemorySystem | None = None
        self._long_term_memory: LongTermMemory | None = None
        self._memory_consolidator: MemoryConsolidator | None = None
        self._context_builder: ContextBuilder | None = None
        self._last_consolidation_tick: float = 0.0

        # Needs system (cognition)
        self._needs_system = NeedsSystem()

        # Behavior evaluation and selection (cognition) - reasoner set in start()
        self._evaluator = BehaviorEvaluator(self._needs_system)

        # Behavior tree execution
        self._executor = BehaviorTreeExecutor(
            action_callback=self._action_dispatcher.create_callback()
        )

        # Timing stats
        self._last_perception_time: float = 0
        self._last_cognition_time: float = 0
        self._last_execution_time: float = 0

        # Behavior history for dashboard (max 10 entries)
        self._behavior_history: deque[dict[str, Any]] = deque(maxlen=10)

        # Requested behavior (set via dashboard control)
        self._requested_behavior: str | None = None

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

        # Initialize LLM integration if configured
        await self._init_llm()

        # Start WebSocket server
        await self._connection.start()
        logger.info(f"WebSocket server started on ws://{self._host}:{self._port}")

        # Start audio receiver (will receive tracks from video receiver)
        await self._audio_receiver.start()
        logger.info("Audio receiver started")

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

    async def _init_llm(self) -> None:
        """Initialize LLM components if configured."""
        # Get config (from parameter or environment)
        if self._llm_config is None:
            from .llm import LLMConfig
            self._llm_config = LLMConfig.from_env()

        # Skip if neither vision nor reasoning enabled
        if not self._llm_config.vision_enabled and not self._llm_config.reasoning_enabled:
            logger.info("LLM integration disabled")
            return

        # Validate config
        issues = self._llm_config.validate()
        if issues:
            logger.warning(f"LLM config issues: {issues}")
            return

        # Create LLM service
        from .llm import BehaviorReasoner, LLMService, VisionAnalyzer

        self._llm_service = LLMService(self._llm_config)

        # Create vision analyzer if enabled
        if self._llm_config.vision_enabled:
            self._vision_analyzer = VisionAnalyzer(self._llm_service, self._llm_config)
            logger.info("LLM vision analyzer initialized")

        # Create behavior reasoner if enabled
        if self._llm_config.reasoning_enabled:
            self._behavior_reasoner = BehaviorReasoner(self._llm_service, self._llm_config)
            # Update evaluator with reasoner
            self._evaluator._reasoner = self._behavior_reasoner
            logger.info("LLM behavior reasoner initialized")

        # Initialize speech service for STT (if API key available)
        if self._llm_config.provider == "nanogpt":
            from .llm.services.speech_service import SpeechService

            self._speech_service = SpeechService(self._llm_config)
            self._audio_receiver.set_speech_service(self._speech_service)
            logger.info("Speech service initialized for STT")

        # Initialize voice command service
        from .llm.services.voice_command_service import VoiceCommandService

        self._voice_command_service = VoiceCommandService(
            llm_service=self._llm_service,
            config=self._llm_config,
        )
        logger.info("Voice command service initialized")

        # Initialize memory consolidation if enabled
        if self._llm_config.consolidation_enabled:
            await self._init_memory_consolidation()

        logger.info(f"LLM integration initialized (provider: {self._llm_config.provider})")

    async def _init_memory_consolidation(self) -> None:
        """Initialize memory consolidation components."""
        from .cognition.memory.consolidation import ConsolidationConfig, MemoryConsolidator
        from .cognition.memory.long_term_memory import LongTermMemory
        from .cognition.memory.memory_system import MemorySystem
        from .llm.services.context_builder import ContextBuilder
        from .storage import Database

        # Create consolidation config from LLM config
        consolidation_config = ConsolidationConfig(
            enabled=self._llm_config.consolidation_enabled,
            consolidation_tick_interval=self._llm_config.consolidation_tick_interval,
            event_summarization_interval=self._llm_config.event_summarization_interval,
            relationship_update_interval=self._llm_config.relationship_update_interval,
            reflection_probability=self._llm_config.reflection_probability,
        )

        # Initialize database and long-term memory
        database = Database()
        await database.initialize()
        self._long_term_memory = LongTermMemory(database)
        await self._long_term_memory.initialize()

        # Initialize memory system with long-term storage
        self._memory_system = MemorySystem(long_term=self._long_term_memory)
        await self._memory_system.initialize_from_database()

        # Connect memory system to vision processor for face recognition
        self._vision_processor.set_memory_system(self._memory_system)

        # Initialize context builder
        self._context_builder = ContextBuilder(
            memory_system=self._memory_system,
            long_term_memory=self._long_term_memory,
            llm_service=self._llm_service,
            config=self._llm_config,
        )

        # Initialize memory consolidator
        self._memory_consolidator = MemoryConsolidator(
            llm_service=self._llm_service,
            memory_system=self._memory_system,
            long_term_memory=self._long_term_memory,
            config=consolidation_config,
        )

        logger.info("Memory consolidation initialized")

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

        # Run final memory consolidation
        if self._memory_consolidator:
            try:
                await self._memory_consolidator.consolidate_session()
                logger.info("Memory consolidation complete")
            except Exception as e:
                logger.error(f"Memory consolidation error: {e}")

        # Shutdown memory system (sync to database)
        if self._memory_system:
            await self._memory_system.shutdown()
            logger.info("Memory system shutdown")

        # Stop LLM service
        if self._llm_service:
            await self._llm_service.close()
            logger.info("LLM service closed")

        # Stop speech service
        if self._speech_service:
            await self._speech_service.close()
            logger.info("Speech service closed")

        # Stop audio receiver
        await self._audio_receiver.stop()

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

                    # LLM vision analysis (periodic, non-blocking)
                    if self._vision_analyzer and frame is not None:
                        scene_analysis = await self._vision_analyzer.analyze_if_ready(frame)
                        if scene_analysis:
                            self._vision_analyzer.update_world_context(
                                self._world_context, scene_analysis
                            )
                else:
                    # Clear vision state if no fresh frame
                    self._vision_processor.update_world_context(self._world_context, None)
                    if self._vision_analyzer:
                        self._vision_analyzer.update_world_context(self._world_context, None)

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

                # Update time since last speech (always increases)
                self._world_context.time_since_last_speech += cycle_time

                # If no behavior running and Pi is connected, select and start one
                if not self._executor.is_running and self._pi_connected:
                    # Check for dashboard-requested behavior first
                    if self._requested_behavior:
                        behavior = self._evaluator.registry.get(self._requested_behavior)
                        self._requested_behavior = None
                        if behavior:
                            # Create a minimal ScoredBehavior for the requested behavior
                            from .cognition.behavior.evaluator import ScoredBehavior
                            best = ScoredBehavior(
                                behavior=behavior,
                                total_score=1.0,
                                base_value=1.0,
                                need_modifier=1.0,
                                personality_modifier=1.0,
                                opportunity_bonus=1.0,
                            )
                            self._start_behavior(best)
                            continue

                    # Use async selection if reasoner available (may consult LLM)
                    if self._behavior_reasoner:
                        best = await self._evaluator.select_best_async(self._world_context)
                    else:
                        best = self._evaluator.select_best(self._world_context)
                    if best:
                        self._start_behavior(best)

                # Periodic memory consolidation tick
                if self._memory_consolidator:
                    current_time = time.time()
                    if current_time - self._last_consolidation_tick > self._llm_config.consolidation_tick_interval:
                        await self._memory_consolidator.tick()
                        self._last_consolidation_tick = current_time

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
        Handle behavior completion - apply need effects and record history.

        Args:
            result: Execution result from the executor
        """
        behavior = self._evaluator.registry.get(result.behavior_name)

        # Determine result status
        if result.succeeded():
            status = "completed"
        elif result.failed():
            status = "failed"
        elif result.was_interrupted():
            status = "interrupted"
        else:
            status = "unknown"

        # Record in behavior history
        self._behavior_history.append({
            "name": result.behavior_name,
            "duration": result.duration,
            "status": status,
            "timestamp": time.time(),
        })

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

        # Trigger experience reflection (async, non-blocking)
        if self._memory_consolidator and behavior:
            need_changes = behavior.need_effects if result.succeeded() else {}
            asyncio.create_task(
                self._memory_consolidator.on_behavior_complete(
                    behavior_name=result.behavior_name,
                    result=status,
                    duration=result.duration,
                    context_snapshot=self._world_context.get_state(),
                    need_changes=need_changes,
                )
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

    def _on_transcription(self, text: str) -> None:
        """
        Callback for transcribed speech from AudioReceiver.

        Updates world context with the heard text and processes voice commands.

        Args:
            text: Transcribed speech text
        """
        logger.info(f"Heard speech: '{text}'")
        self._world_context.last_heard_text = text
        self._world_context.time_since_last_speech = 0.0

        # Process voice command asynchronously
        if self._voice_command_service:
            asyncio.create_task(self._process_voice_command(text))

    async def _process_voice_command(self, text: str) -> None:
        """
        Process transcribed speech for voice commands.

        Args:
            text: Transcribed speech text
        """
        if not self._voice_command_service:
            return

        try:
            from .llm.services.voice_command_service import CommandType

            result = await self._voice_command_service.process_speech(
                text=text,
                world_context=self._world_context,
                needs_system=self._needs_system,
            )

            if result is None:
                # Not a command for Murph (no wake word)
                return

            logger.info(
                f"Voice command result: action={result.command.action if result.command else 'none'}, "
                f"response='{result.response_text}'"
            )

            # Update needs based on social interaction
            for need_name, delta in result.need_adjustments.items():
                if delta > 0:
                    self._needs_system.satisfy_need(need_name, delta)
                else:
                    self._needs_system.deplete_need(need_name, abs(delta))

            # Execute command based on type
            if result.command:
                if result.execute_immediately:
                    # Direct execution (speak, stop)
                    await self._execute_direct_command(result.command)
                elif result.command.command_type == CommandType.BEHAVIOR_TRIGGER:
                    # Trigger behavior via evaluator
                    try:
                        await self.request_behavior(result.command.action)
                    except ValueError as e:
                        logger.warning(f"Could not trigger behavior: {e}")
                elif result.command.command_type == CommandType.FEEDBACK:
                    # Feedback already handled via need adjustments
                    pass

            # Always respond with speech if we have a response
            if result.response_text:
                await self._speak_response(result.response_text)

        except Exception as e:
            logger.error(f"Voice command processing error: {e}", exc_info=True)

    async def _execute_direct_command(self, command: Any) -> None:
        """
        Execute a direct action command (bypassing behavior system).

        Args:
            command: VoiceCommand to execute
        """
        from .llm.services.voice_command_service import VoiceCommand

        if not isinstance(command, VoiceCommand):
            return

        if command.action == "stop":
            # Stop current behavior
            if self._executor.is_running:
                current = self._executor.current_behavior
                if current and current.interruptible:
                    self._executor.interrupt()
                    logger.info("Voice command: stopped current behavior")
                else:
                    logger.info("Voice command: behavior not interruptible")
            else:
                logger.info("Voice command: no behavior running to stop")
        elif command.action == "speak":
            # Speaking is handled by the response
            pass
        else:
            logger.warning(f"Unknown direct action: {command.action}")

    async def _speak_response(self, text: str) -> None:
        """
        Speak a response immediately via TTS.

        Args:
            text: Text to speak
        """
        if not self._speech_service:
            logger.debug(f"Speech service unavailable, would say: '{text}'")
            return

        if not self._connection.is_connected:
            logger.debug(f"Pi not connected, would say: '{text}'")
            return

        try:
            # Get current mood for emotion
            emotion = self._map_need_to_emotion()

            # Synthesize speech
            audio_data = await self._speech_service.synthesize(text, emotion)
            if not audio_data:
                logger.warning("TTS synthesis returned no audio")
                return

            # Send to Pi
            from shared.messages import Command, MessageType, RobotMessage, SpeechCommand

            audio_b64 = self._speech_service.encode_audio_base64(audio_data)
            speech_cmd = SpeechCommand(
                audio_data=audio_b64,
                audio_format="wav",
                sample_rate=22050,
                volume=1.0,
                emotion=emotion,
                text=text[:50],  # Truncate for logging
            )
            msg = RobotMessage(
                message_type=MessageType.SPEECH_COMMAND,
                payload=Command(payload=speech_cmd),
            )
            await self._connection.send_message(msg)
            logger.debug(f"Sent speech response: '{text[:30]}...'")

        except Exception as e:
            logger.error(f"Speech response error: {e}")

    def _map_need_to_emotion(self) -> str:
        """
        Map current need state to emotion for TTS.

        Returns:
            Emotion string for TTS
        """
        most_urgent = self._needs_system.get_most_urgent_need()
        if not most_urgent:
            return "neutral"

        emotion_map = {
            "energy": "sleepy",
            "curiosity": "curious",
            "play": "playful",
            "social": "happy",
            "affection": "love",
            "comfort": "neutral",
            "safety": "scared",
        }

        # If the need is critical, use that emotion
        if most_urgent.is_critical():
            return emotion_map.get(most_urgent.name, "neutral")

        # Otherwise check overall happiness
        happiness = self._needs_system.get_happiness()
        if happiness > 70:
            return "happy"
        elif happiness > 50:
            return "playful"
        elif happiness > 30:
            return "neutral"
        else:
            return "sleepy"

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

    def get_extended_status(self) -> dict[str, Any]:
        """
        Get extended orchestrator status for dashboard.

        Includes behavior suggestions and history in addition to basic status.
        """
        base_status = self.get_status()

        # Add elapsed time for current behavior
        if self._executor.is_running:
            base_status["elapsed_time"] = self._executor.elapsed_time
        else:
            base_status["elapsed_time"] = 0.0

        # Add behavior suggestions
        base_status["behaviors"] = {
            "suggested": self.get_behavior_suggestions(count=10),
            "history": list(self._behavior_history),
        }

        # Add available behavior names for trigger dropdown
        base_status["available_behaviors"] = [
            b.name for b in self._evaluator.registry.get_all()
        ]

        return base_status

    def get_behavior_suggestions(self, count: int = 10) -> list[dict[str, Any]]:
        """
        Get top behavior suggestions with scores.

        Args:
            count: Number of behaviors to return

        Returns:
            List of behavior dictionaries with name, score, and breakdown
        """
        scored = self._evaluator.select_top_n(count, self._world_context)
        return [
            {
                "name": sb.behavior.name,
                "score": round(sb.total_score, 3),
                "breakdown": {
                    "base": round(sb.base_value, 2),
                    "need": round(sb.need_modifier, 2),
                    "personality": round(sb.personality_modifier, 2),
                    "opportunity": round(sb.opportunity_bonus, 2),
                },
            }
            for sb in scored
        ]

    def adjust_need(self, name: str, delta: float) -> None:
        """
        Adjust a need value by delta (for dashboard control).

        Args:
            name: Name of the need (e.g., "energy", "curiosity")
            delta: Amount to adjust (+/-)

        Raises:
            ValueError: If need name is invalid
        """
        need = self._needs_system.get_need(name)
        if need is None:
            raise ValueError(f"Unknown need: {name}")

        if delta > 0:
            self._needs_system.satisfy_need(name, delta)
        else:
            self._needs_system.deplete_need(name, abs(delta))

        logger.info(f"Dashboard adjusted need '{name}' by {delta}")

    async def request_behavior(self, behavior_name: str) -> None:
        """
        Request a specific behavior to be executed next (for dashboard control).

        The behavior will be started on the next cognition cycle if no behavior
        is currently running and the requested behavior exists.

        Args:
            behavior_name: Name of the behavior to trigger

        Raises:
            ValueError: If behavior name is invalid
        """
        behavior = self._evaluator.registry.get(behavior_name)
        if behavior is None:
            raise ValueError(f"Unknown behavior: {behavior_name}")

        self._requested_behavior = behavior_name
        logger.info(f"Dashboard requested behavior: {behavior_name}")

    def __str__(self) -> str:
        return (
            f"CognitionOrchestrator("
            f"running={self._running}, "
            f"pi_connected={self._pi_connected}, "
            f"behavior={self._executor.current_behavior_name})"
        )
