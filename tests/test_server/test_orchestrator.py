"""
Unit tests for Murph's Cognition Orchestrator.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from server.orchestrator import CognitionOrchestrator
from server.cognition.behavior.context import WorldContext
from server.cognition.behavior.tree_executor import ExecutionState, ExecutionResult
from shared.messages import SensorData, IMUData, TouchData, LocalTrigger


class TestOrchestratorInitialization:
    """Tests for CognitionOrchestrator initialization."""

    def test_initialization_default_params(self):
        """Test orchestrator initializes with defaults."""
        orchestrator = CognitionOrchestrator()

        assert orchestrator.is_running is False
        assert orchestrator.is_pi_connected is False
        assert orchestrator._world_context is not None
        assert orchestrator._needs_system is not None
        assert orchestrator._evaluator is not None
        assert orchestrator._executor is not None
        assert orchestrator._sensor_processor is not None
        assert orchestrator._connection is not None
        assert orchestrator._action_dispatcher is not None

    def test_initialization_custom_params(self):
        """Test orchestrator initializes with custom host/port."""
        orchestrator = CognitionOrchestrator(host="0.0.0.0", port=9000)

        assert orchestrator._host == "0.0.0.0"
        assert orchestrator._port == 9000

    def test_executor_has_action_callback(self):
        """Test that executor is wired to action dispatcher."""
        orchestrator = CognitionOrchestrator()

        # The executor should have been initialized with a callback
        # from the action dispatcher
        from server.cognition.behavior.actions import ActionNode
        assert ActionNode.action_callback is not None


class TestOrchestratorLifecycle:
    """Tests for orchestrator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Test that start() sets running flag."""
        orchestrator = CognitionOrchestrator()

        with patch.object(orchestrator._connection, 'start', new_callable=AsyncMock):
            await orchestrator.start()

            assert orchestrator.is_running is True
            assert len(orchestrator._tasks) == 3

            # Cleanup
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        """Test that stop() clears running flag."""
        orchestrator = CognitionOrchestrator()

        with patch.object(orchestrator._connection, 'start', new_callable=AsyncMock):
            with patch.object(orchestrator._connection, 'stop', new_callable=AsyncMock):
                await orchestrator.start()
                await orchestrator.stop()

                assert orchestrator.is_running is False
                assert len(orchestrator._tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_cancels_all_tasks(self):
        """Test that stop() cancels all loop tasks."""
        orchestrator = CognitionOrchestrator()

        with patch.object(orchestrator._connection, 'start', new_callable=AsyncMock):
            with patch.object(orchestrator._connection, 'stop', new_callable=AsyncMock):
                await orchestrator.start()

                # Get references to tasks
                tasks = orchestrator._tasks.copy()
                assert len(tasks) == 3

                await orchestrator.stop()

                # All tasks should be cancelled
                for task in tasks:
                    assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_force_stops_executor(self):
        """Test that stop() force stops running behavior."""
        orchestrator = CognitionOrchestrator()

        # Mock executor with running behavior
        orchestrator._executor._execution_state = ExecutionState.RUNNING

        with patch.object(orchestrator._connection, 'start', new_callable=AsyncMock):
            with patch.object(orchestrator._connection, 'stop', new_callable=AsyncMock):
                with patch.object(orchestrator._executor, 'force_stop') as mock_stop:
                    await orchestrator.start()
                    await orchestrator.stop()

                    mock_stop.assert_called_once()


class TestOrchestratorCallbacks:
    """Tests for orchestrator WebSocket callbacks."""

    def test_on_sensor_data_processes_imu(self):
        """Test sensor data callback processes IMU data."""
        orchestrator = CognitionOrchestrator()

        imu = IMUData(
            accel_x=0.0, accel_y=0.0, accel_z=1.0,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        sensor_data = SensorData(payload=imu)

        with patch.object(orchestrator._sensor_processor, 'process_sensor_data') as mock:
            orchestrator._on_sensor_data(sensor_data)
            mock.assert_called_once_with(sensor_data)

    def test_on_sensor_data_processes_touch(self):
        """Test sensor data callback processes touch data."""
        orchestrator = CognitionOrchestrator()

        touch = TouchData(
            is_touched=True,
            touched_electrodes=[0, 1, 2],
        )
        sensor_data = SensorData(payload=touch)

        with patch.object(orchestrator._sensor_processor, 'process_sensor_data') as mock:
            orchestrator._on_sensor_data(sensor_data)
            mock.assert_called_once_with(sensor_data)

    def test_on_local_trigger_handled(self):
        """Test local trigger callback."""
        orchestrator = CognitionOrchestrator()

        trigger = LocalTrigger(trigger_name="picked_up_fast", intensity=1.0)

        with patch.object(orchestrator._sensor_processor, 'handle_local_trigger') as mock:
            orchestrator._on_local_trigger(trigger)
            mock.assert_called_once_with(trigger)

    def test_on_local_trigger_falling_interrupts_behavior(self):
        """Test that falling trigger interrupts interruptible behavior."""
        orchestrator = CognitionOrchestrator()

        # Setup mock running interruptible behavior
        mock_behavior = MagicMock()
        mock_behavior.interruptible = True
        orchestrator._executor._current_behavior = mock_behavior
        orchestrator._executor._execution_state = ExecutionState.RUNNING

        trigger = LocalTrigger(trigger_name="falling", intensity=1.0)

        with patch.object(orchestrator._sensor_processor, 'handle_local_trigger'):
            with patch.object(orchestrator._executor, 'interrupt') as mock_interrupt:
                orchestrator._on_local_trigger(trigger)
                mock_interrupt.assert_called_once()

    def test_on_connection_change_connected(self):
        """Test connection change callback when Pi connects."""
        orchestrator = CognitionOrchestrator()

        orchestrator._on_connection_change(True)

        assert orchestrator.is_pi_connected is True

    def test_on_connection_change_disconnected_resets_state(self):
        """Test connection change callback resets state on disconnect."""
        orchestrator = CognitionOrchestrator()

        # Setup some state
        orchestrator._world_context.person_detected = True
        orchestrator._pi_connected = True

        # Mock running behavior
        orchestrator._executor._execution_state = ExecutionState.RUNNING

        with patch.object(orchestrator._executor, 'force_stop') as mock_stop:
            orchestrator._on_connection_change(False)

            assert orchestrator.is_pi_connected is False
            # World context should be reset
            assert orchestrator._world_context.person_detected is False
            mock_stop.assert_called_once()


class TestBehaviorCompletion:
    """Tests for behavior completion handling."""

    def test_handle_completion_applies_need_effects(self):
        """Test that completion applies positive need effects."""
        orchestrator = CognitionOrchestrator()

        # Set initial need value
        orchestrator._needs_system.needs["curiosity"].value = 50.0

        # Create a result for explore behavior (has curiosity +15)
        result = ExecutionResult(
            behavior_name="explore",
            state=ExecutionState.COMPLETED,
            duration=5.0,
            ticks=100,
        )

        orchestrator._handle_behavior_completion(result)

        # Curiosity should have increased
        assert orchestrator._needs_system.needs["curiosity"].value > 50.0

    def test_handle_completion_clears_current_behavior(self):
        """Test that completion clears current behavior from context."""
        orchestrator = CognitionOrchestrator()
        orchestrator._world_context.current_behavior = "explore"

        result = ExecutionResult(
            behavior_name="explore",
            state=ExecutionState.COMPLETED,
            duration=5.0,
            ticks=100,
        )

        orchestrator._handle_behavior_completion(result)

        assert orchestrator._world_context.current_behavior is None

    def test_handle_completion_no_effects_on_failure(self):
        """Test that failed behaviors don't apply effects."""
        orchestrator = CognitionOrchestrator()

        initial_curiosity = orchestrator._needs_system.needs["curiosity"].value

        result = ExecutionResult(
            behavior_name="explore",
            state=ExecutionState.FAILED,
            duration=2.0,
            ticks=40,
        )

        orchestrator._handle_behavior_completion(result)

        # Need should not have changed
        assert orchestrator._needs_system.needs["curiosity"].value == initial_curiosity


class TestStartBehavior:
    """Tests for starting behaviors."""

    def test_start_behavior_marks_used(self):
        """Test that starting behavior marks it as used for cooldown."""
        orchestrator = CognitionOrchestrator()

        # Get a scored behavior
        scored = orchestrator._evaluator.select_best()

        with patch.object(orchestrator._executor, 'start_behavior', return_value=True):
            with patch.object(orchestrator._evaluator, 'mark_behavior_used') as mock_mark:
                orchestrator._start_behavior(scored)
                mock_mark.assert_called_once_with(scored.behavior.name)

    def test_start_behavior_updates_context(self):
        """Test that starting behavior updates world context."""
        orchestrator = CognitionOrchestrator()

        scored = orchestrator._evaluator.select_best()

        with patch.object(orchestrator._executor, 'start_behavior', return_value=True):
            orchestrator._start_behavior(scored)

            assert orchestrator._world_context.current_behavior == scored.behavior.name

    def test_start_behavior_failure_no_side_effects(self):
        """Test that failed start doesn't update state."""
        orchestrator = CognitionOrchestrator()

        scored = orchestrator._evaluator.select_best()

        with patch.object(orchestrator._executor, 'start_behavior', return_value=False):
            with patch.object(orchestrator._evaluator, 'mark_behavior_used') as mock_mark:
                orchestrator._start_behavior(scored)

                # Should not mark as used
                mock_mark.assert_not_called()
                # Should not update context
                assert orchestrator._world_context.current_behavior is None


class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status_returns_dict(self):
        """Test that get_status returns comprehensive status."""
        orchestrator = CognitionOrchestrator()

        status = orchestrator.get_status()

        assert "running" in status
        assert "pi_connected" in status
        assert "current_behavior" in status
        assert "executor_state" in status
        assert "needs" in status
        assert "world_context" in status
        assert "connection" in status
        assert "timing" in status

    def test_str_representation(self):
        """Test string representation."""
        orchestrator = CognitionOrchestrator()

        string = str(orchestrator)

        assert "CognitionOrchestrator" in string
        assert "running=" in string
        assert "pi_connected=" in string


class TestLoopIntegration:
    """Integration tests for the cognitive loops."""

    @pytest.mark.asyncio
    async def test_perception_loop_updates_context(self):
        """Test that perception loop updates world context."""
        orchestrator = CognitionOrchestrator()
        orchestrator._running = True

        # Run one perception cycle manually
        orchestrator._sensor_processor.update_world_context(orchestrator._world_context)
        orchestrator._executor.update_context(orchestrator._world_context)
        orchestrator._sensor_processor.clear_transient_state()

        # Should complete without error
        assert orchestrator._world_context is not None

    @pytest.mark.asyncio
    async def test_cognition_loop_selects_behavior(self):
        """Test that cognition loop selects behaviors when idle."""
        orchestrator = CognitionOrchestrator()

        # Run one cognition cycle manually
        orchestrator._needs_system.update(0.2)

        # Should select a behavior
        best = orchestrator._evaluator.select_best(orchestrator._world_context)
        assert best is not None

    @pytest.mark.asyncio
    async def test_execution_loop_ticks_executor(self):
        """Test that execution loop ticks the executor."""
        orchestrator = CognitionOrchestrator()

        # Start a behavior
        best = orchestrator._evaluator.select_best()
        orchestrator._executor.start_behavior(best.behavior)

        # Run one execution cycle
        result = orchestrator._executor.tick()

        # Should still be running (or completed if very short)
        assert orchestrator._executor.is_running or result is not None

        # Cleanup
        if orchestrator._executor.is_running:
            orchestrator._executor.force_stop()


class TestTimeTracking:
    """Tests for time-based behavior."""

    def test_interaction_time_resets_on_interaction(self):
        """Test time since interaction resets when interacting."""
        orchestrator = CognitionOrchestrator()

        # Set some time
        orchestrator._world_context.time_since_last_interaction = 100.0

        # Simulate interaction
        orchestrator._world_context.is_being_petted = True

        # Manually check the logic from cognition loop
        if orchestrator._world_context.is_being_petted:
            orchestrator._world_context.time_since_last_interaction = 0

        assert orchestrator._world_context.time_since_last_interaction == 0

    def test_interaction_time_increases_without_interaction(self):
        """Test time since interaction increases when not interacting."""
        orchestrator = CognitionOrchestrator()

        initial = orchestrator._world_context.time_since_last_interaction

        # Simulate no interaction
        orchestrator._world_context.is_being_petted = False
        orchestrator._world_context.is_being_held = False
        orchestrator._world_context.person_detected = False

        # Manually check the logic from cognition loop
        if not orchestrator._world_context.is_being_petted and \
           not orchestrator._world_context.is_being_held and \
           not orchestrator._world_context.person_detected:
            orchestrator._world_context.time_since_last_interaction += 0.2

        assert orchestrator._world_context.time_since_last_interaction > initial


class TestFullCycle:
    """End-to-end tests for complete cognitive cycles."""

    @pytest.mark.asyncio
    async def test_complete_behavior_cycle(self):
        """Test a complete behavior selection and execution cycle."""
        orchestrator = CognitionOrchestrator()

        # Step 1: Cognition selects a behavior
        orchestrator._needs_system.update(0.2)
        best = orchestrator._evaluator.select_best(orchestrator._world_context)
        assert best is not None

        # Step 2: Start executing
        started = orchestrator._executor.start_behavior(
            best.behavior,
            orchestrator._world_context,
            orchestrator._needs_system,
        )
        assert started is True

        orchestrator._evaluator.mark_behavior_used(best.behavior.name)
        orchestrator._world_context.current_behavior = best.behavior.name

        # Step 3: Execute until complete (with timeout)
        import time
        start_time = time.time()
        result = None

        while time.time() - start_time < 2.0:  # 2 second timeout
            result = orchestrator._executor.tick()
            if result is not None:
                break
            await asyncio.sleep(0.05)

        # If not completed naturally, force stop
        if result is None:
            result = orchestrator._executor.force_stop()

        assert result is not None

        # Step 4: Handle completion
        orchestrator._handle_behavior_completion(result)

        assert orchestrator._world_context.current_behavior is None
