"""
Unit tests for Murph's Behavior Tree Executor system.
"""

import time
import pytest
from server.cognition.behavior import (
    Behavior,
    WorldContext,
    BehaviorRegistry,
    BehaviorTreeFactory,
    BehaviorTreeExecutor,
    ExecutionState,
    ExecutionResult,
    MoveAction,
    TurnAction,
    PlaySoundAction,
    SetExpressionAction,
    WaitAction,
    ScanAction,
    StopAction,
    ActionNode,
)
from server.cognition.needs import NeedsSystem


class TestActionNodes:
    """Tests for action node classes."""

    def test_move_action_creation(self):
        """Test creating a MoveAction."""
        action = MoveAction("forward", speed=0.5, duration=2.0)
        assert "Move" in action.name
        assert action._direction == "forward"
        assert action._speed == 0.5
        assert action._duration == 2.0

    def test_move_action_invalid_direction(self):
        """Test MoveAction rejects invalid directions."""
        with pytest.raises(ValueError):
            MoveAction("up", speed=0.5, duration=1.0)

    def test_move_action_speed_clamping(self):
        """Test MoveAction clamps speed to valid range."""
        action = MoveAction("forward", speed=2.0, duration=1.0)
        assert action._speed == 1.0

        action = MoveAction("forward", speed=-0.5, duration=1.0)
        assert action._speed == 0.0

    def test_move_action_completes_after_duration(self):
        """Test MoveAction returns SUCCESS after duration."""
        action = MoveAction("forward", speed=0.5, duration=0.1)
        action.initialise()
        time.sleep(0.15)
        status = action.update()
        assert status.name == "SUCCESS"

    def test_turn_action_creation(self):
        """Test creating a TurnAction."""
        action = TurnAction(angle=90.0, speed=0.5)
        assert "Turn" in action.name
        assert action._angle == 90.0
        assert action._speed == 0.5

    def test_turn_action_negative_angle(self):
        """Test TurnAction with negative angle."""
        action = TurnAction(angle=-45.0, speed=0.5)
        assert action._angle == -45.0

    def test_play_sound_action(self):
        """Test PlaySoundAction."""
        action = PlaySoundAction("greeting")
        assert "PlaySound" in action.name
        assert action._sound_name == "greeting"
        assert action._duration == 1.0  # Default greeting duration

    def test_play_sound_unknown_uses_default(self):
        """Test PlaySoundAction uses default duration for unknown sounds."""
        action = PlaySoundAction("unknown_sound")
        assert action._duration == PlaySoundAction.DEFAULT_DURATION

    def test_set_expression_action_instant(self):
        """Test SetExpressionAction completes instantly."""
        action = SetExpressionAction("happy")
        action.initialise()
        status = action.update()
        assert status.name == "SUCCESS"

    def test_wait_action(self):
        """Test WaitAction waits for duration."""
        action = WaitAction(0.1)
        action.initialise()

        # Should be running initially
        status = action.update()
        assert status.name == "RUNNING"

        time.sleep(0.15)
        status = action.update()
        assert status.name == "SUCCESS"

    def test_scan_action_types(self):
        """Test ScanAction with different scan types."""
        full = ScanAction("full")
        assert full._duration == 4.0

        partial = ScanAction("partial")
        assert partial._duration == 2.0

        quick = ScanAction("quick")
        assert quick._duration == 1.0

    def test_stop_action_instant(self):
        """Test StopAction completes instantly."""
        action = StopAction()
        action.initialise()
        status = action.update()
        assert status.name == "SUCCESS"

    def test_action_callback_dispatch(self):
        """Test that actions dispatch callbacks."""
        dispatched = []

        def callback(name: str, params: dict) -> bool:
            dispatched.append((name, params))
            return True

        ActionNode.action_callback = callback

        action = MoveAction("forward", speed=0.5, duration=1.0)
        action.initialise()

        assert len(dispatched) == 1
        assert dispatched[0][1]["action"] == "move"
        assert dispatched[0][1]["direction"] == "forward"

        ActionNode.action_callback = None

    def test_action_get_params(self):
        """Test getting action parameters."""
        action = MoveAction("backward", speed=0.3, duration=2.0)
        params = action.get_params()

        assert params["action"] == "move"
        assert params["direction"] == "backward"
        assert params["speed"] == 0.3
        assert params["duration"] == 2.0


class TestBehaviorTreeFactory:
    """Tests for the BehaviorTreeFactory."""

    def test_factory_has_all_default_behaviors(self):
        """Test that factory has trees for all default behaviors."""
        registry = BehaviorRegistry()
        factory_trees = set(BehaviorTreeFactory.available_trees())

        # All behaviors should have trees
        for behavior in registry.get_all():
            assert behavior.name in factory_trees, f"Missing tree for {behavior.name}"

    def test_create_explore_tree(self):
        """Test creating explore behavior tree."""
        tree = BehaviorTreeFactory.create_tree("explore")
        assert tree is not None
        assert tree.name == "explore"

    def test_create_greet_tree(self):
        """Test creating greet behavior tree."""
        tree = BehaviorTreeFactory.create_tree("greet")
        assert tree is not None
        assert tree.name == "greet"

    def test_create_rest_tree(self):
        """Test creating rest behavior tree."""
        tree = BehaviorTreeFactory.create_tree("rest")
        assert tree is not None
        assert tree.name == "rest"

    def test_fallback_tree_for_unknown(self):
        """Test that unknown behaviors get fallback tree."""
        tree = BehaviorTreeFactory.create_tree("unknown_behavior_xyz")
        assert tree is not None
        assert "fallback" in tree.name

    def test_has_tree(self):
        """Test has_tree check."""
        assert BehaviorTreeFactory.has_tree("explore") is True
        assert BehaviorTreeFactory.has_tree("greet") is True
        assert BehaviorTreeFactory.has_tree("nonexistent") is False

    def test_available_trees_count(self):
        """Test that we have trees for all behaviors."""
        trees = BehaviorTreeFactory.available_trees()
        assert len(trees) == 26  # All default behaviors


class TestBehaviorTreeExecutor:
    """Tests for the BehaviorTreeExecutor."""

    def test_executor_creation(self):
        """Test creating an executor."""
        executor = BehaviorTreeExecutor()
        assert executor.is_running is False
        assert executor.current_behavior is None
        assert executor.execution_state == ExecutionState.IDLE

    def test_start_behavior(self):
        """Test starting a behavior."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="idle", display_name="Idle", duration_seconds=1.0)

        result = executor.start_behavior(behavior)

        assert result is True
        assert executor.is_running is True
        assert executor.current_behavior_name == "idle"
        assert executor.execution_state == ExecutionState.RUNNING

    def test_tick_returns_none_while_running(self):
        """Test that tick returns None while behavior is still running."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="idle", display_name="Idle", duration_seconds=10.0)

        executor.start_behavior(behavior)
        result = executor.tick()

        # Should still be running
        assert result is None
        assert executor.is_running is True

    def test_behavior_completes_by_duration(self):
        """Test that behavior completes after max duration."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="idle", display_name="Idle", duration_seconds=0.1)

        executor.start_behavior(behavior)
        time.sleep(0.15)
        result = executor.tick()

        assert result is not None
        assert result.succeeded() is True
        assert result.behavior_name == "idle"
        assert executor.is_running is False

    def test_interrupt_behavior(self):
        """Test interrupting a behavior."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(
            name="idle",
            display_name="Idle",
            duration_seconds=10.0,
            interruptible=True,
        )

        executor.start_behavior(behavior)
        result = executor.interrupt()

        assert result is not None
        assert result.was_interrupted() is True
        assert executor.is_running is False

    def test_cannot_interrupt_non_interruptible(self):
        """Test that non-interruptible behaviors cannot be interrupted."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(
            name="sleep",
            display_name="Sleep",
            duration_seconds=10.0,
            interruptible=False,
        )

        executor.start_behavior(behavior)
        result = executor.interrupt()

        # Should return None, behavior still running
        assert result is None
        assert executor.is_running is True

        # Cleanup
        executor.force_stop()

    def test_force_stop_non_interruptible(self):
        """Test force stopping a non-interruptible behavior."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(
            name="sleep",
            display_name="Sleep",
            duration_seconds=10.0,
            interruptible=False,
        )

        executor.start_behavior(behavior)
        result = executor.force_stop()

        assert result is not None
        assert result.was_interrupted() is True
        assert executor.is_running is False

    def test_start_new_interrupts_interruptible(self):
        """Test that starting new behavior interrupts current interruptible one."""
        executor = BehaviorTreeExecutor()
        behavior1 = Behavior(
            name="explore",
            display_name="Explore",
            duration_seconds=10.0,
            interruptible=True,
        )
        behavior2 = Behavior(
            name="greet",
            display_name="Greet",
            duration_seconds=5.0,
            interruptible=True,
        )

        executor.start_behavior(behavior1)
        assert executor.current_behavior_name == "explore"

        result = executor.start_behavior(behavior2)
        assert result is True
        assert executor.current_behavior_name == "greet"

    def test_start_blocked_by_non_interruptible(self):
        """Test that starting new behavior is blocked by non-interruptible."""
        executor = BehaviorTreeExecutor()
        behavior1 = Behavior(
            name="sleep",
            display_name="Sleep",
            duration_seconds=10.0,
            interruptible=False,
        )
        behavior2 = Behavior(
            name="greet",
            display_name="Greet",
            duration_seconds=5.0,
            interruptible=True,
        )

        executor.start_behavior(behavior1)
        result = executor.start_behavior(behavior2)

        # Should be blocked
        assert result is False
        assert executor.current_behavior_name == "sleep"

        # Cleanup
        executor.force_stop()

    def test_tick_count(self):
        """Test that tick count is tracked."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="idle", display_name="Idle", duration_seconds=10.0)

        executor.start_behavior(behavior)

        for _ in range(5):
            executor.tick()

        state = executor.get_state()
        assert state["tick_count"] == 5

        executor.force_stop()

    def test_elapsed_time_tracking(self):
        """Test that elapsed time is tracked."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="idle", display_name="Idle", duration_seconds=10.0)

        executor.start_behavior(behavior)
        time.sleep(0.1)

        assert executor.elapsed_time >= 0.1

        executor.force_stop()

    def test_update_context_during_execution(self):
        """Test updating world context during execution."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="explore", display_name="Explore", duration_seconds=10.0)

        context1 = WorldContext(person_detected=False)
        executor.start_behavior(behavior, context=context1)

        context2 = WorldContext(person_detected=True)
        executor.update_context(context2)

        # Should not raise
        executor.tick()

        executor.force_stop()

    def test_get_state(self):
        """Test getting executor state."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="greet", display_name="Greet", duration_seconds=5.0)

        executor.start_behavior(behavior)
        state = executor.get_state()

        assert state["execution_state"] == "RUNNING"
        assert state["current_behavior"] == "greet"
        assert "elapsed_time" in state
        assert "tick_count" in state

        executor.force_stop()

    def test_summary(self):
        """Test getting executor summary."""
        executor = BehaviorTreeExecutor()
        behavior = Behavior(name="play", display_name="Play", duration_seconds=5.0)

        executor.start_behavior(behavior)
        summary = executor.summary()

        assert summary["state"] == "RUNNING"
        assert summary["behavior"] == "play"

        executor.force_stop()


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_succeeded(self):
        """Test succeeded check."""
        result = ExecutionResult(
            behavior_name="test",
            state=ExecutionState.COMPLETED,
            duration=1.0,
            ticks=10,
        )
        assert result.succeeded() is True
        assert result.failed() is False
        assert result.was_interrupted() is False

    def test_failed(self):
        """Test failed check."""
        result = ExecutionResult(
            behavior_name="test",
            state=ExecutionState.FAILED,
            duration=1.0,
            ticks=5,
        )
        assert result.succeeded() is False
        assert result.failed() is True
        assert result.was_interrupted() is False

    def test_interrupted(self):
        """Test interrupted check."""
        result = ExecutionResult(
            behavior_name="test",
            state=ExecutionState.INTERRUPTED,
            duration=0.5,
            ticks=3,
        )
        assert result.succeeded() is False
        assert result.failed() is False
        assert result.was_interrupted() is True


class TestIntegration:
    """Integration tests for evaluator + executor flow."""

    def test_evaluator_to_executor_flow(self):
        """Test complete flow from evaluator to executor."""
        from server.cognition.behavior import BehaviorEvaluator

        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)
        executor = BehaviorTreeExecutor()

        # Get best behavior
        best = evaluator.select_best()
        assert best is not None

        # Start executing
        started = executor.start_behavior(best.behavior)
        assert started is True
        assert executor.is_running is True

        # Mark as used for cooldown
        evaluator.mark_behavior_used(best.behavior.name)

        executor.force_stop()

    def test_need_effects_applied_after_completion(self):
        """Test applying need effects after behavior completes."""
        from server.cognition.behavior import BehaviorEvaluator

        needs = NeedsSystem()
        needs.needs["curiosity"].value = 50.0

        evaluator = BehaviorEvaluator(needs)
        executor = BehaviorTreeExecutor()

        # Get explore behavior
        explore = evaluator.registry.get("explore")
        assert explore is not None

        # Execute with very short duration
        started = executor.start_behavior(explore, max_duration=0.05)
        assert started is True

        time.sleep(0.1)
        result = executor.tick()

        assert result is not None
        assert result.succeeded() is True

        # Apply need effects
        if result.succeeded():
            for need_name, effect in explore.need_effects.items():
                if effect > 0:
                    needs.satisfy_need(need_name, effect)
                else:
                    needs.deplete_need(need_name, abs(effect))

        # Curiosity should have increased
        assert needs.needs["curiosity"].value > 50.0

    def test_behavior_switching_scenario(self):
        """Test switching behaviors based on context changes."""
        from server.cognition.behavior import BehaviorEvaluator

        needs = NeedsSystem()
        evaluator = BehaviorEvaluator(needs)
        executor = BehaviorTreeExecutor()

        # Start with no person
        context1 = WorldContext(person_detected=False)
        best1 = evaluator.select_best(context1)
        executor.start_behavior(best1.behavior, context=context1)

        initial_behavior = executor.current_behavior_name

        # Person appears - switch behavior
        context2 = WorldContext(
            person_detected=True,
            person_is_familiar=True,
            person_distance=30.0,
        )
        best2 = evaluator.select_best(context2)

        # If it's a different behavior, switch
        if best2.behavior.name != initial_behavior:
            executor.start_behavior(best2.behavior, context=context2)

        executor.force_stop()


class TestActionCallback:
    """Tests for action callback mechanism."""

    def test_callback_receives_all_actions(self):
        """Test that callback receives all dispatched actions."""
        actions_received = []

        def callback(name: str, params: dict) -> bool:
            actions_received.append(params["action"])
            return True

        executor = BehaviorTreeExecutor(action_callback=callback)
        behavior = Behavior(name="greet", display_name="Greet", duration_seconds=0.1)

        executor.start_behavior(behavior)

        # Tick enough times to go through the sequence
        for _ in range(20):
            result = executor.tick()
            if result:
                break
            time.sleep(0.05)

        # Should have received some actions
        assert len(actions_received) > 0
        assert "set_expression" in actions_received

        # Cleanup
        ActionNode.action_callback = None
