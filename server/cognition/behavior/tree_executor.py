"""
Murph - Behavior Tree Executor
Manages behavior tree execution lifecycle.
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import py_trees
from py_trees.common import Status

from .behavior import Behavior
from .context import WorldContext
from .trees import BehaviorTreeFactory
from .actions import ActionNode


class ExecutionState(Enum):
    """Current state of the executor."""
    IDLE = auto()        # No behavior running
    RUNNING = auto()     # Behavior in progress
    COMPLETED = auto()   # Behavior finished successfully
    FAILED = auto()      # Behavior failed
    INTERRUPTED = auto()  # Behavior was interrupted


@dataclass
class ExecutionResult:
    """
    Result of a behavior execution.

    Attributes:
        behavior_name: Name of the behavior that was executed
        state: Final execution state
        duration: How long the behavior ran (seconds)
        ticks: Number of ticks executed
        final_status: Final py_trees status
    """
    behavior_name: str
    state: ExecutionState
    duration: float
    ticks: int
    final_status: Status | None = None

    def succeeded(self) -> bool:
        """Check if behavior completed successfully."""
        return self.state == ExecutionState.COMPLETED

    def failed(self) -> bool:
        """Check if behavior failed."""
        return self.state == ExecutionState.FAILED

    def was_interrupted(self) -> bool:
        """Check if behavior was interrupted."""
        return self.state == ExecutionState.INTERRUPTED


class BehaviorTreeExecutor:
    """
    Manages behavior tree execution.

    The executor:
    - Creates behavior trees from behavior definitions
    - Ticks the current tree each cognitive cycle
    - Handles behavior switching (interruption)
    - Tracks execution state and duration
    - Provides blackboard data to condition nodes

    Usage:
        executor = BehaviorTreeExecutor()
        executor.start_behavior(greet_behavior, context, needs_system)

        while True:
            result = executor.tick()
            if result is not None:
                # Behavior completed
                break
    """

    def __init__(
        self,
        action_callback: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            action_callback: Optional callback to dispatch actions to the robot.
                            Signature: callback(action_name: str, params: dict) -> bool
        """
        self._current_tree: py_trees.behaviour.Behaviour | None = None
        self._current_behavior: Behavior | None = None
        self._execution_state: ExecutionState = ExecutionState.IDLE

        self._start_time: float | None = None
        self._tick_count: int = 0
        self._max_duration: float | None = None

        # Blackboard for sharing data with tree nodes
        self._blackboard = py_trees.blackboard.Client(name="executor")
        self._blackboard.register_key(
            key="world_context",
            access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="needs_system",
            access=py_trees.common.Access.WRITE
        )

        # Set action callback on ActionNode class
        if action_callback:
            ActionNode.action_callback = action_callback

    @property
    def is_running(self) -> bool:
        """Check if a behavior is currently running."""
        return self._execution_state == ExecutionState.RUNNING

    @property
    def current_behavior(self) -> Behavior | None:
        """Get the currently executing behavior."""
        return self._current_behavior

    @property
    def current_behavior_name(self) -> str | None:
        """Get the name of the currently executing behavior."""
        return self._current_behavior.name if self._current_behavior else None

    @property
    def execution_state(self) -> ExecutionState:
        """Get the current execution state."""
        return self._execution_state

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since behavior started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start_behavior(
        self,
        behavior: Behavior,
        context: WorldContext | None = None,
        needs_system: Any = None,
        max_duration: float | None = None,
    ) -> bool:
        """
        Start executing a behavior.

        If another behavior is running, it will be interrupted if
        it is interruptible, otherwise this call fails.

        Args:
            behavior: The behavior to execute
            context: Current world context for conditions
            needs_system: NeedsSystem for need-based conditions
            max_duration: Override behavior's duration_seconds

        Returns:
            True if behavior started, False if blocked by non-interruptible behavior
        """
        # Check if we need to interrupt current behavior
        if self.is_running:
            if self._current_behavior and not self._current_behavior.interruptible:
                return False
            self._interrupt_current()

        # Create the behavior tree
        tree = BehaviorTreeFactory.create_tree(behavior.name)
        if tree is None:
            return False

        # Setup the tree
        tree.setup_with_descendants()

        # Update blackboard
        if context:
            self._blackboard.world_context = context
        if needs_system:
            self._blackboard.needs_system = needs_system

        # Initialize execution state
        self._current_tree = tree
        self._current_behavior = behavior
        self._execution_state = ExecutionState.RUNNING
        self._start_time = time.time()
        self._tick_count = 0
        self._max_duration = max_duration or behavior.duration_seconds

        return True

    def tick(self) -> ExecutionResult | None:
        """
        Execute one tick of the current behavior tree.

        Should be called each cognitive cycle (e.g., 10Hz).

        Returns:
            ExecutionResult if behavior completed/failed, None if still running
        """
        if not self.is_running or self._current_tree is None:
            return None

        self._tick_count += 1

        # Check duration limit
        if self._max_duration and self.elapsed_time >= self._max_duration:
            return self._complete(ExecutionState.COMPLETED)

        # Tick the tree
        self._current_tree.tick_once()
        status = self._current_tree.status

        # Check result
        if status == Status.SUCCESS:
            return self._complete(ExecutionState.COMPLETED)
        elif status == Status.FAILURE:
            return self._complete(ExecutionState.FAILED)

        # Still running
        return None

    def interrupt(self) -> ExecutionResult | None:
        """
        Interrupt the current behavior.

        Returns:
            ExecutionResult if behavior was interrupted, None if nothing running
        """
        if not self.is_running:
            return None

        if self._current_behavior and not self._current_behavior.interruptible:
            return None

        return self._interrupt_current()

    def force_stop(self) -> ExecutionResult | None:
        """
        Force stop the current behavior, even if non-interruptible.

        Use sparingly - for safety overrides.

        Returns:
            ExecutionResult if behavior was stopped, None if nothing running
        """
        if not self.is_running:
            return None

        return self._interrupt_current()

    def update_context(self, context: WorldContext) -> None:
        """
        Update the world context during execution.

        Allows condition nodes to react to changing world state.

        Args:
            context: Updated world context
        """
        self._blackboard.world_context = context

    def _complete(self, state: ExecutionState) -> ExecutionResult:
        """Complete execution with given state."""
        result = ExecutionResult(
            behavior_name=self._current_behavior.name if self._current_behavior else "unknown",
            state=state,
            duration=self.elapsed_time,
            ticks=self._tick_count,
            final_status=self._current_tree.status if self._current_tree else None,
        )

        self._cleanup()
        return result

    def _interrupt_current(self) -> ExecutionResult:
        """Interrupt and cleanup current behavior."""
        result = ExecutionResult(
            behavior_name=self._current_behavior.name if self._current_behavior else "unknown",
            state=ExecutionState.INTERRUPTED,
            duration=self.elapsed_time,
            ticks=self._tick_count,
            final_status=self._current_tree.status if self._current_tree else None,
        )

        self._cleanup()
        return result

    def _cleanup(self) -> None:
        """Clean up after execution ends."""
        if self._current_tree:
            # Shutdown the tree properly
            self._current_tree.shutdown()

        self._current_tree = None
        self._current_behavior = None
        self._execution_state = ExecutionState.IDLE
        self._start_time = None
        self._tick_count = 0
        self._max_duration = None

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for debugging."""
        return {
            "execution_state": self._execution_state.name,
            "current_behavior": self._current_behavior.name if self._current_behavior else None,
            "elapsed_time": self.elapsed_time,
            "tick_count": self._tick_count,
            "tree_status": self._current_tree.status.name if self._current_tree else None,
        }

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "state": self._execution_state.name,
            "behavior": self._current_behavior.name if self._current_behavior else None,
            "elapsed": round(self.elapsed_time, 2),
            "ticks": self._tick_count,
        }

    def __str__(self) -> str:
        if self._current_behavior:
            return (
                f"BehaviorTreeExecutor("
                f"running={self._current_behavior.name}, "
                f"elapsed={self.elapsed_time:.1f}s)"
            )
        return "BehaviorTreeExecutor(idle)"
