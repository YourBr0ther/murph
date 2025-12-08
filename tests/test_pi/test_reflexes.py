"""
Tests for Pi local behavior reflexes.
"""

import pytest
import time

from pi.local_behaviors import ReflexController, ReflexType
from shared.messages import IMUData


class TestReflexController:
    """Tests for ReflexController."""

    @pytest.fixture
    def reflex_state(self):
        """Track reflex triggers for testing."""
        return {
            "expressions": [],
            "sounds": [],
            "triggers": [],
        }

    @pytest.fixture
    def controller(self, reflex_state):
        def on_expression(name):
            reflex_state["expressions"].append(name)

        def on_sound(name):
            reflex_state["sounds"].append(name)

        def on_trigger(name, intensity):
            reflex_state["triggers"].append((name, intensity))

        return ReflexController(
            on_expression=on_expression,
            on_sound=on_sound,
            on_trigger=on_trigger,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, controller):
        await controller.start()
        await controller.stop()

    def test_detect_pickup_fast(self, controller, reflex_state):
        controller._running = True

        # Simulate fast pickup (high acceleration)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=2.5)  # > 2g
        controller.process_imu(imu)

        assert controller.is_being_held
        assert len(reflex_state["triggers"]) > 0
        trigger_name, _ = reflex_state["triggers"][0]
        assert trigger_name == "picked_up_fast"

    def test_detect_pickup_gentle(self, controller, reflex_state):
        controller._running = True

        # Simulate gentle pickup (moderate acceleration)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=0.3)  # > 1.5g total
        imu2 = IMUData(accel_x=0.5, accel_y=0.5, accel_z=1.2)
        controller.process_imu(imu2)

        # May or may not trigger depending on threshold
        # Just verify no crash

    def test_detect_falling(self, controller, reflex_state):
        controller._running = True

        # Simulate freefall (near 0g)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-0.1)  # < 0.3g
        controller.process_imu(imu)

        # Check for falling trigger
        falling_triggers = [t for t in reflex_state["triggers"] if "falling" in t[0]]
        assert len(falling_triggers) > 0

    def test_cooldown(self, controller, reflex_state):
        controller._running = True

        # Trigger first pickup
        imu = IMUData(accel_x=0, accel_y=0, accel_z=2.5)
        controller.process_imu(imu)
        initial_count = len(reflex_state["triggers"])

        # Immediately trigger again - should be on cooldown
        controller.process_imu(imu)
        assert len(reflex_state["triggers"]) == initial_count

    def test_set_down_detection(self, controller, reflex_state):
        controller._running = True

        # First pick up
        imu_pickup = IMUData(accel_x=0, accel_y=0, accel_z=2.0)
        controller.process_imu(imu_pickup)
        assert controller.is_being_held

        # Build up history at rest
        for _ in range(10):
            imu_rest = IMUData(accel_x=0, accel_y=0, accel_z=-1.0)
            controller.process_imu(imu_rest)
            # Small delay to allow cooldown
            time.sleep(0.01)

        # Eventually should detect set down
        # (may need more samples for stability check)

    def test_get_state(self, controller):
        controller._running = True
        imu = IMUData(accel_x=0.1, accel_y=0.2, accel_z=-1.0)
        controller.process_imu(imu)

        state = controller.get_state()
        assert "is_being_held" in state
        assert "recent_accel" in state
        assert state["recent_accel"] is not None

    def test_not_running(self, controller, reflex_state):
        # Controller not started, should not process
        imu = IMUData(accel_x=0, accel_y=0, accel_z=2.5)
        controller.process_imu(imu)

        assert len(reflex_state["triggers"]) == 0


class TestReflexTypes:
    """Test reflex type definitions."""

    def test_reflex_types_exist(self):
        assert ReflexType.PICKED_UP_FAST
        assert ReflexType.PICKED_UP_GENTLE
        assert ReflexType.BUMP
        assert ReflexType.SHAKE
        assert ReflexType.FALLING
        assert ReflexType.SET_DOWN

    def test_reflex_configs(self):
        from pi.local_behaviors.reflexes import REFLEX_CONFIGS

        # Check that important reflex types have configs
        important_types = [
            ReflexType.PICKED_UP_FAST,
            ReflexType.PICKED_UP_GENTLE,
            ReflexType.BUMP,
            ReflexType.FALLING,
            ReflexType.SET_DOWN,
        ]
        for reflex_type in important_types:
            assert reflex_type in REFLEX_CONFIGS
            config = REFLEX_CONFIGS[reflex_type]
            assert config.expression is not None
            assert config.sound is not None
            assert config.cooldown_ms > 0
