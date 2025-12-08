"""
Tests for Pi sensor mock implementations.
"""

import pytest
import asyncio

from pi.sensors import MockIMUSensor, MockTouchSensor
from shared.messages import IMUData, TouchData


class TestMockIMUSensor:
    """Tests for MockIMUSensor."""

    @pytest.fixture
    def imu(self):
        return MockIMUSensor()

    @pytest.mark.asyncio
    async def test_initialize(self, imu):
        result = await imu.initialize()
        assert result is True
        assert imu.is_ready()

    @pytest.mark.asyncio
    async def test_shutdown(self, imu):
        await imu.initialize()
        await imu.shutdown()
        assert not imu.is_ready()

    @pytest.mark.asyncio
    async def test_read(self, imu):
        await imu.initialize()
        data = await imu.read()
        assert isinstance(data, IMUData)
        # At rest, should be approximately 1g downward
        assert -1.1 < data.accel_z < -0.9

    @pytest.mark.asyncio
    async def test_streaming(self, imu):
        await imu.initialize()
        readings = []

        def callback(data):
            readings.append(data)

        await imu.start_streaming(callback, interval_ms=50)
        assert imu.is_streaming()

        await asyncio.sleep(0.2)  # Should get ~4 readings
        await imu.stop_streaming()

        assert not imu.is_streaming()
        assert len(readings) >= 3

    @pytest.mark.asyncio
    async def test_simulate_pickup(self, imu):
        await imu.initialize()
        imu.simulate_pickup(duration=0.3)

        data = await imu.read()
        # Should have elevated z acceleration
        assert data.accel_z > -0.5

    @pytest.mark.asyncio
    async def test_simulate_bump(self, imu):
        await imu.initialize()
        imu.simulate_bump(duration=0.3)

        data = await imu.read()
        # Should have elevated x acceleration
        assert data.accel_x > 1.5

    @pytest.mark.asyncio
    async def test_set_motion_state(self, imu):
        await imu.initialize()
        imu.set_motion_state(accel=(0.5, 0.5, 0.5))

        data = await imu.read()
        # Should be close to set values (with noise)
        assert 0.3 < data.accel_x < 0.7


class TestMockTouchSensor:
    """Tests for MockTouchSensor."""

    @pytest.fixture
    def touch(self):
        return MockTouchSensor()

    @pytest.mark.asyncio
    async def test_initialize(self, touch):
        result = await touch.initialize()
        assert result is True
        assert touch.is_ready()

    @pytest.mark.asyncio
    async def test_read_no_touch(self, touch):
        await touch.initialize()
        data = await touch.read()
        assert isinstance(data, TouchData)
        assert not data.is_touched
        assert data.touched_electrodes == []

    @pytest.mark.asyncio
    async def test_simulate_touch(self, touch):
        await touch.initialize()
        touch.simulate_touch([3, 4, 5])

        data = await touch.read()
        assert data.is_touched
        assert 4 in data.touched_electrodes

    @pytest.mark.asyncio
    async def test_simulate_release(self, touch):
        await touch.initialize()
        touch.simulate_touch([1, 2])
        touch.simulate_release()

        data = await touch.read()
        assert not data.is_touched

    @pytest.mark.asyncio
    async def test_simulate_pet(self, touch):
        await touch.initialize()
        touch.simulate_pet()

        data = await touch.read()
        assert data.is_touched
        assert len(data.touched_electrodes) >= 2

    @pytest.mark.asyncio
    async def test_electrode_validation(self, touch):
        await touch.initialize()
        # Should only accept electrodes 0-11
        touch.simulate_touch([0, 5, 11, 15, -1])

        data = await touch.read()
        # Should filter out invalid electrodes
        assert 15 not in data.touched_electrodes
        assert -1 not in data.touched_electrodes
        assert 5 in data.touched_electrodes

    @pytest.mark.asyncio
    async def test_streaming(self, touch):
        await touch.initialize()
        readings = []

        def callback(data):
            readings.append(data)

        await touch.start_streaming(callback, interval_ms=50)
        assert touch.is_streaming()

        touch.simulate_touch([1, 2, 3])
        await asyncio.sleep(0.2)

        await touch.stop_streaming()
        assert not touch.is_streaming()

        # At least some readings should show touch
        touched_readings = [r for r in readings if r.is_touched]
        assert len(touched_readings) > 0
