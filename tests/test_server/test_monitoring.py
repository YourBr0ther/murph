"""Tests for the monitoring dashboard server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server.monitoring import MonitoringServer, NeedAdjustRequest, TriggerBehaviorRequest


class MockNeedsSystem:
    """Mock NeedsSystem for testing."""

    def __init__(self):
        self._needs = {
            "energy": MagicMock(value=75.0),
            "curiosity": MagicMock(value=50.0),
            "play": MagicMock(value=60.0),
            "social": MagicMock(value=45.0),
            "affection": MagicMock(value=80.0),
            "comfort": MagicMock(value=70.0),
            "safety": MagicMock(value=100.0),
        }

    def get_need(self, name):
        return self._needs.get(name)

    def satisfy_need(self, name, amount):
        pass

    def deplete_need(self, name, amount):
        pass

    def summary(self):
        return {
            "happiness": 65.0,
            "mood": "content",
            "critical_needs": [],
            "most_urgent": "social",
            "needs": {k: v.value for k, v in self._needs.items()},
        }


class MockBehavior:
    """Mock behavior for testing."""

    def __init__(self, name):
        self.name = name


class MockBehaviorRegistry:
    """Mock BehaviorRegistry for testing."""

    def __init__(self):
        self._behaviors = {
            "idle": MockBehavior("idle"),
            "explore": MockBehavior("explore"),
            "greet": MockBehavior("greet"),
        }

    def get(self, name):
        return self._behaviors.get(name)

    def get_all(self):
        return list(self._behaviors.values())


class MockScoredBehavior:
    """Mock ScoredBehavior for testing."""

    def __init__(self, name, score):
        self.behavior = MockBehavior(name)
        self.total_score = score
        self.base_value = 1.0
        self.need_modifier = score
        self.personality_modifier = 1.0
        self.opportunity_bonus = 1.0


class MockEvaluator:
    """Mock BehaviorEvaluator for testing."""

    def __init__(self):
        self.registry = MockBehaviorRegistry()

    def select_top_n(self, count, context=None):
        return [
            MockScoredBehavior("idle", 0.8),
            MockScoredBehavior("explore", 0.6),
            MockScoredBehavior("greet", 0.4),
        ][:count]


class MockExecutor:
    """Mock BehaviorTreeExecutor for testing."""

    def __init__(self):
        self.is_running = False
        self.current_behavior_name = None
        self.execution_state = MagicMock(name="IDLE")
        self.elapsed_time = 0.0


class MockWorldContext:
    """Mock WorldContext for testing."""

    def summary(self):
        return {
            "person": {"detected": False},
            "environment": {"is_dark": False, "is_loud": False, "near_edge": False},
            "physical": {"is_being_held": False, "is_being_petted": False},
            "spatial": {"current_zone_type": None, "at_home_base": False},
            "active_triggers": ["idle"],
        }


class MockConnection:
    """Mock PiConnectionManager for testing."""

    def get_status(self):
        return {"connected": True, "pending_commands": 0}


class MockOrchestrator:
    """Mock CognitionOrchestrator for testing."""

    def __init__(self):
        self._running = True
        self._pi_connected = True
        self._needs_system = MockNeedsSystem()
        self._evaluator = MockEvaluator()
        self._executor = MockExecutor()
        self._world_context = MockWorldContext()
        self._connection = MockConnection()
        self._behavior_history = []
        self._last_perception_time = 0.0
        self._last_cognition_time = 0.0
        self._last_execution_time = 0.0
        self._requested_behavior = None

    def get_status(self):
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

    def get_extended_status(self):
        status = self.get_status()
        status["elapsed_time"] = self._executor.elapsed_time
        status["behaviors"] = {
            "suggested": self.get_behavior_suggestions(count=10),
            "history": list(self._behavior_history),
        }
        status["available_behaviors"] = [
            b.name for b in self._evaluator.registry.get_all()
        ]
        return status

    def get_behavior_suggestions(self, count=10):
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

    def adjust_need(self, name, delta):
        need = self._needs_system.get_need(name)
        if need is None:
            raise ValueError(f"Unknown need: {name}")
        if delta > 0:
            self._needs_system.satisfy_need(name, delta)
        else:
            self._needs_system.deplete_need(name, abs(delta))

    async def request_behavior(self, behavior_name):
        behavior = self._evaluator.registry.get(behavior_name)
        if behavior is None:
            raise ValueError(f"Unknown behavior: {behavior_name}")
        self._requested_behavior = behavior_name


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    return MockOrchestrator()


@pytest.fixture
def monitoring_server(mock_orchestrator):
    """Create a monitoring server with mock orchestrator."""
    return MonitoringServer(mock_orchestrator)


@pytest.fixture
def client(monitoring_server):
    """Create a test client for the monitoring server."""
    return TestClient(monitoring_server.app)


class TestMonitoringServerEndpoints:
    """Test REST API endpoints."""

    def test_get_status(self, client):
        """Test GET /api/status returns orchestrator status."""
        response = client.get("/api/status")
        assert response.status_code == 200

        data = response.json()
        assert data["running"] is True
        assert data["pi_connected"] is True
        assert "needs" in data
        assert "world_context" in data
        assert "behaviors" in data
        assert "available_behaviors" in data

    def test_get_behaviors(self, client):
        """Test GET /api/behaviors returns behavior suggestions."""
        response = client.get("/api/behaviors")
        assert response.status_code == 200

        data = response.json()
        assert "behaviors" in data
        behaviors = data["behaviors"]
        assert len(behaviors) > 0
        assert behaviors[0]["name"] == "idle"
        assert "score" in behaviors[0]
        assert "breakdown" in behaviors[0]

    def test_adjust_need_positive(self, client, mock_orchestrator):
        """Test POST /api/control/need with positive delta."""
        response = client.post(
            "/api/control/need",
            json={"name": "energy", "delta": 10.0}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_adjust_need_negative(self, client, mock_orchestrator):
        """Test POST /api/control/need with negative delta."""
        response = client.post(
            "/api/control/need",
            json={"name": "curiosity", "delta": -5.0}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_adjust_need_invalid(self, client):
        """Test POST /api/control/need with invalid need name."""
        response = client.post(
            "/api/control/need",
            json={"name": "invalid_need", "delta": 10.0}
        )
        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert "Unknown need" in data["message"]

    def test_trigger_behavior_valid(self, client, mock_orchestrator):
        """Test POST /api/control/trigger with valid behavior."""
        response = client.post(
            "/api/control/trigger",
            json={"behavior_name": "explore"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert mock_orchestrator._requested_behavior == "explore"

    def test_trigger_behavior_invalid(self, client):
        """Test POST /api/control/trigger with invalid behavior."""
        response = client.post(
            "/api/control/trigger",
            json={"behavior_name": "invalid_behavior"}
        )
        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert "Unknown behavior" in data["message"]


class TestMonitoringServerBroadcast:
    """Test WebSocket broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_loop_runs(self, monitoring_server):
        """Test that broadcast loop runs and can be stopped."""
        # Start broadcast loop
        task = asyncio.create_task(monitoring_server.start_broadcast_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop it
        await monitoring_server.stop_broadcast_loop()

        # Task should complete
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_broadcast_behavior_change(self, monitoring_server):
        """Test behavior change broadcast creates correct message."""
        # Add a mock client
        mock_ws = AsyncMock()
        monitoring_server._clients.add(mock_ws)

        await monitoring_server.broadcast_behavior_change(
            previous="idle",
            current="explore",
            reason="curiosity_high"
        )

        # Verify message was sent
        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "behavior_change"
        assert call_args["data"]["previous"] == "idle"
        assert call_args["data"]["current"] == "explore"
        assert call_args["data"]["reason"] == "curiosity_high"


class TestNeedAdjustRequest:
    """Test NeedAdjustRequest model."""

    def test_valid_request(self):
        """Test creating a valid request."""
        request = NeedAdjustRequest(name="energy", delta=10.0)
        assert request.name == "energy"
        assert request.delta == 10.0

    def test_negative_delta(self):
        """Test request with negative delta."""
        request = NeedAdjustRequest(name="curiosity", delta=-5.0)
        assert request.delta == -5.0


class TestTriggerBehaviorRequest:
    """Test TriggerBehaviorRequest model."""

    def test_valid_request(self):
        """Test creating a valid request."""
        request = TriggerBehaviorRequest(behavior_name="explore")
        assert request.behavior_name == "explore"


class TestOrchestratorExtensions:
    """Test orchestrator extension methods."""

    def test_get_extended_status_includes_behaviors(self, mock_orchestrator):
        """Test get_extended_status includes behavior data."""
        status = mock_orchestrator.get_extended_status()

        assert "behaviors" in status
        assert "suggested" in status["behaviors"]
        assert "history" in status["behaviors"]
        assert "available_behaviors" in status
        assert "elapsed_time" in status

    def test_get_behavior_suggestions_returns_list(self, mock_orchestrator):
        """Test get_behavior_suggestions returns properly formatted list."""
        suggestions = mock_orchestrator.get_behavior_suggestions(count=3)

        assert len(suggestions) == 3
        assert suggestions[0]["name"] == "idle"
        assert "score" in suggestions[0]
        assert "breakdown" in suggestions[0]
        assert all(k in suggestions[0]["breakdown"] for k in ["base", "need", "personality", "opportunity"])

    def test_adjust_need_calls_system(self, mock_orchestrator):
        """Test adjust_need calls the needs system correctly."""
        # Test positive adjustment
        mock_orchestrator.adjust_need("energy", 10)

        # Test negative adjustment
        mock_orchestrator.adjust_need("curiosity", -5)

    def test_adjust_need_invalid_raises(self, mock_orchestrator):
        """Test adjust_need raises for invalid need."""
        with pytest.raises(ValueError, match="Unknown need"):
            mock_orchestrator.adjust_need("invalid", 10)

    @pytest.mark.asyncio
    async def test_request_behavior_sets_flag(self, mock_orchestrator):
        """Test request_behavior sets the requested behavior."""
        await mock_orchestrator.request_behavior("explore")
        assert mock_orchestrator._requested_behavior == "explore"

    @pytest.mark.asyncio
    async def test_request_behavior_invalid_raises(self, mock_orchestrator):
        """Test request_behavior raises for invalid behavior."""
        with pytest.raises(ValueError, match="Unknown behavior"):
            await mock_orchestrator.request_behavior("invalid_behavior")
