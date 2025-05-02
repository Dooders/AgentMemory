"""Unit tests for memory API models."""

import pytest
from memory.api.models import AgentState, ActionData, ActionResult


class TestAgentState:
    """Test suite for the AgentState class."""

    def test_initialization(self):
        """Test basic initialization of AgentState."""
        state = AgentState(
            agent_id="agent-1",
            step_number=42,
            health=0.8,
            reward=10.5,
            position_x=1.0,
            position_y=2.0,
            position_z=3.0,
            resource_level=0.7,
            extra_data={"inventory": ["sword", "shield"]}
        )
        
        assert state.agent_id == "agent-1"
        assert state.step_number == 42
        assert state.health == 0.8
        assert state.reward == 10.5
        assert state.position_x == 1.0
        assert state.position_y == 2.0
        assert state.position_z == 3.0
        assert state.resource_level == 0.7
        assert state.extra_data == {"inventory": ["sword", "shield"]}

    def test_initialization_minimal(self):
        """Test initialization with only required fields."""
        state = AgentState(agent_id="agent-1", step_number=10)
        
        assert state.agent_id == "agent-1"
        assert state.step_number == 10
        assert state.health is None
        assert state.reward is None
        assert state.position_x is None
        assert state.position_y is None
        assert state.position_z is None
        assert state.resource_level is None
        assert state.extra_data == {}

    def test_as_dict_with_all_fields(self):
        """Test as_dict method with all fields populated."""
        state = AgentState(
            agent_id="agent-1",
            step_number=42,
            health=0.8,
            reward=10.5,
            position_x=1.0,
            position_y=2.0,
            position_z=3.0,
            resource_level=0.7,
            extra_data={"inventory": ["sword", "shield"]}
        )
        
        state_dict = state.as_dict()
        
        assert state_dict["agent_id"] == "agent-1"
        assert state_dict["step_number"] == 42
        assert state_dict["health"] == 0.8
        assert state_dict["reward"] == 10.5
        assert state_dict["position_x"] == 1.0
        assert state_dict["position_y"] == 2.0
        assert state_dict["position_z"] == 3.0
        assert state_dict["resource_level"] == 0.7
        assert state_dict["extra_data"] == {"inventory": ["sword", "shield"]}

    def test_as_dict_with_none_values(self):
        """Test as_dict method excludes None values."""
        state = AgentState(
            agent_id="agent-1",
            step_number=42,
            health=None,
            reward=None,
            position_x=None,
            position_y=None,
            position_z=None,
            resource_level=None
        )
        
        state_dict = state.as_dict()
        
        assert "agent_id" in state_dict
        assert "step_number" in state_dict
        assert "health" not in state_dict
        assert "reward" not in state_dict
        assert "position_x" not in state_dict
        assert "position_y" not in state_dict
        assert "position_z" not in state_dict
        assert "resource_level" not in state_dict
        assert "extra_data" not in state_dict

    def test_as_dict_with_empty_extra_data(self):
        """Test as_dict method excludes empty extra_data."""
        state = AgentState(
            agent_id="agent-1",
            step_number=42,
            health=0.8,
            extra_data={}
        )
        
        state_dict = state.as_dict()
        
        assert "agent_id" in state_dict
        assert "step_number" in state_dict
        assert "health" in state_dict
        assert "extra_data" not in state_dict


class TestActionData:
    """Test suite for the ActionData class."""

    def test_initialization(self):
        """Test basic initialization of ActionData."""
        action_data = ActionData(
            action_type="move",
            action_params={"direction": "north", "distance": 2},
            state_before={"position": [0, 0], "health": 1.0},
            state_after={"position": [0, 2], "health": 0.9},
            reward=5.0,
            execution_time=0.25,
            step_number=42
        )
        
        assert action_data.action_type == "move"
        assert action_data.action_params == {"direction": "north", "distance": 2}
        assert action_data.state_before == {"position": [0, 0], "health": 1.0}
        assert action_data.state_after == {"position": [0, 2], "health": 0.9}
        assert action_data.reward == 5.0
        assert action_data.execution_time == 0.25
        assert action_data.step_number == 42

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        action_data = ActionData(
            action_type="move",
            state_before={"position": [0, 0], "health": 1.0},
            state_after={"position": [0, 2], "health": 0.9},
            execution_time=0.25,
            step_number=42
        )
        
        assert action_data.action_type == "move"
        assert action_data.action_params == {}
        assert action_data.reward == 0.0

    def test_get_state_difference_numeric(self):
        """Test get_state_difference method with numeric values."""
        action_data = ActionData(
            action_type="move",
            state_before={"health": 1.0, "position_x": 10, "energy": 100},
            state_after={"health": 0.8, "position_x": 15, "energy": 90},
            execution_time=0.25,
            step_number=42
        )
        
        diff = action_data.get_state_difference()
        
        assert "health" in diff
        assert "position_x" in diff
        assert "energy" in diff
        assert pytest.approx(diff["health"]) == -0.2
        assert diff["position_x"] == 5
        assert diff["energy"] == -10

    def test_get_state_difference_non_numeric(self):
        """Test get_state_difference method ignores non-numeric values."""
        action_data = ActionData(
            action_type="move",
            state_before={
                "health": 1.0, 
                "position": [0, 0], 
                "inventory": ["sword"]
            },
            state_after={
                "health": 0.8, 
                "position": [0, 2], 
                "inventory": ["sword", "shield"]
            },
            execution_time=0.25,
            step_number=42
        )
        
        diff = action_data.get_state_difference()
        
        assert "health" in diff
        assert pytest.approx(diff["health"]) == -0.2
        assert "position" not in diff
        assert "inventory" not in diff

    def test_get_state_difference_missing_keys(self):
        """Test get_state_difference method with keys present in only one state."""
        action_data = ActionData(
            action_type="move",
            state_before={"health": 1.0, "energy": 100},
            state_after={"health": 0.8, "mana": 50},
            execution_time=0.25,
            step_number=42
        )
        
        diff = action_data.get_state_difference()
        
        assert "health" in diff
        assert pytest.approx(diff["health"]) == -0.2
        assert "energy" not in diff
        assert "mana" not in diff


class TestActionResult:
    """Test suite for the ActionResult class."""

    def test_initialization(self):
        """Test basic initialization of ActionResult."""
        result = ActionResult(
            action_type="attack",
            params={"target": "enemy-1", "weapon": "sword"},
            reward=10.0
        )
        
        assert result.action_type == "attack"
        assert result.params == {"target": "enemy-1", "weapon": "sword"}
        assert result.reward == 10.0

    def test_initialization_default_values(self):
        """Test initialization with default values."""
        result = ActionResult(action_type="observe")
        
        assert result.action_type == "observe"
        assert result.params == {}
        assert result.reward == 0.0 