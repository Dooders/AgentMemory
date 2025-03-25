"""Unit tests for memory hooks implementation."""

import time
from unittest.mock import Mock, patch

import pytest

from agent_memory.api.hooks import ActionResult, BaseAgent, install_memory_hooks, with_memory
from agent_memory.config import MemoryConfig
from agent_memory.core import AgentMemorySystem


class TestAgent(BaseAgent):
    """Test agent class for hook testing."""

    def __init__(self, config=None):
        self.config = config or Mock()
        self.agent_id = "test_agent"
        self.step_number = 0
        
    def act(self, observation=None, **kwargs):
        """Test implementation of act method."""
        self.step_number += 1
        return ActionResult(action_type="test_action", params={}, reward=5.0)
        
    def get_state(self):
        """Test implementation of get_state method."""
        return {"health": 0.8, "reward": 5.0}


class TestMemoryHooks:
    """Test suite for memory hooks functionality."""

    @pytest.fixture
    def memory_config(self):
        """Create a test memory configuration."""
        return MemoryConfig(enable_memory_hooks=True)

    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock memory system."""
        with patch.object(AgentMemorySystem, "get_instance") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def test_agent(self, memory_config):
        """Create a test agent with memory config."""
        agent = TestAgent(config=Mock(memory_config=memory_config))
        return agent

    def test_install_memory_hooks_initialization(self, test_agent, mock_memory_system):
        """Test memory hooks installation during initialization."""
        decorated_agent = install_memory_hooks(TestAgent)(
            config=Mock(memory_config=MemoryConfig(enable_memory_hooks=True))
        )

        assert hasattr(decorated_agent, "memory_system")
        assert decorated_agent._memory_hooks_installed
        assert not decorated_agent._memory_recording

    def test_install_memory_hooks_disabled(self):
        """Test memory hooks are not installed when disabled in config."""
        config = Mock(memory_config=MemoryConfig(enable_memory_hooks=False))
        agent = install_memory_hooks(TestAgent)(config=config)

        assert not hasattr(agent, "memory_system")

    def test_act_with_memory_recording(self, test_agent, mock_memory_system):
        """Test act method with memory recording."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        result = decorated_agent.act()

        # Verify memory system calls
        mock_memory_system.store_agent_action.assert_called_once()
        call_args = mock_memory_system.store_agent_action.call_args[0]

        assert call_args[0] == "test_agent"  # agent_id
        assert call_args[1]["action_type"] == "test_action"
        assert call_args[1]["state_before"] == {"health": 0.8, "reward": 5.0}
        assert call_args[1]["state_after"] == {"health": 0.8, "reward": 5.0}

    def test_get_state_with_memory(self, test_agent, mock_memory_system):
        """Test get_state method with memory integration."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        state = decorated_agent.get_state()

        # Verify memory system calls when not recording
        mock_memory_system.store_agent_state.assert_called_once_with(
            "test_agent", {"health": 0.8, "reward": 5.0}, 0, priority=1.0
        )

    def test_calculate_state_difference(self, test_agent):
        """Test state difference calculation."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)

        state1 = {"health": 1.0, "reward": 0.0}
        state2 = {"health": 0.5, "reward": 10.0}

        diff = decorated_agent._calculate_state_difference(state1, state2)
        assert 0.0 <= diff <= 1.0

    def test_memory_error_handling(self, test_agent):
        """Test error handling in memory operations."""
        with patch.object(AgentMemorySystem, "get_instance") as mock:
            mock.side_effect = Exception("Test error")

            decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
            assert decorated_agent.memory_system is None

            # Should still work without memory system
            result = decorated_agent.act()
            assert result is not None

    def test_with_memory_decorator(self, test_agent, mock_memory_system):
        """Test with_memory instance decorator."""
        decorated_agent = with_memory(test_agent)

        assert hasattr(decorated_agent, "memory_system")
        assert decorated_agent.__class__.__name__.endswith("WithMemory")

    def test_duplicate_hook_installation(self, test_agent):
        """Test that hooks are not installed multiple times."""
        agent_class = type(test_agent)
        decorated_once = install_memory_hooks(agent_class)
        decorated_twice = install_memory_hooks(decorated_once)

        assert decorated_once == decorated_twice

    def test_importance_calculation(self, test_agent, mock_memory_system):
        """Test importance score calculation for state storage."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)

        # Test with high reward
        test_agent.get_state = lambda: {"health": 1.0, "reward": 10.0}
        state = decorated_agent.get_state()

        last_call = mock_memory_system.store_agent_state.call_args
        assert last_call[1]["priority"] > 0.5  # Higher importance due to high reward

        # Test with low health
        test_agent.get_state = lambda: {"health": 0.1, "reward": 0.0}
        state = decorated_agent.get_state()

        last_call = mock_memory_system.store_agent_state.call_args
        assert last_call[1]["priority"] > 0.5  # Higher importance due to low health

    def test_memory_recording_flag(self, test_agent, mock_memory_system):
        """Test memory recording flag management."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)

        assert not decorated_agent._memory_recording
        decorated_agent.act()
        assert not decorated_agent._memory_recording  # Should be reset after act

    def test_error_time_throttling(self, test_agent):
        """Test error logging throttling."""
        with patch.object(AgentMemorySystem, "get_instance") as mock:
            mock.side_effect = Exception("Test error")

            decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
            decorated_agent._memory_last_error_time = time.time()

            # Should not log error within 60 seconds
            with patch("logging.getLogger") as mock_logger:
                decorated_agent.act()
                mock_logger.assert_not_called()

    @pytest.mark.parametrize(
        "state_before,state_after,expected_range",
        [
            ({"val": 0}, {"val": 1}, (0.9, 1.0)),  # Large relative change
            ({"val": 100}, {"val": 101}, (0.0, 0.1)),  # Small relative change
            ({}, {}, (0.4, 0.6)),  # Empty states
            ({"val": 0}, {"val": 0}, (0.0, 0.1)),  # No change
            ({"val": -1}, {"val": 1}, (1.0, 1.0)),  # Sign change
            ({"val": 0.1}, {"val": 0.2}, (0.0, 0.2)),  # Small relative change with decimals
            ({"val": 1000}, {"val": 2000}, (0.9, 1.0)),  # Large absolute change
            ({"a": 1, "b": 2}, {"a": 2, "b": 3}, (0.7, 0.8)),  # Multiple values
            ({"a": 1}, {"b": 1}, (0.4, 0.6)),  # Different keys
            ({"val": "string"}, {"val": "string"}, (0.4, 0.6)),  # Non-numeric values
        ],
    )
    def test_state_difference_scenarios(
        self, test_agent, state_before, state_after, expected_range
    ):
        """Test state difference calculation with various scenarios."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        diff = decorated_agent._calculate_state_difference(state_before, state_after)

        assert expected_range[0] <= diff <= expected_range[1]

    @pytest.mark.parametrize(
        "state,expected_importance_range",
        [
            ({"health": 1.0, "reward": 0.0}, (0.5, 1.0)),  # Default case
            ({"health": 0.1, "reward": 0.0}, (0.8, 1.0)),  # Low health
            ({"health": 1.0, "reward": 10.0}, (0.8, 1.0)),  # High reward
            ({"health": 0.1, "reward": 10.0}, (0.8, 1.0)),  # Both low health and high reward
            ({"health": 0.5, "reward": 5.0}, (0.8, 1.0)),  # Moderate changes
            ({"health": 0.8, "reward": -5.0}, (0.8, 1.0)),  # Negative reward
            ({}, (0.5, 1.0)),  # Empty state
        ],
    )
    def test_importance_calculation_scenarios(
        self, test_agent, mock_memory_system, state, expected_importance_range
    ):
        """Test importance score calculation for different state scenarios."""
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        test_agent.get_state = lambda: state
        decorated_agent.get_state()

        last_call = mock_memory_system.store_agent_state.call_args
        assert expected_importance_range[0] <= last_call[1]["priority"] <= expected_importance_range[1]

    def test_with_memory_error_handling(self, test_agent):
        """Test error handling in with_memory decorator."""
        # Create a fresh agent to avoid interference from other tests
        fresh_agent = TestAgent(config=Mock(memory_config=MemoryConfig(enable_memory_hooks=True)))
        
        # Need to patch at module level to catch the decorator's call to get_instance
        with patch('agent_memory.api.hooks.AgentMemorySystem') as mock_cls:
            mock_cls.get_instance.side_effect = Exception("Test error")
            
            try:
                decorated_agent = with_memory(fresh_agent)
                # The implementation should handle errors gracefully
                assert decorated_agent.__class__.__name__.endswith("WithMemory")
            except Exception:
                # If we reach here, the error handling failed
                pytest.fail("with_memory should handle memory errors gracefully")

    def test_execution_time_recording(self, test_agent, mock_memory_system):
        """Test execution time recording in act method."""
        # Create a decorated agent
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        
        # Call the act method without any special timing
        decorated_agent.act()
        
        # Verify that an action was stored
        mock_memory_system.store_agent_action.assert_called_once()
        
        # Check that execution_time field exists in the stored data
        action_data = mock_memory_system.store_agent_action.call_args[0][1]
        assert "execution_time" in action_data
        assert isinstance(action_data["execution_time"], float)
        assert action_data["execution_time"] >= 0.0

    def test_step_number_tracking(self, test_agent, mock_memory_system):
        """Test step number tracking in state storage."""
        # Create decorated agent with our mocked memory system
        decorated_agent = install_memory_hooks(TestAgent)(test_agent.config)
        decorated_agent.step_number = 0  # Reset step number to start clean
        
        # Act once and check that step number is 1 in the call to store_agent_action
        decorated_agent.act()
        
        # The original should have incremented step_number
        assert decorated_agent.step_number == 1
        
        # Check the exact function arguments to store_agent_action
        mock_memory_system.store_agent_action.assert_called()
        
        # Use a simpler test approach: just check that step_number was passed to the memory system
        # through the action_data
        action_data = mock_memory_system.store_agent_action.call_args[0][1]
        assert action_data["step_number"] == 0  # The step_number before act was called
