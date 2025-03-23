"""Unit tests for the Agent Memory System module.

This test suite covers the functionality of the AgentMemorySystem class, which serves
as the central manager for agent memory components.

The tests use pytest fixtures and mocks to isolate the AgentMemorySystem from its
dependencies for focused testing of the system's logic.

To run these tests:
    pytest tests/test_agent_memory_system.py

To run with coverage:
    pytest tests/test_agent_memory_system.py --cov=agent_memory.core

Test categories:
- TestAgentMemorySystemBasics: Tests for initialization and configuration
- TestAgentManagement: Tests for agent creation and management
- TestMemoryStorage: Tests for memory storage operations
- TestMemoryRetrieval: Tests for memory retrieval operations
- TestMemoryMaintenance: Tests for memory maintenance operations
- TestMemoryHooks: Tests for the event hook mechanism
"""

import unittest.mock as mock
import pytest

from agent_memory.config import MemoryConfig
from agent_memory.core import AgentMemorySystem
from agent_memory.memory_agent import MemoryAgent


@pytest.fixture
def mock_memory_agent():
    """Mock the MemoryAgent class."""
    agent = mock.MagicMock(spec=MemoryAgent)
    agent.store_state.return_value = True
    agent.store_interaction.return_value = True
    agent.store_action.return_value = True
    agent.retrieve_similar_states.return_value = []
    agent.retrieve_by_time_range.return_value = []
    agent.retrieve_by_attributes.return_value = []
    agent.search_by_embedding.return_value = []
    agent.search_by_content.return_value = []
    agent.get_memory_statistics.return_value = {"size": 1000}
    agent.force_maintenance.return_value = True
    agent.register_hook.return_value = True
    agent.trigger_event.return_value = True
    agent.clear_memory.return_value = True
    return agent


@pytest.fixture
def memory_system(mock_memory_agent):
    """Create an AgentMemorySystem with mocked MemoryAgent."""
    config = MemoryConfig()
    config.ltm_config.db_path = ":memory:"  # Use in-memory SQLite to avoid file system issues
    
    # Override singleton instance if it exists
    AgentMemorySystem._instance = None
    
    # Mock MemoryAgent creation
    with mock.patch("agent_memory.core.MemoryAgent", return_value=mock_memory_agent):
        system = AgentMemorySystem.get_instance(config)
    
    return system


class TestAgentMemorySystemBasics:
    """Tests for basic AgentMemorySystem functionality."""

    def test_init(self):
        """Test AgentMemorySystem initialization."""
        config = MemoryConfig()
        config.ltm_config.db_path = ":memory:"  # Use in-memory SQLite to avoid file system issues
        
        # Override singleton instance if it exists
        AgentMemorySystem._instance = None
        
        system = AgentMemorySystem(config)
        
        assert system.config == config
        assert isinstance(system.agents, dict)
        assert len(system.agents) == 0

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        config = MemoryConfig()
        config.ltm_config.db_path = ":memory:"  # Use in-memory SQLite to avoid file system issues
        
        # Override singleton instance if it exists
        AgentMemorySystem._instance = None
        
        system1 = AgentMemorySystem.get_instance(config)
        system2 = AgentMemorySystem.get_instance()
        
        assert system1 is system2
        assert id(system1) == id(system2)

    def test_get_instance_with_config(self):
        """Test get_instance with new config if no instance exists."""
        config = MemoryConfig()
        config.ltm_config.db_path = ":memory:"  # Use in-memory SQLite to avoid file system issues
        config.logging_level = "DEBUG"
        
        # Override singleton instance if it exists
        AgentMemorySystem._instance = None
        
        system = AgentMemorySystem.get_instance(config)
        
        assert system.config.logging_level == "DEBUG"


class TestAgentManagement:
    """Tests for agent creation and management."""

    def test_get_memory_agent_new(self, memory_system, mock_memory_agent):
        """Test getting a new memory agent."""
        agent_id = "test-agent-1"
        
        assert agent_id not in memory_system.agents
        
        # Mock the MemoryAgent constructor to return a specific agent
        with mock.patch("agent_memory.core.MemoryAgent", return_value=mock_memory_agent):
            agent = memory_system.get_memory_agent(agent_id)
        
        assert agent_id in memory_system.agents
        assert memory_system.agents[agent_id] is agent
        assert agent is mock_memory_agent

    def test_get_memory_agent_existing(self, memory_system, mock_memory_agent):
        """Test getting an existing memory agent."""
        agent_id = "test-agent-2"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        agent = memory_system.get_memory_agent(agent_id)
        
        assert agent is mock_memory_agent
        assert len(memory_system.agents) == 1


class TestMemoryStorage:
    """Tests for memory storage operations."""

    def test_store_agent_state(self, memory_system, mock_memory_agent):
        """Test storing agent state."""
        agent_id = "test-agent"
        state_data = {"location": "home", "energy": 100, "mood": "happy"}
        step_number = 42
        priority = 0.8
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.store_agent_state(
            agent_id, state_data, step_number, priority
        )
        
        assert result is True
        mock_memory_agent.store_state.assert_called_once_with(
            state_data, step_number, priority
        )

    def test_store_agent_interaction(self, memory_system, mock_memory_agent):
        """Test storing agent interaction."""
        agent_id = "test-agent"
        interaction_data = {
            "target_agent": "agent-2",
            "type": "conversation", 
            "content": "Hello!"
        }
        step_number = 42
        priority = 0.9
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.store_agent_interaction(
            agent_id, interaction_data, step_number, priority
        )
        
        assert result is True
        mock_memory_agent.store_interaction.assert_called_once_with(
            interaction_data, step_number, priority
        )

    def test_store_agent_action(self, memory_system, mock_memory_agent):
        """Test storing agent action."""
        agent_id = "test-agent"
        action_data = {
            "action_type": "move", 
            "direction": "north", 
            "speed": "fast"
        }
        step_number = 42
        priority = 0.7
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.store_agent_action(
            agent_id, action_data, step_number, priority
        )
        
        assert result is True
        mock_memory_agent.store_action.assert_called_once_with(
            action_data, step_number, priority
        )


class TestMemoryRetrieval:
    """Tests for memory retrieval operations."""

    def test_retrieve_similar_states(self, memory_system, mock_memory_agent):
        """Test retrieving similar states."""
        agent_id = "test-agent"
        query_state = {"location": "store", "energy": 50}
        k = 3
        memory_type = "state"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.retrieve_similar_states.return_value = [
            {"memory_id": "mem1", "contents": {"location": "store", "energy": 60}},
            {"memory_id": "mem2", "contents": {"location": "store", "energy": 70}},
        ]
        
        result = memory_system.retrieve_similar_states(
            agent_id, query_state, k, memory_type
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        mock_memory_agent.retrieve_similar_states.assert_called_once_with(
            query_state, k, memory_type
        )

    def test_retrieve_by_time_range(self, memory_system, mock_memory_agent):
        """Test retrieving memories by time range."""
        agent_id = "test-agent"
        start_step = 10
        end_step = 20
        memory_type = "action"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.retrieve_by_time_range.return_value = [
            {"memory_id": "mem1", "step_number": 12},
            {"memory_id": "mem2", "step_number": 15},
            {"memory_id": "mem3", "step_number": 18},
        ]
        
        result = memory_system.retrieve_by_time_range(
            agent_id, start_step, end_step, memory_type
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        mock_memory_agent.retrieve_by_time_range.assert_called_once_with(
            start_step, end_step, memory_type
        )

    def test_retrieve_by_attributes(self, memory_system, mock_memory_agent):
        """Test retrieving memories by attributes."""
        agent_id = "test-agent"
        attributes = {"location": "home", "mood": "happy"}
        memory_type = "state"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.retrieve_by_attributes.return_value = [
            {"memory_id": "mem1", "contents": {"location": "home", "mood": "happy", "energy": 80}},
        ]
        
        result = memory_system.retrieve_by_attributes(
            agent_id, attributes, memory_type
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        mock_memory_agent.retrieve_by_attributes.assert_called_once_with(
            attributes, memory_type
        )

    def test_get_memory_statistics(self, memory_system, mock_memory_agent):
        """Test getting memory statistics."""
        agent_id = "test-agent"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.get_memory_statistics.return_value = {
            "stm_count": 100,
            "im_count": 500,
            "ltm_count": 1000,
            "total_count": 1600,
            "total_size_kb": 1024,
        }
        
        result = memory_system.get_memory_statistics(agent_id)
        
        assert isinstance(result, dict)
        assert result["stm_count"] == 100
        assert result["total_count"] == 1600
        mock_memory_agent.get_memory_statistics.assert_called_once()


class TestMemoryMaintenance:
    """Tests for memory maintenance operations."""

    def test_force_memory_maintenance_single_agent(self, memory_system, mock_memory_agent):
        """Test forcing memory maintenance for a single agent."""
        agent_id = "test-agent"
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.force_memory_maintenance(agent_id)
        
        assert result is True
        mock_memory_agent.force_maintenance.assert_called_once()

    def test_force_memory_maintenance_all_agents(self, memory_system, mock_memory_agent):
        """Test forcing memory maintenance for all agents."""
        # Add multiple agents to the system
        memory_system.agents["agent1"] = mock_memory_agent
        memory_system.agents["agent2"] = mock.MagicMock(spec=MemoryAgent)
        memory_system.agents["agent2"].force_maintenance.return_value = True
        
        result = memory_system.force_memory_maintenance()
        
        assert result is True
        assert mock_memory_agent.force_maintenance.call_count == 1
        memory_system.agents["agent2"].force_maintenance.assert_called_once()

    def test_force_memory_maintenance_failure(self, memory_system, mock_memory_agent):
        """Test forcing memory maintenance with a failure."""
        # Add multiple agents to the system
        memory_system.agents["agent1"] = mock_memory_agent
        memory_system.agents["agent2"] = mock.MagicMock(spec=MemoryAgent)
        memory_system.agents["agent2"].force_maintenance.return_value = False
        
        result = memory_system.force_memory_maintenance()
        
        assert result is False
        assert mock_memory_agent.force_maintenance.call_count == 1
        memory_system.agents["agent2"].force_maintenance.assert_called_once()

    def test_search_by_embedding(self, memory_system, mock_memory_agent):
        """Test searching by embedding vector."""
        agent_id = "test-agent"
        embedding = [0.1, 0.2, 0.3, 0.4]
        k = 3
        memory_tiers = ["stm", "im"]
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.search_by_embedding.return_value = [
            {"memory_id": "mem1", "similarity": 0.95},
            {"memory_id": "mem2", "similarity": 0.85},
            {"memory_id": "mem3", "similarity": 0.75},
        ]
        
        result = memory_system.search_by_embedding(
            agent_id, embedding, k, memory_tiers
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        mock_memory_agent.search_by_embedding.assert_called_once_with(
            embedding, k, memory_tiers
        )

    def test_search_by_content(self, memory_system, mock_memory_agent):
        """Test searching by content."""
        agent_id = "test-agent"
        content_query = "find memories about the store"
        k = 5
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.search_by_content.return_value = [
            {"memory_id": "mem1", "relevance": 0.92},
            {"memory_id": "mem2", "relevance": 0.85},
        ]
        
        result = memory_system.search_by_content(
            agent_id, content_query, k
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        mock_memory_agent.search_by_content.assert_called_once_with(
            content_query, k
        )

    def test_clear_all_memories(self, memory_system, mock_memory_agent):
        """Test clearing all memories for all agents."""
        # Add multiple agents to the system
        memory_system.agents["agent1"] = mock_memory_agent
        mock_agent2 = mock.MagicMock(spec=MemoryAgent)
        mock_agent2.clear_memory.return_value = True
        memory_system.agents["agent2"] = mock_agent2
        
        # Store the mocks before clearing
        agent1_mock = memory_system.agents["agent1"]
        agent2_mock = memory_system.agents["agent2"]
        
        result = memory_system.clear_all_memories()
        
        assert result is True
        agent1_mock.clear_memory.assert_called_once()
        agent2_mock.clear_memory.assert_called_once()
        assert len(memory_system.agents) == 0

    def test_clear_all_memories_failure(self, memory_system, mock_memory_agent):
        """Test clearing all memories with a failure."""
        # Add multiple agents to the system
        memory_system.agents["agent1"] = mock_memory_agent
        mock_agent2 = mock.MagicMock(spec=MemoryAgent)
        mock_agent2.clear_memory.return_value = False
        memory_system.agents["agent2"] = mock_agent2
        
        # Store the mocks before clearing
        agent1_mock = memory_system.agents["agent1"]
        agent2_mock = memory_system.agents["agent2"]
        
        result = memory_system.clear_all_memories()
        
        assert result is False
        agent1_mock.clear_memory.assert_called_once()
        agent2_mock.clear_memory.assert_called_once()
        assert len(memory_system.agents) == 0


class TestMemoryHooks:
    """Tests for memory hook mechanism."""

    def test_register_memory_hook(self, memory_system, mock_memory_agent):
        """Test registering a memory hook."""
        agent_id = "test-agent"
        event_type = "memory_created"
        priority = 7
        
        def hook_function(event_data, agent):
            return True
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.register_hook.return_value = True
        
        result = memory_system.register_memory_hook(
            agent_id, event_type, hook_function, priority
        )
        
        assert result is True
        mock_memory_agent.register_hook.assert_called_once_with(
            event_type, hook_function, priority
        )

    def test_register_memory_hook_disabled(self, memory_system, mock_memory_agent):
        """Test registering a memory hook when hooks are disabled."""
        agent_id = "test-agent"
        event_type = "memory_created"
        priority = 7
        
        def hook_function(event_data, agent):
            return True
        
        # Disable memory hooks
        memory_system.config.enable_memory_hooks = False
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.register_memory_hook(
            agent_id, event_type, hook_function, priority
        )
        
        assert result is False
        mock_memory_agent.register_hook.assert_not_called()

    def test_trigger_memory_event(self, memory_system, mock_memory_agent):
        """Test triggering a memory event."""
        agent_id = "test-agent"
        event_type = "memory_accessed"
        event_data = {"memory_id": "mem1", "access_time": 1234567890}
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        mock_memory_agent.trigger_event.return_value = True
        
        result = memory_system.trigger_memory_event(
            agent_id, event_type, event_data
        )
        
        assert result is True
        mock_memory_agent.trigger_event.assert_called_once_with(
            event_type, event_data
        )

    def test_trigger_memory_event_disabled(self, memory_system, mock_memory_agent):
        """Test triggering a memory event when hooks are disabled."""
        agent_id = "test-agent"
        event_type = "memory_accessed"
        event_data = {"memory_id": "mem1", "access_time": 1234567890}
        
        # Disable memory hooks
        memory_system.config.enable_memory_hooks = False
        
        # Add agent to the system
        memory_system.agents[agent_id] = mock_memory_agent
        
        result = memory_system.trigger_memory_event(
            agent_id, event_type, event_data
        )
        
        assert result is False
        mock_memory_agent.trigger_event.assert_not_called()


def test_add_memory(memory_system, sample_memory):
    """Test adding a memory entry."""
    # Add a memory
    memory_id = memory_system.add_memory(sample_memory)

    # Verify it was assigned a memory_id if none was provided
    assert memory_id is not None

    # Retrieve and check
    retrieved = memory_system.get_memory(memory_id)
    assert retrieved["memory_id"] == memory_id
    assert retrieved["content"] == sample_memory["content"]
    assert retrieved["type"] == sample_memory["type"] 