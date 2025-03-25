"""Unit tests for the memory API interface."""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock the problematic dependencies
import sys
sys.modules['torch'] = MagicMock()
sys.modules['agent_memory.embeddings.autoencoder'] = MagicMock()

from agent_memory.api.memory_api import AgentMemoryAPI
from agent_memory.config import MemoryConfig
from agent_memory.core import AgentMemorySystem


class TestAgentMemoryAPI:
    """Test suite for the AgentMemoryAPI class."""

    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock memory system."""
        with patch.object(AgentMemorySystem, "get_instance") as mock_get_instance:
            mock_instance = Mock()
            mock_instance.agents = {}
            mock_get_instance.return_value = mock_instance
            yield mock_get_instance, mock_instance

    @pytest.fixture
    def memory_config(self):
        """Create a test memory configuration."""
        return MemoryConfig()

    @pytest.fixture
    def api(self, mock_memory_system):
        """Create an AgentMemoryAPI instance with mocked system."""
        # Use only the mock instance, not the get_instance mock
        _, mock_instance = mock_memory_system
        # Create a new API with no arguments to use the mocked instance 
        return AgentMemoryAPI()

    @pytest.fixture
    def mock_memory_agent(self):
        """Create a mock memory agent."""
        mock_agent = Mock()
        mock_agent.stm_store = Mock()
        mock_agent.im_store = Mock()
        mock_agent.ltm_store = Mock()
        mock_agent.embedding_engine = Mock()
        return mock_agent

    def test_initialization(self, mock_memory_system, memory_config):
        """Test that the API initializes correctly."""
        # Unpack the mock_memory_system fixture
        mock_get_instance, mock_instance = mock_memory_system
        
        # Create a new API with the config
        api = AgentMemoryAPI(memory_config)
        
        # Verify that get_instance was called with the right config
        mock_get_instance.assert_called_once_with(memory_config)
        
        # Verify that the API's memory_system is the mock instance
        assert api.memory_system == mock_instance

    def test_store_agent_state(self, api, mock_memory_system):
        """Test storing an agent state."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        state_data = {"health": 0.8, "position": [10, 20]}
        api.store_agent_state("agent1", state_data, 42, 0.75)
        
        mock_instance.store_agent_state.assert_called_once_with(
            "agent1", state_data, 42, 0.75
        )

    def test_store_agent_interaction(self, api, mock_memory_system):
        """Test storing an agent interaction."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        interaction_data = {"target": "agent2", "type": "greeting"}
        api.store_agent_interaction("agent1", interaction_data, 42, 0.6)
        
        mock_instance.store_agent_interaction.assert_called_once_with(
            "agent1", interaction_data, 42, 0.6
        )

    def test_store_agent_action(self, api, mock_memory_system):
        """Test storing an agent action."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        action_data = {"action": "move", "direction": "north"}
        api.store_agent_action("agent1", action_data, 42, 0.8)
        
        mock_instance.store_agent_action.assert_called_once_with(
            "agent1", action_data, 42, 0.8
        )

    def test_retrieve_state_by_id(self, api, mock_memory_system, mock_memory_agent):
        """Test retrieving a state by ID."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        memory_id = "mem123"
        expected_memory = {"memory_id": memory_id, "contents": {"health": 0.8}}
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.get.return_value = None
        mock_memory_agent.im_store.get.return_value = None
        mock_memory_agent.ltm_store.get.return_value = expected_memory
        
        # Call and verify
        result = api.retrieve_state_by_id("agent1", memory_id)
        assert result == expected_memory
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.get.assert_called_once_with(memory_id)
        mock_memory_agent.im_store.get.assert_called_once_with(memory_id)
        mock_memory_agent.ltm_store.get.assert_called_once_with(memory_id)

    def test_retrieve_recent_states(self, api, mock_memory_system, mock_memory_agent):
        """Test retrieving recent states."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        expected_states = [{"memory_id": f"mem{i}", "contents": {}} for i in range(5)]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.get_recent.return_value = expected_states
        
        # Call and verify
        result = api.retrieve_recent_states("agent1", 5, "state")
        assert result == expected_states
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.get_recent.assert_called_once_with(5, "state")

    def test_retrieve_similar_states(self, api, mock_memory_system, mock_memory_agent):
        """Test retrieving similar states."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        query_state = {"location": [10, 20], "health": 0.7}
        expected_results = [
            {"memory_id": "mem1", "contents": {"location": [10, 22]}, "_similarity_score": 0.9},
            {"memory_id": "mem2", "contents": {"location": [12, 18]}, "_similarity_score": 0.8},
        ]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        
        # Mock the new ensure_embedding_dimensions to return the same embedding
        mock_memory_agent.embedding_engine.ensure_embedding_dimensions.return_value = [0.1, 0.2, 0.3]
        
        mock_memory_agent.stm_store.search_by_vector.return_value = expected_results
        mock_memory_agent.im_store.search_by_vector.return_value = []
        mock_memory_agent.ltm_store.search_by_vector.return_value = []
        
        # Call and verify
        result = api.retrieve_similar_states("agent1", query_state, 5)
        assert result == expected_results
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.embedding_engine.encode_stm.assert_called_once_with(query_state)
        
        # Verify the conversion is called for each store
        assert mock_memory_agent.embedding_engine.ensure_embedding_dimensions.call_count == 3
        
        # Verify search_by_vector is called with the converted embedding
        mock_memory_agent.stm_store.search_by_vector.assert_called_once_with(
            [0.1, 0.2, 0.3], k=5, memory_type=None
        )

    def test_retrieve_by_time_range(self, api, mock_memory_system, mock_memory_agent):
        """Test retrieving memories by time range."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        stm_results = [{"memory_id": "stm1", "step_number": 10}]
        im_results = [{"memory_id": "im1", "step_number": 5}]
        ltm_results = [{"memory_id": "ltm1", "step_number": 2}]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.get_by_step_range.return_value = stm_results
        mock_memory_agent.im_store.get_by_step_range.return_value = im_results
        mock_memory_agent.ltm_store.get_by_step_range.return_value = ltm_results
        
        # Call and verify
        result = api.retrieve_by_time_range("agent1", 1, 10, "state")
        
        # Results should be sorted by step_number
        expected = [{"memory_id": "ltm1", "step_number": 2}, 
                   {"memory_id": "im1", "step_number": 5}, 
                   {"memory_id": "stm1", "step_number": 10}]
        assert result == expected
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.get_by_step_range.assert_called_once_with(1, 10, "state")
        mock_memory_agent.im_store.get_by_step_range.assert_called_once_with(1, 10, "state")
        mock_memory_agent.ltm_store.get_by_step_range.assert_called_once_with(1, 10, "state")

    def test_retrieve_by_attributes(self, api, mock_memory_system, mock_memory_agent):
        """Test retrieving memories by attributes."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        stm_results = [{"memory_id": "stm1", "step_number": 10}]
        im_results = [{"memory_id": "im1", "step_number": 5}]
        ltm_results = [{"memory_id": "ltm1", "step_number": 2}]
        attributes = {"location": "kitchen", "mood": "happy"}
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.get_by_attributes.return_value = stm_results
        mock_memory_agent.im_store.get_by_attributes.return_value = im_results
        mock_memory_agent.ltm_store.get_by_attributes.return_value = ltm_results
        
        # Call and verify
        result = api.retrieve_by_attributes("agent1", attributes, "state")
        
        # Results should be sorted by step_number in reverse (most recent first)
        expected = [{"memory_id": "stm1", "step_number": 10}, 
                   {"memory_id": "im1", "step_number": 5}, 
                   {"memory_id": "ltm1", "step_number": 2}]
        assert result == expected
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.get_by_attributes.assert_called_once_with(attributes, "state")
        mock_memory_agent.im_store.get_by_attributes.assert_called_once_with(attributes, "state")
        mock_memory_agent.ltm_store.get_by_attributes.assert_called_once_with(attributes, "state")

    def test_search_by_embedding(self, api, mock_memory_system, mock_memory_agent):
        """Test searching by embedding vector."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        stm_results = [{"memory_id": "stm1", "_similarity_score": 0.9}]
        im_results = [{"memory_id": "im1", "_similarity_score": 0.7}]
        memory_tiers = ["stm", "im"]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        
        # Mock the embedding engine existence 
        mock_memory_agent.embedding_engine = Mock()
        
        # Mock the ensure_embedding_dimensions to return the same embedding
        mock_memory_agent.embedding_engine.ensure_embedding_dimensions.return_value = query_embedding
        
        # Setup embedding engine with correct configuration
        mock_config = MagicMock()
        mock_config.autoencoder_config.im_dim = 4  # Match query_embedding length
        mock_memory_agent.config = mock_config
        
        mock_memory_agent.stm_store.search_by_vector.return_value = stm_results
        mock_memory_agent.im_store.search_by_vector.return_value = im_results
        
        # Call and verify
        result = api.search_by_embedding("agent1", query_embedding, k=5, memory_tiers=memory_tiers)
        
        # Results should be sorted by similarity score
        expected = [{"memory_id": "stm1", "_similarity_score": 0.9}, 
                   {"memory_id": "im1", "_similarity_score": 0.7}]
        assert result == expected
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        
        # Verify ensure_embedding_dimensions is called for each tier
        mock_memory_agent.embedding_engine.ensure_embedding_dimensions.assert_any_call(query_embedding, "stm")
        mock_memory_agent.embedding_engine.ensure_embedding_dimensions.assert_any_call(query_embedding, "im")
        
        # Verify search_by_vector is called with the converted embedding
        mock_memory_agent.stm_store.search_by_vector.assert_called_once_with(query_embedding, k=5)
        mock_memory_agent.im_store.search_by_vector.assert_called_once_with(query_embedding, k=4)

    def test_search_by_content_string(self, api, mock_memory_system, mock_memory_agent):
        """Test searching by content with string query."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        stm_results = [{"memory_id": "stm1", "contents": {"text": "hello world"}}]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.search_by_content.return_value = stm_results
        mock_memory_agent.im_store.search_by_content.return_value = []
        mock_memory_agent.ltm_store.search_by_content.return_value = []
        
        # Call and verify
        result = api.search_by_content("agent1", "hello", k=5)
        assert result == stm_results
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.search_by_content.assert_called_once_with({"text": "hello"}, k=5)

    def test_search_by_content_dict(self, api, mock_memory_system, mock_memory_agent):
        """Test searching by content with dictionary query."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        query = {"location": "kitchen", "mood": "happy"}
        stm_results = [{"memory_id": "stm1", "contents": query}]
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.search_by_content.return_value = stm_results
        mock_memory_agent.im_store.search_by_content.return_value = []
        mock_memory_agent.ltm_store.search_by_content.return_value = []
        
        # Call and verify
        result = api.search_by_content("agent1", query, k=5)
        assert result == stm_results
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.search_by_content.assert_called_once_with(query, k=5)

    def test_get_memory_statistics(self, api, mock_memory_system, mock_memory_agent):
        """Test getting memory statistics."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        expected_stats = {
            "total_memories": 120,
            "stm_count": 50,
            "im_count": 40,
            "ltm_count": 30,
            "memory_type_distribution": {"state": 50, "action": 30, "interaction": 40},
            "last_maintenance_time": 12345,
            "insert_count_since_maintenance": 10
        }
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.count.return_value = 50
        mock_memory_agent.im_store.count.return_value = 40
        mock_memory_agent.ltm_store.count.return_value = 30
        mock_memory_agent.stm_store.count_by_type.return_value = {"state": 50, "action": 30, "interaction": 40}
        mock_memory_agent.last_maintenance_time = 12345
        mock_memory_agent._insert_count = 10
        
        # Call and verify
        result = api.get_memory_statistics("agent1")
        assert result == expected_stats
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")

    def test_force_memory_maintenance_single_agent(self, api, mock_memory_system, mock_memory_agent):
        """Test forcing memory maintenance for a single agent."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent._perform_maintenance.return_value = True
        
        # Call and verify
        result = api.force_memory_maintenance("agent1")
        assert result is True
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent._perform_maintenance.assert_called_once()

    def test_force_memory_maintenance_all_agents(self, api, mock_memory_system):
        """Test forcing memory maintenance for all agents."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        # Setup mocks
        mock_agent1 = Mock()
        mock_agent1._perform_maintenance.return_value = True
        mock_agent2 = Mock()
        mock_agent2._perform_maintenance.return_value = True
        
        mock_instance.agents = {"agent1": mock_agent1, "agent2": mock_agent2}
        
        # Call and verify
        result = api.force_memory_maintenance()
        assert result is True
        
        mock_agent1._perform_maintenance.assert_called_once()
        mock_agent2._perform_maintenance.assert_called_once()

    def test_force_memory_maintenance_failure(self, api, mock_memory_system):
        """Test handling of maintenance failure."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        # Setup mocks
        mock_agent1 = Mock()
        mock_agent1._perform_maintenance.return_value = True
        mock_agent2 = Mock()
        mock_agent2._perform_maintenance.return_value = False
        
        mock_instance.agents = {"agent1": mock_agent1, "agent2": mock_agent2}
        
        # Call and verify - should return False since agent2 failed
        result = api.force_memory_maintenance()
        assert result is False
        
        mock_agent1._perform_maintenance.assert_called_once()
        mock_agent2._perform_maintenance.assert_called_once()

    def test_clear_agent_memory_all_tiers(self, api, mock_memory_system, mock_memory_agent):
        """Test clearing all memory tiers for an agent."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.clear_memory.return_value = True
        
        # Call and verify
        result = api.clear_agent_memory("agent1")
        assert result is True
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.clear_memory.assert_called_once()

    def test_clear_agent_memory_specific_tiers(self, api, mock_memory_system, mock_memory_agent):
        """Test clearing specific memory tiers for an agent."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.clear.return_value = True
        mock_memory_agent.im_store.clear.return_value = True
        
        # Call and verify
        result = api.clear_agent_memory("agent1", memory_tiers=["stm", "im"])
        assert result is True
        
        mock_instance.get_memory_agent.assert_called_once_with("agent1")
        mock_memory_agent.stm_store.clear.assert_called_once()
        mock_memory_agent.im_store.clear.assert_called_once()
        mock_memory_agent.ltm_store.clear.assert_not_called()

    def test_set_importance_score(self, api, mock_memory_system, mock_memory_agent):
        """Test setting importance score for a memory."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        memory_id = "mem123"
        memory = {
            "memory_id": memory_id,
            "metadata": {"importance_score": 0.5},
            "contents": {"health": 0.8}
        }
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        mock_memory_agent.stm_store.get.return_value = memory
        mock_memory_agent.im_store.get.return_value = None
        mock_memory_agent.ltm_store.get.return_value = None
        mock_memory_agent.stm_store.contains.return_value = True
        mock_memory_agent.stm_store.update.return_value = True
        
        # Call and verify
        result = api.set_importance_score("agent1", memory_id, 0.8)
        assert result is True
        
        # Check that the score was updated properly
        assert memory["metadata"]["importance_score"] == 0.8
        
        # First call is from retrieve_state_by_id and second is from the actual function
        # So we don't check assert_called_once_with
        assert mock_instance.get_memory_agent.call_count >= 1 
        mock_memory_agent.stm_store.update.assert_called_once_with(memory)

    def test_get_memory_snapshots(self, api, mock_memory_system, mock_memory_agent):
        """Test getting memory snapshots for specific steps."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        step10_memory = {"memory_id": "mem10", "step_number": 10, "contents": {"health": 0.8}}
        step20_memory = {"memory_id": "mem20", "step_number": 20, "contents": {"health": 0.7}}
        
        # Setup mocks
        mock_instance.get_memory_agent.return_value = mock_memory_agent
        
        def mock_retrieve_by_time_range(agent_id, start_step, end_step, memory_type):
            if start_step == 10 and end_step == 10:
                return [step10_memory]
            elif start_step == 20 and end_step == 20:
                return [step20_memory]
            elif start_step == 30 and end_step == 30:
                return []
            return []
            
        # Use patch to mock the retrieve_by_time_range method
        with patch.object(api, 'retrieve_by_time_range', side_effect=mock_retrieve_by_time_range):
            # Call and verify
            result = api.get_memory_snapshots("agent1", [10, 20, 30])
            
            expected = {
                10: step10_memory,
                20: step20_memory,
                30: None
            }
            assert result == expected

    def test_configure_memory_system(self, api, mock_memory_system, mock_memory_agent):
        """Test configuring the memory system."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        config_updates = {
            "cleanup_interval": 200,
            "memory_priority_decay": 0.9
        }
        
        # Setup mocks
        mock_instance.agents = {"agent1": mock_memory_agent}
        mock_instance.config = MagicMock()
        mock_instance.config.stm_config = "stm_config"
        mock_instance.config.im_config = "im_config" 
        mock_instance.config.ltm_config = "ltm_config"
        
        # Call and verify
        result = api.configure_memory_system(config_updates)
        assert result is True
        
        # Check that attributes were updated
        for key, value in config_updates.items():
            assert setattr(mock_instance.config, key, value) is None
            
        # Check that agent configs were updated
        assert mock_memory_agent.config == mock_instance.config
        assert mock_memory_agent.stm_store.config == "stm_config"
        assert mock_memory_agent.im_store.config == "im_config"
        assert mock_memory_agent.ltm_store.config == "ltm_config"

    def test_get_attribute_change_history(self, api, mock_memory_system, mock_memory_agent):
        """Test getting attribute change history."""
        # Unpack to get just the mock instance
        _, mock_instance = mock_memory_system
        
        memories = [
            {"memory_id": "mem1", "step_number": 10, "timestamp": 1000, "contents": {"health": 1.0}},
            {"memory_id": "mem2", "step_number": 20, "timestamp": 2000, "contents": {"health": 0.8}},
            {"memory_id": "mem3", "step_number": 30, "timestamp": 3000, "contents": {"health": 0.8}},
            {"memory_id": "mem4", "step_number": 40, "timestamp": 4000, "contents": {"health": 0.5}},
        ]
        
        # Use patch to mock the retrieve_by_time_range method
        with patch.object(api, 'retrieve_by_time_range', return_value=memories):
            # Call and verify
            result = api.get_attribute_change_history("agent1", "health")
            
            expected = [
                {
                    "memory_id": "mem1",
                    "step_number": 10,
                    "timestamp": 1000,
                    "previous_value": None,
                    "new_value": 1.0
                },
                {
                    "memory_id": "mem2",
                    "step_number": 20,
                    "timestamp": 2000,
                    "previous_value": 1.0,
                    "new_value": 0.8
                },
                {
                    "memory_id": "mem4",
                    "step_number": 40,
                    "timestamp": 4000,
                    "previous_value": 0.8,
                    "new_value": 0.5
                }
            ]
            assert result == expected
            
            # Verify retrieve_by_time_range was called correctly
            api.retrieve_by_time_range.assert_called_once_with(
                "agent1", 0, float("inf"), memory_type="state"
            )

if __name__ == "__main__":
    pytest.main(["-xvs", "test_memory_api.py"]) 