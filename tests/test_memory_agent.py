"""Unit tests for the Memory Agent module.

This test suite covers the functionality of the MemoryAgent class, which manages
hierarchical memory storage across short-term (STM), intermediate (IM), and 
long-term memory (LTM) tiers.

The tests use pytest fixtures and mocks to isolate the MemoryAgent from its
dependencies, allowing focused testing of the agent's logic.

To run these tests:
    pytest tests/test_memory_agent.py

To run with coverage:
    pytest tests/test_memory_agent.py --cov=agent_memory

Test categories:
- TestMemoryAgentBasics: Tests for initialization and configuration
- TestMemoryStorage: Tests for memory storage operations
- TestMemoryTransitions: Tests for memory movement between tiers
- TestMemoryRetrieval: Tests for memory retrieval operations
- TestEventHooks: Tests for the event hook mechanism
- TestUtilityFunctions: Tests for utility and maintenance functions
"""

import time
import unittest.mock as mock
import pytest

from agent_memory.config import MemoryConfig, AutoencoderConfig
from agent_memory.memory_agent import MemoryAgent


@pytest.fixture
def mock_stm_store():
    """Mock the Short-Term Memory store."""
    store = mock.MagicMock()
    store.store.return_value = True
    store.count.return_value = 10
    store.get_all.return_value = []
    store.get_size.return_value = 1000
    store.clear.return_value = True
    return store


@pytest.fixture
def mock_im_store():
    """Mock the Intermediate Memory store."""
    store = mock.MagicMock()
    store.store.return_value = True
    store.count.return_value = 10
    store.get_all.return_value = []
    store.get_size.return_value = 1000
    store.clear.return_value = True
    return store


@pytest.fixture
def mock_ltm_store():
    """Mock the Long-Term Memory store."""
    store = mock.MagicMock()
    store.store_batch.return_value = True
    store.count.return_value = 10
    store.get_all.return_value = []
    store.get_size.return_value = 1000
    store.clear.return_value = True
    return store


@pytest.fixture
def mock_compression_engine():
    """Mock the Compression Engine."""
    engine = mock.MagicMock()
    engine.compress.return_value = {"metadata": {}}
    engine.compress_embedding.return_value = [0.1, 0.2, 0.3]
    return engine


@pytest.fixture
def mock_embedding_engine():
    """Mock the Embedding Engine."""
    engine = mock.MagicMock()
    engine.encode_stm.return_value = [0.1, 0.2, 0.3, 0.4]
    engine.encode_im.return_value = [0.1, 0.2, 0.3]
    engine.encode_ltm.return_value = [0.1, 0.2]
    return engine


@pytest.fixture
def memory_agent(mock_stm_store, mock_im_store, mock_ltm_store, 
                mock_compression_engine, mock_embedding_engine):
    """Create a memory agent with mocked dependencies."""
    agent_id = "test-agent"
    config = MemoryConfig()
    config.autoencoder_config.use_neural_embeddings = True
    config.ltm_config.db_path = "test_agent_memory.db"  # Set a valid db path
    
    # Mock store classes before instantiating the agent
    with mock.patch("agent_memory.memory_agent.RedisSTMStore") as mock_stm_class, \
         mock.patch("agent_memory.memory_agent.RedisIMStore") as mock_im_class, \
         mock.patch("agent_memory.memory_agent.SQLiteLTMStore") as mock_ltm_class, \
         mock.patch("agent_memory.memory_agent.CompressionEngine"), \
         mock.patch("agent_memory.memory_agent.AutoencoderEmbeddingEngine"):
        
        # Configure the mock classes to return our mock instances
        mock_stm_class.return_value = mock_stm_store
        mock_im_class.return_value = mock_im_store
        mock_ltm_class.return_value = mock_ltm_store
        
        agent = MemoryAgent(agent_id, config)
        
        # No need to replace stores as they are already our mocks
        agent.compression_engine = mock_compression_engine
        agent.embedding_engine = mock_embedding_engine
        
        return agent


# Base test cases follow
class TestMemoryAgentBasics:
    """Basic tests for Memory Agent initialization and configuration."""
    
    def test_init(self):
        """Test memory agent initialization."""
        agent_id = "test-agent"
        config = MemoryConfig()
        config.autoencoder_config.use_neural_embeddings = True
        config.ltm_config.db_path = "test_agent_memory.db"  # Set a valid db path
        
        with mock.patch("agent_memory.memory_agent.RedisSTMStore") as mock_stm, \
             mock.patch("agent_memory.memory_agent.RedisIMStore") as mock_im, \
             mock.patch("agent_memory.memory_agent.SQLiteLTMStore") as mock_ltm, \
             mock.patch("agent_memory.memory_agent.CompressionEngine") as mock_ce, \
             mock.patch("agent_memory.memory_agent.AutoencoderEmbeddingEngine") as mock_ae:
            
            agent = MemoryAgent(agent_id, config)
            
            # Verify stores were initialized
            mock_stm.assert_called_once_with(config.stm_config)
            mock_im.assert_called_once_with(config.im_config)
            mock_ltm.assert_called_once_with(agent_id, config.ltm_config)
            mock_ce.assert_called_once_with(config.autoencoder_config)
            mock_ae.assert_called_once()
            
            assert agent.agent_id == agent_id
            assert agent.config == config
    
    def test_init_without_neural_embeddings(self):
        """Test memory agent initialization without neural embeddings."""
        agent_id = "test-agent"
        config = MemoryConfig()
        config.autoencoder_config.use_neural_embeddings = False
        config.ltm_config.db_path = "test_agent_memory.db"  # Set a valid db path
        
        with mock.patch("agent_memory.memory_agent.RedisSTMStore"), \
             mock.patch("agent_memory.memory_agent.RedisIMStore"), \
             mock.patch("agent_memory.memory_agent.SQLiteLTMStore"), \
             mock.patch("agent_memory.memory_agent.CompressionEngine"):
            
            agent = MemoryAgent(agent_id, config)
            assert agent.embedding_engine is None


class TestMemoryStorage:
    """Tests for memory storage operations."""
    
    def test_store_state(self, memory_agent, mock_stm_store):
        """Test storing a state memory."""
        state_data = {"position": [1, 2, 3], "inventory": ["sword", "shield"]}
        step_number = 42
        priority = 0.8
        
        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, '_create_memory_entry') as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-1"}
            
            result = memory_agent.store_state(state_data, step_number, priority)
            
            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(state_data, step_number, "state", priority)
            
            # Check the store call
            mock_stm_store.store.assert_called_once_with(memory_agent.agent_id, {"memory_id": "test-memory-1"})
            
            assert result is True
            assert memory_agent._insert_count == 1
    
    def test_store_interaction(self, memory_agent, mock_stm_store):
        """Test storing an interaction memory."""
        interaction_data = {"agent": "agent1", "target": "agent2", "action": "greet"}
        step_number = 42
        priority = 0.5
        
        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, '_create_memory_entry') as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-2"}
            
            result = memory_agent.store_interaction(interaction_data, step_number, priority)
            
            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(interaction_data, step_number, "interaction", priority)
            
            # Check the store call
            mock_stm_store.store.assert_called_once_with(memory_agent.agent_id, {"memory_id": "test-memory-2"})
            
            assert result is True
            assert memory_agent._insert_count == 1
    
    def test_store_action(self, memory_agent, mock_stm_store):
        """Test storing an action memory."""
        action_data = {"action_type": "move", "direction": "north", "result": "success"}
        step_number = 42
        priority = 0.7
        
        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, '_create_memory_entry') as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-3"}
            
            result = memory_agent.store_action(action_data, step_number, priority)
            
            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(action_data, step_number, "action", priority)
            
            # Check the store call
            mock_stm_store.store.assert_called_once_with(memory_agent.agent_id, {"memory_id": "test-memory-3"})
            
            assert result is True
            assert memory_agent._insert_count == 1
    
    def test_cleanup_triggered(self, memory_agent):
        """Test that cleanup is triggered after multiple insertions."""
        memory_agent.config.cleanup_interval = 5
        memory_agent._insert_count = 0
        
        # Patch _check_memory_transition
        with mock.patch.object(memory_agent, '_check_memory_transition') as mock_check:
            # Do 5 insertions
            for i in range(5):
                with mock.patch.object(memory_agent, '_create_memory_entry') as mock_create:
                    mock_create.return_value = {"memory_id": f"test-memory-{i}"}
                    memory_agent.store_state({"test": i}, i, 0.5)
            
            # Verify _check_memory_transition was called once
            mock_check.assert_called_once()
            assert memory_agent._insert_count == 5
    
    def test_create_memory_entry(self, memory_agent, mock_embedding_engine):
        """Test memory entry creation."""
        test_data = {"test": "data"}
        step_number = 42
        memory_type = "state"
        priority = 0.9
        
        # Mock time.time()
        with mock.patch('time.time', return_value=12345):
            entry = memory_agent._create_memory_entry(test_data, step_number, memory_type, priority)
            
            # Check structure
            assert entry["memory_id"] == f"{memory_agent.agent_id}-{step_number}-12345"
            assert entry["agent_id"] == memory_agent.agent_id
            assert entry["step_number"] == step_number
            assert entry["timestamp"] == 12345
            assert entry["content"] == test_data
            
            # Check metadata
            metadata = entry["metadata"]
            assert metadata["creation_time"] == 12345
            assert metadata["last_access_time"] == 12345
            assert metadata["compression_level"] == 0
            assert metadata["importance_score"] == priority
            assert metadata["retrieval_count"] == 0
            assert metadata["memory_type"] == memory_type
            
            # Check embeddings
            assert "embeddings" in entry
            
            # Verify embedding engine calls
            mock_embedding_engine.encode_stm.assert_called_once_with(test_data)
            mock_embedding_engine.encode_im.assert_called_once_with(test_data)
            mock_embedding_engine.encode_ltm.assert_called_once_with(test_data)


class TestMemoryTransitions:
    """Tests for memory transitions between tiers."""
    
    def test_check_memory_transition_stm_to_im(self, memory_agent, mock_stm_store, mock_im_store):
        """Test transitioning memories from STM to IM when STM is at capacity."""
        # Configure mocks
        memory_agent.config.stm_config.memory_limit = 5
        mock_stm_store.count.return_value = 10  # Over capacity
        
        # Create sample memories in STM
        stm_memories = []
        for i in range(10):
            stm_memories.append({
                "memory_id": f"mem-{i}",
                "metadata": {
                    "creation_time": time.time() - (i * 1000),  # Older memories have higher i
                    "importance_score": 0.5,
                    "retrieval_count": i % 3  # Some variation in retrieval count
                }
            })
        
        mock_stm_store.get_all.return_value = stm_memories
        
        # Call the transition method
        with mock.patch.object(memory_agent, '_calculate_importance', 
                              side_effect=lambda m: m["metadata"]["importance_score"]):
            memory_agent._check_memory_transition()
            
            # Check that IM store was called with compressed memories
            assert mock_im_store.store.call_count == 5  # Should transition 5 memories
            
            # Verify the compression engine was used
            assert memory_agent.compression_engine.compress.call_count == 5
            
            # Verify memories were deleted from STM
            assert mock_stm_store.delete.call_count == 5
    
    def test_check_memory_transition_im_to_ltm(self, memory_agent, mock_im_store, mock_ltm_store, mock_stm_store):
        """Test transitioning memories from IM to LTM when IM is at capacity."""
        # Configure mocks
        memory_agent.config.im_config.memory_limit = 5
        memory_agent.config.ltm_config.batch_size = 3
        mock_stm_store.count.return_value = 3  # Under capacity
        mock_im_store.count.return_value = 8  # Over capacity
        
        # Create sample memories in IM
        im_memories = []
        for i in range(8):
            im_memories.append({
                "memory_id": f"mem-{i}",
                "metadata": {
                    "creation_time": time.time() - (i * 1000),  # Older memories have higher i
                    "importance_score": 0.5,
                    "retrieval_count": i % 3  # Some variation in retrieval count
                }
            })
        
        mock_im_store.get_all.return_value = im_memories
        
        # Call the transition method
        with mock.patch.object(memory_agent, '_calculate_importance', 
                              side_effect=lambda m: m["metadata"]["importance_score"]):
            memory_agent._check_memory_transition()
            
            # Check that LTM store was called with batches
            # Should be 1 call with 3 memories, plus 1 more call with remaining 0 memories
            calls = mock_ltm_store.store_batch.call_args_list
            assert len(calls) == 1  # One batch of 3
            assert len(calls[0][0][0]) == 3  # First batch has 3 items
            
            # Verify the compression engine was used
            assert memory_agent.compression_engine.compress.call_count == 3
            
            # Verify memories were deleted from IM
            assert mock_im_store.delete.call_count == 3
    
    def test_calculate_importance(self, memory_agent):
        """Test importance calculation logic."""
        # Create a test memory with relevant fields
        current_time = time.time()
        memory = {
            "content": {"reward": 5.0},
            "metadata": {
                "retrieval_count": 3,
                "creation_time": current_time - 500,
                "surprise_factor": 0.7
            }
        }
        
        importance = memory_agent._calculate_importance(memory)
        
        # Verify importance calculation components and bounds
        assert 0.0 <= importance <= 1.0
        
        # Test reward component (40% of score)
        memory_high_reward = {
            "content": {"reward": 20.0},  # Above the cap of 10
            "metadata": {
                "retrieval_count": 0,
                "creation_time": current_time,
                "surprise_factor": 0.0
            }
        }
        
        high_reward_importance = memory_agent._calculate_importance(memory_high_reward)

        # Should have full reward component (40%) and full recency component (20%)
        assert high_reward_importance == pytest.approx(0.6, abs=0.01)
        
        # Test retrieval frequency component (30% of score)
        memory_high_retrieval = {
            "content": {"reward": 0.0},
            "metadata": {
                "retrieval_count": 10,  # Above the cap of 5
                "creation_time": current_time,
                "surprise_factor": 0.0
            }
        }
        
        high_retrieval_importance = memory_agent._calculate_importance(memory_high_retrieval)
        # Should have full retrieval component (30%) and full recency component (20%)
        assert high_retrieval_importance == pytest.approx(0.5, abs=0.01)


class TestMemoryRetrieval:
    """Tests for memory retrieval operations."""
    
    # def test_retrieve_similar_states(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
    #     """Test retrieving similar states across memory tiers."""
    #     # Configure the mock stores to return results
    #     stm_results = [{"memory_id": "stm1", "similarity_score": 0.9}, 
    #                   {"memory_id": "stm2", "similarity_score": 0.8}]
    #     im_results = [{"memory_id": "im1", "similarity_score": 0.7}]
    #     ltm_results = [{"memory_id": "ltm1", "similarity_score": 0.6}, 
    #                   {"memory_id": "ltm2", "similarity_score": 0.5}]
        
    #     mock_stm_store.search_similar.return_value = stm_results
    #     mock_im_store.search_similar.return_value = im_results
    #     mock_ltm_store.search_similar.return_value = ltm_results
        
    #     # Mock the embedding engine
    #     query_state = {"position": [1, 2, 3]}
        
    #     # Set k=5 to ensure we search all tiers
    #     results = memory_agent.retrieve_similar_states(query_state, k=5)
        
    #     # Should return top 5 results from all stores combined, sorted by similarity
    #     assert len(results) == 5
    #     assert results[0]["memory_id"] == "stm1"  # Highest similarity
    #     assert results[1]["memory_id"] == "stm2"
    #     assert results[2]["memory_id"] == "im1"
    #     assert results[3]["memory_id"] == "ltm1"
    #     assert results[4]["memory_id"] == "ltm2"  # Lowest similarity
        
    #     # Verify embedding engine calls
    #     memory_agent.embedding_engine.encode_stm.assert_called_with(query_state, None)
    #     memory_agent.embedding_engine.encode_im.assert_called_with(query_state, None)
    #     memory_agent.embedding_engine.encode_ltm.assert_called_with(query_state, None)
        
    #     # Verify search calls on stores with correct parameters
    #     stm_query = memory_agent.embedding_engine.encode_stm.return_value
    #     im_query = memory_agent.embedding_engine.encode_im.return_value
    #     ltm_query = memory_agent.embedding_engine.encode_ltm.return_value
        
    #     mock_stm_store.search_similar.assert_called_once_with(memory_agent.agent_id, stm_query, k=5, memory_type=None)
    #     mock_im_store.search_similar.assert_called_once_with(memory_agent.agent_id, im_query, k=3, memory_type=None)
    #     mock_ltm_store.search_similar.assert_called_once_with(ltm_query, k=2, memory_type=None)
    
    def test_retrieve_similar_states_with_type_filter(self, memory_agent, mock_stm_store):
        """Test retrieving similar states with a memory type filter."""
        # Configure the stores
        mock_stm_store.search_similar.return_value = [{"memory_id": "stm1", "similarity_score": 0.9}]
        
        query_state = {"position": [1, 2, 3]}
        memory_type = "action"
        
        memory_agent.retrieve_similar_states(query_state, k=5, memory_type=memory_type)
        
        # Verify memory_type was passed to the store
        mock_stm_store.search_similar.assert_called_once()
        args, kwargs = mock_stm_store.search_similar.call_args
        assert kwargs.get("memory_type") == memory_type
    
    def test_retrieve_by_time_range(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test retrieving memories within a time range."""
        # Configure the stores
        stm_results = [{"memory_id": "stm1", "step_number": 5}, 
                      {"memory_id": "stm2", "step_number": 7}]
        im_results = [{"memory_id": "im1", "step_number": 3}]
        ltm_results = [{"memory_id": "ltm1", "step_number": 2}]
        
        mock_stm_store.search_by_step_range.return_value = stm_results
        mock_im_store.search_by_step_range.return_value = im_results
        mock_ltm_store.search_by_step_range.return_value = ltm_results
        
        results = memory_agent.retrieve_by_time_range(1, 10)
        
        # Results should be sorted by step number
        assert len(results) == 4
        assert results[0]["memory_id"] == "ltm1"  # Earliest step
        assert results[1]["memory_id"] == "im1"
        assert results[2]["memory_id"] == "stm1"
        assert results[3]["memory_id"] == "stm2"  # Latest step
        
        # Verify calls to stores
        mock_stm_store.search_by_step_range.assert_called_once_with(memory_agent.agent_id, 1, 10, None)
        mock_im_store.search_by_step_range.assert_called_once_with(memory_agent.agent_id, 1, 10, None)
        mock_ltm_store.search_by_step_range.assert_called_once_with(1, 10, None)
    
    def test_retrieve_by_attributes(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test retrieving memories by attribute matching."""
        # Configure the stores
        stm_results = [{"memory_id": "stm1", "timestamp": 1000}]
        im_results = [{"memory_id": "im1", "timestamp": 2000}]
        ltm_results = [{"memory_id": "ltm1", "timestamp": 500}]
        
        mock_stm_store.search_by_attributes.return_value = stm_results
        mock_im_store.search_by_attributes.return_value = im_results
        mock_ltm_store.search_by_attributes.return_value = ltm_results
        
        attributes = {"location": "forest", "action_type": "gather"}
        
        results = memory_agent.retrieve_by_attributes(attributes)
        
        # Results should be sorted by timestamp (most recent first)
        assert len(results) == 3
        assert results[0]["memory_id"] == "im1"   # Most recent
        assert results[1]["memory_id"] == "stm1"
        assert results[2]["memory_id"] == "ltm1"  # Oldest
        
        # Verify calls to stores
        mock_stm_store.search_by_attributes.assert_called_once_with(memory_agent.agent_id, attributes, None)
        mock_im_store.search_by_attributes.assert_called_once_with(memory_agent.agent_id, attributes, None)
        mock_ltm_store.search_by_attributes.assert_called_once_with(attributes, None)
    
    def test_search_by_embedding(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test searching by raw embedding vector."""
        # Configure the stores
        stm_results = [{"memory_id": "stm1", "similarity_score": 0.9}]
        im_results = [{"memory_id": "im1", "similarity_score": 0.7}]
        ltm_results = [{"memory_id": "ltm1", "similarity_score": 0.5}]
        
        mock_stm_store.search_by_embedding.return_value = stm_results
        mock_im_store.search_by_embedding.return_value = im_results
        mock_ltm_store.search_by_embedding.return_value = ltm_results
        
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        results = memory_agent.search_by_embedding(query_embedding, k=3)
        
        # Results should be sorted by similarity score
        assert len(results) == 3
        assert results[0]["memory_id"] == "stm1"  # Highest similarity
        assert results[1]["memory_id"] == "im1"
        assert results[2]["memory_id"] == "ltm1"  # Lowest similarity
        
        # Verify compression engine calls for IM and LTM
        memory_agent.compression_engine.compress_embedding.assert_any_call(query_embedding, level=1)
        memory_agent.compression_engine.compress_embedding.assert_any_call(query_embedding, level=2)
        
        # Verify search calls include agent_id for Redis stores
        mock_stm_store.search_by_embedding.assert_called_once_with(memory_agent.agent_id, query_embedding, k=3)
        mock_im_store.search_by_embedding.assert_called_once()  # Already passes but could be more specific
        mock_ltm_store.search_by_embedding.assert_called_once()  # Already passes but could be more specific
    
    def test_search_by_content(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test searching by content text or attributes."""
        # Configure the stores
        stm_results = [{"memory_id": "stm1", "relevance_score": 0.9}]
        im_results = [{"memory_id": "im1", "relevance_score": 0.7}]
        ltm_results = [{"memory_id": "ltm1", "relevance_score": 0.5}]
        
        mock_stm_store.search_by_content.return_value = stm_results
        mock_im_store.search_by_content.return_value = im_results
        mock_ltm_store.search_by_content.return_value = ltm_results
        
        # Test with string query
        text_query = "forest exploration"
        results = memory_agent.search_by_content(text_query, k=3)
        
        # Verify string query was converted to dict
        mock_stm_store.search_by_content.assert_called_with(memory_agent.agent_id, {"text": text_query}, 3)
        
        # Results should be sorted by relevance score
        assert len(results) == 3
        assert results[0]["memory_id"] == "stm1"  # Highest relevance
        assert results[1]["memory_id"] == "im1"
        assert results[2]["memory_id"] == "ltm1"  # Lowest relevance
        
        # Test with dict query
        dict_query = {"location": "forest", "action": "explore"}
        memory_agent.search_by_content(dict_query, k=2)
        
        # Verify dict query was passed as is
        mock_stm_store.search_by_content.assert_called_with(memory_agent.agent_id, dict_query, 2)


class TestEventHooks:
    """Tests for memory event hooks mechanism."""
    
    def test_register_hook(self, memory_agent):
        """Test registering event hooks."""
        # Define a sample hook function
        def test_hook(event_data, agent):
            return {"result": "success"}
        
        # Register the hook
        result = memory_agent.register_hook("memory_storage", test_hook, priority=7)
        
        assert result is True
        assert hasattr(memory_agent, "_event_hooks")
        assert "memory_storage" in memory_agent._event_hooks
        assert len(memory_agent._event_hooks["memory_storage"]) == 1
        assert memory_agent._event_hooks["memory_storage"][0]["function"] == test_hook
        assert memory_agent._event_hooks["memory_storage"][0]["priority"] == 7
    
    def test_register_multiple_hooks_with_priority(self, memory_agent):
        """Test registering multiple hooks with priority ordering."""
        # Define sample hook functions
        def hook1(event_data, agent): return {"result": "hook1"}
        def hook2(event_data, agent): return {"result": "hook2"}
        def hook3(event_data, agent): return {"result": "hook3"}
        
        # Register hooks with different priorities
        memory_agent.register_hook("test_event", hook1, priority=3)
        memory_agent.register_hook("test_event", hook2, priority=8)
        memory_agent.register_hook("test_event", hook3, priority=5)
        
        # Verify hooks are stored in priority order (highest first)
        hooks = memory_agent._event_hooks["test_event"]
        assert len(hooks) == 3
        assert hooks[0]["function"] == hook2  # Priority 8
        assert hooks[1]["function"] == hook3  # Priority 5
        assert hooks[2]["function"] == hook1  # Priority 3
    
    def test_trigger_event(self, memory_agent):
        """Test triggering events and executing hooks."""
        # Define sample hook functions with different behaviors
        event_results = []
        
        def hook1(event_data, agent):
            event_results.append("hook1 executed")
            return {"result": "hook1 done"}
        
        def hook2(event_data, agent):
            event_results.append("hook2 executed")
            # This hook requests memory storage
            return {
                "store_memory": True,
                "memory_data": {"hook_generated": True},
                "step_number": 42,
                "priority": 0.9
            }
        
        # Register both hooks
        memory_agent.register_hook("test_event", hook1, priority=5)
        memory_agent.register_hook("test_event", hook2, priority=3)
        
        # Mock store_state method
        with mock.patch.object(memory_agent, 'store_state') as mock_store:
            # Trigger the event
            event_data = {"source": "test", "action": "move"}
            result = memory_agent.trigger_event("test_event", event_data)
            
            # Verify both hooks executed in order
            assert result is True
            assert len(event_results) == 2
            assert event_results[0] == "hook1 executed"
            assert event_results[1] == "hook2 executed"
            
            # Verify memory storage was requested by hook2
            mock_store.assert_called_once_with(
                {"hook_generated": True},  # memory_data
                42,                        # step_number
                0.9                        # priority
            )
    
    def test_trigger_event_with_exception(self, memory_agent):
        """Test handling exceptions in event hooks."""
        # Define a hook that raises an exception
        def failing_hook(event_data, agent):
            raise ValueError("Test exception")
        
        # Define a normal hook
        normal_executed = False
        def normal_hook(event_data, agent):
            nonlocal normal_executed
            normal_executed = True
            return {"result": "success"}
        
        # Register hooks
        memory_agent.register_hook("test_event", failing_hook, priority=5)
        memory_agent.register_hook("test_event", normal_hook, priority=3)
        
        # Trigger the event - should continue after exception
        result = memory_agent.trigger_event("test_event", {"source": "test"})
        
        # First hook failed but second one executed
        assert result is False  # Overall failure
        assert normal_executed is True  # Second hook still ran


class TestUtilityFunctions:
    """Tests for memory utility functions."""
    
    def test_clear_memory(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test clearing all memory stores."""
        result = memory_agent.clear_memory()
        
        assert result is True
        mock_stm_store.clear.assert_called_once()
        mock_im_store.clear.assert_called_once()
        mock_ltm_store.clear.assert_called_once()
    
    def test_clear_memory_failure(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test handling failure in clearing memory."""
        mock_stm_store.clear.return_value = False
        
        result = memory_agent.clear_memory()
        
        assert result is False
    
    def test_force_maintenance(self, memory_agent):
        """Test forcing memory maintenance."""
        with mock.patch.object(memory_agent, '_check_memory_transition') as mock_check:
            result = memory_agent.force_maintenance()
            
            assert result is True
            mock_check.assert_called_once()
    
    def test_force_maintenance_failure(self, memory_agent):
        """Test handling failure in forced maintenance."""
        with mock.patch.object(memory_agent, '_check_memory_transition', 
                              side_effect=Exception("Test exception")):
            result = memory_agent.force_maintenance()
            
            assert result is False
    
    def test_get_memory_statistics(self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test retrieving memory statistics."""
        # Configure the mocks
        mock_stm_store.count.return_value = 10
        mock_im_store.count.return_value = 20
        mock_ltm_store.count.return_value = 30
        
        with mock.patch.object(memory_agent, '_calculate_tier_importance', return_value=0.5), \
             mock.patch.object(memory_agent, '_calculate_compression_ratio', return_value=2.5), \
             mock.patch.object(memory_agent, '_get_memory_type_distribution', 
                              return_value={"state": 30, "action": 20, "interaction": 10}), \
             mock.patch.object(memory_agent, '_get_access_patterns', return_value={"most_accessed": []}):
            
            stats = memory_agent.get_memory_statistics()
            
            # Verify statistics structure
            assert stats["total_memories"] == 60  # 10 + 20 + 30
            assert "timestamp" in stats
            
            # Check tier stats
            assert stats["tiers"]["stm"]["count"] == 10
            assert stats["tiers"]["im"]["count"] == 20
            assert stats["tiers"]["ltm"]["count"] == 30
            
            # Check memory types
            assert stats["memory_types"]["state"] == 30
            assert stats["memory_types"]["action"] == 20
            assert stats["memory_types"]["interaction"] == 10
            
            # Verify helper method calls
            memory_agent._calculate_tier_importance.assert_any_call("stm")
            memory_agent._calculate_tier_importance.assert_any_call("im")
            memory_agent._calculate_tier_importance.assert_any_call("ltm")
            
            memory_agent._calculate_compression_ratio.assert_any_call("im")
            memory_agent._calculate_compression_ratio.assert_any_call("ltm")


if __name__ == "__main__":
    pytest.main(["-xvs", "test_memory_agent.py"]) 