"""Unit tests for the Memory Agent module.

This test suite covers the functionality of the MemoryAgent class, which manages
hierarchical memory storage across short-term (STM), intermediate (IM), and
long-term memory (LTM) tiers.

The tests use pytest fixtures and mocks to isolate the MemoryAgent from its
dependencies, allowing focused testing of the agent's logic.

To run these tests:
    pytest tests/test_memory_agent.py

To run with coverage:
    pytest tests/test_memory_agent.py --cov=memory

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

from memory.agent_memory import MemoryAgent
from memory.config import MemoryConfig


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
def memory_agent(
    mock_stm_store,
    mock_im_store,
    mock_ltm_store,
    mock_compression_engine,
    mock_embedding_engine,
):
    """Create a memory agent with mocked dependencies."""
    agent_id = "test-agent"
    config = MemoryConfig()
    config.autoencoder_config.use_neural_embeddings = True
    config.text_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use a real model
    config.ltm_config.db_path = "test_memory.db"  # Set a valid db path

    with mock.patch("memory.agent_memory.RedisSTMStore") as mock_stm, mock.patch(
        "memory.agent_memory.RedisIMStore"
    ) as mock_im, mock.patch(
        "memory.agent_memory.SQLiteLTMStore"
    ) as mock_ltm, mock.patch(
        "memory.agent_memory.CompressionEngine"
    ) as mock_ce, mock.patch(
        "memory.agent_memory.TextEmbeddingEngine"
    ) as mock_ae:

        # Configure the mock classes to return our mock instances
        mock_stm.return_value = mock_stm_store
        mock_im.return_value = mock_im_store
        mock_ltm.return_value = mock_ltm_store

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
        config.use_embedding_engine = True
        config.text_model_name = (
            "sentence-transformers/all-MiniLM-L6-v2"  # Use a real model
        )
        config.ltm_config.db_path = "test_memory.db"  # Set a valid db path

        with mock.patch("memory.agent_memory.RedisSTMStore") as mock_stm, mock.patch(
            "memory.agent_memory.RedisIMStore"
        ) as mock_im, mock.patch(
            "memory.agent_memory.SQLiteLTMStore"
        ) as mock_ltm, mock.patch(
            "memory.agent_memory.CompressionEngine"
        ) as mock_ce, mock.patch(
            "memory.agent_memory.TextEmbeddingEngine"
        ) as mock_ae:

            agent = MemoryAgent(agent_id, config)

            # Verify stores were initialized
            mock_stm.assert_called_once_with(config.stm_config)
            mock_im.assert_called_once_with(config.im_config)
            mock_ltm.assert_called_once_with(agent_id, config.ltm_config)
            mock_ce.assert_called_once_with(config.autoencoder_config)
            mock_ae.assert_called_once_with(model_name=config.text_model_name)

            assert agent.agent_id == agent_id
            assert agent.config == config

    def test_init_without_neural_embeddings(self):
        """Test memory agent initialization without neural embeddings."""
        agent_id = "test-agent"
        config = MemoryConfig()
        config.use_embedding_engine = False
        config.ltm_config.db_path = "test_memory.db"  # Set a valid db path

        with mock.patch("memory.agent_memory.RedisSTMStore"), mock.patch(
            "memory.agent_memory.RedisIMStore"
        ), mock.patch("memory.agent_memory.SQLiteLTMStore"), mock.patch(
            "memory.agent_memory.CompressionEngine"
        ), mock.patch(
            "memory.agent_memory.TextEmbeddingEngine"
        ) as mock_te:

            agent = MemoryAgent(agent_id, config)
            assert agent.embedding_engine is None
            mock_te.assert_not_called()


class TestMemoryStorage:
    """Tests for memory storage operations."""

    def test_store_state(self, memory_agent, mock_stm_store):
        """Test storing a state memory in the default tier (STM)."""
        state_data = {"position": [1, 2, 3], "inventory": ["sword", "shield"]}
        step_number = 42
        priority = 0.8

        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-1"}

            result = memory_agent.store_state(state_data, step_number, priority)

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                state_data, step_number, "state", priority, "stm"
            )

            # Check the store call
            mock_stm_store.store.assert_called_once_with(
                memory_agent.agent_id, {"memory_id": "test-memory-1"}
            )

            assert result is True
            assert memory_agent._insert_count == 1

    def test_store_state_custom_tier(self, memory_agent, mock_im_store, mock_ltm_store):
        """Test storing a state memory in a custom tier."""
        state_data = {"position": [1, 2, 3], "inventory": ["sword", "shield"]}
        step_number = 42
        priority = 0.8

        # Configure mocks to return True
        mock_im_store.store.return_value = True
        mock_ltm_store.store.return_value = True

        # Test storing in IM tier
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-im"}

            result = memory_agent.store_state(
                state_data, step_number, priority, tier="im"
            )

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                state_data, step_number, "state", priority, "im"
            )

            # Check the store call to IM store
            mock_im_store.store.assert_called_once_with(
                memory_agent.agent_id, {"memory_id": "test-memory-im"}
            )

            assert result is True

        # Test storing in LTM tier
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-ltm"}

            result = memory_agent.store_state(
                state_data, step_number, priority, tier="ltm"
            )

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                state_data, step_number, "state", priority, "ltm"
            )

            # Check the store call to LTM store
            mock_ltm_store.store.assert_called_once_with(
                {"memory_id": "test-memory-ltm"}
            )

            assert result is True

    def test_store_interaction(self, memory_agent, mock_stm_store):
        """Test storing an interaction memory."""
        interaction_data = {"agent": "agent1", "target": "agent2", "action": "greet"}
        step_number = 42
        priority = 0.5

        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-2"}

            result = memory_agent.store_interaction(
                interaction_data, step_number, priority
            )

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                interaction_data, step_number, "interaction", priority, "stm"
            )

            # Check the store call
            mock_stm_store.store.assert_called_once_with(
                memory_agent.agent_id, {"memory_id": "test-memory-2"}
            )

            assert result is True
            assert memory_agent._insert_count == 1

    def test_store_interaction_custom_tier(self, memory_agent, mock_im_store):
        """Test storing an interaction memory in a custom tier."""
        interaction_data = {"agent": "agent1", "target": "agent2", "action": "greet"}
        step_number = 42
        priority = 0.5
        tier = "im"

        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-2-im"}

            result = memory_agent.store_interaction(
                interaction_data, step_number, priority, tier
            )

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                interaction_data, step_number, "interaction", priority, tier
            )

            # Check the store call
            mock_im_store.store.assert_called_once_with(
                memory_agent.agent_id, {"memory_id": "test-memory-2-im"}
            )

            assert result is True
            assert memory_agent._insert_count == 1

    def test_store_action(self, memory_agent, mock_stm_store):
        """Test storing an action memory."""
        action_data = {"action_type": "move", "direction": "north", "result": "success"}
        step_number = 42
        priority = 0.7

        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-3"}

            result = memory_agent.store_action(action_data, step_number, priority)

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                action_data, step_number, "action", priority, "stm"
            )

            # Check the store call
            mock_stm_store.store.assert_called_once_with(
                memory_agent.agent_id, {"memory_id": "test-memory-3"}
            )

            assert result is True
            assert memory_agent._insert_count == 1

    def test_store_action_custom_tier(self, memory_agent, mock_ltm_store):
        """Test storing an action memory in a custom tier."""
        action_data = {"action_type": "move", "direction": "north", "result": "success"}
        step_number = 42
        priority = 0.7
        tier = "ltm"

        # Configure mock to return True
        mock_ltm_store.store.return_value = True

        # Patch the _create_memory_entry method
        with mock.patch.object(memory_agent, "_create_memory_entry") as mock_create:
            mock_create.return_value = {"memory_id": "test-memory-3-ltm"}

            result = memory_agent.store_action(action_data, step_number, priority, tier)

            # Check the _create_memory_entry call
            mock_create.assert_called_once_with(
                action_data, step_number, "action", priority, tier
            )

            # Check the store call
            mock_ltm_store.store.assert_called_once_with(
                {"memory_id": "test-memory-3-ltm"}
            )

            assert result is True

    def test_store_in_tier(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test the _store_in_tier method for different tiers."""
        memory_entry = {"memory_id": "test-memory-tier", "content": {"test": "data"}}

        # Configure mocks to return True
        mock_stm_store.store.return_value = True
        mock_im_store.store.return_value = True
        mock_ltm_store.store.return_value = True

        # Test storing in STM
        result_stm = memory_agent._store_in_tier("stm", memory_entry)
        mock_stm_store.store.assert_called_with(memory_agent.agent_id, memory_entry)
        assert result_stm is True

        # Test storing in IM
        result_im = memory_agent._store_in_tier("im", memory_entry)
        mock_im_store.store.assert_called_with(memory_agent.agent_id, memory_entry)
        assert result_im is True

        # Test storing in LTM
        result_ltm = memory_agent._store_in_tier("ltm", memory_entry)
        mock_ltm_store.store.assert_called_with(memory_entry)
        assert result_ltm is True

        # Test invalid tier
        with mock.patch("memory.agent_memory.logger.warning") as mock_logger:
            result_invalid = memory_agent._store_in_tier("invalid_tier", memory_entry)
            mock_logger.assert_called_once()
            mock_stm_store.store.assert_called_with(memory_agent.agent_id, memory_entry)
            assert result_invalid is True

    def test_cleanup_triggered(self, memory_agent):
        """Test that cleanup is triggered after multiple insertions."""
        memory_agent.config.cleanup_interval = 5
        memory_agent._insert_count = 0

        # Patch _check_memory_transition
        with mock.patch.object(memory_agent, "_check_memory_transition") as mock_check:
            # Do 5 insertions
            for i in range(5):
                with mock.patch.object(
                    memory_agent, "_create_memory_entry"
                ) as mock_create:
                    mock_create.return_value = {"memory_id": f"test-memory-{i}"}
                    memory_agent.store_state({"test": i}, i, 0.5)

            # Verify _check_memory_transition was called once
            mock_check.assert_called_once()
            assert memory_agent._insert_count == 5

    def test_create_memory_entry(self, memory_agent, mock_embedding_engine):
        """Test memory entry creation with default tier."""
        test_data = {"test": "data"}
        step_number = 42
        memory_type = "state"
        priority = 0.9

        # Mock time.time()
        with mock.patch("time.time", return_value=12345):
            entry = memory_agent._create_memory_entry(
                test_data, step_number, memory_type, priority
            )

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
            assert metadata["current_tier"] == "stm"

            # Check embeddings
            assert "embeddings" in entry

            # Verify embedding engine calls
            mock_embedding_engine.encode_stm.assert_called_once_with(test_data)
            mock_embedding_engine.encode_im.assert_called_once_with(test_data)
            mock_embedding_engine.encode_ltm.assert_called_once_with(test_data)

    def test_create_memory_entry_with_tier(
        self, memory_agent, mock_embedding_engine, mock_compression_engine
    ):
        """Test memory entry creation with different tiers and compression."""
        test_data = {"test": "data"}
        step_number = 42
        memory_type = "state"
        priority = 0.9

        # Test IM tier
        with mock.patch("time.time", return_value=12345):
            entry_im = memory_agent._create_memory_entry(
                test_data, step_number, memory_type, priority, tier="im"
            )

            # Check compression level and tier
            assert entry_im["metadata"]["compression_level"] == 1
            assert entry_im["metadata"]["current_tier"] == "im"

            # Verify compression engine was called for IM
            mock_compression_engine.compress.assert_called_with(test_data, level=1)

        # Test LTM tier
        with mock.patch("time.time", return_value=12346):
            entry_ltm = memory_agent._create_memory_entry(
                test_data, step_number, memory_type, priority, tier="ltm"
            )

            # Check compression level and tier
            assert entry_ltm["metadata"]["compression_level"] == 2
            assert entry_ltm["metadata"]["current_tier"] == "ltm"

            # Verify compression engine was called for LTM
            mock_compression_engine.compress.assert_called_with(test_data, level=2)


class TestMemoryTransitions:
    """Tests for memory transitions between tiers."""

    def test_check_memory_transition_stm_to_im(
        self, memory_agent, mock_stm_store, mock_im_store
    ):
        """Test transitioning memories from STM to IM when STM is at capacity."""
        # Configure mocks
        memory_agent.config.stm_config.memory_limit = 5
        mock_stm_store.count.return_value = 10  # Over capacity

        # Create sample memories in STM
        stm_memories = []
        for i in range(10):
            stm_memories.append(
                {
                    "memory_id": f"mem-{i}",
                    "metadata": {
                        "creation_time": time.time()
                        - (i * 1000),  # Older memories have higher i
                        "importance_score": 0.5,
                        "retrieval_count": i % 3,  # Some variation in retrieval count
                        "current_tier": "stm",  # Current tier is STM
                    },
                }
            )

        mock_stm_store.get_all.return_value = stm_memories

        # Mock the compression engine to capture the metadata updates
        def mock_compress_func(memory, level):
            return {
                "memory_id": memory["memory_id"],
                "metadata": memory[
                    "metadata"
                ].copy(),  # Create a copy to detect changes
            }

        memory_agent.compression_engine.compress.side_effect = mock_compress_func

        # Call the transition method
        with mock.patch.object(
            memory_agent,
            "_calculate_importance",
            side_effect=lambda m: m["metadata"]["importance_score"],
        ):
            memory_agent._check_memory_transition()

            # Check that IM store was called with compressed memories
            assert mock_im_store.store.call_count == 5  # Should transition 5 memories

            # Verify the compression engine was used
            assert memory_agent.compression_engine.compress.call_count == 5

            # Verify current_tier was updated in the metadata
            for call in mock_im_store.store.call_args_list:
                args, _ = call
                memory = args[1]
                assert memory["metadata"]["current_tier"] == "im"

            # Verify memories were deleted from STM
            assert mock_stm_store.delete.call_count == 5

    def test_check_memory_transition_im_to_ltm(
        self, memory_agent, mock_im_store, mock_ltm_store, mock_stm_store
    ):
        """Test transitioning memories from IM to LTM when IM is at capacity."""
        # Configure mocks
        memory_agent.config.im_config.memory_limit = 5
        memory_agent.config.ltm_config.batch_size = 3
        mock_stm_store.count.return_value = 3  # Under capacity
        mock_im_store.count.return_value = 8  # Over capacity

        # Create sample memories in IM
        im_memories = []
        for i in range(8):
            im_memories.append(
                {
                    "memory_id": f"mem-{i}",
                    "metadata": {
                        "creation_time": time.time()
                        - (i * 1000),  # Older memories have higher i
                        "importance_score": 0.5,
                        "retrieval_count": i % 3,  # Some variation in retrieval count
                        "current_tier": "im",  # Current tier is IM
                    },
                }
            )

        mock_im_store.get_all.return_value = im_memories

        # Mock the compression engine to capture the metadata updates
        def mock_compress_func(memory, level):
            return {
                "memory_id": memory["memory_id"],
                "metadata": memory[
                    "metadata"
                ].copy(),  # Create a copy to detect changes
            }

        memory_agent.compression_engine.compress.side_effect = mock_compress_func

        # Call the transition method
        with mock.patch.object(
            memory_agent,
            "_calculate_importance",
            side_effect=lambda m: m["metadata"]["importance_score"],
        ):
            memory_agent._check_memory_transition()

            # Check that LTM store was called with batches
            # Should be 1 call with 3 memories, plus 1 more call with remaining 0 memories
            calls = mock_ltm_store.store_batch.call_args_list
            assert len(calls) == 1  # One batch of 3
            assert len(calls[0][0][0]) == 3  # First batch has 3 items

            # Verify the compression engine was used
            assert memory_agent.compression_engine.compress.call_count == 3

            # Verify current_tier was updated in the metadata
            batch = calls[0][0][0]  # Get the batch of memories
            for memory in batch:
                assert memory["metadata"]["current_tier"] == "ltm"

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
                "surprise_factor": 0.7,
            },
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
                "surprise_factor": 0.0,
            },
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
                "surprise_factor": 0.0,
            },
        }

        high_retrieval_importance = memory_agent._calculate_importance(
            memory_high_retrieval
        )
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

    def test_retrieve_similar_states_with_type_filter(
        self, memory_agent, mock_stm_store
    ):
        """Test retrieving similar states with a memory type filter."""
        # Configure the stores
        mock_stm_store.search_similar.return_value = [
            {"memory_id": "stm1", "similarity_score": 0.9}
        ]

        query_state = {"position": [1, 2, 3]}
        memory_type = "action"

        memory_agent.retrieve_similar_states(query_state, k=5, memory_type=memory_type)

        # Verify memory_type was passed to the store
        mock_stm_store.search_similar.assert_called_once()
        args, kwargs = mock_stm_store.search_similar.call_args
        assert kwargs.get("memory_type") == memory_type

    def test_retrieve_by_time_range(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test retrieving memories within a time range."""
        # Configure the stores
        stm_results = [
            {"memory_id": "stm1", "step_number": 5},
            {"memory_id": "stm2", "step_number": 7},
        ]
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
        mock_stm_store.search_by_step_range.assert_called_once_with(
            memory_agent.agent_id, 1, 10, None
        )
        mock_im_store.search_by_step_range.assert_called_once_with(
            memory_agent.agent_id, 1, 10, None
        )
        mock_ltm_store.search_by_step_range.assert_called_once_with(1, 10, None)

    def test_retrieve_by_attributes(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
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
        assert results[0]["memory_id"] == "im1"  # Most recent
        assert results[1]["memory_id"] == "stm1"
        assert results[2]["memory_id"] == "ltm1"  # Oldest

        # Verify calls to stores
        mock_stm_store.search_by_attributes.assert_called_once_with(
            memory_agent.agent_id, attributes, None
        )
        mock_im_store.search_by_attributes.assert_called_once_with(
            memory_agent.agent_id, attributes, None
        )
        mock_ltm_store.search_by_attributes.assert_called_once_with(
            memory_agent.agent_id, attributes, None
        )

    def test_search_by_embedding(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
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
        memory_agent.compression_engine.compress_embedding.assert_any_call(
            query_embedding, level=1
        )
        memory_agent.compression_engine.compress_embedding.assert_any_call(
            query_embedding, level=2
        )

        # Verify search calls include agent_id for Redis stores
        mock_stm_store.search_by_embedding.assert_called_once_with(
            memory_agent.agent_id, query_embedding, k=3
        )
        mock_im_store.search_by_embedding.assert_called_once()  # Already passes but could be more specific
        mock_ltm_store.search_by_embedding.assert_called_once()  # Already passes but could be more specific

    def test_search_by_content(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
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
        mock_stm_store.search_by_content.assert_called_with(
            memory_agent.agent_id, {"text": text_query}, 3
        )

        # Results should be sorted by relevance score
        assert len(results) == 3
        assert results[0]["memory_id"] == "stm1"  # Highest relevance
        assert results[1]["memory_id"] == "im1"
        assert results[2]["memory_id"] == "ltm1"  # Lowest relevance

        # Test with dict query
        dict_query = {"location": "forest", "action": "explore"}
        memory_agent.search_by_content(dict_query, k=2)

        # Verify dict query was passed as is
        mock_stm_store.search_by_content.assert_called_with(
            memory_agent.agent_id, dict_query, 2
        )


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
        def hook1(event_data, agent):
            return {"result": "hook1"}

        def hook2(event_data, agent):
            return {"result": "hook2"}

        def hook3(event_data, agent):
            return {"result": "hook3"}

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
                "priority": 0.9,
            }

        # Register both hooks
        memory_agent.register_hook("test_event", hook1, priority=5)
        memory_agent.register_hook("test_event", hook2, priority=3)

        # Mock store_state method
        with mock.patch.object(memory_agent, "store_state") as mock_store:
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
                42,  # step_number
                0.9,  # priority
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

    def test_clear_memory(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test clearing all memory stores."""
        result = memory_agent.clear_memory()

        assert result is True
        mock_stm_store.clear.assert_called_once()
        mock_im_store.clear.assert_called_once()
        mock_ltm_store.clear.assert_called_once()

    def test_clear_memory_failure(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test handling failure in clearing memory."""
        mock_stm_store.clear.return_value = False

        result = memory_agent.clear_memory()

        assert result is False

    def test_force_maintenance(self, memory_agent):
        """Test forcing memory maintenance."""
        with mock.patch.object(memory_agent, "_check_memory_transition") as mock_check:
            result = memory_agent.force_maintenance()

            assert result is True
            mock_check.assert_called_once()

    def test_force_maintenance_failure(self, memory_agent):
        """Test handling failure in forced maintenance."""
        with mock.patch.object(
            memory_agent,
            "_check_memory_transition",
            side_effect=Exception("Test exception"),
        ):
            result = memory_agent.force_maintenance()

            assert result is False

    def test_get_memory_statistics(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test retrieving memory statistics."""
        # Configure the mocks
        mock_stm_store.count.return_value = 10
        mock_im_store.count.return_value = 20
        mock_ltm_store.count.return_value = 30

        with mock.patch.object(
            memory_agent, "_calculate_tier_importance", return_value=0.5
        ), mock.patch.object(
            memory_agent, "_calculate_compression_ratio", return_value=2.5
        ), mock.patch.object(
            memory_agent,
            "_get_memory_type_distribution",
            return_value={"state": 30, "action": 20, "interaction": 10},
        ), mock.patch.object(
            memory_agent, "_get_access_patterns", return_value={"most_accessed": []}
        ):

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

    def test_calculate_reward_score(self, memory_agent):
        """Test calculating reward score from a memory."""
        # Memory with no reward
        memory_no_reward = {"content": {}}

        # Memory with reward
        memory_with_reward = {"content": {"reward": 5.0}}

        # Memory with negative reward
        memory_negative_reward = {"content": {"reward": -2.0}}

        # Test with max_reward_score of 10 (the default)
        memory_agent.config.autoencoder_config.max_reward_score = 10.0

        # Test cases
        no_reward_score = memory_agent.calculate_reward_score(memory_no_reward)
        assert no_reward_score == 0.0

        reward_score = memory_agent.calculate_reward_score(memory_with_reward)
        assert reward_score == 0.5  # 5/10 = 0.5

        # Negative rewards should be normalized to 0
        negative_score = memory_agent.calculate_reward_score(memory_negative_reward)
        assert negative_score == 0.0

        # Test with higher reward than max (should cap at 1.0)
        memory_high_reward = {"content": {"reward": 15.0}}
        high_score = memory_agent.calculate_reward_score(memory_high_reward)
        assert high_score == 1.0

    def test_hybrid_retrieve(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test hybrid retrieval combining similarity and attribute matching."""
        # Setup query state
        query_state = {
            "position": {"location": "kitchen"},
            "health": 100,
            "inventory": ["sword", "potion"],
        }

        # Mock vector results
        vector_results = [
            {
                "memory_id": "vector1",
                "similarity_score": 0.9,
                "content": {"position": {"location": "kitchen"}},
            },
            {
                "memory_id": "vector2",
                "similarity_score": 0.7,
                "content": {"position": {"location": "living_room"}},
            },
        ]

        # Mock attribute results
        attr_results = [
            {"memory_id": "attr1", "content": {"position": {"location": "kitchen"}}},
            {
                "memory_id": "vector1",  # Overlapping with vector results
                "content": {"position": {"location": "kitchen"}},
            },
        ]

        # Configure the mocks
        with mock.patch.object(
            memory_agent, "retrieve_similar_states", return_value=vector_results
        ), mock.patch.object(
            memory_agent, "retrieve_by_attributes", return_value=attr_results
        ):

            # Test hybrid retrieval with default weights
            results = memory_agent.hybrid_retrieve(query_state, k=3)

            # Verify the function calls
            memory_agent.retrieve_similar_states.assert_called_with(
                query_state, k=6, memory_type=None, threshold=0.2
            )

            memory_agent.retrieve_by_attributes.assert_called()

            # Check results
            assert len(results) <= 3  # Should not exceed k

            # First result should be vector1 which has both high similarity and attribute match
            if results:
                assert results[0]["memory_id"] == "vector1"
                assert "hybrid_score" in results[0]

            # Test with custom weights
            results_custom = memory_agent.hybrid_retrieve(
                query_state, k=2, vector_weight=0.8, attribute_weight=0.2
            )

            # Results should be ordered by weighted scores
            assert len(results_custom) <= 2

    def test_hybrid_retrieve_no_embeddings(self, memory_agent, mock_stm_store):
        """Test hybrid retrieval when embeddings are not available."""
        # Query state with attributes
        query_state = {"position": {"location": "bedroom"}, "energy": 50}

        # Mock attribute results
        attr_results = [
            {"memory_id": "attr1", "content": {"position": {"location": "bedroom"}}}
        ]

        # Temporarily set embedding_engine to None
        memory_agent.embedding_engine = None

        with mock.patch.object(
            memory_agent, "retrieve_by_attributes", return_value=attr_results
        ):
            results = memory_agent.hybrid_retrieve(query_state, k=3)

            # Should fall back to attribute-based search only
            memory_agent.retrieve_by_attributes.assert_called_once()

            # Check results
            assert len(results) <= 3
            if results:
                assert results[0]["memory_id"] == "attr1"

    def test_flush_to_ltm(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test flushing memories from STM and IM to LTM."""
        # Setup memory data
        stm_memories = [
            {
                "memory_id": "stm1",
                "content": {"data": "stm_data1"},
                "metadata": {"current_tier": "stm"},
            },
            {
                "memory_id": "stm2",
                "content": {"data": "stm_data2"},
                "metadata": {"current_tier": "stm"},
            },
        ]

        im_memories = [
            {
                "memory_id": "im1",
                "content": {"data": "im_data1"},
                "metadata": {"current_tier": "im"},
            },
            {
                "memory_id": "im2",
                "content": {"data": "im_data2"},
                "metadata": {"current_tier": "im"},
            },
        ]

        # Configure mocks
        mock_stm_store.get_all.return_value = stm_memories
        mock_im_store.get_all.return_value = im_memories

        # LTM store flush_memories returns (stored_count, filtered_count)
        mock_ltm_store.flush_memories.return_value = (2, 0)

        # Test flushing both STM and IM
        result = memory_agent.flush_to_ltm(include_stm=True, include_im=True)

        # Verify the calls
        mock_stm_store.get_all.assert_called_once()
        mock_im_store.get_all.assert_called_once()

        # Capture the actual arguments passed to flush_memories to verify tier updates
        stm_call_args = mock_ltm_store.flush_memories.call_args_list[0][0][0]
        im_call_args = mock_ltm_store.flush_memories.call_args_list[1][0][0]

        # Verify tier was updated in all memories before being sent to LTM
        for memory in stm_call_args:
            assert memory["metadata"]["current_tier"] == "ltm"

        for memory in im_call_args:
            assert memory["metadata"]["current_tier"] == "ltm"

        # Verify the clear calls
        mock_stm_store.clear.assert_called_once()
        mock_im_store.clear.assert_called_once()

        # Check result structure
        assert result["stm_stored"] == 2
        assert result["stm_filtered"] == 0
        assert result["im_stored"] == 2
        assert result["im_filtered"] == 0

    def test_flush_to_ltm_with_filtering(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test flushing memories with filtering applied."""
        # Setup memory data
        stm_memories = [
            {
                "memory_id": "stm1",
                "content": {"data": "stm_data1"},
                "metadata": {"current_tier": "stm"},
            },
            {
                "memory_id": "stm2",
                "content": {"data": "stm_data2"},
                "metadata": {"current_tier": "stm"},
            },
        ]

        # Configure mocks
        mock_stm_store.get_all.return_value = stm_memories
        mock_im_store.get_all.return_value = []

        # LTM store flush_memories returns different counts based on force parameter
        mock_ltm_store.flush_memories.side_effect = [
            (1, 1),  # With force=False: 1 stored, 1 filtered
            (2, 0),  # With force=True: 2 stored, 0 filtered
        ]

        # Test flushing with normal filtering
        result1 = memory_agent.flush_to_ltm(
            include_stm=True, include_im=False, force=False
        )

        # Verify call with force=False
        mock_ltm_store.flush_memories.assert_called_with(stm_memories, force=False)

        # Check filtering results
        assert result1["stm_stored"] == 1
        assert result1["stm_filtered"] == 1

        # Reset mocks
        mock_ltm_store.flush_memories.reset_mock()
        mock_stm_store.clear.reset_mock()

        # Test flushing with force=True to bypass filtering
        result2 = memory_agent.flush_to_ltm(
            include_stm=True, include_im=False, force=True
        )

        # Verify call with force=True
        mock_ltm_store.flush_memories.assert_called_with(stm_memories, force=True)

        # Check forced results
        assert result2["stm_stored"] == 2
        assert result2["stm_filtered"] == 0

    def test_flush_to_ltm_error_handling(
        self, memory_agent, mock_stm_store, mock_ltm_store
    ):
        """Test error handling during memory flush operations."""
        # Setup memory data
        stm_memories = [{"memory_id": "stm1", "metadata": {"current_tier": "stm"}}]
        mock_stm_store.get_all.return_value = stm_memories

        # Configure LTM store to raise an exception on first call, then succeed
        mock_ltm_store.flush_memories.side_effect = [
            Exception("Database error"),
            (1, 0),  # Second attempt succeeds
        ]

        # Mock time.sleep to avoid waiting during test
        with mock.patch("time.sleep"):
            # Test flush with retry logic
            result = memory_agent.flush_to_ltm(include_stm=True, include_im=False)

            # Should have called flush_memories twice due to retry
            assert mock_ltm_store.flush_memories.call_count == 2

            # Final result should show the successful second attempt
            assert result["stm_stored"] == 1
            assert result["stm_filtered"] == 0

    def test_calculate_tier_importance(self, memory_agent, mock_stm_store):
        """Test calculating the average importance score for a memory tier."""
        # Create test memories with importance scores
        test_memories = [
            {"metadata": {"importance_score": 0.3}},
            {"metadata": {"importance_score": 0.7}},
            {"metadata": {"importance_score": 0.5}},
        ]

        # Configure the store to return our test memories
        mock_stm_store.get_all.return_value = test_memories

        # Calculate tier importance for STM
        importance = memory_agent._calculate_tier_importance("stm")

        # Average should be (0.3 + 0.7 + 0.5) / 3 = 0.5
        assert importance == 0.5

        # Test with empty store
        mock_stm_store.get_all.return_value = []
        empty_importance = memory_agent._calculate_tier_importance("stm")
        assert empty_importance == 0.0

    def test_calculate_compression_ratio(self, memory_agent, mock_im_store):
        """Test calculating the compression ratio for a memory tier."""
        # Create test memories with original size metadata
        test_memories = [
            {"metadata": {"original_size": 1000}},
            {"metadata": {"original_size": 2000}},
        ]

        # Configure the store
        mock_im_store.get_all.return_value = test_memories
        mock_im_store.get_size.return_value = 1200  # Compressed size

        # Calculate compression ratio for IM
        ratio = memory_agent._calculate_compression_ratio("im")

        # Ratio should be (1000 + 2000) / 1200 = 2.5
        assert ratio == 2.5

        # Test with zero compressed size (should avoid division by zero)
        mock_im_store.get_size.return_value = 0
        zero_ratio = memory_agent._calculate_compression_ratio("im")
        assert zero_ratio == 0.0

        # Test with empty store
        mock_im_store.get_all.return_value = []
        empty_ratio = memory_agent._calculate_compression_ratio("im")
        assert empty_ratio == 0.0

    def test_get_memory_type_distribution(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test getting the distribution of memory types across all tiers."""
        # Create test memories for each store
        stm_memories = [
            {"metadata": {"memory_type": "state"}},
            {"metadata": {"memory_type": "action"}},
        ]

        im_memories = [
            {"metadata": {"memory_type": "state"}},
            {"metadata": {"memory_type": "interaction"}},
            {"metadata": {"memory_type": "interaction"}},
        ]

        ltm_memories = [
            {"metadata": {"memory_type": "state"}},
            {"metadata": {"memory_type": "action"}},
            {"metadata": {"memory_type": "action"}},
            {"metadata": {"memory_type": "interaction"}},
        ]

        # Configure the stores
        mock_stm_store.get_all.return_value = stm_memories
        mock_im_store.get_all.return_value = im_memories
        mock_ltm_store.get_all.return_value = ltm_memories

        # Get distribution
        distribution = memory_agent._get_memory_type_distribution()

        # Check results
        assert distribution["state"] == 3  # 1 from STM, 1 from IM, 1 from LTM
        assert distribution["action"] == 3  # 1 from STM, 0 from IM, 2 from LTM
        assert distribution["interaction"] == 3  # 0 from STM, 2 from IM, 1 from LTM

        # Test with empty stores
        mock_stm_store.get_all.return_value = []
        mock_im_store.get_all.return_value = []
        mock_ltm_store.get_all.return_value = []

        empty_distribution = memory_agent._get_memory_type_distribution()
        assert len(empty_distribution) == 0

    def test_get_access_patterns(
        self, memory_agent, mock_stm_store, mock_im_store, mock_ltm_store
    ):
        """Test getting statistics about memory access patterns."""
        # Create test memories with various access counts
        memories = [
            {"metadata": {"retrieval_count": 10}},
            {"metadata": {"retrieval_count": 5}},
            {"metadata": {"retrieval_count": 0}},
            {"metadata": {"retrieval_count": 3}},
            {"metadata": {"retrieval_count": 8}},
        ]

        # Configure stores to return subsets of the memories
        mock_stm_store.get_all.return_value = memories[0:2]  # 10, 5
        mock_im_store.get_all.return_value = memories[2:3]  # 0
        mock_ltm_store.get_all.return_value = memories[3:5]  # 3, 8

        # Get access patterns
        patterns = memory_agent._get_access_patterns()

        # Check total accesses: 10 + 5 + 0 + 3 + 8 = 26
        assert patterns["total_accesses"] == 26

        # Check average accesses: 26 / 5 = 5.2
        assert patterns["avg_accesses"] == 5.2

        # Check most accessed (should be sorted by retrieval_count, highest first)
        assert len(patterns["most_accessed"]) <= 5
        if patterns["most_accessed"]:
            # First one should be the one with count 10
            assert patterns["most_accessed"][0]["metadata"]["retrieval_count"] == 10

        # Check least accessed
        assert len(patterns["least_accessed"]) <= 5

        # Verify that the least_accessed list contains the memory with retrieval_count 0
        least_accessed_counts = [
            item["metadata"]["retrieval_count"] for item in patterns["least_accessed"]
        ]
        assert 0 in least_accessed_counts

        # Test with empty stores
        mock_stm_store.get_all.return_value = []
        mock_im_store.get_all.return_value = []
        mock_ltm_store.get_all.return_value = []

        empty_patterns = memory_agent._get_access_patterns()
        assert empty_patterns["total_accesses"] == 0
        assert empty_patterns["avg_accesses"] == 0
        assert len(empty_patterns["most_accessed"]) == 0
        assert len(empty_patterns["least_accessed"]) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", "test_memory_agent.py"])
