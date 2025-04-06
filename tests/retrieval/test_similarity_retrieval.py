"""Unit tests for the similarity-based memory retrieval mechanisms."""

from unittest.mock import Mock

import pytest

from memory.retrieval.similarity import SimilarityRetrieval


class TestSimilarityRetrieval:
    """Test suite for the SimilarityRetrieval class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        mock = Mock(name="mock_vector_store")
        return mock

    @pytest.fixture
    def mock_embedding_engine(self):
        """Create a mock embedding engine."""
        mock = Mock(name="mock_embedding_engine")
        # Set up encoding methods to return test vectors
        mock.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock.encode_im.return_value = [0.4, 0.5]
        mock.encode_ltm.return_value = [0.6]
        return mock

    @pytest.fixture
    def mock_stm_store(self):
        """Create a mock STM store."""
        mock = Mock(name="mock_stm_store")
        return mock

    @pytest.fixture
    def mock_im_store(self):
        """Create a mock IM store."""
        mock = Mock(name="mock_im_store")
        return mock

    @pytest.fixture
    def mock_ltm_store(self):
        """Create a mock LTM store."""
        mock = Mock(name="mock_ltm_store")
        return mock

    @pytest.fixture
    def retriever(
        self,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        mock_im_store,
        mock_ltm_store,
    ):
        """Create a SimilarityRetrieval instance with mocked dependencies."""
        return SimilarityRetrieval(
            mock_vector_store,
            mock_embedding_engine,
            mock_stm_store,
            mock_im_store,
            mock_ltm_store,
        )

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            {
                "memory_id": "mem1",
                "memory_type": "observation",
                "importance": 0.8,
                "metadata": {"location": "kitchen", "mood": "happy"},
                "contents": {"text": "I saw something interesting in the kitchen"},
                "embeddings": {
                    "full_vector": [0.1, 0.2, 0.3],
                    "compressed_vector": [0.4, 0.5],
                    "abstract_vector": [0.6],
                },
            },
            {
                "memory_id": "mem2",
                "memory_type": "action",
                "importance": 0.6,
                "metadata": {"location": "living_room", "mood": "excited"},
                "contents": {"text": "I moved to the living room"},
                "embeddings": {
                    "full_vector": [0.2, 0.3, 0.4],
                    "compressed_vector": [0.5, 0.6],
                    "abstract_vector": [0.7],
                },
            },
            {
                "memory_id": "mem3",
                "memory_type": "dialog",
                "importance": 0.9,
                "metadata": {"location": "living_room", "target": "agent2"},
                "contents": {"text": "I talked with agent2"},
                "embeddings": {
                    "full_vector": [0.3, 0.4, 0.5],
                    "compressed_vector": [0.6, 0.7],
                    "abstract_vector": [0.8],
                },
            },
        ]

    @pytest.fixture
    def vector_search_results(self):
        """Create sample vector search results."""
        return [
            {"id": "mem1", "score": 0.95, "metadata": {"memory_type": "observation"}},
            {"id": "mem2", "score": 0.85, "metadata": {"memory_type": "action"}},
            {"id": "mem3", "score": 0.75, "metadata": {"memory_type": "dialog"}},
            {"id": "mem4", "score": 0.65, "metadata": {"memory_type": "observation"}},
        ]

    def test_initialization(
        self,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        mock_im_store,
        mock_ltm_store,
    ):
        """Test that SimilarityRetrieval initializes correctly."""
        retriever = SimilarityRetrieval(
            mock_vector_store,
            mock_embedding_engine,
            mock_stm_store,
            mock_im_store,
            mock_ltm_store,
        )

        assert retriever.vector_store == mock_vector_store
        assert retriever.embedding_engine == mock_embedding_engine
        assert retriever.stm_store == mock_stm_store
        assert retriever.im_store == mock_im_store
        assert retriever.ltm_store == mock_ltm_store

    def test_retrieve_similar_to_state_stm(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a state from STM."""
        # Set up mocks
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a simple state
        test_state = {"text": "Looking for something similar"}
        result = retriever.retrieve_similar_to_state(
            test_state, limit=2, min_score=0.8, tier="stm"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_stm.assert_called_once_with(test_state)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.1, 0.2, 0.3],
            tier="stm",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
            metadata_filter={},
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"
        assert result[0]["metadata"]["similarity_score"] == 0.95
        assert result[1]["metadata"]["similarity_score"] == 0.85

    def test_retrieve_similar_to_state_im(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_im_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a state from IM."""
        # Set up mocks
        mock_embedding_engine.encode_im.return_value = [0.4, 0.5]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_im_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a simple state
        test_state = {"text": "Looking for something similar"}
        result = retriever.retrieve_similar_to_state(
            test_state, limit=2, min_score=0.8, tier="im"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_im.assert_called_once_with(test_state)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.4, 0.5],
            tier="im",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
            metadata_filter={},
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"
        assert result[0]["metadata"]["similarity_score"] == 0.95
        assert result[1]["metadata"]["similarity_score"] == 0.85

    def test_retrieve_similar_to_state_ltm(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_ltm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a state from LTM."""
        # Set up mocks
        mock_embedding_engine.encode_ltm.return_value = [0.6]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_ltm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a simple state
        test_state = {"text": "Looking for something similar"}
        result = retriever.retrieve_similar_to_state(
            test_state, limit=2, min_score=0.8, tier="ltm"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_ltm.assert_called_once_with(test_state)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.6],
            tier="ltm",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
            metadata_filter={},
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"
        assert result[0]["metadata"]["similarity_score"] == 0.95
        assert result[1]["metadata"]["similarity_score"] == 0.85

    def test_retrieve_similar_to_state_with_memory_type(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a state with memory_type filter."""
        # Set up mocks
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with memory_type filter
        test_state = {"text": "Looking for observation"}
        result = retriever.retrieve_similar_to_state(
            test_state, limit=2, min_score=0.7, memory_type="observation", tier="stm"
        )

        # Verify metadata filter was used
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.1, 0.2, 0.3],
            tier="stm",
            limit=4,
            metadata_filter={"memory_type": "observation"},
        )

    def test_retrieve_similar_to_memory_stm(
        self,
        retriever,
        mock_vector_store,
        mock_stm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a memory from STM."""
        # Set up vector store and memory retrieval
        mock_vector_store.find_similar_memories.return_value = vector_search_results
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a memory ID
        result = retriever.retrieve_similar_to_memory(
            "mem1", limit=2, min_score=0.8, tier="stm"
        )

        # Verify vector store was called with the right vector
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.1, 0.2, 0.3],  # the full_vector from mem1
            tier="stm",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
        )

        # Verify results (at least one result, mem1 is excluded by default)
        assert len(result) >= 1
        # We should get at least mem2, but not mem1 (excluded by default)
        assert "mem1" not in [m["memory_id"] for m in result]
        assert result[0]["memory_id"] == "mem2"

    def test_retrieve_similar_to_memory_im(
        self,
        retriever,
        mock_vector_store,
        mock_im_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a memory from IM."""
        # Set up vector store and memory retrieval
        mock_vector_store.find_similar_memories.return_value = vector_search_results
        mock_im_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a memory ID
        result = retriever.retrieve_similar_to_memory(
            "mem1", limit=2, min_score=0.8, tier="im"
        )

        # Verify vector store was called with the right vector
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.4, 0.5],  # the compressed_vector from mem1
            tier="im",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
        )

        # Verify results (at least one result, mem1 is excluded by default)
        assert len(result) >= 1
        assert "mem1" not in [m["memory_id"] for m in result]

    def test_retrieve_similar_to_memory_ltm(
        self,
        retriever,
        mock_vector_store,
        mock_ltm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a memory from LTM."""
        # Set up vector store and memory retrieval
        mock_vector_store.find_similar_memories.return_value = vector_search_results
        mock_ltm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with a memory ID
        result = retriever.retrieve_similar_to_memory(
            "mem1", limit=2, min_score=0.8, tier="ltm"
        )

        # Verify vector store was called with the right vector
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.6],  # the abstract_vector from mem1
            tier="ltm",
            limit=4,  # 2*2 as we request 2 but code doubles to allow filtering
        )

        # Verify results (at least one result, mem1 is excluded by default)
        assert len(result) >= 1
        assert "mem1" not in [m["memory_id"] for m in result]

    def test_retrieve_similar_to_memory_include_self(
        self,
        retriever,
        mock_vector_store,
        mock_stm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories similar to a memory including self."""
        # Set up vector store and memory retrieval
        mock_vector_store.find_similar_memories.return_value = vector_search_results
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with exclude_self=False
        result = retriever.retrieve_similar_to_memory(
            "mem1", limit=2, min_score=0.8, exclude_self=False, tier="stm"
        )

        # Verify we get the memory itself in the results if it meets score criteria
        assert len(result) == 2
        # First result will be mem1 itself since it has highest similarity score
        assert result[0]["memory_id"] == "mem1"

    def test_retrieve_similar_to_memory_nonexistent(self, retriever, mock_stm_store):
        """Test behavior when retrieving similar to a non-existent memory."""
        # Set up the store to return None (memory not found)
        mock_stm_store.get.return_value = None

        # Test with a non-existent memory ID
        result = retriever.retrieve_similar_to_memory("nonexistent", tier="stm")

        # Should return empty list
        assert result == []

    def test_retrieve_by_example_stm(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories by example in STM."""
        # Set up mocks
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with an example
        example = {"text": "Example pattern to match"}
        result = retriever.retrieve_by_example(
            example, limit=2, min_score=0.8, tier="stm"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_stm.assert_called_once_with(example)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.1, 0.2, 0.3],
            tier="stm",
            limit=4,
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"
        assert result[0]["metadata"]["similarity_score"] == 0.95
        assert result[1]["metadata"]["similarity_score"] == 0.85

    def test_retrieve_by_example_im(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_im_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories by example in IM."""
        # Set up mocks
        mock_embedding_engine.encode_im.return_value = [0.4, 0.5]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_im_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with an example
        example = {"text": "Example pattern to match"}
        result = retriever.retrieve_by_example(
            example, limit=2, min_score=0.8, tier="im"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_im.assert_called_once_with(example)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.4, 0.5],
            tier="im",
            limit=4,
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"

    def test_retrieve_by_example_ltm(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_ltm_store,
        vector_search_results,
        sample_memories,
    ):
        """Test retrieving memories by example in LTM."""
        # Set up mocks
        mock_embedding_engine.encode_ltm.return_value = [0.6]
        mock_vector_store.find_similar_memories.return_value = vector_search_results

        # Set up memory retrieval
        mock_ltm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with an example
        example = {"text": "Example pattern to match"}
        result = retriever.retrieve_by_example(
            example, limit=2, min_score=0.8, tier="ltm"
        )

        # Verify mocks were called correctly
        mock_embedding_engine.encode_ltm.assert_called_once_with(example)
        mock_vector_store.find_similar_memories.assert_called_once_with(
            [0.6],
            tier="ltm",
            limit=4,
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"

    def test_score_filtering(
        self,
        retriever,
        mock_vector_store,
        mock_embedding_engine,
        mock_stm_store,
        sample_memories,
    ):
        """Test that results are properly filtered by score."""
        # Mock vector search with varying scores
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.95, "metadata": {}},
            {"id": "mem2", "score": 0.85, "metadata": {}},
            {"id": "mem3", "score": 0.75, "metadata": {}},
            {"id": "mem4", "score": 0.65, "metadata": {}},
        ]

        # Set up memory retrieval
        mock_stm_store.get.side_effect = lambda id: next(
            (m for m in sample_memories if m["memory_id"] == id), None
        )

        # Test with higher min_score
        result = retriever.retrieve_similar_to_state(
            {"text": "Test state"},
            limit=10,  # High limit to not restrict by count
            min_score=0.8,  # Should only return mem1 and mem2
            tier="stm",
        )

        # Should have filtered out lower scores
        assert len(result) == 2
        assert [m["memory_id"] for m in result] == ["mem1", "mem2"]

    def test_memory_lookup_failure(
        self, retriever, mock_vector_store, mock_embedding_engine, mock_stm_store
    ):
        """Test handling of memory lookup failures."""
        # Set up mocks
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.95, "metadata": {}},
            {"id": "mem2", "score": 0.85, "metadata": {}},
        ]

        # Set up memory store to return None for some IDs (simulating lookup failure)
        mock_stm_store.get.side_effect = lambda id: (
            {"memory_id": id} if id == "mem1" else None
        )

        # Test retrieval
        result = retriever.retrieve_similar_to_state(
            {"text": "Test state"}, limit=2, min_score=0.8, tier="stm"
        )

        # Should only return mem1, since mem2 lookup failed
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    def test_memory_without_embeddings(
        self, retriever, mock_vector_store, mock_embedding_engine, mock_stm_store
    ):
        """Test retrieving similar to a memory without embeddings."""
        # Create a memory without embeddings
        memory_without_embeddings = {
            "memory_id": "no_embed",
            "contents": {"text": "Memory without embeddings"},
        }

        # Set up the store to return this memory
        mock_stm_store.get.return_value = memory_without_embeddings

        # Set up the embedding engine to generate a vector
        mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]

        # Set up vector store results
        mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.95, "metadata": {}},
        ]

        # Test retrieval
        result = retriever.retrieve_similar_to_memory("no_embed", tier="stm")

        # Should generate embedding from contents
        mock_embedding_engine.encode_stm.assert_called_once_with(
            memory_without_embeddings["contents"]
        )

        # Should return results
        assert len(result) == 1
