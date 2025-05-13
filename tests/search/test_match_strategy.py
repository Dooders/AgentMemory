"""Tests for the ExampleMatchingStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.match import ExampleMatchingStrategy


class TestExampleMatchingStrategy(unittest.TestCase):
    """Tests for the ExampleMatchingStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_vector_store = MagicMock()
        self.mock_embedding_engine = MagicMock()
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create mock memory system
        self.mock_memory_system = MagicMock()
        self.mock_memory_system.vector_store = self.mock_vector_store
        self.mock_memory_system.embedding_engine = self.mock_embedding_engine
        
        # Create mock memory agent
        self.mock_agent = MagicMock()
        self.mock_agent.stm_store = self.mock_stm_store
        self.mock_agent.im_store = self.mock_im_store
        self.mock_agent.ltm_store = self.mock_ltm_store
        
        # Configure memory system to return mock agent
        self.mock_memory_system.get_memory_agent.return_value = self.mock_agent

        # Create strategy with mock memory system
        self.strategy = ExampleMatchingStrategy(self.mock_memory_system)

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "match")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("match", self.strategy.description().lower())

    def test_search_with_example_object(self):
        """Test search with an example object."""
        # Set up example memory to match against
        example_memory = {
            "content": {"type": "task", "priority": "high", "status": "pending"},
            "metadata": {"tags": ["important", "project"]},
        }

        # Mock vector search
        self.mock_embedding_engine.encode.return_value = [0.1, 0.2, 0.3]
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.9},
            {"id": "mem2", "score": 0.8},
            {"id": "mem3", "score": 0.7},
        ]

        # Mock memory retrieval
        self.mock_stm_store.get.side_effect = lambda agent_id, memory_id: {
            "mem1": {
                "id": "mem1",
                "content": {"type": "task", "priority": "high", "status": "pending"},
                "metadata": {"tags": ["important", "urgent"]},
            },
            "mem2": {
                "id": "mem2",
                "content": {
                    "type": "task",
                    "priority": "medium",
                    "status": "in_progress",
                },
                "metadata": {"tags": ["project"]},
            },
            "mem3": {
                "id": "mem3",
                "content": {"type": "note", "priority": "low"},
                "metadata": {"tags": ["reference"]},
            },
        }.get(memory_id)

        # Perform search
        results = self.strategy.search(
            query={"example": example_memory}, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify embedding engine and vector store were called
        self.mock_embedding_engine.encode_stm.assert_called_once()
        self.mock_vector_store.find_similar_memories.assert_called_once()

        # Verify results are ordered by similarity score
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "mem1")  # highest score 0.9
        self.assertEqual(results[1]["id"], "mem2")  # score 0.8
        self.assertEqual(results[2]["id"], "mem3")  # score 0.7

        # Check that score is added to metadata
        self.assertEqual(results[0]["metadata"]["match_score"], 0.9)

    def test_search_with_fields_mask(self):
        """Test search with fields mask for matching."""
        # Set up example memory with fields mask
        example_memory = {"content": {"type": "task", "priority": "high"}}

        fields_mask = ["content.type", "content.priority"]

        # Mock vector search
        self.mock_embedding_engine.encode_im.return_value = [0.4, 0.5, 0.6]
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.95},
            {"id": "mem2", "score": 0.75},
        ]

        # Mock memory retrieval
        self.mock_im_store.get.side_effect = lambda agent_id, memory_id: {
            "mem1": {
                "id": "mem1",
                "content": {
                    "type": "task",
                    "priority": "high",
                    "status": "pending",
                    "description": "unrelated field",
                },
                "metadata": {},
            },
            "mem2": {
                "id": "mem2",
                "content": {
                    "type": "task",
                    "priority": "medium",
                    "status": "completed",
                },
                "metadata": {},
            },
        }.get(memory_id)

        # Perform search with fields mask
        results = self.strategy.search(
            query={"example": example_memory, "fields": fields_mask},
            agent_id="agent-1",
            tier="im",
            limit=5,
        )

        # Verify embedding engine was called with only the masked fields
        masked_example = {"content": {"type": "task", "priority": "high"}}
        self.mock_embedding_engine.encode_im.assert_called_with(masked_example)

        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem2")

    def test_search_with_min_score(self):
        """Test search with minimum score threshold."""
        # Set up example
        example_memory = {"content": "example content"}

        # Mock vector search
        self.mock_embedding_engine.encode_ltm.return_value = [0.7, 0.8, 0.9]
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.9},
            {"id": "mem2", "score": 0.6},  # Below min_score
            {"id": "mem3", "score": 0.8},
        ]

        # Mock memory retrieval
        self.mock_ltm_store.get.side_effect = lambda agent_id, memory_id: {
            "mem1": {"id": "mem1", "content": "matching content 1", "metadata": {}},
            "mem2": {"id": "mem2", "content": "matching content 2", "metadata": {}},
            "mem3": {"id": "mem3", "content": "matching content 3", "metadata": {}},
        }.get(memory_id)

        # Perform search with min_score
        results = self.strategy.search(
            query={"example": example_memory},
            agent_id="agent-1",
            tier="ltm",
            min_score=0.7,
            limit=5,
        )

        # Verify results only include those above min_score
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)  # score 0.9
        self.assertIn("mem3", result_ids)  # score 0.8
        self.assertNotIn("mem2", result_ids)  # score 0.6 (below threshold)

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        # Set up example
        example_memory = {"content": "example content"}

        # Mock vector search
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.9},
            {"id": "mem2", "score": 0.8},
        ]

        # Mock memory retrieval
        self.mock_stm_store.get.side_effect = lambda agent_id, memory_id: {
            "mem1": {
                "id": "mem1",
                "content": "content 1",
                "metadata": {"type": "task"},
            },
            "mem2": {
                "id": "mem2",
                "content": "content 2",
                "metadata": {"type": "note"},
            },
        }.get(memory_id)

        # Perform search with metadata filter
        metadata_filter = {"type": "task"}
        results = self.strategy.search(
            query={"example": example_memory},
            agent_id="agent-1",
            tier="stm",
            metadata_filter=metadata_filter,
            limit=5,
        )

        # Verify metadata filter was passed to vector store
        self.mock_vector_store.find_similar_memories.assert_called_with(
            [0.1, 0.2, 0.3],
            tier="stm",
            limit=10,  # 2x the requested limit
            metadata_filter=metadata_filter,
        )

        # Verify results are filtered
        self.assertEqual(len(results), 2)

    def test_extract_fields(self):
        """Test extracting fields from a memory based on a mask."""
        # Set up a test memory
        memory = {
            "content": {
                "title": "Test document",
                "description": "This is a test",
                "metadata": {"author": "Test User", "tags": ["test", "document"]},
                "attributes": {"importance": 5, "visibility": "public"},
            },
            "metadata": {"created_at": "2023-01-01", "type": "document"},
        }

        # Test with content fields only
        fields = ["content.title", "content.attributes.importance"]
        extracted = self.strategy._extract_fields(memory, fields)

        # Verify extraction
        expected = {
            "content": {"title": "Test document", "attributes": {"importance": 5}}
        }
        self.assertEqual(extracted, expected)

        # Test with both content and metadata fields
        fields = ["content.title", "metadata.type"]
        extracted = self.strategy._extract_fields(memory, fields)

        # Verify extraction
        expected = {
            "content": {"title": "Test document"},
            "metadata": {"type": "document"},
        }
        self.assertEqual(extracted, expected)

    def test_invalid_example_format(self):
        """Test handling of invalid example format."""
        # Test with missing example
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={}, agent_id="agent-1", tier="stm"  # Missing example
            )

        # Test with invalid example type
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"example": "not a dict or memory object"},
                agent_id="agent-1",
                tier="stm",
            )

        # Test with invalid fields format
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"example": {}, "fields": "not_a_list"},
                agent_id="agent-1",
                tier="stm",
            )

    def test_search_with_none_vector(self):
        """Test search when the embedding engine returns None for the vector."""
        # Set up example memory
        example_memory = {"content": "example content"}

        # Mock embedding engine to return None
        self.mock_embedding_engine.encode_stm.return_value = None

        # Perform search
        results = self.strategy.search(
            query={"example": example_memory}, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify embedding engine was called
        self.mock_embedding_engine.encode_stm.assert_called_once()

        # Verify no call to find_similar_memories and empty results
        self.mock_vector_store.find_similar_memories.assert_not_called()
        self.assertEqual(results, [])

    def test_search_with_empty_similar_memories(self):
        """Test search when vector store returns empty similar memories."""
        # Set up example memory
        example_memory = {"content": "example content"}

        # Mock embedding engine and vector store
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        self.mock_vector_store.find_similar_memories.return_value = []  # Empty results

        # Perform search
        results = self.strategy.search(
            query={"example": example_memory}, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify calls were made but no memory retrieval attempts
        self.mock_embedding_engine.encode_stm.assert_called_once()
        self.mock_vector_store.find_similar_memories.assert_called_once()
        self.mock_stm_store.get.assert_not_called()

        # Verify empty results
        self.assertEqual(results, [])

    def test_get_store_for_tier_default(self):
        """Test _get_store_for_tier when tier is None (default case)."""
        # Get store with None tier
        store = self.strategy._get_store_for_tier(self.mock_agent, None)

        # Verify default store is stm_store
        self.assertEqual(store, self.mock_stm_store)

    def test_search_with_none_memory(self):
        """Test search when store.get() returns None for a memory."""
        # Set up example memory
        example_memory = {"content": "example content"}

        # Mock embedding engine and vector store
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "mem1", "score": 0.9},
            {"id": "mem2", "score": 0.8},  # This memory won't be found
            {"id": "mem3", "score": 0.7},
        ]

        # Mock memory retrieval with one memory missing
        self.mock_stm_store.get.side_effect = lambda agent_id, memory_id: {
            "mem1": {"id": "mem1", "content": "content 1", "metadata": {}},
            "mem2": None,  # Missing memory
            "mem3": {"id": "mem3", "content": "content 3", "metadata": {}},
        }.get(memory_id)

        # Perform search
        results = self.strategy.search(
            query={"example": example_memory}, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify all memory IDs were looked up
        self.assertEqual(self.mock_stm_store.get.call_count, 3)

        # Verify only found memories are in results
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem3", result_ids)
        self.assertNotIn("mem2", result_ids)


if __name__ == "__main__":
    unittest.main()
