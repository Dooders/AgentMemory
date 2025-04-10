"""Tests for the SimilaritySearchStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.similarity import SimilaritySearchStrategy


class TestSimilaritySearchStrategy(unittest.TestCase):
    """Tests for the SimilaritySearchStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_vector_store = MagicMock()
        self.mock_embedding_engine = MagicMock()
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()
        
        # Create strategy with mock dependencies
        self.strategy = SimilaritySearchStrategy(
            self.mock_vector_store,
            self.mock_embedding_engine,
            self.mock_stm_store,
            self.mock_im_store,
            self.mock_ltm_store
        )
    
    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "similarity")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("similarity", self.strategy.description().lower())
    
    def test_search_text_query(self):
        """Test search with a text query string."""
        # Set up mocks
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        self.mock_embedding_engine.encode_im.return_value = [0.1, 0.2, 0.3]
        self.mock_embedding_engine.encode_ltm.return_value = [0.1, 0.2, 0.3]
        
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "memory1", "score": 0.9},
            {"id": "memory2", "score": 0.8}
        ]
        
        # Mock the store retrieval methods
        self.mock_stm_store.get.return_value = {"id": "memory1", "content": "test content 1", "metadata": {}}
        self.mock_im_store.get.return_value = {"id": "memory2", "content": "test content 2", "metadata": {}}
        
        # Perform search
        results = self.strategy.search(
            query="test query",
            agent_id="agent-1",
            limit=5
        )
        
        # Verify vector store was called for each tier
        self.assertEqual(self.mock_vector_store.find_similar_memories.call_count, 3)
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        # Check that score and tier were added to metadata
        self.assertEqual(results[0]["metadata"]["similarity_score"], 0.9)
        self.assertIn("memory_tier", results[0]["metadata"])
    
    def test_search_with_existing_vector(self):
        """Test search with an existing vector."""
        # Set up mocks
        query_vector = [0.4, 0.5, 0.6]
        
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "memory3", "score": 0.95}
        ]
        
        self.mock_ltm_store.get.return_value = {"id": "memory3", "content": "test content 3", "metadata": {}}
        
        # Perform search with a specific tier and vector
        results = self.strategy.search(
            query=query_vector,
            agent_id="agent-1",
            tier="ltm",
            limit=5
        )
        
        # Verify vector store was called only for the specified tier
        self.assertEqual(self.mock_vector_store.find_similar_memories.call_count, 1)
        
        # Verify the vector was passed directly
        self.mock_vector_store.find_similar_memories.assert_called_with(
            query_vector,
            tier="ltm",
            limit=10,  # 2x the requested limit
            metadata_filter={},
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["similarity_score"], 0.95)
        self.assertEqual(results[0]["metadata"]["memory_tier"], "ltm")
    
    def test_search_with_min_score_filtering(self):
        """Test search with minimum score filtering."""
        # Set up mocks
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "memory1", "score": 0.9},
            {"id": "memory2", "score": 0.6},  # Below min_score
            {"id": "memory3", "score": 0.8}
        ]
        
        self.mock_stm_store.get.side_effect = lambda id: {
            "memory1": {"id": "memory1", "content": "content 1", "metadata": {}},
            "memory3": {"id": "memory3", "content": "content 3", "metadata": {}}
        }.get(id)
        
        # Perform search with min_score
        results = self.strategy.search(
            query="test query",
            agent_id="agent-1",
            tier="stm",
            min_score=0.7,
            limit=5
        )
        
        # Verify results only include those above min_score
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory2", [r["id"] for r in results])
    
    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        # Set up mocks
        self.mock_embedding_engine.encode_im.return_value = [0.1, 0.2, 0.3]
        
        # Perform search with metadata filter
        metadata_filter = {"type": "note", "tags": ["important"]}
        self.strategy.search(
            query="test query",
            agent_id="agent-1",
            tier="im",
            metadata_filter=metadata_filter,
            limit=5
        )
        
        # Verify metadata filter was passed to vector store
        self.mock_vector_store.find_similar_memories.assert_called_with(
            [0.1, 0.2, 0.3],
            tier="im",
            limit=10,  # 2x the requested limit
            metadata_filter=metadata_filter,
        )
    
    def test_generate_query_vector_text(self):
        """Test generating a query vector from text."""
        # Set up mocks
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        
        # Generate vector
        vector = self.strategy._generate_query_vector("test query", "stm")
        
        # Verify embedding engine was called
        self.mock_embedding_engine.encode_stm.assert_called_once()
        
        # Verify result
        self.assertEqual(vector, [0.1, 0.2, 0.3])
    
    def test_generate_query_vector_dict(self):
        """Test generating a query vector from a dictionary."""
        # Set up mocks
        self.mock_embedding_engine.encode_ltm.return_value = [0.4, 0.5, 0.6]
        
        # Generate vector
        query_dict = {"content": "test", "type": "note"}
        vector = self.strategy._generate_query_vector(query_dict, "ltm")
        
        # Verify embedding engine was called with the dict
        self.mock_embedding_engine.encode_ltm.assert_called_once_with(query_dict)
        
        # Verify result
        self.assertEqual(vector, [0.4, 0.5, 0.6])
    
    def test_generate_query_vector_existing_vector(self):
        """Test generating a query vector with an existing vector."""
        # Input is already a vector
        existing_vector = [0.7, 0.8, 0.9]
        
        # Generate vector
        vector = self.strategy._generate_query_vector(existing_vector, "im")
        
        # Verify embedding engine was not called
        self.mock_embedding_engine.encode_im.assert_not_called()
        
        # Verify the vector was returned as-is
        self.assertEqual(vector, existing_vector)
    
    def test_generate_query_vector_invalid_input(self):
        """Test generating a query vector with invalid input."""
        # Set up an invalid input (neither string, dict, nor vector)
        invalid_input = None
        
        # Generate vector
        vector = self.strategy._generate_query_vector(invalid_input, "stm")
        
        # Verify no embedding was generated
        self.assertIsNone(vector)
    
    def test_search_result_sorting(self):
        """Test that search results are sorted by score."""
        # Set up mocks
        self.mock_embedding_engine.encode_stm.return_value = [0.1, 0.2, 0.3]
        
        # Set up vector store to return results in non-sorted order
        self.mock_vector_store.find_similar_memories.return_value = [
            {"id": "memory1", "score": 0.7},
            {"id": "memory2", "score": 0.9},
            {"id": "memory3", "score": 0.8}
        ]
        
        # Set up memory store get method
        self.mock_stm_store.get.side_effect = lambda id: {
            "memory1": {"id": "memory1", "content": "content 1", "metadata": {}},
            "memory2": {"id": "memory2", "content": "content 2", "metadata": {}},
            "memory3": {"id": "memory3", "content": "content 3", "metadata": {}}
        }.get(id)
        
        # Perform search
        results = self.strategy.search(
            query="test query",
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Verify results are sorted by score (descending)
        self.assertEqual(results[0]["id"], "memory2")  # score 0.9
        self.assertEqual(results[1]["id"], "memory3")  # score 0.8
        self.assertEqual(results[2]["id"], "memory1")  # score 0.7


if __name__ == "__main__":
    unittest.main() 