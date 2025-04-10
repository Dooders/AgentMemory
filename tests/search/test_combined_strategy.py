"""Tests for the CombinedSearchStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.combined import CombinedSearchStrategy
from memory.search.strategies.base import SearchStrategy


class MockSearchStrategy(SearchStrategy):
    """Mock search strategy for testing."""
    
    def __init__(self, name_str, description_str="Mock strategy for testing", results=None):
        self._name = name_str
        self._description = description_str
        self._results = results or []
    
    def search(self, query, agent_id, limit=10, metadata_filter=None, tier=None, **kwargs):
        """Mock implementation of search."""
        return self._results
    
    def name(self):
        """Return the strategy name."""
        return self._name
    
    def description(self):
        """Return the strategy description."""
        return self._description


class TestCombinedSearchStrategy(unittest.TestCase):
    """Tests for the CombinedSearchStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock results for different strategies
        self.similarity_results = [
            {
                "id": "memory1",
                "content": "Meeting about project",
                "metadata": {"similarity_score": 0.9, "type": "meeting"}
            },
            {
                "id": "memory2",
                "content": "Email to client",
                "metadata": {"similarity_score": 0.8, "type": "email"}
            }
        ]
        
        self.temporal_results = [
            {
                "id": "memory3",
                "content": "Recent call with team",
                "metadata": {"similarity_score": 0.75, "type": "call"}
            },
            {
                "id": "memory1",  # Duplicate with different score
                "content": "Meeting about project",
                "metadata": {"similarity_score": 0.6, "type": "meeting"}
            }
        ]
        
        self.attribute_results = [
            {
                "id": "memory4",
                "content": "Task planning session",
                "metadata": {"similarity_score": 0.85, "type": "meeting"}
            },
            {
                "id": "memory2",  # Duplicate with different score
                "content": "Email to client",
                "metadata": {"similarity_score": 0.7, "type": "email"}
            }
        ]
        
        # Create mock strategies
        self.mock_similarity = MockSearchStrategy(
            "similarity", "Similarity search", self.similarity_results
        )
        self.mock_temporal = MockSearchStrategy(
            "temporal", "Temporal search", self.temporal_results
        )
        self.mock_attribute = MockSearchStrategy(
            "attribute", "Attribute search", self.attribute_results
        )
        
        # Create a dictionary of strategies
        self.strategies = {
            "similarity": self.mock_similarity,
            "temporal": self.mock_temporal,
            "attribute": self.mock_attribute
        }
        
        # Create strategy with equal weights
        self.strategy = CombinedSearchStrategy(self.strategies)
    
    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "combined")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("similarity", self.strategy.description().lower())
        self.assertIn("temporal", self.strategy.description().lower())
        self.assertIn("attribute", self.strategy.description().lower())
    
    def test_default_weights(self):
        """Test default weights assignment."""
        # Each strategy should have weight 1.0 by default
        self.assertEqual(self.strategy.weights["similarity"], 1.0)
        self.assertEqual(self.strategy.weights["temporal"], 1.0)
        self.assertEqual(self.strategy.weights["attribute"], 1.0)
    
    def test_custom_weights(self):
        """Test initializing with custom weights."""
        weights = {
            "similarity": 2.0,
            "temporal": 1.5,
            # Note: no weight for attribute, should default to 1.0
        }
        
        strategy = CombinedSearchStrategy(self.strategies, weights)
        
        self.assertEqual(strategy.weights["similarity"], 2.0)
        self.assertEqual(strategy.weights["temporal"], 1.5)
        self.assertEqual(strategy.weights["attribute"], 1.0)  # Default
    
    def test_set_weights(self):
        """Test setting weights after initialization."""
        new_weights = {
            "similarity": 3.0,
            "temporal": 0.5,
            "nonexistent": 1.0  # Should be ignored
        }
        
        result = self.strategy.set_weights(new_weights)
        
        # Should return True
        self.assertTrue(result)
        
        # Weights should be updated
        self.assertEqual(self.strategy.weights["similarity"], 3.0)
        self.assertEqual(self.strategy.weights["temporal"], 0.5)
        self.assertEqual(self.strategy.weights["attribute"], 1.0)  # Unchanged
        
        # Nonexistent strategy should not be added
        self.assertNotIn("nonexistent", self.strategy.weights)
    
    def test_search_combines_results(self):
        """Test that search combines results from all strategies."""
        # Create a new mock_search_strategy for more predictable results
        similarity_results = [
            {"id": "memory1", "content": "Content 1", "metadata": {"similarity_score": 0.9}},
            {"id": "memory2", "content": "Content 2", "metadata": {"similarity_score": 0.8}}
        ]
        
        temporal_results = [
            {"id": "memory3", "content": "Content 3", "metadata": {"similarity_score": 0.9}},
            {"id": "memory4", "content": "Content 4", "metadata": {"similarity_score": 0.8}}
        ]
        
        # Create strategies with predictable results
        mock_similarity = MockSearchStrategy("similarity", results=similarity_results)
        mock_temporal = MockSearchStrategy("temporal", results=temporal_results)
        
        strategies = {
            "similarity": mock_similarity,
            "temporal": mock_temporal
        }
        
        # Create strategy with mock strategies
        strategy = CombinedSearchStrategy(strategies)
        
        # Perform search
        results = strategy.search(
            query="test query",
            agent_id="agent-1",
            limit=4
        )
        
        # Should contain results from both strategies
        self.assertEqual(len(results), 4)
        
        # Check that all memory IDs are present
        result_ids = [r["id"] for r in results]
        self.assertIn("memory1", result_ids)
        self.assertIn("memory2", result_ids)
        self.assertIn("memory3", result_ids)
        self.assertIn("memory4", result_ids)
        
        # Verify source strategy and combined score added to metadata
        for result in results:
            self.assertIn("source_strategy", result["metadata"])
            self.assertIn("combined_score", result["metadata"])
    
    def test_search_deduplicates_results(self):
        """Test that search deduplicates results with the same ID."""
        # Create results with duplicate IDs but different scores
        similarity_results = [
            {"id": "memory1", "content": "Content 1", "metadata": {"similarity_score": 0.9}},
            {"id": "memory2", "content": "Content 2", "metadata": {"similarity_score": 0.8}}
        ]
        
        temporal_results = [
            {"id": "memory1", "content": "Content 1", "metadata": {"similarity_score": 0.7}},
            {"id": "memory3", "content": "Content 3", "metadata": {"similarity_score": 0.6}}
        ]
        
        # Create strategies with duplicate results
        mock_similarity = MockSearchStrategy("similarity", results=similarity_results)
        mock_temporal = MockSearchStrategy("temporal", results=temporal_results)
        
        strategies = {
            "similarity": mock_similarity,
            "temporal": mock_temporal
        }
        
        # Create strategy with mock strategies
        strategy = CombinedSearchStrategy(strategies)
        
        # Perform search
        results = strategy.search(
            query="test query",
            agent_id="agent-1",
            limit=3
        )
        
        # Should contain 3 unique results (memory1 should appear only once)
        self.assertEqual(len(results), 3)
        
        # Check that all memory IDs are present
        result_ids = [r["id"] for r in results]
        self.assertEqual(result_ids.count("memory1"), 1)  # memory1 should appear only once
        self.assertIn("memory2", result_ids)
        self.assertIn("memory3", result_ids)
        
        # The duplicate with the higher score should be kept
        # Find memory1 in the results
        memory1 = next(r for r in results if r["id"] == "memory1")
        
        # Should have source_strategy from the similarity strategy (higher score)
        self.assertEqual(memory1["metadata"]["source_strategy"], "similarity")
    
    def test_search_respects_limit(self):
        """Test that search respects the limit parameter."""
        # Create many results
        many_results = [
            {"id": f"memory{i}", "content": f"Content {i}", "metadata": {"similarity_score": 0.9 - (i * 0.1)}}
            for i in range(10)
        ]
        
        # Create strategy with many results
        mock_strategy = MockSearchStrategy("mock", results=many_results)
        strategies = {"mock": mock_strategy}
        
        strategy = CombinedSearchStrategy(strategies)
        
        # Perform search with limit
        results = strategy.search(
            query="test query",
            agent_id="agent-1",
            limit=3
        )
        
        # Should respect the limit
        self.assertEqual(len(results), 3)
    
    def test_search_with_strategy_params(self):
        """Test search with strategy-specific parameters."""
        # Create a mock strategy with a spy on the search method
        mock_strategy = MagicMock()
        mock_strategy.name.return_value = "mock"
        mock_strategy.description.return_value = "Mock strategy"
        mock_strategy.search.return_value = []
        
        strategies = {"mock": mock_strategy}
        
        strategy = CombinedSearchStrategy(strategies)
        
        # Perform search with strategy-specific parameters
        strategy_params = {
            "mock": {
                "custom_param": "custom_value"
            }
        }
        
        strategy.search(
            query="test query",
            agent_id="agent-1",
            strategy_params=strategy_params,
            shared_param="shared_value"
        )
        
        # Verify the strategy's search method was called with both specific and shared params
        mock_strategy.search.assert_called_once()
        call_args = mock_strategy.search.call_args[1]
        
        self.assertEqual(call_args["custom_param"], "custom_value")
        self.assertEqual(call_args["shared_param"], "shared_value")


if __name__ == "__main__":
    unittest.main() 