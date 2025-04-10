"""Tests for the SearchModel class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.config import MemoryConfig
from memory.search.model import SearchModel
from memory.search.strategies.base import SearchStrategy


class MockSearchStrategy(SearchStrategy):
    """Mock search strategy for testing."""
    
    def __init__(self, name_str="mock_strategy", description_str="Mock strategy for testing"):
        self._name = name_str
        self._description = description_str
        # Create a mock method that we can use to check calls
        self.search_mock = MagicMock(return_value=[])
    
    def search(self, query, agent_id, limit=10, metadata_filter=None, tier=None, **kwargs):
        """Mock implementation of search."""
        return self.search_mock(query, agent_id, limit, metadata_filter, tier, **kwargs)
    
    def name(self):
        """Return the strategy name."""
        return self._name
    
    def description(self):
        """Return the strategy description."""
        return self._description


class TestSearchModel(unittest.TestCase):
    """Tests for the SearchModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MemoryConfig()
        self.search_model = SearchModel(self.config)
        self.mock_strategy = MockSearchStrategy()
    
    def test_register_strategy(self):
        """Test registering a search strategy."""
        # Register the strategy
        result = self.search_model.register_strategy(self.mock_strategy)
        
        # Check registration was successful
        self.assertTrue(result)
        self.assertIn("mock_strategy", self.search_model.strategies)
        self.assertEqual(self.search_model.default_strategy, "mock_strategy")
    
    def test_register_strategy_make_default(self):
        """Test registering a strategy and explicitly making it default."""
        # Register a strategy
        strategy1 = MockSearchStrategy("strategy1")
        self.search_model.register_strategy(strategy1)
        
        # Register another strategy and make it default
        strategy2 = MockSearchStrategy("strategy2")
        result = self.search_model.register_strategy(strategy2, make_default=True)
        
        # Check that the second strategy is the default
        self.assertTrue(result)
        self.assertEqual(self.search_model.default_strategy, "strategy2")
    
    def test_unregister_strategy(self):
        """Test unregistering a search strategy."""
        # Register a strategy
        self.search_model.register_strategy(self.mock_strategy)
        
        # Unregister it
        result = self.search_model.unregister_strategy("mock_strategy")
        
        # Check unregistration was successful
        self.assertTrue(result)
        self.assertNotIn("mock_strategy", self.search_model.strategies)
        self.assertIsNone(self.search_model.default_strategy)
    
    def test_unregister_unknown_strategy(self):
        """Test unregistering an unknown strategy."""
        result = self.search_model.unregister_strategy("unknown_strategy")
        
        # Should return False for unknown strategy
        self.assertFalse(result)
    
    def test_unregister_default_strategy_with_fallback(self):
        """Test unregistering the default strategy when other strategies exist."""
        # Register two strategies
        strategy1 = MockSearchStrategy("strategy1")
        strategy2 = MockSearchStrategy("strategy2")
        
        self.search_model.register_strategy(strategy1)
        self.search_model.register_strategy(strategy2, make_default=True)
        
        # Unregister the default one
        self.search_model.unregister_strategy("strategy2")
        
        # Default should fall back to the other strategy
        self.assertEqual(self.search_model.default_strategy, "strategy1")
    
    def test_set_default_strategy(self):
        """Test setting the default search strategy."""
        # Register two strategies
        strategy1 = MockSearchStrategy("strategy1")
        strategy2 = MockSearchStrategy("strategy2")
        
        self.search_model.register_strategy(strategy1)
        self.search_model.register_strategy(strategy2)
        
        # Set the default
        result = self.search_model.set_default_strategy("strategy2")
        
        # Check that the default was set
        self.assertTrue(result)
        self.assertEqual(self.search_model.default_strategy, "strategy2")
    
    def test_set_unknown_default_strategy(self):
        """Test setting an unknown strategy as default."""
        result = self.search_model.set_default_strategy("unknown_strategy")
        
        # Should return False for unknown strategy
        self.assertFalse(result)
    
    def test_search_with_default_strategy(self):
        """Test search using the default strategy."""
        # Register a strategy and make it default
        self.search_model.register_strategy(self.mock_strategy, make_default=True)
        
        # Perform search
        query = "test query"
        agent_id = "agent-1"
        
        self.search_model.search(query, agent_id)
        
        # Check that the strategy's search method was called with correct params
        self.mock_strategy.search_mock.assert_called_once_with(
            query, agent_id, 10, None, None
        )
    
    def test_search_with_specified_strategy(self):
        """Test search using a specified strategy."""
        # Register two strategies
        strategy1 = MockSearchStrategy("strategy1")
        strategy2 = MockSearchStrategy("strategy2")
        
        self.search_model.register_strategy(strategy1, make_default=True)
        self.search_model.register_strategy(strategy2)
        
        # Perform search with the non-default strategy
        query = "test query"
        agent_id = "agent-1"
        
        self.search_model.search(query, agent_id, strategy_name="strategy2")
        
        # Check that strategy2's search method was called
        strategy2.search_mock.assert_called_once()
        # And strategy1's search method was not called
        self.assertEqual(strategy1.search_mock.call_count, 0)
    
    def test_search_with_additional_params(self):
        """Test search with additional parameters."""
        # Register a strategy
        self.search_model.register_strategy(self.mock_strategy, make_default=True)
        
        # Perform search with additional params
        query = "test query"
        agent_id = "agent-1"
        metadata_filter = {"type": "note"}
        tier = "stm"
        custom_param = "custom_value"
        
        self.search_model.search(
            query, 
            agent_id, 
            limit=5, 
            metadata_filter=metadata_filter, 
            tier=tier, 
            custom_param=custom_param
        )
        
        # Check that the strategy's search method was called with all params
        self.mock_strategy.search_mock.assert_called_once_with(
            query, agent_id, 5, metadata_filter, tier, custom_param=custom_param
        )
    
    def test_search_with_no_valid_strategy(self):
        """Test search when no valid strategy is available."""
        # Don't register any strategies
        
        # Perform search
        result = self.search_model.search("query", "agent-1")
        
        # Should return empty list
        self.assertEqual(result, [])
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        # Register two strategies
        strategy1 = MockSearchStrategy("strategy1", "First mock strategy")
        strategy2 = MockSearchStrategy("strategy2", "Second mock strategy")
        
        self.search_model.register_strategy(strategy1)
        self.search_model.register_strategy(strategy2)
        
        # Get available strategies
        strategies = self.search_model.get_available_strategies()
        
        # Check that both strategies are included with correct descriptions
        self.assertEqual(len(strategies), 2)
        self.assertEqual(strategies["strategy1"], "First mock strategy")
        self.assertEqual(strategies["strategy2"], "Second mock strategy")


if __name__ == "__main__":
    unittest.main() 