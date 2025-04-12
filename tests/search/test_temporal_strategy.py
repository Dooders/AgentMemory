"""Tests for the TemporalSearchStrategy class."""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from memory.search.strategies.temporal import TemporalSearchStrategy


class TestTemporalSearchStrategy(unittest.TestCase):
    """Tests for the TemporalSearchStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()
        
        # Create strategy with mock dependencies
        self.strategy = TemporalSearchStrategy(
            self.mock_stm_store,
            self.mock_im_store,
            self.mock_ltm_store
        )
        
        # Create reference times for testing
        self.now = datetime.now()
        
        # Sample memories for testing with different creation times
        self.older_memory = {
            "id": "memory1",
            "content": "This is an older memory",
            "created_at": (self.now - timedelta(days=7)).isoformat(),
            "metadata": {
                "type": "note",
                "importance": "medium"
            }
        }
        
        self.recent_memory = {
            "id": "memory2",
            "content": "This is a more recent memory",
            "created_at": (self.now - timedelta(hours=12)).isoformat(),
            "metadata": {
                "type": "meeting",
                "importance": "high"
            }
        }
        
        self.very_recent_memory = {
            "id": "memory3",
            "content": "This is a very recent memory",
            "created_at": (self.now - timedelta(hours=1)).isoformat(),
            "metadata": {
                "type": "note",
                "importance": "low"
            }
        }
        
        self.sample_memories = [
            self.older_memory,
            self.recent_memory,
            self.very_recent_memory
        ]
    
    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "temporal")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("temporal", self.strategy.description().lower())
    
    def test_search_with_time_range(self):
        """Test search with explicit time range."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories
        self.mock_im_store.get_all.return_value = []
        self.mock_ltm_store.get_all.return_value = []
        
        # Define a time range that includes only the two more recent memories
        start_time = (self.now - timedelta(days=1)).isoformat()
        end_time = self.now.isoformat()
        
        # Perform search with time range
        results = self.strategy.search(
            query={},  # Empty query, just use time range
            agent_id="agent-1",
            tier="stm",
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        # Should find only the memories within the time range
        self.assertEqual(len(results), 2)
        self.assertIn("memory2", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory1", [r["id"] for r in results])
        
        # Verify temporal score was added to metadata
        self.assertIn("temporal_score", results[0]["metadata"])
    
    def test_search_with_dict_query(self):
        """Test search with a dictionary query specifying time range."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = []
        self.mock_im_store.get_all.return_value = self.sample_memories
        self.mock_ltm_store.get_all.return_value = []
        
        # Define time range in the query
        query = {
            "start_time": (self.now - timedelta(days=2)).isoformat(),
            "end_time": (self.now - timedelta(hours=2)).isoformat()
        }
        
        # Perform search
        results = self.strategy.search(
            query=query,
            agent_id="agent-1",
            tier="im",
            limit=5
        )
        
        # Should find memories within the time range in query
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory2", [r["id"] for r in results])
        self.assertNotIn("memory3", [r["id"] for r in results])
    
    def test_search_with_recency_weight(self):
        """Test search with recency weighting."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = []
        self.mock_im_store.get_all.return_value = []
        self.mock_ltm_store.get_all.return_value = self.sample_memories
        
        # Perform search with high recency weight
        results_high_recency = self.strategy.search(
            query={},
            agent_id="agent-1",
            tier="ltm",
            recency_weight=2.0,  # High weight on recency
            limit=3
        )
        
        # Perform search with low recency weight
        results_low_recency = self.strategy.search(
            query={},
            agent_id="agent-1",
            tier="ltm",
            recency_weight=0.1,  # Low weight on recency
            limit=3
        )
        
        # With high recency weight, the most recent memory should be first
        self.assertEqual(results_high_recency[0]["id"], "memory3")
        
        # With low recency weight, order might be different based on other factors
        # but the first memory may not be the most recent one
        # This is hard to test conclusively without knowing the exact scoring algorithm,
        # so we'll just verify we get results
        self.assertEqual(len(results_low_recency), 3)
    
    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories
        
        # Perform search with metadata filter for importance=high
        results = self.strategy.search(
            query={},
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"importance": "high"},
            limit=5
        )
        
        # Should only find the memory with high importance
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "memory2")
    
    def test_search_all_tiers(self):
        """Test search across all memory tiers."""
        # Set up stores to return different memories in each tier
        self.mock_stm_store.get_all.return_value = [self.sample_memories[0]]
        self.mock_im_store.get_all.return_value = [self.sample_memories[1]]
        self.mock_ltm_store.get_all.return_value = [self.sample_memories[2]]
        
        # Perform search across all tiers
        results = self.strategy.search(
            query={},
            agent_id="agent-1",
            # No tier specified, should search all
            limit=5
        )
        
        # Verify results include memories from all tiers
        self.assertEqual(len(results), 3)
        
        # Check that tier information was added to metadata
        tiers = [r.get("metadata", {}).get("memory_tier") for r in results]
        self.assertIn("stm", tiers)
        self.assertIn("im", tiers)
        self.assertIn("ltm", tiers)
    
    def test_parse_datetime(self):
        """Test parsing datetime from various formats."""
        # Test with datetime object
        dt_obj = datetime(2023, 6, 15, 12, 30, 0)
        parsed_dt = self.strategy._parse_datetime(dt_obj)
        self.assertEqual(parsed_dt, dt_obj)
        
        # Test with ISO format string
        iso_str = "2023-06-15T12:30:00"
        parsed_iso = self.strategy._parse_datetime(iso_str)
        self.assertEqual(parsed_iso.year, 2023)
        self.assertEqual(parsed_iso.month, 6)
        self.assertEqual(parsed_iso.day, 15)
        
        # Test with date string
        date_str = "2023-06-15"
        parsed_date = self.strategy._parse_datetime(date_str)
        self.assertEqual(parsed_date.year, 2023)
        self.assertEqual(parsed_date.month, 6)
        self.assertEqual(parsed_date.day, 15)
        
        # Test with invalid string
        invalid_str = "not-a-date"
        parsed_invalid = self.strategy._parse_datetime(invalid_str)
        self.assertIsNone(parsed_invalid)
    
    def test_process_query_with_string(self):
        """Test processing a string query."""
        # A date string query
        query_str = "2023-06-15"
        params = self.strategy._process_query(query_str, None, None)
        
        # Reference time should be parsed from the string
        self.assertIsNotNone(params["reference_time"])
        self.assertEqual(params["reference_time"].year, 2023)
        self.assertEqual(params["reference_time"].month, 6)
        self.assertEqual(params["reference_time"].day, 15)
        
        # Start and end time should still be None
        self.assertIsNone(params["start_time"])
        self.assertIsNone(params["end_time"])


if __name__ == "__main__":
    unittest.main() 