"""Tests for the ImportanceStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.importance import ImportanceStrategy


class TestImportanceStrategy(unittest.TestCase):
    """Tests for the ImportanceStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create strategy with mock dependencies
        self.strategy = ImportanceStrategy(
            self.mock_stm_store, self.mock_im_store, self.mock_ltm_store
        )

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "importance")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("importance", self.strategy.description().lower())

    def test_search_min_importance(self):
        """Test search with minimum importance threshold."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Perform search with min_importance
        results = self.strategy.search(
            query={"min_importance": 7}, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify results only include memories with importance >= 7
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)  # importance 8
        self.assertIn("mem3", result_ids)  # importance 9

        # Verify results are sorted by importance (descending)
        self.assertEqual(results[0]["id"], "mem3")  # importance 9
        self.assertEqual(results[1]["id"], "mem1")  # importance 8

    def test_search_max_importance(self):
        """Test search with maximum importance threshold."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Perform search with max_importance
        results = self.strategy.search(
            query={"max_importance": 5}, agent_id="agent-1", tier="im", limit=5
        )

        # Verify results only include memories with importance <= 5
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem2", result_ids)  # importance 3
        self.assertIn("mem4", result_ids)  # importance 2

        # Verify results are still sorted by importance (descending)
        self.assertEqual(results[0]["id"], "mem2")  # importance 3
        self.assertEqual(results[1]["id"], "mem4")  # importance 2

    def test_search_importance_range(self):
        """Test search with an importance range."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {
                "id": "mem4",
                "content": "Semi-important report",
                "metadata": {"importance": 6},
            },
            {"id": "mem5", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Perform search with importance range
        results = self.strategy.search(
            query={"min_importance": 3, "max_importance": 8},
            agent_id="agent-1",
            tier="ltm",
            limit=5,
        )

        # Verify results only include memories with importance between 3 and 8 (inclusive)
        self.assertEqual(len(results), 3)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)  # importance 8
        self.assertIn("mem2", result_ids)  # importance 3
        self.assertIn("mem4", result_ids)  # importance 6

        # Verify results are sorted by importance (descending)
        self.assertEqual(results[0]["id"], "mem1")  # importance 8
        self.assertEqual(results[1]["id"], "mem4")  # importance 6
        self.assertEqual(results[2]["id"], "mem2")  # importance 3

    def test_search_with_top_n(self):
        """Test search getting top N most important memories."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {
                "id": "mem4",
                "content": "Semi-important report",
                "metadata": {"importance": 6},
            },
            {"id": "mem5", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Perform search with top_n
        results = self.strategy.search(
            query={"top_n": 3}, agent_id="agent-1", tier="stm"
        )

        # Verify only top 3 most important memories are returned
        self.assertEqual(len(results), 3)
        result_ids = [r["id"] for r in results]
        self.assertEqual(results[0]["id"], "mem3")  # importance 9
        self.assertEqual(results[1]["id"], "mem1")  # importance 8
        self.assertEqual(results[2]["id"], "mem4")  # importance 6

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8, "type": "meeting"},
            },
            {
                "id": "mem2",
                "content": "Regular task",
                "metadata": {"importance": 3, "type": "task"},
            },
            {
                "id": "mem3",
                "content": "Critical alert",
                "metadata": {"importance": 9, "type": "alert"},
            },
            {
                "id": "mem4",
                "content": "Important task",
                "metadata": {"importance": 7, "type": "task"},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Perform search with metadata filter
        results = self.strategy.search(
            query={"min_importance": 5},
            agent_id="agent-1",
            tier="im",
            metadata_filter={"type": "task"},
        )

        # Verify only high importance tasks are returned
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem4")  # important task

    def test_missing_importance_in_metadata(self):
        """Test handling of memories without importance in metadata."""
        # Set up mock memory data with some memories missing importance
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Missing importance", "metadata": {}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Also missing", "metadata": {"type": "task"}},
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Perform search
        results = self.strategy.search(
            query={"min_importance": 1}, agent_id="agent-1", tier="ltm", limit=5
        )

        # Verify only memories with importance metadata are included
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem3", result_ids)
        self.assertNotIn("mem2", result_ids)
        self.assertNotIn("mem4", result_ids)

    def test_invalid_query_parameters(self):
        """Test handling of invalid query parameters."""
        # Test with non-numeric importance value
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"min_importance": "high"},  # Not a number
                agent_id="agent-1",
                tier="stm",
            )

        # Test with negative importance value
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"min_importance": -1},  # Negative value
                agent_id="agent-1",
                tier="stm",
            )

        # Test with min > max
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"min_importance": 8, "max_importance": 5},  # min > max
                agent_id="agent-1",
                tier="stm",
            )

        # Test with invalid top_n
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"top_n": 0}, agent_id="agent-1", tier="stm"  # Should be > 0
            )

    def test_query_as_direct_value(self):
        """Test search with direct numeric value for query."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Perform search with direct numeric value (minimum importance of 5)
        results = self.strategy.search(
            query=5.0, agent_id="agent-1", tier="stm", limit=5
        )

        # Verify results only include memories with importance >= 5
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)  # importance 8
        self.assertIn("mem3", result_ids)  # importance 9

    def test_sort_order_parameter(self):
        """Test search with different sort order parameters."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "Important meeting",
                "metadata": {"importance": 8},
            },
            {"id": "mem2", "content": "Regular task", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Critical alert", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Routine check", "metadata": {"importance": 2}},
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Test with descending sort order (default)
        desc_results = self.strategy.search(
            query={"min_importance": 1}, agent_id="agent-1", tier="im", limit=5
        )

        # Verify descending order (highest importance first)
        self.assertEqual(desc_results[0]["id"], "mem3")  # importance 9
        self.assertEqual(desc_results[1]["id"], "mem1")  # importance 8
        self.assertEqual(desc_results[2]["id"], "mem2")  # importance 3
        self.assertEqual(desc_results[3]["id"], "mem4")  # importance 2

        # Test with ascending sort order
        asc_results = self.strategy.search(
            query={"min_importance": 1},
            agent_id="agent-1",
            tier="im",
            limit=5,
            sort_order="asc",
        )

        # Verify ascending order (lowest importance first)
        self.assertEqual(asc_results[0]["id"], "mem4")  # importance 2
        self.assertEqual(asc_results[1]["id"], "mem2")  # importance 3
        self.assertEqual(asc_results[2]["id"], "mem1")  # importance 8
        self.assertEqual(asc_results[3]["id"], "mem3")  # importance 9

    def test_all_tiers_search(self):
        """Test search across all memory tiers."""
        # Set up mock memory data for each tier
        stm_memories = [
            {"id": "stm1", "content": "STM memory", "metadata": {"importance": 7}},
        ]
        im_memories = [
            {"id": "im1", "content": "IM memory", "metadata": {"importance": 5}},
        ]
        ltm_memories = [
            {"id": "ltm1", "content": "LTM memory", "metadata": {"importance": 8}},
        ]

        # Configure mocks to return tier-specific memories
        self.mock_stm_store.list.return_value = stm_memories
        self.mock_im_store.list.return_value = im_memories
        self.mock_ltm_store.list.return_value = ltm_memories

        # Perform search without specifying tier (should search all)
        results = self.strategy.search(
            query={"min_importance": 6}, agent_id="agent-1", limit=5
        )

        # Verify results include memories from all tiers that meet the criteria
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("ltm1", result_ids)  # importance 8 from LTM
        self.assertIn("stm1", result_ids)  # importance 7 from STM
        self.assertNotIn("im1", result_ids)  # importance 5 from IM (below threshold)

    def test_combined_top_n_and_importance_filters(self):
        """Test search with both top_n and min/max importance filters."""
        # Set up mock memory data
        memories = [
            {"id": "mem1", "content": "Memory 1", "metadata": {"importance": 8}},
            {"id": "mem2", "content": "Memory 2", "metadata": {"importance": 3}},
            {"id": "mem3", "content": "Memory 3", "metadata": {"importance": 9}},
            {"id": "mem4", "content": "Memory 4", "metadata": {"importance": 6}},
            {"id": "mem5", "content": "Memory 5", "metadata": {"importance": 5}},
            {"id": "mem6", "content": "Memory 6", "metadata": {"importance": 7}},
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Test combining top_n with min/max importance filters
        results = self.strategy.search(
            query={"top_n": 2, "min_importance": 5, "max_importance": 8},
            agent_id="agent-1",
            tier="ltm",
        )

        # Should return top 2 memories in the importance range 5-8
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "mem1")  # importance 8
        self.assertEqual(results[1]["id"], "mem6")  # importance 7

    def test_get_stores_for_tier(self):
        """Test the _get_stores_for_tier method with all possible inputs."""
        # Test each tier specifically
        stm_stores = self.strategy._get_stores_for_tier("stm")
        self.assertEqual(len(stm_stores), 1)
        self.assertEqual(stm_stores[0], self.mock_stm_store)

        im_stores = self.strategy._get_stores_for_tier("im")
        self.assertEqual(len(im_stores), 1)
        self.assertEqual(im_stores[0], self.mock_im_store)

        ltm_stores = self.strategy._get_stores_for_tier("ltm")
        self.assertEqual(len(ltm_stores), 1)
        self.assertEqual(ltm_stores[0], self.mock_ltm_store)

        # Test None or invalid tier (should return all stores)
        all_stores = self.strategy._get_stores_for_tier(None)
        self.assertEqual(len(all_stores), 3)
        self.assertIn(self.mock_stm_store, all_stores)
        self.assertIn(self.mock_im_store, all_stores)
        self.assertIn(self.mock_ltm_store, all_stores)

        # Test with invalid tier value
        invalid_stores = self.strategy._get_stores_for_tier("invalid_tier")
        self.assertEqual(len(invalid_stores), 3)  # Should default to all stores


if __name__ == "__main__":
    unittest.main()
