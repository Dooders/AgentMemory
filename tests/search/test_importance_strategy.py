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


if __name__ == "__main__":
    unittest.main()
