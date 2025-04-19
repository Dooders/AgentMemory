"""Tests for the CompoundQueryStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.compound import CompoundQueryStrategy


class TestCompoundQueryStrategy(unittest.TestCase):
    """Tests for the CompoundQueryStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create strategy with mock dependencies
        self.strategy = CompoundQueryStrategy(
            self.mock_stm_store, self.mock_im_store, self.mock_ltm_store
        )

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "compound")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("compound", self.strategy.description().lower())

    def test_search_with_and_condition(self):
        """Test search with AND condition."""
        # Mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "meeting notes",
                "metadata": {"type": "note", "importance": 3},
            },
            {
                "id": "mem2",
                "content": "email about project",
                "metadata": {"type": "email", "importance": 5},
            },
            {
                "id": "mem3",
                "content": "task reminder",
                "metadata": {"type": "task", "importance": 4},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Create AND query
        query = {
            "operator": "AND",
            "conditions": [
                {"field": "metadata.type", "comparison": "equals", "value": "note"},
                {
                    "field": "metadata.importance",
                    "comparison": "greater_than",
                    "value": 2,
                },
            ],
        }

        # Perform search
        results = self.strategy.search(
            query=query, agent_id="agent-1", tier="stm", limit=5
        )

        # Should return only memory 1 (note with importance 3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

    def test_search_with_or_condition(self):
        """Test search with OR condition."""
        # Mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "meeting notes",
                "metadata": {"type": "note", "importance": 3},
            },
            {
                "id": "mem2",
                "content": "email about project",
                "metadata": {"type": "email", "importance": 5},
            },
            {
                "id": "mem3",
                "content": "task reminder",
                "metadata": {"type": "task", "importance": 4},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Create OR query
        query = {
            "operator": "OR",
            "conditions": [
                {"field": "metadata.type", "comparison": "equals", "value": "note"},
                {
                    "field": "metadata.importance",
                    "comparison": "greater_than",
                    "value": 4,
                },
            ],
        }

        # Perform search
        results = self.strategy.search(
            query=query, agent_id="agent-1", tier="im", limit=5
        )

        # Should return memories 1 and 2 (notes or importance > 4)
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)

    def test_search_with_nested_conditions(self):
        """Test search with nested conditions."""
        # Mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "meeting notes",
                "metadata": {"type": "note", "importance": 3, "tags": ["work"]},
            },
            {
                "id": "mem2",
                "content": "email about project",
                "metadata": {"type": "email", "importance": 5, "tags": ["work"]},
            },
            {
                "id": "mem3",
                "content": "task reminder",
                "metadata": {"type": "task", "importance": 4, "tags": ["personal"]},
            },
            {
                "id": "mem4",
                "content": "shopping list",
                "metadata": {"type": "note", "importance": 2, "tags": ["personal"]},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Create nested query
        query = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "metadata.type",
                            "comparison": "equals",
                            "value": "note",
                        },
                        {
                            "field": "metadata.tags",
                            "comparison": "contains",
                            "value": "work",
                        },
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "metadata.importance",
                            "comparison": "greater_than",
                            "value": 3,
                        },
                        {
                            "field": "metadata.tags",
                            "comparison": "contains",
                            "value": "personal",
                        },
                    ],
                },
            ],
        }

        # Perform search
        results = self.strategy.search(
            query=query, agent_id="agent-1", tier="ltm", limit=5
        )

        # Should return memories 1 and 3 (work notes OR personal with importance > 3)
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem3", result_ids)

    def test_field_comparison_operators(self):
        """Test different field comparison operators."""
        # Mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "text with keyword",
                "metadata": {"count": 5, "created": "2023-01-01"},
            },
            {
                "id": "mem2",
                "content": "another text sample",
                "metadata": {"count": 10, "created": "2023-02-15"},
            },
            {
                "id": "mem3",
                "content": "final example",
                "metadata": {"count": 15, "created": "2023-03-20"},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Test equals
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "equals", "value": 10},
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem2")

        # Test not_equals
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "not_equals", "value": 10},
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 2)

        # Test contains
        results = self.strategy.search(
            query={"field": "content", "comparison": "contains", "value": "keyword"},
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

        # Test less_than
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "less_than", "value": 10},
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

        # Test greater_than
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "greater_than", "value": 5},
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 2)

        # Test regex
        results = self.strategy.search(
            query={
                "field": "metadata.created",
                "comparison": "regex",
                "value": "2023-0[1-2]-.*",
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 2)

    def test_invalid_query_structure(self):
        """Test handling of invalid query structures."""
        # Missing required fields
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"operator": "AND"},  # missing conditions
                agent_id="agent-1",
                tier="stm",
            )

        # Invalid operator
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"operator": "XOR", "conditions": []},
                agent_id="agent-1",
                tier="stm",
            )

        # Invalid field comparison
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"field": "content", "comparison": "invalid_op", "value": "test"},
                agent_id="agent-1",
                tier="stm",
            )

    def test_metadata_filter(self):
        """Test metadata filtering in search results."""
        # Mock memory data
        memories = [
            {
                "id": "mem1",
                "content": "text one",
                "metadata": {"category": "work", "priority": "high"},
            },
            {
                "id": "mem2",
                "content": "text two",
                "metadata": {"category": "personal", "priority": "high"},
            },
            {
                "id": "mem3",
                "content": "text three",
                "metadata": {"category": "work", "priority": "low"},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Test with metadata filter
        results = self.strategy.search(
            query={"field": "content", "comparison": "contains", "value": "text"},
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"category": "work"},
        )

        # Should return memories 1 and 3 (content contains "text" AND metadata.category is "work")
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem3", result_ids)

        # Test with more specific metadata filter
        results = self.strategy.search(
            query={"field": "content", "comparison": "contains", "value": "text"},
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"category": "work", "priority": "high"},
        )

        # Should return only memory 1
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

    def test_default_tier_handling(self):
        """Test default tier handling when tier=None."""
        # Mock memory data
        memories = [
            {"id": "mem1", "content": "text one"},
            {"id": "mem2", "content": "text two"},
        ]

        # Configure mock to return memories
        self.mock_stm_store.list.return_value = memories

        # Test search with tier=None (should default to stm_store)
        results = self.strategy.search(
            query={"field": "content", "comparison": "contains", "value": "text"},
            agent_id="agent-1",
            tier=None,  # No tier specified
        )

        # Verify stm_store was used
        self.mock_stm_store.list.assert_called_once()
        self.mock_im_store.list.assert_not_called()
        self.mock_ltm_store.list.assert_not_called()

        # Should return all memories
        self.assertEqual(len(results), 2)

    def test_list_indexing_in_field_value(self):
        """Test accessing values by array index in _get_field_value."""
        # Mock memory data with nested arrays
        memories = [
            {
                "id": "mem1",
                "content": "text with array",
                "metadata": {
                    "tags": ["important", "meeting", "project"],
                    "nested": [{"key": "value1"}, {"key": "value2"}],
                },
            }
        ]

        # Configure mock to return memories
        self.mock_stm_store.list.return_value = memories

        # Test accessing array index
        results = self.strategy.search(
            query={
                "field": "metadata.tags.0",
                "comparison": "equals",
                "value": "important",
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 1)

        # Test accessing nested array objects
        results = self.strategy.search(
            query={
                "field": "metadata.nested.1.key",
                "comparison": "equals",
                "value": "value2",
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 1)

        # Test with invalid index
        results = self.strategy.search(
            query={
                "field": "metadata.tags.10",
                "comparison": "equals",
                "value": "nonexistent",
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 0)

    def test_remaining_comparison_operators(self):
        """Test the remaining comparison operators: greater_than_equal and less_than_equal."""
        # Mock memory data
        memories = [
            {"id": "mem1", "metadata": {"score": 5}},
            {"id": "mem2", "metadata": {"score": 10}},
            {"id": "mem3", "metadata": {"score": 15}},
        ]

        # Configure mock to return memories
        self.mock_stm_store.list.return_value = memories

        # Test greater_than_equal
        results = self.strategy.search(
            query={
                "field": "metadata.score",
                "comparison": "greater_than_equal",
                "value": 10,
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem2", result_ids)
        self.assertIn("mem3", result_ids)

        # Test less_than_equal
        results = self.strategy.search(
            query={
                "field": "metadata.score",
                "comparison": "less_than_equal",
                "value": 10,
            },
            agent_id="agent-1",
            tier="stm",
        )
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)


if __name__ == "__main__":
    unittest.main()
