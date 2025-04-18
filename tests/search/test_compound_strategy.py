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
            self.mock_stm_store,
            self.mock_im_store,
            self.mock_ltm_store
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
            {"id": "mem1", "content": "meeting notes", "metadata": {"type": "note", "importance": 3}},
            {"id": "mem2", "content": "email about project", "metadata": {"type": "email", "importance": 5}},
            {"id": "mem3", "content": "task reminder", "metadata": {"type": "task", "importance": 4}}
        ]
        
        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories
        
        # Create AND query
        query = {
            "operator": "AND",
            "conditions": [
                {"field": "metadata.type", "comparison": "equals", "value": "note"},
                {"field": "metadata.importance", "comparison": "greater_than", "value": 2}
            ]
        }
        
        # Perform search
        results = self.strategy.search(
            query=query,
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Should return only memory 1 (note with importance 3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
    
    def test_search_with_or_condition(self):
        """Test search with OR condition."""
        # Mock memory data
        memories = [
            {"id": "mem1", "content": "meeting notes", "metadata": {"type": "note", "importance": 3}},
            {"id": "mem2", "content": "email about project", "metadata": {"type": "email", "importance": 5}},
            {"id": "mem3", "content": "task reminder", "metadata": {"type": "task", "importance": 4}}
        ]
        
        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories
        
        # Create OR query
        query = {
            "operator": "OR",
            "conditions": [
                {"field": "metadata.type", "comparison": "equals", "value": "note"},
                {"field": "metadata.importance", "comparison": "greater_than", "value": 4}
            ]
        }
        
        # Perform search
        results = self.strategy.search(
            query=query,
            agent_id="agent-1",
            tier="im",
            limit=5
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
            {"id": "mem1", "content": "meeting notes", "metadata": {"type": "note", "importance": 3, "tags": ["work"]}},
            {"id": "mem2", "content": "email about project", "metadata": {"type": "email", "importance": 5, "tags": ["work"]}},
            {"id": "mem3", "content": "task reminder", "metadata": {"type": "task", "importance": 4, "tags": ["personal"]}},
            {"id": "mem4", "content": "shopping list", "metadata": {"type": "note", "importance": 2, "tags": ["personal"]}}
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
                        {"field": "metadata.type", "comparison": "equals", "value": "note"},
                        {"field": "metadata.tags", "comparison": "contains", "value": "work"}
                    ]
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "metadata.importance", "comparison": "greater_than", "value": 3},
                        {"field": "metadata.tags", "comparison": "contains", "value": "personal"}
                    ]
                }
            ]
        }
        
        # Perform search
        results = self.strategy.search(
            query=query,
            agent_id="agent-1",
            tier="ltm",
            limit=5
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
            {"id": "mem1", "content": "text with keyword", "metadata": {"count": 5, "created": "2023-01-01"}},
            {"id": "mem2", "content": "another text sample", "metadata": {"count": 10, "created": "2023-02-15"}},
            {"id": "mem3", "content": "final example", "metadata": {"count": 15, "created": "2023-03-20"}}
        ]
        
        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories
        
        # Test equals
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "equals", "value": 10},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem2")
        
        # Test not_equals
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "not_equals", "value": 10},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 2)
        
        # Test contains
        results = self.strategy.search(
            query={"field": "content", "comparison": "contains", "value": "keyword"},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
        
        # Test less_than
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "less_than", "value": 10},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
        
        # Test greater_than
        results = self.strategy.search(
            query={"field": "metadata.count", "comparison": "greater_than", "value": 5},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 2)
        
        # Test regex
        results = self.strategy.search(
            query={"field": "metadata.created", "comparison": "regex", "value": "2023-0[1-2]-.*"},
            agent_id="agent-1",
            tier="stm"
        )
        self.assertEqual(len(results), 2)
    
    def test_invalid_query_structure(self):
        """Test handling of invalid query structures."""
        # Missing required fields
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"operator": "AND"},  # missing conditions
                agent_id="agent-1",
                tier="stm"
            )
        
        # Invalid operator
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"operator": "XOR", "conditions": []},
                agent_id="agent-1",
                tier="stm"
            )
        
        # Invalid field comparison
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"field": "content", "comparison": "invalid_op", "value": "test"},
                agent_id="agent-1",
                tier="stm"
            )


if __name__ == "__main__":
    unittest.main() 