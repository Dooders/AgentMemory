"""Tests for the AttributeSearchStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.attribute import AttributeSearchStrategy


class TestAttributeSearchStrategy(unittest.TestCase):
    """Tests for the AttributeSearchStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create strategy with mock dependencies
        self.strategy = AttributeSearchStrategy(
            self.mock_stm_store, self.mock_im_store, self.mock_ltm_store
        )

        # Sample memories for testing
        self.sample_memories = [
            {
                "id": "memory1",
                "content": {
                    "content": "Meeting with John about the project timeline",
                    "metadata": {
                        "type": "meeting",
                        "tags": ["project", "timeline"],
                        "importance": "high",
                    },
                },
            },
            {
                "id": "memory2",
                "content": {
                    "content": "Sent email to team about status update",
                    "metadata": {
                        "type": "communication",
                        "tags": ["email", "team", "status"],
                        "importance": "medium",
                    },
                },
            },
            {
                "id": "memory3",
                "content": {
                    "content": "Phone call with client to discuss requirements",
                    "metadata": {
                        "type": "meeting",
                        "tags": ["client", "requirements"],
                        "importance": "high",
                    },
                },
            },
        ]

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "attribute")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("attribute", self.strategy.description().lower())

    def test_search_string_query(self):
        """Test search with a simple string query."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories
        self.mock_im_store.get_all.return_value = []
        self.mock_ltm_store.get_all.return_value = []

        # Perform search
        results = self.strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory2", [r["id"] for r in results])

        # Verify score was added to metadata
        self.assertIn("attribute_score", results[0]["metadata"])

    def test_search_dict_query(self):
        """Test search with a dictionary query."""
        # Set up store to return only the meeting-type memories
        meeting_memories = [
            mem
            for mem in self.sample_memories
            if mem["content"]["metadata"]["type"] == "meeting"
        ]

        self.mock_stm_store.get_all.return_value = []
        self.mock_im_store.get_all.return_value = meeting_memories
        self.mock_ltm_store.get_all.return_value = []

        # Perform search with string query instead
        results = self.strategy.search(
            query="meeting",
            agent_id="agent-1",
            tier="im",
            content_fields=[],
            metadata_fields=["content.metadata.type"],
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory2", [r["id"] for r in results])

    def test_search_with_metadata_filter(self):
        """Test search with metadata filter."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = []
        self.mock_im_store.get_all.return_value = []
        self.mock_ltm_store.get_all.return_value = self.sample_memories

        # Perform search with metadata filter
        results = self.strategy.search(
            query="client",
            agent_id="agent-1",
            tier="ltm",
            metadata_filter={"content.metadata.importance": "high"},
            limit=5,
        )

        # Verify search found matching memories that also match the metadata filter
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "memory3")

    def test_search_with_regex(self):
        """Test search with regex pattern."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories

        # Perform search with regex
        results = self.strategy.search(
            query="meet.*john|phone.*client",
            agent_id="agent-1",
            tier="stm",
            use_regex=True,
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

    def test_search_with_match_all(self):
        """Test search requiring all conditions to match."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories

        # Query that should match only when all conditions are met
        query = {
            "content": "meeting",
            "metadata": {"type": "meeting", "importance": "high"},
        }

        # Perform search requiring all conditions to match
        results = self.strategy.search(
            query=query, agent_id="agent-1", tier="stm", match_all=True, limit=5
        )

        # Verify search found only memories matching all conditions
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

    def test_search_with_case_sensitive(self):
        """Test search with case sensitivity."""
        # Modify a sample memory for case-sensitive testing
        case_sensitive_memories = self.sample_memories.copy()
        case_sensitive_memories[0]["content"][
            "content"
        ] = "MEETING with John about the project"

        self.mock_stm_store.get_all.return_value = case_sensitive_memories

        # Case-insensitive search (default)
        results_insensitive = self.strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Should find both memories with "meeting" regardless of case
        self.assertEqual(len(results_insensitive), 2)

        # Case-sensitive search
        results_sensitive = self.strategy.search(
            query="MEETING",
            agent_id="agent-1",
            tier="stm",
            case_sensitive=True,
            limit=5,
        )

        # Should only find the memory with exact case match
        self.assertEqual(len(results_sensitive), 1)
        self.assertEqual(results_sensitive[0]["id"], "memory1")

    def test_search_with_custom_fields(self):
        """Test search with custom content and metadata fields."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories

        # Perform search with specific content and metadata fields
        results = self.strategy.search(
            query="high",
            agent_id="agent-1",
            tier="stm",
            content_fields=[],  # Don't search in content
            metadata_fields=[
                "content.metadata.importance"
            ],  # Only search in importance field
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

        # Should not find this even though "high" could match in tags or somewhere else
        self.assertNotIn("memory2", [r["id"] for r in results])

    def test_search_all_tiers(self):
        """Test search across all memory tiers."""
        # Set up stores to return different memories in each tier
        self.mock_stm_store.get_all.return_value = [self.sample_memories[0]]
        self.mock_im_store.get_all.return_value = [self.sample_memories[1]]
        self.mock_ltm_store.get_all.return_value = [self.sample_memories[2]]

        # Perform search across all tiers
        results = self.strategy.search(
            query="meeting",
            agent_id="agent-1",
            # No tier specified, should search all
            limit=5,
        )

        # Verify results include memories from both tiers
        self.assertEqual(len(results), 2)

        # Check that tier was added to metadata
        tiers = [r.get("metadata", {}).get("memory_tier") for r in results]
        self.assertIn("stm", tiers)
        self.assertIn("ltm", tiers)

    def test_get_field_value(self):
        """Test retrieving field values from nested paths."""
        memory = {
            "content": "test content",
            "metadata": {"nested": {"field": "nested value"}, "array": [1, 2, 3]},
        }

        # Test simple field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "content"), "test content"
        )

        # Test nested field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.nested.field"),
            "nested value",
        )

        # Test array field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.array"), [1, 2, 3]
        )

        # Test nonexistent field
        self.assertIsNone(self.strategy._get_field_value(memory, "nonexistent.field"))

    def test_empty_query(self):
        """Test search with empty query."""
        # Set up store to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories

        # Perform search with empty string query
        results_empty_string = self.strategy.search(
            query="", agent_id="agent-1", tier="stm", limit=5
        )

        # Should return no results with empty string
        self.assertEqual(len(results_empty_string), 0)

        # Perform search with empty dict query
        results_empty_dict = self.strategy.search(
            query={}, agent_id="agent-1", tier="stm", limit=5
        )

        # Should return no results with empty dict
        self.assertEqual(len(results_empty_dict), 0)

    def test_special_characters_in_query(self):
        """Test search with special characters in query."""
        # Create memories with special characters
        special_char_memories = [
            {
                "id": "memory7",
                "content": {
                    "content": "Email from john.doe@example.com",
                    "metadata": {"type": "email", "tags": ["email", "contact"]},
                },
            },
            {
                "id": "memory8",
                "content": {
                    "content": "Meeting about C++ & Python integration",
                    "metadata": {"type": "meeting", "tags": ["C++", "Python"]},
                },
            },
            {
                "id": "memory9",
                "content": {
                    "content": "Query: SELECT * FROM table WHERE id='123'",
                    "metadata": {"type": "database", "tags": ["SQL", "query"]},
                },
            },
        ]

        # Set up store to return special character memories
        self.mock_stm_store.get_all.return_value = special_char_memories

        # Test with email address in query
        email_results = self.strategy.search(
            query="john.doe@example.com", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(email_results), 1)
        self.assertEqual(email_results[0]["id"], "memory7")

        # Test with SQL-like query with quotes and special chars
        sql_results = self.strategy.search(
            query="SELECT * FROM", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(sql_results), 1)
        self.assertEqual(sql_results[0]["id"], "memory9")

        # Test with symbols in query
        symbol_results = self.strategy.search(
            query="C++ & Python", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(symbol_results), 1)
        self.assertEqual(symbol_results[0]["id"], "memory8")

    def test_mixed_data_types(self):
        """Test search handling mixed data types in memory fields."""
        # Create memories with mixed data types
        mixed_type_memories = [
            {
                "id": "memory10",
                "content": {
                    "content": "Temperature is 72 degrees",
                    "metadata": {
                        "type": "weather",
                        "temperature": 72,  # Numeric value
                        "tags": ["weather", "temperature"],
                    },
                },
            },
            {
                "id": "memory11",
                "content": {
                    "content": "Task completed",
                    "metadata": {
                        "type": "task",
                        "completed": True,  # Boolean value
                        "tags": ["task", "completed"],
                    },
                },
            },
            {
                "id": "memory12",
                "content": {
                    "content": "Complex data structure",
                    "metadata": {
                        "type": "data",
                        "nested": {
                            "array": [1, "two", 3.0, True],  # Mixed array
                            "object": {"key": "value"},
                        },
                    },
                },
            },
        ]

        # Set up store to return mixed type memories
        self.mock_stm_store.get_all.return_value = mixed_type_memories

        # Test search with numeric value in query
        numeric_results = self.strategy.search(
            query="72", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(numeric_results), 1)
        self.assertEqual(numeric_results[0]["id"], "memory10")

        # Test with boolean in metadata filter
        boolean_filter_results = self.strategy.search(
            query="task",
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"content.metadata.completed": True},
            limit=5,
        )
        self.assertEqual(len(boolean_filter_results), 1)
        self.assertEqual(boolean_filter_results[0]["id"], "memory11")


if __name__ == "__main__":
    unittest.main()
