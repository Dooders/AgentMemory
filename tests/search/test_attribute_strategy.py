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
            self.mock_stm_store,
            self.mock_im_store,
            self.mock_ltm_store
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
                        "importance": "high"
                    }
                }
            },
            {
                "id": "memory2",
                "content": {
                    "content": "Sent email to team about status update",
                    "metadata": {
                        "type": "communication",
                        "tags": ["email", "team", "status"],
                        "importance": "medium"
                    }
                }
            },
            {
                "id": "memory3",
                "content": {
                    "content": "Phone call with client to discuss requirements",
                    "metadata": {
                        "type": "meeting",
                        "tags": ["client", "requirements"],
                        "importance": "high"
                    }
                }
            }
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
            query="meeting",
            agent_id="agent-1",
            tier="stm",
            limit=5
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
        meeting_memories = [mem for mem in self.sample_memories 
                           if mem["content"]["metadata"]["type"] == "meeting"]
        
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
            limit=5
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
            limit=5
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
            limit=5
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
            "metadata": {
                "type": "meeting",
                "importance": "high"
            }
        }
        
        # Perform search requiring all conditions to match
        results = self.strategy.search(
            query=query,
            agent_id="agent-1",
            tier="stm",
            match_all=True,
            limit=5
        )
        
        # Verify search found only memories matching all conditions
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
    
    def test_search_with_case_sensitive(self):
        """Test search with case sensitivity."""
        # Modify a sample memory for case-sensitive testing
        case_sensitive_memories = self.sample_memories.copy()
        case_sensitive_memories[0]["content"]["content"] = "MEETING with John about the project"
        
        self.mock_stm_store.get_all.return_value = case_sensitive_memories
        
        # Case-insensitive search (default)
        results_insensitive = self.strategy.search(
            query="meeting",
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Should find both memories with "meeting" regardless of case
        self.assertEqual(len(results_insensitive), 2)
        
        # Case-sensitive search
        results_sensitive = self.strategy.search(
            query="MEETING",
            agent_id="agent-1",
            tier="stm",
            case_sensitive=True,
            limit=5
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
            metadata_fields=["content.metadata.importance"],  # Only search in importance field
            limit=5
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
            limit=5
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
            "metadata": {
                "nested": {
                    "field": "nested value"
                },
                "array": [1, 2, 3]
            }
        }
        
        # Test simple field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "content"),
            "test content"
        )
        
        # Test nested field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.nested.field"),
            "nested value"
        )
        
        # Test array field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.array"),
            [1, 2, 3]
        )
        
        # Test nonexistent field
        self.assertIsNone(
            self.strategy._get_field_value(memory, "nonexistent.field")
        )


if __name__ == "__main__":
    unittest.main() 