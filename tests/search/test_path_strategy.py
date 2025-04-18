"""Tests for the ContentPathStrategy class."""

import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.path import ContentPathStrategy


class TestContentPathStrategy(unittest.TestCase):
    """Tests for the ContentPathStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()
        
        # Create strategy with mock dependencies
        self.strategy = ContentPathStrategy(
            self.mock_stm_store,
            self.mock_im_store,
            self.mock_ltm_store
        )
    
    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "path")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("path", self.strategy.description().lower())
    
    def test_search_exact_path_match(self):
        """Test search for exact path match."""
        # Set up mock memory data with nested content
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "title": "Meeting notes",
                    "tags": ["work", "project"],
                    "details": {
                        "location": "Conference room",
                        "attendees": ["Alice", "Bob", "Charlie"]
                    }
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "title": "Task list",
                    "tags": ["personal"],
                    "items": ["Buy groceries", "Clean house"]
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories
        
        # Perform search for exact path match
        results = self.strategy.search(
            query={"path": "content.details.location", "value": "Conference room"},
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
    
    def test_search_array_item_match(self):
        """Test search for item in array."""
        # Set up mock memory data with arrays
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "title": "Meeting notes",
                    "tags": ["work", "project"],
                    "attendees": ["Alice", "Bob", "Charlie"]
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "title": "Task list",
                    "tags": ["work", "personal"],
                    "items": ["Buy groceries", "Clean house"]
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories
        
        # Perform search for array item match
        results = self.strategy.search(
            query={"path": "content.tags", "value": "work"},
            agent_id="agent-1",
            tier="im",
            limit=5
        )
        
        # Both memories have "work" in their tags array
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
    
    def test_search_with_regex_match(self):
        """Test search with regex pattern matching."""
        # Set up mock memory data
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "title": "Meeting 2023-01-15",
                    "status": "completed"
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "title": "Meeting 2023-02-20",
                    "status": "pending"
                },
                "metadata": {}
            },
            {
                "id": "mem3", 
                "content": {
                    "title": "Call 2023-03-10",
                    "status": "completed"
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories
        
        # Perform search with regex pattern
        results = self.strategy.search(
            query={"path": "content.title", "regex": "Meeting 2023-\\d{2}-\\d{2}"},
            agent_id="agent-1",
            tier="ltm",
            limit=5
        )
        
        # Should match the two meeting entries
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertNotIn("mem3", result_ids)
    
    def test_search_with_exists_check(self):
        """Test search checking if a path exists."""
        # Set up mock memory data with different structures
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "title": "Document",
                    "comments": ["First comment", "Second comment"]
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "title": "Report",
                    "tags": ["important"]
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories
        
        # Perform search to check path existence
        results = self.strategy.search(
            query={"path": "content.comments", "exists": True},
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Only mem1 has the comments path
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
        
        # Check for non-existence
        results = self.strategy.search(
            query={"path": "content.comments", "exists": False},
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Only mem2 doesn't have the comments path
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem2")
    
    def test_search_with_comparison_operators(self):
        """Test search with comparison operators."""
        # Set up mock memory data with numeric values
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "title": "Project A",
                    "status": "completed",
                    "metrics": {
                        "progress": 100,
                        "score": 85
                    }
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "title": "Project B",
                    "status": "in_progress",
                    "metrics": {
                        "progress": 60,
                        "score": 92
                    }
                },
                "metadata": {}
            },
            {
                "id": "mem3", 
                "content": {
                    "title": "Project C",
                    "status": "planned",
                    "metrics": {
                        "progress": 0,
                        "score": 0
                    }
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories
        
        # Test greater than
        results = self.strategy.search(
            query={"path": "content.metrics.progress", "gt": 50},
            agent_id="agent-1",
            tier="im",
            limit=5
        )
        
        # Should match projects A and B
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        
        # Test less than
        results = self.strategy.search(
            query={"path": "content.metrics.progress", "lt": 70},
            agent_id="agent-1",
            tier="im",
            limit=5
        )
        
        # Should match projects B and C
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem2", result_ids)
        self.assertIn("mem3", result_ids)
        
        # Test range (combination of gt and lt)
        results = self.strategy.search(
            query={"path": "content.metrics.progress", "gt": 30, "lt": 70},
            agent_id="agent-1",
            tier="im",
            limit=5
        )
        
        # Should match only project B
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem2")
    
    def test_search_with_nested_path(self):
        """Test search with deeply nested path."""
        # Set up mock memory data with deeply nested structure
        memories = [
            {
                "id": "mem1", 
                "content": {
                    "project": {
                        "details": {
                            "client": {
                                "name": "Acme Corp",
                                "contact": {
                                    "email": "contact@acme.com"
                                }
                            }
                        }
                    }
                },
                "metadata": {}
            },
            {
                "id": "mem2", 
                "content": {
                    "project": {
                        "details": {
                            "client": {
                                "name": "Beta Inc",
                                "contact": {
                                    "email": "info@beta.com"
                                }
                            }
                        }
                    }
                },
                "metadata": {}
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories
        
        # Perform search with deeply nested path
        results = self.strategy.search(
            query={"path": "content.project.details.client.name", "value": "Acme Corp"},
            agent_id="agent-1",
            tier="ltm",
            limit=5
        )
        
        # Should match only mem1
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
    
    def test_invalid_path_format(self):
        """Test handling of invalid path format."""
        # Configure mock to return memories
        self.mock_stm_store.list.return_value = []
        
        # Test with invalid path format (missing required fields)
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={},  # Missing path
                agent_id="agent-1",
                tier="stm"
            )
        
        # Test with invalid path string
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"path": "invalid[path"},  # Invalid path syntax
                agent_id="agent-1",
                tier="stm"
            )
    
    def test_search_with_metadata_path(self):
        """Test search using a path in the metadata field."""
        # Set up mock memory data with structured metadata
        memories = [
            {
                "id": "mem1", 
                "content": "Content A",
                "metadata": {
                    "source": {
                        "type": "email",
                        "address": "sender@example.com"
                    }
                }
            },
            {
                "id": "mem2", 
                "content": "Content B",
                "metadata": {
                    "source": {
                        "type": "chat",
                        "platform": "Slack"
                    }
                }
            }
        ]
        
        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories
        
        # Perform search with metadata path
        results = self.strategy.search(
            query={"path": "metadata.source.type", "value": "email"},
            agent_id="agent-1",
            tier="stm",
            limit=5
        )
        
        # Should match only mem1
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")


if __name__ == "__main__":
    unittest.main() 