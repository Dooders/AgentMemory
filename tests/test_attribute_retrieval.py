"""Unit tests for the attribute-based memory retrieval mechanisms."""

import re
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_memory.retrieval.attribute import AttributeRetrieval
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore


class TestAttributeRetrieval:
    """Test suite for the AttributeRetrieval class."""

    @pytest.fixture
    def mock_stm_store(self):
        """Create a mock STM store."""
        mock = Mock(name="mock_stm_store", autospec=False)
        return mock

    @pytest.fixture
    def mock_im_store(self):
        """Create a mock IM store."""
        mock = Mock(name="mock_im_store", autospec=False)
        return mock

    @pytest.fixture
    def mock_ltm_store(self):
        """Create a mock LTM store."""
        mock = Mock(name="mock_ltm_store", autospec=False)
        return mock

    @pytest.fixture
    def retriever(self, mock_stm_store, mock_im_store, mock_ltm_store):
        """Create an AttributeRetrieval instance with mocked stores."""
        return AttributeRetrieval(mock_stm_store, mock_im_store, mock_ltm_store)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            {
                "memory_id": "mem1",
                "memory_type": "state",
                "step_number": 10,
                "importance": 0.8,
                "metadata": {"location": "kitchen", "mood": "happy"},
                "contents": {
                    "agent": "agent1",
                    "location": {"name": "kitchen", "coordinates": [10, 20]},
                    "dialog": {"text": "Hello there!"}
                }
            },
            {
                "memory_id": "mem2",
                "memory_type": "action",
                "step_number": 11,
                "importance": 0.6,
                "metadata": {"location": "living_room", "mood": "excited"},
                "contents": {
                    "agent": "agent1",
                    "action": "move",
                    "location": {"name": "living_room", "coordinates": [15, 30]},
                    "dialog": {"text": "I'm moving to the living room"}
                }
            },
            {
                "memory_id": "mem3",
                "memory_type": "interaction",
                "step_number": 12,
                "importance": 0.9,
                "metadata": {"location": "living_room", "target": "agent2"},
                "contents": {
                    "agent": "agent1",
                    "target": "agent2",
                    "location": {"name": "living_room", "coordinates": [15, 30]},
                    "dialog": {"text": "Hi agent2, how are you?"}
                }
            }
        ]

    def test_initialization(self, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test that AttributeRetrieval initializes correctly."""
        retriever = AttributeRetrieval(mock_stm_store, mock_im_store, mock_ltm_store)
        
        assert retriever.stm_store == mock_stm_store
        assert retriever.im_store == mock_im_store
        assert retriever.ltm_store == mock_ltm_store

    def test_retrieve_by_memory_type_stm(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories by type from STM."""
        mock_stm_store.get_by_type.return_value = [m for m in sample_memories if m["memory_type"] == "state"]
        
        result = retriever.retrieve_by_memory_type("state", limit=10, tier="stm")
        
        mock_stm_store.get_by_type.assert_called_once_with("state", limit=10)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    def test_retrieve_by_memory_type_im(self, retriever, mock_im_store, sample_memories):
        """Test retrieving memories by type from IM."""
        mock_im_store.get_by_type.return_value = [m for m in sample_memories if m["memory_type"] == "action"]
        
        result = retriever.retrieve_by_memory_type("action", limit=10, tier="im")
        
        mock_im_store.get_by_type.assert_called_once_with("action", limit=10)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem2"

    def test_retrieve_by_memory_type_ltm(self, retriever, mock_ltm_store, sample_memories):
        """Test retrieving memories by type from LTM."""
        mock_ltm_store.get_by_type.return_value = [m for m in sample_memories if m["memory_type"] == "interaction"]
        
        result = retriever.retrieve_by_memory_type("interaction", limit=10, tier="ltm")
        
        mock_ltm_store.get_by_type.assert_called_once_with("interaction", limit=10)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem3"

    def test_retrieve_by_importance(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories by importance score."""
        high_importance = [m for m in sample_memories if m["importance"] >= 0.8]
        mock_stm_store.get_by_importance.return_value = high_importance
        
        result = retriever.retrieve_by_importance(min_importance=0.8, limit=10, tier="stm")
        
        mock_stm_store.get_by_importance.assert_called_once_with(0.8, limit=10)
        assert len(result) == 2
        assert "mem1" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result]

    def test_retrieve_by_metadata(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories by metadata filters."""
        # Setup
        mock_stm_store.get_all.return_value = sample_memories
        metadata_filters = {"location": "living_room"}
        
        # Call the method
        result = retriever.retrieve_by_metadata(metadata_filters, limit=10, tier="stm")
        
        # Verify
        mock_stm_store.get_all.assert_called_once()
        assert len(result) == 2
        assert "mem2" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result]

    def test_retrieve_by_content_value(self, retriever, mock_im_store, sample_memories):
        """Test retrieving memories with specific content value."""
        # Setup
        mock_im_store.get_all.return_value = sample_memories
        
        # Call the method
        result = retriever.retrieve_by_content_value(
            path="location.name", 
            value="kitchen", 
            limit=10, 
            tier="im"
        )
        
        # Verify
        mock_im_store.get_all.assert_called_once()
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    def test_retrieve_by_content_pattern(self, retriever, mock_ltm_store, sample_memories):
        """Test retrieving memories matching a content pattern."""
        # Setup
        mock_ltm_store.get_all.return_value = sample_memories
        
        # Call the method
        result = retriever.retrieve_by_content_pattern(
            path="dialog.text", 
            pattern="agent2", 
            limit=10, 
            tier="ltm"
        )
        
        # Verify
        mock_ltm_store.get_all.assert_called_once()
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem3"

    def test_retrieve_by_custom_filter(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories using a custom filter function."""
        # Setup
        mock_stm_store.get_all.return_value = sample_memories
        
        # Define a custom filter function
        def filter_fn(memory):
            # Find memories with step_number > 10
            return memory.get("step_number", 0) > 10
        
        # Call the method
        result = retriever.retrieve_by_custom_filter(
            filter_fn=filter_fn,
            limit=10,
            tier="stm"
        )
        
        # Verify
        mock_stm_store.get_all.assert_called_once()
        assert len(result) == 2
        assert "mem2" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result]

    def test_retrieve_by_tag(self, retriever, mock_im_store):
        """Test retrieving memories with a specific tag."""
        # Setup expected data
        expected_memories = [{"memory_id": "tag1", "tags": ["important", "conversation"]}]
        mock_im_store.get_by_tag.return_value = expected_memories
        
        # Call the method
        result = retriever.retrieve_by_tag("important", limit=10, tier="im")
        
        # Verify
        mock_im_store.get_by_tag.assert_called_once_with("important", limit=10)
        assert result == expected_memories

    def test_retrieve_by_compound_query_and(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories matching a compound AND query."""
        # Setup
        mock_stm_store.get_all.return_value = sample_memories
        
        # Define compound conditions
        conditions = [
            {
                "type": "metadata",
                "key": "location",
                "value": "living_room"
            },
            {
                "type": "content",
                "path": "agent",
                "value": "agent1"
            }
        ]
        
        # Call the method
        result = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="AND",
            limit=10,
            tier="stm"
        )
        
        # Verify
        mock_stm_store.get_all.assert_called_once()
        assert len(result) == 2
        assert "mem2" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result]

    def test_retrieve_by_compound_query_or(self, retriever, mock_ltm_store, sample_memories):
        """Test retrieving memories matching a compound OR query."""
        # Setup
        mock_ltm_store.get_all.return_value = sample_memories
        
        # Define compound conditions
        conditions = [
            {
                "type": "metadata",
                "key": "mood",
                "value": "happy"
            },
            {
                "type": "metadata",
                "key": "target",
                "value": "agent2"
            }
        ]
        
        # Call the method
        result = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="OR",
            limit=10,
            tier="ltm"
        )
        
        # Verify
        mock_ltm_store.get_all.assert_called_once()
        assert len(result) == 2
        assert "mem1" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result]

    def test_retrieve_by_compound_query_with_pattern(self, retriever, mock_im_store, sample_memories):
        """Test retrieving memories with a compound query including pattern matching."""
        # Setup
        mock_im_store.get_all.return_value = sample_memories
        
        # Define compound conditions with pattern
        conditions = [
            {
                "type": "metadata",
                "key": "location",
                "value": "living_room"
            },
            {
                "type": "pattern",
                "path": "dialog.text",
                "pattern": "moving"
            }
        ]
        
        # Call the method
        result = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="AND",
            limit=10,
            tier="im"
        )
        
        # Verify
        mock_im_store.get_all.assert_called_once()
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem2"

    def test_matches_metadata_filters(self, retriever):
        """Test the _matches_metadata_filters internal method."""
        # Create test memory
        memory = {
            "metadata": {
                "location": "kitchen",
                "mood": "happy",
                "energy": 0.8
            }
        }
        
        # Test with matching filters
        assert retriever._matches_metadata_filters(memory, {"location": "kitchen"})
        assert retriever._matches_metadata_filters(memory, {"location": "kitchen", "mood": "happy"})
        
        # Test with non-matching filters
        assert not retriever._matches_metadata_filters(memory, {"location": "bedroom"})
        assert not retriever._matches_metadata_filters(memory, {"location": "kitchen", "mood": "sad"})
        
        # Test with missing metadata key
        assert not retriever._matches_metadata_filters(memory, {"missing_key": "value"})
        
        # Test with empty memory
        empty_memory = {}
        assert not retriever._matches_metadata_filters(empty_memory, {"location": "kitchen"})

    def test_get_value_at_path(self, retriever):
        """Test the _get_value_at_path internal method."""
        # Create test object
        obj = {
            "name": "Agent1",
            "location": {
                "name": "kitchen",
                "coordinates": [10, 20]
            },
            "stats": {
                "health": 100,
                "energy": {
                    "current": 80,
                    "max": 100
                }
            },
            "inventory": ["apple", "book"]
        }
        
        # Test simple paths
        assert retriever._get_value_at_path(obj, "name") == "Agent1"
        
        # Test nested paths
        assert retriever._get_value_at_path(obj, "location.name") == "kitchen"
        assert retriever._get_value_at_path(obj, "stats.energy.current") == 80
        
        # Test paths to arrays/lists
        assert retriever._get_value_at_path(obj, "location.coordinates") == [10, 20]
        
        # Test nonexistent paths
        assert retriever._get_value_at_path(obj, "nonexistent") is None
        assert retriever._get_value_at_path(obj, "location.nonexistent") is None
        assert retriever._get_value_at_path(obj, "location.coordinates.nonexistent") is None

    def test_get_store_for_tier(self, retriever, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test the _get_store_for_tier internal method."""
        # Test each tier
        assert retriever._get_store_for_tier("stm") == mock_stm_store
        assert retriever._get_store_for_tier("im") == mock_im_store
        assert retriever._get_store_for_tier("ltm") == mock_ltm_store
        
        # Test default case (invalid tier defaults to STM)
        assert retriever._get_store_for_tier("invalid_tier") == mock_stm_store

    def test_retrieve_by_custom_filter_with_exception(self, retriever, mock_stm_store, sample_memories):
        """Test custom filter handling when the filter function raises an exception."""
        # Setup
        mock_stm_store.get_all.return_value = sample_memories
        
        # Define a filter function that raises an exception for certain memories
        def buggy_filter(memory):
            if memory["memory_id"] == "mem2":
                raise ValueError("Test exception")
            return memory["importance"] > 0.7
        
        # Call the method - should not raise an exception
        result = retriever.retrieve_by_custom_filter(
            filter_fn=buggy_filter,
            limit=10,
            tier="stm"
        )
        
        # Verify - only memories that didn't raise exceptions and matched the filter
        assert len(result) == 2
        assert "mem1" in [m["memory_id"] for m in result]
        assert "mem3" in [m["memory_id"] for m in result] 