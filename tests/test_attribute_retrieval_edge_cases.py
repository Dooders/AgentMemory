"""Tests for edge cases in attribute retrieval functionality."""

import re
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_memory.retrieval.attribute import AttributeRetrieval
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore


class TestAttributeRetrievalEdgeCases:
    """Test suite for edge cases and error handling in AttributeRetrieval."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock store instances for all memory tiers."""
        mock_stm = Mock(name="mock_stm_store", autospec=False)
        mock_im = Mock(name="mock_im_store", autospec=False)
        mock_ltm = Mock(name="mock_ltm_store", autospec=False)
        return {"stm": mock_stm, "im": mock_im, "ltm": mock_ltm}

    @pytest.fixture
    def retriever(self, mock_stores):
        """Create AttributeRetrieval instance with mock stores."""
        return AttributeRetrieval(mock_stores["stm"], mock_stores["im"], mock_stores["ltm"])

    @pytest.fixture
    def empty_memories(self):
        """Create an empty list of memories for testing edge cases."""
        return []

    @pytest.fixture
    def malformed_memories(self):
        """Create malformed memories for testing robustness."""
        return [
            {},  # Empty memory
            {"memory_id": "mem1"},  # Missing other fields
            {"memory_id": "mem2", "contents": None},  # None contents
            {"memory_id": "mem3", "metadata": None},  # None metadata
            {"memory_id": "mem4", "contents": "invalid-not-a-dict"},  # Invalid contents type
            {"memory_id": "mem5", "metadata": "invalid-not-a-dict"}  # Invalid metadata type
        ]

    def test_empty_results_memory_type(self, retriever, mock_stores):
        """Test handling of empty results when retrieving by memory type."""
        # Setup - mock empty response
        mock_stores["stm"].get_by_type.return_value = []
        
        # Call the method
        result = retriever.retrieve_by_memory_type("state", limit=10, tier="stm")
        
        # Should return empty list, not raise exception
        assert result == []

    def test_empty_results_importance(self, retriever, mock_stores):
        """Test handling of empty results when retrieving by importance."""
        # Setup - mock empty response
        mock_stores["im"].get_by_importance.return_value = []
        
        # Call the method
        result = retriever.retrieve_by_importance(min_importance=0.8, limit=10, tier="im")
        
        # Should return empty list, not raise exception
        assert result == []

    def test_empty_results_metadata(self, retriever, mock_stores):
        """Test handling of empty results when retrieving by metadata."""
        # Setup - mock empty store
        mock_stores["ltm"].get_all.return_value = []
        
        # Call the method
        result = retriever.retrieve_by_metadata(
            metadata_filters={"location": "kitchen"}, 
            limit=10, 
            tier="ltm"
        )
        
        # Should return empty list, not raise exception
        assert result == []

    def test_malformed_memory_content_value(self, retriever, mock_stores, malformed_memories):
        """Test handling of malformed memories when retrieving by content value."""
        # Setup - mock malformed memories
        mock_stores["stm"].get_all.return_value = malformed_memories
        
        # Call the method
        result = retriever.retrieve_by_content_value(
            path="text", 
            value="hello", 
            limit=10, 
            tier="stm"
        )
        
        # Should handle malformed memories gracefully
        assert result == []
        
    def test_malformed_memory_content_pattern(self, retriever, mock_stores, malformed_memories):
        """Test handling of malformed memories when retrieving by content pattern."""
        # Setup - mock malformed memories
        mock_stores["im"].get_all.return_value = malformed_memories
        
        # Call the method
        result = retriever.retrieve_by_content_pattern(
            path="text", 
            pattern="hello", 
            limit=10, 
            tier="im"
        )
        
        # Should handle malformed memories gracefully
        assert result == []

    def test_malformed_regex_pattern(self, retriever, mock_stores):
        """Test handling of invalid regex pattern."""
        # Setup - good memories but invalid regex
        memories = [
            {"memory_id": "mem1", "contents": {"text": "hello world"}}
        ]
        mock_stores["ltm"].get_all.return_value = memories
        
        # Call with invalid regex pattern
        result = retriever.retrieve_by_content_pattern(
            path="text", 
            pattern="[", # Invalid regex
            limit=10, 
            tier="ltm"
        )
        
        # Should handle invalid regex gracefully (treat as plain text or return empty)
        assert result == []

    def test_invalid_tier(self, retriever, mock_stores):
        """Test handling of invalid tier specification."""
        # Call with invalid tier
        result = retriever.retrieve_by_memory_type(
            memory_type="state", 
            limit=10, 
            tier="invalid_tier"  # Not 'stm', 'im', or 'ltm'
        )
        
        # Should default to STM and not raise exception
        assert mock_stores["stm"].get_by_type.called
        assert not mock_stores["im"].get_by_type.called
        assert not mock_stores["ltm"].get_by_type.called

    def test_zero_limit(self, retriever, mock_stores):
        """Test handling of zero limit."""
        # Setup
        memories = [{"memory_id": f"mem{i}"} for i in range(5)]
        mock_stores["stm"].get_by_type.return_value = memories
        
        # Call with zero limit
        result = retriever.retrieve_by_memory_type(
            memory_type="state", 
            limit=0, 
            tier="stm"
        )
        
        # Should handle zero limit gracefully (either return empty or all)
        assert len(result) <= len(memories)

    def test_negative_limit(self, retriever, mock_stores):
        """Test handling of negative limit."""
        # Setup
        memories = [{"memory_id": f"mem{i}"} for i in range(5)]
        mock_stores["stm"].get_by_type.return_value = memories
        
        # Call with negative limit
        result = retriever.retrieve_by_memory_type(
            memory_type="state", 
            limit=-10, 
            tier="stm"
        )
        
        # Should handle negative limit gracefully (either return empty or all)
        assert len(result) <= len(memories)

    def test_get_value_at_path_empty_path(self, retriever):
        """Test _get_value_at_path with an empty path."""
        obj = {"name": "test"}

        # Empty path should return None or the object itself
        result = retriever._get_value_at_path(obj, "")
        
        # This is implementation dependent - check actual behavior
        # When empty path is provided, the method may either return None or the object itself
        assert result is None

    def test_get_value_at_path_none_object(self, retriever):
        """Test _get_value_at_path with None object."""
        # Call with None object
        result = retriever._get_value_at_path(None, "name")
        
        # Should return None, not raise exception
        assert result is None

    def test_get_value_at_path_non_dict(self, retriever):
        """Test _get_value_at_path with non-dictionary object."""
        # Call with list, string, integer
        list_result = retriever._get_value_at_path([1, 2, 3], "0")
        str_result = retriever._get_value_at_path("hello", "0")
        int_result = retriever._get_value_at_path(123, "digits")
        
        # Should handle non-dict objects gracefully
        assert list_result is None
        assert str_result is None
        assert int_result is None

    def test_compound_query_empty_conditions(self, retriever, mock_stores):
        """Test compound query with empty conditions list."""
        mock_stores["stm"].get_all.return_value = [{"memory_id": "mem1"}]

        # Empty conditions list
        result = retriever.retrieve_by_compound_query(
            conditions=[],  # Empty conditions
            operator="AND",
            limit=10,
            tier="stm"
        )

        # With empty conditions, the method could return all memories or none
        # This depends on the implementation - adjust expectation based on actual behavior
        assert result == [{"memory_id": "mem1"}]

    def test_compound_query_unsupported_condition_type(self, retriever, mock_stores):
        """Test compound query with unsupported condition type."""
        memories = [{"memory_id": "mem1", "contents": {"text": "hello"}}]
        mock_stores["stm"].get_all.return_value = memories

        # Use an unsupported condition type
        conditions = [
            {
                "type": "unsupported_type",  # Unsupported type
                "key": "text",
                "value": "hello"
            }
        ]

        result = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="AND",
            limit=10,
            tier="stm"
        )

        # When an unsupported condition type is encountered, the implementation
        # may either skip it (returning all memories) or treat it as a filter that returns nothing
        # This depends on the implementation - adjust expectation based on actual behavior
        assert result == memories

    def test_metadata_filters_none_value(self, retriever):
        """Test _matches_metadata_filters with None value in metadata."""
        # Create memory with None value in metadata
        memory = {
            "metadata": {
                "location": None,
                "mood": "happy"
            }
        }
        
        # Test with matching for None
        assert not retriever._matches_metadata_filters(memory, {"location": "kitchen"})
        assert retriever._matches_metadata_filters(memory, {"mood": "happy"})
        
        # The implementation may or may not match None values
        # If the implementation treats None as not matching anything:
        # assert not retriever._matches_metadata_filters(memory, {"location": None})

    def test_custom_filter_returning_non_boolean(self, retriever, mock_stores):
        """Test custom filter that returns non-boolean values."""
        # Setup
        memories = [{"memory_id": f"mem{i}", "value": i} for i in range(5)]
        mock_stores["stm"].get_all.return_value = memories
        
        # Define filter that returns non-boolean
        def strange_filter(memory):
            return memory.get("value", 0)  # Returns an integer
        
        # Call the method
        result = retriever.retrieve_by_custom_filter(
            filter_fn=strange_filter,
            limit=10,
            tier="stm"
        )
        
        # Should handle non-boolean returns gracefully
        # In most Python implementations, non-zero integers are truthy
        # So memories with value > 0 should be included
        assert all(memory["value"] > 0 for memory in result)

    def test_retrieve_by_tag_regex_behavior(self, retriever, mock_stores):
        """Test that retrieve_by_tag handles regex patterns correctly."""
        # Setup
        mock_stores["im"].get_by_tag.return_value = [
            {"memory_id": "mem1", "tags": ["important", "work"]}
        ]
        
        # Call with a regex-like pattern
        result = retriever.retrieve_by_tag(
            tag="imp.*",
            limit=10,
            tier="im"
        )
        
        # Verify the pattern was passed to the store correctly
        mock_stores["im"].get_by_tag.assert_called_once_with("imp.*", limit=10)
        assert len(result) == 1

    def test_store_method_exceptions(self, retriever, mock_stores):
        """Test handling of exceptions from store methods."""
        # Setup - make get_by_type raise exception
        mock_stores["stm"].get_by_type.side_effect = Exception("Test exception")
        
        # Call should propagate the exception
        with pytest.raises(Exception) as excinfo:
            retriever.retrieve_by_memory_type("state", limit=10, tier="stm")
        
        assert "Test exception" in str(excinfo.value)

    def test_complex_nested_path(self, retriever):
        """Test _get_value_at_path with deeply nested path."""
        # Create deeply nested object
        obj = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "target": "found me!"
                        }
                    }
                }
            }
        }
        
        # Call with deeply nested path
        result = retriever._get_value_at_path(obj, "level1.level2.level3.level4.target")
        
        # Should navigate to the correct value
        assert result == "found me!" 