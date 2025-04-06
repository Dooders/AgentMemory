"""Unit tests for the temporal-based memory retrieval mechanisms."""

import time
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock

from memory.retrieval.temporal import TemporalRetrieval
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.redis_im import RedisIMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore


class TestTemporalRetrieval:
    """Test suite for the TemporalRetrieval class."""

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
        """Create a TemporalRetrieval instance with mocked stores."""
        return TemporalRetrieval(mock_stm_store, mock_im_store, mock_ltm_store)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            {
                "memory_id": "mem1",
                "step_number": 10,
                "timestamp": int(time.time()) - 3600,  # 1 hour ago
                "metadata": {"memory_type": "observation"},
                "content": {"text": "Agent observed something"},
            },
            {
                "memory_id": "mem2",
                "step_number": 11,
                "timestamp": int(time.time()) - 1800,  # 30 minutes ago
                "metadata": {"memory_type": "action"},
                "content": {"text": "Agent did something"},
            },
            {
                "memory_id": "mem3",
                "step_number": 12,
                "timestamp": int(time.time()) - 900,  # 15 minutes ago
                "metadata": {"memory_type": "reflection"},
                "content": {"text": "Agent thought about something"},
            },
            {
                "memory_id": "mem4",
                "step_number": 13,
                "timestamp": int(time.time()) - 60,  # 1 minute ago
                "metadata": {"memory_type": "observation"},
                "content": {"text": "Agent observed something else"},
            },
            {
                "memory_id": "mem5",
                "step_number": 14,
                "timestamp": int(time.time()),  # Now
                "metadata": {"memory_type": "action"},
                "content": {"text": "Agent did something else"},
            },
        ]

    def test_initialization(self, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test that TemporalRetrieval initializes correctly."""
        retriever = TemporalRetrieval(mock_stm_store, mock_im_store, mock_ltm_store)
        assert retriever.stm_store is mock_stm_store
        assert retriever.im_store is mock_im_store
        assert retriever.ltm_store is mock_ltm_store

    def test_get_store_for_tier(self, retriever, mock_stm_store, mock_im_store, mock_ltm_store):
        """Test that the correct store is returned for each tier."""
        assert retriever._get_store_for_tier("stm") is mock_stm_store
        assert retriever._get_store_for_tier("im") is mock_im_store
        assert retriever._get_store_for_tier("ltm") is mock_ltm_store
        # Test default value
        assert retriever._get_store_for_tier("unknown") is mock_stm_store

    def test_retrieve_recent_stm(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving recent memories from STM."""
        mock_stm_store.get_recent.return_value = sample_memories
        
        # Test with default values
        result = retriever.retrieve_recent()
        mock_stm_store.get_recent.assert_called_once_with(count=20)
        assert result == sample_memories[:10]
        
        # Reset mock
        mock_stm_store.reset_mock()
        
        # Test with custom count
        result = retriever.retrieve_recent(count=2)
        mock_stm_store.get_recent.assert_called_once_with(count=4)
        assert result == sample_memories[:2]

    def test_retrieve_recent_with_memory_type(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving recent memories filtered by memory type."""
        mock_stm_store.get_recent.return_value = sample_memories
        
        # Filter by observation type
        result = retriever.retrieve_recent(memory_type="observation")
        mock_stm_store.get_recent.assert_called_once_with(count=20)
        
        # Should only contain observations
        assert len(result) == 2
        assert all(mem.get("metadata", {}).get("memory_type") == "observation" for mem in result)

    def test_retrieve_recent_from_different_tiers(self, retriever, mock_stm_store, mock_im_store, mock_ltm_store, sample_memories):
        """Test retrieving recent memories from different tiers."""
        mock_stm_store.get_recent.return_value = sample_memories
        mock_im_store.get_recent.return_value = sample_memories
        mock_ltm_store.get_recent.return_value = sample_memories
        
        # Test STM tier
        result_stm = retriever.retrieve_recent(tier="stm")
        mock_stm_store.get_recent.assert_called_once_with(count=20)
        assert result_stm == sample_memories[:10]
        
        # Test IM tier
        result_im = retriever.retrieve_recent(tier="im")
        mock_im_store.get_recent.assert_called_once_with(count=20)
        assert result_im == sample_memories[:10]
        
        # Test LTM tier
        result_ltm = retriever.retrieve_recent(tier="ltm")
        mock_ltm_store.get_recent.assert_called_once_with(count=20)
        assert result_ltm == sample_memories[:10]

    def test_retrieve_by_step(self, retriever, mock_stm_store):
        """Test retrieving memory by step number."""
        memory = {"memory_id": "test", "step_number": 42}
        mock_stm_store.get_by_step.return_value = memory
        
        result = retriever.retrieve_by_step(step=42)
        mock_stm_store.get_by_step.assert_called_once_with(42)
        assert result == memory
        
        # Test with a different tier
        mock_im_store = retriever.im_store
        mock_im_store.get_by_step.return_value = memory
        
        result = retriever.retrieve_by_step(step=42, tier="im")
        mock_im_store.get_by_step.assert_called_once_with(42)
        assert result == memory

    def test_retrieve_step_range(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories within a step range."""
        mock_stm_store.get_step_range.return_value = sample_memories
        
        # Test basic step range retrieval
        result = retriever.retrieve_step_range(start_step=10, end_step=14)
        mock_stm_store.get_step_range.assert_called_once_with(10, 14)
        assert result == sample_memories
        
        # Reset mock
        mock_stm_store.reset_mock()
        
        # Test with memory type filter
        result = retriever.retrieve_step_range(start_step=10, end_step=14, memory_type="observation")
        mock_stm_store.get_step_range.assert_called_once_with(10, 14)
        assert len(result) == 2
        assert all(mem.get("metadata", {}).get("memory_type") == "observation" for mem in result)

    def test_retrieve_time_range(self, retriever, mock_im_store, sample_memories):
        """Test retrieving memories within a time range."""
        mock_im_store.get_time_range.return_value = sample_memories
        
        # Test basic time range retrieval
        start_time = int(time.time()) - 3600
        end_time = int(time.time())
        
        result = retriever.retrieve_time_range(start_time=start_time, end_time=end_time, tier="im")
        mock_im_store.get_time_range.assert_called_once_with(start_time, end_time)
        assert result == sample_memories
        
        # Reset mock
        mock_im_store.reset_mock()
        
        # Test with memory type filter
        result = retriever.retrieve_time_range(
            start_time=start_time, 
            end_time=end_time, 
            memory_type="action", 
            tier="im"
        )
        mock_im_store.get_time_range.assert_called_once_with(start_time, end_time)
        assert len(result) == 2
        assert all(mem.get("metadata", {}).get("memory_type") == "action" for mem in result)

    def test_retrieve_last_n_minutes(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving memories from the last N minutes."""
        mock_stm_store.get_time_range.return_value = sample_memories
        
        with patch('time.time') as mock_time:
            current_time = 1625097600  # Fixed timestamp for testing
            mock_time.return_value = current_time
            
            # Test retrieving from last 60 minutes
            result = retriever.retrieve_last_n_minutes(minutes=60)
            
            expected_start_time = current_time - (60 * 60)
            mock_stm_store.get_time_range.assert_called_once_with(
                expected_start_time, current_time
            )
            assert result == sample_memories
            
            # Reset mock
            mock_stm_store.reset_mock()
            
            # Test with memory type filter
            result = retriever.retrieve_last_n_minutes(minutes=30, memory_type="reflection")
            
            expected_start_time = current_time - (30 * 60)
            mock_stm_store.get_time_range.assert_called_once_with(
                expected_start_time, current_time
            )
            assert len(result) == 1
            assert all(mem.get("metadata", {}).get("memory_type") == "reflection" for mem in result)

    def test_retrieve_oldest(self, retriever, mock_ltm_store, sample_memories):
        """Test retrieving the oldest memories."""
        mock_ltm_store.get_oldest.return_value = sample_memories
        
        # Test with default values
        result = retriever.retrieve_oldest(tier="ltm")
        mock_ltm_store.get_oldest.assert_called_once_with(count=20)
        assert result == sample_memories[:10]
        
        # Reset mock
        mock_ltm_store.reset_mock()
        
        # Test with custom count
        result = retriever.retrieve_oldest(count=3, tier="ltm")
        mock_ltm_store.get_oldest.assert_called_once_with(count=6)
        assert result == sample_memories[:3]
        
        # Reset mock
        mock_ltm_store.reset_mock()
        
        # Test with memory type filter
        result = retriever.retrieve_oldest(memory_type="action", tier="ltm")
        mock_ltm_store.get_oldest.assert_called_once_with(count=20)
        assert len(result) == 2
        assert all(mem.get("metadata", {}).get("memory_type") == "action" for mem in result)

    def test_retrieve_narrative_sequence_successful(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving a narrative sequence when central memory is found."""
        central_memory = sample_memories[2]  # Memory with step 12
        before_memories = sample_memories[:2]  # Memories with steps 10-11
        after_memories = sample_memories[3:]  # Memories with steps 13-14
        
        mock_stm_store.get.return_value = central_memory
        mock_stm_store.get_step_range.side_effect = [before_memories, after_memories]
        
        result = retriever.retrieve_narrative_sequence(memory_id="mem3")
        
        # Check that the correct store methods were called
        mock_stm_store.get.assert_called_once_with("mem3")
        
        # Check that get_step_range was called for before and after
        assert mock_stm_store.get_step_range.call_count == 2
        mock_stm_store.get_step_range.assert_any_call(9, 11)  # 3 steps before
        mock_stm_store.get_step_range.assert_any_call(13, 15)  # 3 steps after
        
        # Check the result sequence
        assert len(result) == 5
        assert [m["step_number"] for m in result] == [10, 11, 12, 13, 14]

    def test_retrieve_narrative_sequence_memory_not_found(self, retriever, mock_stm_store):
        """Test retrieving a narrative sequence when central memory is not found."""
        mock_stm_store.get.return_value = None
        
        with patch('agent_memory.retrieval.temporal.logger') as mock_logger:
            result = retriever.retrieve_narrative_sequence(memory_id="nonexistent")
            
            # Check that the correct warnings were logged
            mock_logger.warning.assert_called_once_with(
                "Central memory %s not found in %s", "nonexistent", "stm"
            )
            
            # Result should be an empty list
            assert result == []

    def test_retrieve_narrative_sequence_no_step_number(self, retriever, mock_stm_store):
        """Test retrieving a narrative sequence when central memory has no step number."""
        # Memory without step_number
        central_memory = {
            "memory_id": "mem_no_step",
            "timestamp": int(time.time())
        }
        
        mock_stm_store.get.return_value = central_memory
        
        with patch('agent_memory.retrieval.temporal.logger') as mock_logger:
            result = retriever.retrieve_narrative_sequence(memory_id="mem_no_step")
            
            # Check that the correct warnings were logged
            mock_logger.warning.assert_called_once_with(
                "Central memory %s has no step number", "mem_no_step"
            )
            
            # Result should be an empty list
            assert result == []

    def test_retrieve_narrative_sequence_custom_context(self, retriever, mock_stm_store, sample_memories):
        """Test retrieving a narrative sequence with custom context sizes."""
        central_memory = sample_memories[2]  # Memory with step 12
        before_memories = [sample_memories[1]]  # Just memory with step 11
        after_memories = [sample_memories[3]]  # Just memory with step 13
        
        mock_stm_store.get.return_value = central_memory
        mock_stm_store.get_step_range.side_effect = [before_memories, after_memories]
        
        result = retriever.retrieve_narrative_sequence(
            memory_id="mem3", 
            context_before=1, 
            context_after=1
        )
        
        # Check that get_step_range was called with correct parameters
        assert mock_stm_store.get_step_range.call_count == 2
        mock_stm_store.get_step_range.assert_any_call(11, 11)  # 1 step before
        mock_stm_store.get_step_range.assert_any_call(13, 13)  # 1 step after
        
        # Check the result sequence
        assert len(result) == 3
        assert [m["step_number"] for m in result] == [11, 12, 13]
        
    def test_retrieve_narrative_sequence_different_tier(self, retriever, mock_im_store, sample_memories):
        """Test retrieving a narrative sequence from a different tier."""
        central_memory = sample_memories[2]  # Memory with step 12
        before_memories = sample_memories[:2]  # Memories with steps 10-11
        after_memories = sample_memories[3:]  # Memories with steps 13-14
        
        mock_im_store.get.return_value = central_memory
        mock_im_store.get_step_range.side_effect = [before_memories, after_memories]
        
        result = retriever.retrieve_narrative_sequence(memory_id="mem3", tier="im")
        
        # Check that the correct store methods were called
        mock_im_store.get.assert_called_once_with("mem3")
        
        # Check that get_step_range was called for before and after
        assert mock_im_store.get_step_range.call_count == 2
        
        # Check the result sequence
        assert len(result) == 5
        assert [m["step_number"] for m in result] == [10, 11, 12, 13, 14]

    def test_empty_memories_handling(self, retriever, mock_stm_store):
        """Test handling of empty memory results."""
        # Store returns empty lists
        mock_stm_store.get_recent.return_value = []
        mock_stm_store.get_step_range.return_value = []
        mock_stm_store.get_time_range.return_value = []
        mock_stm_store.get_oldest.return_value = []
        
        # Test various retrieval methods with empty results
        assert retriever.retrieve_recent() == []
        assert retriever.retrieve_step_range(1, 10) == []
        assert retriever.retrieve_time_range(0, int(time.time())) == []
        assert retriever.retrieve_last_n_minutes(30) == []
        assert retriever.retrieve_oldest() == []

    def test_retrieve_recent(self, retriever, sample_memories):
        """Test retrieving recent memories."""
        query = {"time_window": 3600}  # Last hour
        results = retriever.retrieve(sample_memories, query, limit=5)

        # Should retrieve most recent memories
        assert len(results) == 5
        # Check that results are sorted by recency (newest first)
        timestamps = [r["timestamp"] for r in results]
        assert sorted(timestamps, reverse=True) == timestamps
        # Check that memory content is preserved
        assert all("content" in r for r in results)

    def test_retrieve_before_timestamp(self, retriever, sample_memories):
        """Test retrieving memories before a timestamp."""
        # Get a timestamp from the middle of our test set
        middle_index = len(sample_memories) // 2
        middle_timestamp = sample_memories[middle_index]["timestamp"]

        query = {"before": middle_timestamp}
        results = retriever.retrieve(sample_memories, query, limit=10)

        # Check that all results are before the timestamp
        assert all(r["timestamp"] < middle_timestamp for r in results)
        # Check that memory content is preserved
        assert all("content" in r for r in results)

    def test_retrieve_after_timestamp(self, retriever, sample_memories):
        """Test retrieving memories after a timestamp."""
        # Get a timestamp from the middle of our test set
        middle_index = len(sample_memories) // 2
        middle_timestamp = sample_memories[middle_index]["timestamp"]

        query = {"after": middle_timestamp}
        results = retriever.retrieve(sample_memories, query, limit=10)

        # Check that all results are after the timestamp
        assert all(r["timestamp"] > middle_timestamp for r in results)
        # Check that memory content is preserved
        assert all("content" in r for r in results)

    def test_retrieve_time_range(self, retriever, sample_memories):
        """Test retrieving memories in a time range."""
        # Get timestamps from first quarter and third quarter
        quarter_index = len(sample_memories) // 4
        third_quarter_index = 3 * quarter_index
        start_timestamp = sample_memories[quarter_index]["timestamp"]
        end_timestamp = sample_memories[third_quarter_index]["timestamp"]

        query = {"after": start_timestamp, "before": end_timestamp}
        results = retriever.retrieve(sample_memories, query, limit=10)

        # Check that all results are within the time range
        assert all(
            start_timestamp < r["timestamp"] < end_timestamp for r in results
        )
        # Check that memory content is preserved
        assert all("content" in r for r in results)

    def test_retrieve_by_step_number(self, retriever, sample_memories):
        """Test retrieving memories by step number."""
        query = {"step_number": 5}
        results = retriever.retrieve(sample_memories, query, limit=10)

        # Check that all results have the requested step number
        assert all(r.get("step_number") == 5 for r in results)
        # Check that memory content is preserved
        assert all("content" in r for r in results) 