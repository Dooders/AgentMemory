"""Unit tests specifically for enhanced error handling in Redis IM Store.

This module tests the detailed exception handling capabilities of the RedisIMStore
class, verifying that different exception types are caught appropriately and
that proper logging occurs.
"""

import json
import logging
import time
from unittest import mock

import pytest
import redis

from agent_memory.config import RedisIMConfig
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.utils.error_handling import (
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing error handling."""
    with mock.patch(
        "agent_memory.storage.redis_client.ResilientRedisClient"
    ) as mock_client:
        # Configure the mock with basic functionality
        mock_client.return_value.pipeline.return_value = mock.MagicMock()
        yield mock_client.return_value


@pytest.fixture
def im_store(mock_redis_client):
    """Create a RedisIMStore instance with mock Redis client."""
    config = RedisIMConfig(
        host="localhost",
        port=6379,
        db=1,
        namespace="test_agent_memory:im",
        ttl=604800,  # 7 days
    )
    store = RedisIMStore(config)
    store.redis = mock_redis_client
    # Mock Lua scripting availability for testing
    store._lua_scripting_available = False
    return store


@pytest.fixture
def mock_logger():
    """Create a mock logger to test logging behavior."""
    with mock.patch("agent_memory.storage.redis_im.logger") as mock_logger:
        yield mock_logger


def create_test_memory():
    """Create a test memory entry."""
    return {
        "memory_id": "test-memory-id",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1, "importance_score": 0.5},
    }


class TestImprovedErrorHandling:
    """Test cases for improved error handling in RedisIMStore."""

    def test_store_memory_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in _store_memory_entry."""
        # Create a test memory
        memory = create_test_memory()

        # Configure the pipeline to raise a RedisError
        mock_pipe = mock.MagicMock()
        mock_pipe.execute.side_effect = redis.RedisError("Connection refused")
        mock_redis_client.pipeline.return_value = mock_pipe

        # Execute the method
        result = im_store._store_memory_entry("test-agent", memory)

        # Verify the result is False
        assert result is False

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when storing memory entry" in args[0]
        assert memory["memory_id"] in args[1]

    def test_store_memory_json_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of JSONDecodeError in _store_memory_entry."""
        # Create a test memory
        memory = create_test_memory()

        # Configure the pipeline
        mock_pipe = mock.MagicMock()
        # Set up the pipeline.execute to raise JSONDecodeError during hash preparation
        mock_pipe.hset.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_redis_client.pipeline.return_value = mock_pipe

        # Execute the method
        result = im_store._store_memory_entry("test-agent", memory)

        # Verify the result is False
        assert result is False

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "JSON encoding error when storing memory entry" in args[0]
        assert memory["memory_id"] in args[1]

    def test_store_memory_generic_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of generic Exception in _store_memory_entry."""
        # Create a test memory
        memory = create_test_memory()

        # Configure the pipeline to raise a generic exception
        mock_pipe = mock.MagicMock()
        mock_pipe.execute.side_effect = Exception("Generic error")
        mock_redis_client.pipeline.return_value = mock_pipe

        # Execute the method
        result = im_store._store_memory_entry("test-agent", memory)

        # Verify the result is False
        assert result is False

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Unexpected error storing memory entry" in args[0]
        assert memory["memory_id"] in args[1]

    def test_get_memory_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in get method."""
        # Configure hgetall to raise a RedisError
        mock_redis_client.hgetall.side_effect = redis.RedisError("Connection refused")

        # Execute the method
        result = im_store.get("test-agent", "test-memory-id")

        # Verify the result is None
        assert result is None

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error retrieving memory" in args[0]
        assert "test-memory-id" in args[1]
        assert "test-agent" in args[2]

    def test_get_memory_json_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of JSONDecodeError in get method."""
        # Configure hgetall to return data that will cause JSONDecodeError
        mock_redis_client.hgetall.return_value = {
            "memory_id": "test-memory-id",
            "metadata": "{invalid",  # Deliberately invalid JSON
        }

        # Mock _hash_to_memory_entry to raise a JSONDecodeError
        original_hash_to_memory_entry = im_store._hash_to_memory_entry

        def mock_hash_to_memory_entry(hash_data):
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        im_store._hash_to_memory_entry = mock_hash_to_memory_entry

        try:
            # Execute the method
            result = im_store.get("test-agent", "test-memory-id")

            # Verify the result is None
            assert result is None

            # Verify logger.exception was called with the right message
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            assert "JSON decoding error for memory" in args[0]
            assert "test-memory-id" in args[1]
            assert "test-agent" in args[2]
        finally:
            # Restore the original method
            im_store._hash_to_memory_entry = original_hash_to_memory_entry

    def test_get_memory_generic_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of generic Exception in get method."""
        # Configure hgetall to raise a generic exception
        mock_redis_client.hgetall.side_effect = Exception("Generic error")

        # Execute the method
        result = im_store.get("test-agent", "test-memory-id")

        # Verify the result is None
        assert result is None

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Unexpected error retrieving memory" in args[0]
        assert "test-memory-id" in args[1]
        assert "test-agent" in args[2]

    def test_get_by_timerange_redis_error(
        self, im_store, mock_logger, mock_redis_client
    ):
        """Test handling of RedisError in get_by_timerange method."""
        # Configure the method to raise a RedisError
        mock_redis_client.zrangebyscore.side_effect = redis.RedisError(
            "Connection refused"
        )

        # Execute the method
        result = im_store.get_by_timerange("test-agent", 0, time.time())

        # Verify the result is an empty list
        assert result == []

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when retrieving memories by time range" in args[0]
        assert "test-agent" in args[1]

    def test_delete_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in delete method."""
        # Configure the pipeline to raise a RedisError
        mock_pipe = mock.MagicMock()
        mock_pipe.execute.side_effect = redis.RedisError("Connection refused")
        mock_redis_client.pipeline.return_value = mock_pipe

        # Execute the method
        result = im_store.delete("test-agent", "test-memory-id")

        # Verify the result is False
        assert result is False

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when deleting memory" in args[0]
        assert "test-memory-id" in args[1]
        assert "test-agent" in args[2]

    def test_count_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in count method."""
        # Configure zcard to raise a RedisError
        mock_redis_client.zcard.side_effect = redis.RedisError("Connection refused")

        # Execute the method
        result = im_store.count("test-agent")

        # Verify the result is 0
        assert result == 0

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when counting memories" in args[0]
        assert "test-agent" in args[1]

    def test_clear_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in clear method."""
        # Configure the zrange to raise a RedisError
        mock_redis_client.zrange.side_effect = redis.RedisError("Connection refused")

        # Execute the method
        result = im_store.clear("test-agent")

        # Verify the result is False
        assert result is False

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when clearing memories" in args[0]
        assert "test-agent" in args[1]

    def test_redis_unavailable_error_propagation(
        self, im_store, mock_logger, mock_redis_client
    ):
        """Test that RedisUnavailableError is properly logged and propagated."""
        # Create a test memory
        memory = create_test_memory()

        # Configure eval to raise RedisUnavailableError
        mock_redis_client.eval.side_effect = RedisUnavailableError(
            "Redis is unavailable"
        )
        im_store._lua_scripting_available = True

        # Verify the exception is propagated for retry handling
        with pytest.raises(RedisUnavailableError):
            im_store._store_memory_entry("test-agent", memory)

    def test_redis_timeout_error_propagation(
        self, im_store, mock_logger, mock_redis_client
    ):
        """Test that RedisTimeoutError is properly logged and propagated."""
        # Create a test memory
        memory = create_test_memory()

        # Configure eval to raise RedisTimeoutError
        mock_redis_client.eval.side_effect = RedisTimeoutError(
            "Redis operation timed out"
        )
        im_store._lua_scripting_available = True

        # Verify the exception is propagated for retry handling
        with pytest.raises(RedisTimeoutError):
            im_store._store_memory_entry("test-agent", memory)

    def test_get_by_importance_redis_error(
        self, im_store, mock_logger, mock_redis_client
    ):
        """Test handling of RedisError in get_by_importance method."""
        # Configure the method to raise a RedisError
        mock_redis_client.zrangebyscore.side_effect = redis.RedisError(
            "Connection refused"
        )

        # Execute the method
        result = im_store.get_by_importance("test-agent", 0.0, 1.0)

        # Verify the result is an empty list
        assert result == []

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Redis error when retrieving memories by importance" in args[0]
        assert "test-agent" in args[1]

    def test_get_by_importance_json_error(
        self, im_store, mock_logger, mock_redis_client
    ):
        """Test handling of JSONDecodeError in get_by_importance method."""
        # Configure zrangebyscore to return memory IDs
        mock_redis_client.zrangebyscore.return_value = ["memory1"]

        # Configure pipeline to return hash data
        mock_pipe = mock.MagicMock()
        mock_pipe.execute.return_value = [{"memory_id": "memory1"}]
        mock_redis_client.pipeline.return_value = mock_pipe

        # Mock _hash_to_memory_entry to raise a JSONDecodeError
        original_hash_to_memory_entry = im_store._hash_to_memory_entry

        def mock_hash_to_memory_entry(hash_data):
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        im_store._hash_to_memory_entry = mock_hash_to_memory_entry

        try:
            # Execute the method
            result = im_store.get_by_importance("test-agent", 0.0, 1.0)

            # Verify the result is an empty list
            assert result == []

            # Verify logger.exception was called with the right message
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            assert (
                "JSON decoding error when processing memories by importance" in args[0]
            )
            assert "test-agent" in args[1]
        finally:
            # Restore the original method
            im_store._hash_to_memory_entry = original_hash_to_memory_entry

    def test_get_all_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in get_all method."""
        # Configure zrange to raise a RedisError
        mock_redis_client.zrange.side_effect = redis.RedisError("Connection refused")

        # Execute the method
        result = im_store.get_all("test-agent")

        # Verify the result is an empty list
        assert result == []

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Error retrieving all memories for agent" in args[0]
        assert "test-agent" in args[1]

    def test_get_size_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test handling of RedisError in get_size method."""
        # Configure scan_iter to raise a RedisError
        mock_redis_client.scan_iter.side_effect = redis.RedisError("Connection refused")

        # Execute the method
        result = im_store.get_size("test-agent")

        # Verify the result is 0
        assert result == 0

        # Verify logger.exception was called with the right message
        mock_logger.exception.assert_called_once()
        args = mock_logger.exception.call_args[0]
        assert "Error retrieving memory size for agent" in args[0]
        assert "test-agent" in args[1]

    def test_search_similar_redis_error(self, im_store, mock_logger, mock_redis_client):
        """Test fallback to Python implementation when Redis vector search fails."""
        # Enable vector search for this test
        im_store._vector_search_available = True

        # Configure execute_command to raise a RedisError
        mock_redis_client.execute_command.side_effect = redis.RedisError(
            "Connection refused"
        )

        # Configure get_all to return empty list for simplicity
        im_store.get_all = mock.MagicMock(return_value=[])

        # Execute the method
        result = im_store.search_similar("test-agent", [0.1, 0.2, 0.3])

        # Verify the result is an empty list
        assert result == []

        # Verify logger.warning was called for the fallback
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        assert (
            "Redis vector search failed, falling back to Python implementation"
            in args[0]
        )

        # Verify get_all was called as part of the fallback
        im_store.get_all.assert_called_once_with("test-agent")

    def test_cosine_similarity_error(self, im_store, mock_logger):
        """Test error handling in _cosine_similarity method."""
        # Configure numpy to raise an error
        with mock.patch("numpy.linalg.norm") as mock_norm:
            mock_norm.side_effect = Exception("Math error")

            # Execute the method
            result = im_store._cosine_similarity([0.1, 0.2], [0.3, 0.4])

            # Verify the result is 0.0
            assert result == 0.0

            # Verify logger.exception was called
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            assert "Error calculating cosine similarity" in args[0]
