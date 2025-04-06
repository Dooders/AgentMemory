"""Unit tests for Redis Intermediate Memory (IM) storage.

This module contains tests for the RedisIMStore class which provides Redis-based
storage for the intermediate memory tier.
"""

import json
import time
from unittest import mock

import numpy as np
import pytest

from agent_memory.config import RedisIMConfig
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.utils.error_handling import (
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    with mock.patch(
        "agent_memory.storage.redis_client.ResilientRedisClient"
    ) as mock_client:
        # Configure the mock to return success for store operations
        mock_client.return_value.store_with_retry.return_value = True
        mock_client.return_value.set.return_value = True
        mock_client.return_value.hset.return_value = True
        mock_client.return_value.zadd.return_value = 1
        mock_client.return_value.expire.return_value = True
        mock_client.return_value.zrange.return_value = ["memory1", "memory2"]
        mock_client.return_value.zrangebyscore.return_value = ["memory1", "memory2"]
        mock_client.return_value.zcard.return_value = 2
        mock_client.return_value.delete.return_value = 1
        mock_client.return_value.zrem.return_value = 1

        # Configure get to return a mock memory entry (keep for backward compatibility)
        def get_side_effect(key):
            if "memory:memory1" in key:
                return json.dumps(
                    {
                        "memory_id": "memory1",
                        "content": "Test memory 1",
                        "timestamp": time.time(),
                        "metadata": {
                            "compression_level": 1,
                            "importance_score": 0.7,
                            "retrieval_count": 0,
                        },
                    }
                )
            elif "memory:memory2" in key:
                return json.dumps(
                    {
                        "memory_id": "memory2",
                        "content": "Test memory 2",
                        "timestamp": time.time(),
                        "metadata": {
                            "compression_level": 1,
                            "importance_score": 0.5,
                            "retrieval_count": 0,
                        },
                    }
                )
            return None

        mock_client.return_value.get.side_effect = get_side_effect

        # Configure hgetall to return mock hash data
        def hgetall_side_effect(key):
            if "memory:memory1" in key:
                return {
                    "memory_id": "memory1",
                    "content": json.dumps("Test memory 1"),
                    "timestamp": str(time.time()),
                    "metadata": json.dumps(
                        {
                            "compression_level": 1,
                            "importance_score": 0.7,
                            "retrieval_count": 0,
                        }
                    ),
                    "compression_level": "1",
                    "importance_score": "0.7",
                    "retrieval_count": "0",
                }
            elif "memory:memory2" in key:
                return {
                    "memory_id": "memory2",
                    "content": json.dumps("Test memory 2"),
                    "timestamp": str(time.time()),
                    "metadata": json.dumps(
                        {
                            "compression_level": 1,
                            "importance_score": 0.5,
                            "retrieval_count": 0,
                        }
                    ),
                    "compression_level": "1",
                    "importance_score": "0.5",
                    "retrieval_count": "0",
                }
            return {}

        mock_client.return_value.hgetall.side_effect = hgetall_side_effect

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
    # Mock vector search availability for testing
    store._vector_search_available = False
    # Mock Lua scripting availability for testing
    store._lua_scripting_available = True
    # Configure eval to return success by default
    mock_redis_client.eval.return_value = 1
    return store


def test_init():
    """Test RedisIMStore initialization."""
    config = RedisIMConfig(
        host="test-host", port=1234, db=2, namespace="test:namespace", ttl=300
    )

    with mock.patch(
        "agent_memory.storage.redis_im.RedisFactory.create_client"
    ) as mock_factory:
        store = RedisIMStore(config)

        # Check that Redis client was initialized with correct parameters
        mock_factory.assert_called_once_with(
            client_name="im",
            use_mock=config.use_mock,
            host="test-host",
            port=1234,
            db=2,
            password=None,
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )

        # Check store attributes
        assert store.config == config
        assert store._key_prefix == "test:namespace"


def test_store_success(im_store):
    """Test successful memory storage."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1, "importance_score": 0.8},
    }

    im_store.redis.store_with_retry = mock.MagicMock(return_value=True)

    result = im_store.store("agent1", memory_entry)

    # Check the result
    assert result is True

    # Verify store_with_retry was called correctly with keyword arguments
    im_store.redis.store_with_retry.assert_called_once()

    # Get the call arguments
    call_args = im_store.redis.store_with_retry.call_args
    kwargs = call_args.kwargs

    # Verify the kwargs contain the expected values
    assert kwargs["agent_id"] == "agent1"
    assert kwargs["state_data"] == memory_entry
    assert kwargs["store_func"] == im_store._store_memory_entry
    assert kwargs["priority"] == Priority.NORMAL


def test_store_missing_memory_id(im_store):
    """Test storing a memory without a memory_id."""
    memory_entry = {
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1},
    }

    result = im_store.store("agent1", memory_entry)

    # Should fail and not call store_with_retry
    assert result is False
    im_store.redis.store_with_retry.assert_not_called()


def test_store_invalid_compression_level(im_store):
    """Test storing a memory with incorrect compression level."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 2},  # Should be 1 for IM
    }

    result = im_store.store("agent1", memory_entry)

    # Should fail and not call store_with_retry
    assert result is False
    im_store.redis.store_with_retry.assert_not_called()


def test_store_memory_entry(im_store):
    """Test the internal _store_memory_entry method."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": 1234567890.0,
        "metadata": {"compression_level": 1, "importance_score": 0.8},
    }

    # Mock eval method
    im_store.redis.eval = mock.MagicMock(return_value=1)

    result = im_store._store_memory_entry("agent1", memory_entry)

    # Check the result
    assert result is True

    # Verify eval was called with correct arguments
    im_store.redis.eval.assert_called_once()

    # Get the args from the call
    eval_args = im_store.redis.eval.call_args[0]

    # Verify the Lua script is provided (first argument)
    assert "local memory_key = KEYS[1]" in eval_args[0]

    # Verify correct number of keys
    assert eval_args[1] == 4  # Number of keys should be 4

    # Verify the keys are correct
    assert eval_args[2] == f"{im_store._key_prefix}:agent1:memory:test-memory-1"
    assert eval_args[3] == f"{im_store._key_prefix}:agent1:memories"
    assert eval_args[4] == f"{im_store._key_prefix}:agent1:timeline"
    assert eval_args[5] == f"{im_store._key_prefix}:agent1:importance"

    # Verify hash values in JSON format
    hash_values = json.loads(eval_args[6])
    assert hash_values["memory_id"] == "test-memory-1"
    assert hash_values["timestamp"] == str(1234567890.0)
    assert hash_values["importance_score"] == "0.8"


def test_store_memory_entry_redis_error(im_store):
    """Test handling of Redis errors in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1},
    }

    # Mock eval and make it raise an exception
    im_store.redis.eval = mock.MagicMock(side_effect=Exception("Redis error"))
    # Disable Lua scripting to test fallback
    im_store._lua_scripting_available = False
    
    # Mock pipeline to throw an exception when execute is called
    mock_pipeline = mock.MagicMock()
    mock_pipeline.execute.side_effect = Exception("Redis pipeline error")
    im_store.redis.pipeline.return_value = mock_pipeline

    # Should catch exception and return False
    result = im_store._store_memory_entry("agent1", memory_entry)
    assert result is False


def test_store_memory_entry_redis_timeout(im_store):
    """Test handling of Redis timeout in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1},
    }

    # Mock eval and make it raise a timeout error
    im_store.redis.eval = mock.MagicMock(side_effect=RedisTimeoutError("Redis timeout"))
    # Ensure Lua scripting is enabled for this test
    im_store._lua_scripting_available = True

    # Should raise the timeout error for retry handling
    with pytest.raises(RedisTimeoutError):
        im_store._store_memory_entry("agent1", memory_entry)


def test_store_memory_entry_redis_unavailable(im_store):
    """Test handling of Redis unavailability in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1},
    }

    # Mock eval and make it raise an unavailability error
    im_store.redis.eval = mock.MagicMock(
        side_effect=RedisUnavailableError("Redis unavailable")
    )
    # Ensure Lua scripting is enabled for this test
    im_store._lua_scripting_available = True

    # Should raise the unavailability error for retry handling
    with pytest.raises(RedisUnavailableError):
        im_store._store_memory_entry("agent1", memory_entry)


def test_get_memory(im_store):
    """Test retrieving a memory by ID."""
    # Mock eval for _update_access_metadata
    im_store.redis.eval = mock.MagicMock(return_value=1)

    memory = im_store.get("agent1", "memory1")

    # Verify we got the memory
    assert memory is not None
    assert memory["memory_id"] == "memory1"
    assert memory["content"] == "Test memory 1"

    # Verify hgetall was called with correct key
    im_store.redis.hgetall.assert_called_with(
        "test_agent_memory:im:agent1:memory:memory1"
    )

    # Verify eval was called for update metadata
    im_store.redis.eval.assert_called_once()

    # Get the args from the call
    eval_args = im_store.redis.eval.call_args[0]

    # Verify the Lua script is provided (first argument)
    assert "local key = KEYS[1]" in eval_args[0]

    # Verify correct number of keys
    assert eval_args[1] == 2  # Number of keys should be 2

    # Verify the keys are correct
    assert eval_args[2] == f"{im_store._key_prefix}:agent1:memory:memory1"
    assert eval_args[3] == f"{im_store._key_prefix}:agent1:importance"


def test_get_nonexistent_memory(im_store):
    """Test retrieving a memory that doesn't exist."""
    # Make the hgetall call return an empty dict for a non-existent memory
    im_store.redis.hgetall.return_value = {}

    memory = im_store.get("agent1", "nonexistent")

    # Should return None for non-existent memory
    assert memory is None


def test_get_memory_redis_error(im_store):
    """Test retrieving a memory when Redis raises an error."""
    # Make Redis hgetall raise an exception
    im_store.redis.hgetall.side_effect = Exception("Redis error")

    memory = im_store.get("agent1", "memory1")

    # Should catch exception and return None
    assert memory is None


def test_get_memory_redis_timeout(im_store):
    """Test retrieving a memory when Redis times out."""
    # Make Redis hgetall raise a timeout exception
    im_store.redis.hgetall.side_effect = RedisTimeoutError("Operation timed out")

    memory = im_store.get("agent1", "memory1")

    # Should catch exception and return None
    assert memory is None


def test_get_memory_redis_unavailable(im_store):
    """Test retrieving a memory when Redis is unavailable."""
    # Make Redis hgetall raise an unavailability exception
    im_store.redis.hgetall.side_effect = RedisUnavailableError("Redis unavailable")

    memory = im_store.get("agent1", "memory1")

    # Should catch exception and return None
    assert memory is None


def test_get_by_timerange(im_store):
    """Test retrieving memories by time range."""
    # Mock pipeline
    mock_pipe = mock.MagicMock()
    # Configure pipeline to return two memory hash entries
    mock_pipe.execute.return_value = [
        {
            "memory_id": "memory1",
            "content": json.dumps("Test memory 1"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.7}),
            "compression_level": "1",
            "importance_score": "0.7",
        },
        {
            "memory_id": "memory2",
            "content": json.dumps("Test memory 2"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.5}),
            "compression_level": "1",
            "importance_score": "0.5",
        },
    ]
    im_store.redis.pipeline.return_value = mock_pipe

    # Mock update_access_metadata
    im_store._update_access_metadata = mock.MagicMock()

    memories = im_store.get_by_timerange(
        agent_id="agent1", start_time=0, end_time=time.time() + 1000, limit=10
    )

    # Should return list of memories
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "memory1"
    assert memories[1]["memory_id"] == "memory2"

    # Verify zrangebyscore was called with correct parameters
    im_store.redis.zrangebyscore.assert_called_once()
    args, kwargs = im_store.redis.zrangebyscore.call_args
    assert args[0] == "test_agent_memory:im:agent1:timeline"
    assert kwargs["min"] == 0
    assert kwargs["max"] > time.time()
    assert kwargs["start"] == 0
    assert kwargs["num"] == 10

    # Verify pipeline was created and executed
    im_store.redis.pipeline.assert_called_once()
    assert mock_pipe.hgetall.call_count == 2  # Now calling hgetall instead of get
    mock_pipe.execute.assert_called_once()

    # Verify _update_access_metadata was called for each memory
    assert im_store._update_access_metadata.call_count == 2


def test_get_by_timerange_redis_error(im_store):
    """Test time range query when Redis raises an error."""
    # Make Redis zrangebyscore raise an exception
    im_store.redis.zrangebyscore.side_effect = Exception("Redis error")

    memories = im_store.get_by_timerange(
        agent_id="agent1", start_time=0, end_time=time.time(), limit=10
    )

    # Should catch exception and return empty list
    assert memories == []


def test_get_by_importance(im_store):
    """Test retrieving memories by importance score range."""
    # Mock pipeline
    mock_pipe = mock.MagicMock()
    # Configure pipeline to return two memory hash entries
    mock_pipe.execute.return_value = [
        {
            "memory_id": "memory1",
            "content": json.dumps("Test memory 1"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.7}),
            "compression_level": "1",
            "importance_score": "0.7",
        },
        {
            "memory_id": "memory2",
            "content": json.dumps("Test memory 2"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.5}),
            "compression_level": "1",
            "importance_score": "0.5",
        },
    ]
    im_store.redis.pipeline.return_value = mock_pipe

    # Mock update_access_metadata
    im_store._update_access_metadata = mock.MagicMock()

    memories = im_store.get_by_importance(
        agent_id="agent1", min_importance=0.4, max_importance=0.8, limit=10
    )

    # Should return list of memories
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "memory1"
    assert memories[1]["memory_id"] == "memory2"

    # Verify zrangebyscore was called with correct parameters
    im_store.redis.zrangebyscore.assert_called_once()
    args, kwargs = im_store.redis.zrangebyscore.call_args
    assert args[0] == "test_agent_memory:im:agent1:importance"
    assert kwargs["min"] == 0.4
    assert kwargs["max"] == 0.8
    assert kwargs["start"] == 0
    assert kwargs["num"] == 10

    # Verify pipeline was created and executed
    im_store.redis.pipeline.assert_called_once()
    assert mock_pipe.hgetall.call_count == 2  # Now calling hgetall instead of get
    mock_pipe.execute.assert_called_once()

    # Verify _update_access_metadata was called for each memory
    assert im_store._update_access_metadata.call_count == 2


def test_get_by_importance_redis_error(im_store):
    """Test importance query when Redis raises an error."""
    # Make Redis zrangebyscore raise an exception
    im_store.redis.zrangebyscore.side_effect = Exception("Redis error")

    memories = im_store.get_by_importance(
        agent_id="agent1", min_importance=0.0, max_importance=1.0, limit=10
    )

    # Should catch exception and return empty list
    assert memories == []


def test_store_with_different_priorities(im_store):
    """Test storing memories with different priority levels."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1, "importance_score": 0.8},
    }

    im_store.redis.store_with_retry = mock.MagicMock(return_value=True)

    # Test with low priority
    result = im_store.store("agent1", memory_entry, priority=Priority.LOW)
    assert result is True
    call_args = im_store.redis.store_with_retry.call_args
    assert call_args.kwargs["priority"] == Priority.LOW

    # Test with high priority
    result = im_store.store("agent1", memory_entry, priority=Priority.HIGH)
    assert result is True
    call_args = im_store.redis.store_with_retry.call_args
    assert call_args.kwargs["priority"] == Priority.HIGH

    # Test with critical priority
    result = im_store.store("agent1", memory_entry, priority=Priority.CRITICAL)
    assert result is True
    call_args = im_store.redis.store_with_retry.call_args
    assert call_args.kwargs["priority"] == Priority.CRITICAL


def test_delete(im_store):
    """Test deleting a memory."""
    # Mock eval method
    im_store.redis.eval = mock.MagicMock(return_value=1)
    # Ensure Lua scripting is enabled
    im_store._lua_scripting_available = True

    result = im_store.delete("agent1", "memory1")

    # Should return True
    assert result is True

    # Verify eval was called with correct arguments
    im_store.redis.eval.assert_called_once()

    # Get the args from the call
    eval_args = im_store.redis.eval.call_args[0]

    # Verify the Lua script is provided (first argument)
    assert "local memory_key = KEYS[1]" in eval_args[0]

    # Verify correct number of keys
    assert eval_args[1] == 4  # Number of keys should be 4

    # Verify the keys are correct
    assert eval_args[2] == f"{im_store._key_prefix}:agent1:memory:memory1"
    assert eval_args[3] == f"{im_store._key_prefix}:agent1:memories"
    assert eval_args[4] == f"{im_store._key_prefix}:agent1:timeline"
    assert eval_args[5] == f"{im_store._key_prefix}:agent1:importance"

    # Verify memory_id parameter
    assert eval_args[6] == "memory1"


def test_delete_redis_error(im_store):
    """Test deleting a memory when Redis raises an error."""
    # Mock eval and make it raise an exception
    im_store.redis.eval = mock.MagicMock(side_effect=Exception("Redis error"))

    result = im_store.delete("agent1", "memory1")

    # Should catch exception and return False
    assert result is False


def test_count(im_store):
    """Test counting memories for an agent."""
    count = im_store.count("agent1")

    # Should return count from zcard
    assert count == 2

    # Verify zcard was called with correct key
    im_store.redis.zcard.assert_called_once_with("test_agent_memory:im:agent1:memories")


def test_count_redis_error(im_store):
    """Test counting when Redis raises an error."""
    # Make Redis zcard raise an exception
    im_store.redis.zcard.side_effect = Exception("Redis error")

    count = im_store.count("agent1")

    # Should catch exception and return 0
    assert count == 0


def test_clear(im_store):
    """Test clearing all memories for an agent."""
    # Mock eval method
    im_store.redis.eval = mock.MagicMock(return_value=1)
    # Ensure Lua scripting is enabled
    im_store._lua_scripting_available = True

    result = im_store.clear("agent1")

    # Should return True
    assert result is True

    # Verify eval was called with correct arguments
    im_store.redis.eval.assert_called_once()

    # Get the args from the call
    eval_args = im_store.redis.eval.call_args[0]

    # Verify the Lua script is provided (first argument)
    assert "local memories_key = KEYS[1]" in eval_args[0]

    # Verify correct number of keys
    assert eval_args[1] == 3  # Number of keys should be 3

    # Verify the keys are correct
    agent_pattern = f"{im_store._key_prefix}:agent1"
    assert eval_args[2] == f"{agent_pattern}:memories"
    assert eval_args[3] == f"{agent_pattern}:timeline"
    assert eval_args[4] == f"{agent_pattern}:importance"

    # Verify agent pattern parameter
    assert eval_args[5] == agent_pattern


def test_clear_redis_error(im_store):
    """Test clearing memories when Redis raises an error."""
    # Make Redis eval raise an exception to simulate error
    im_store.redis.eval = mock.MagicMock(side_effect=Exception("Redis error"))

    result = im_store.clear("agent1")

    # Should catch exception and return False
    assert result is False


def test_update_access_metadata(im_store):
    """Test updating access metadata for a memory."""
    memory_entry = {
        "memory_id": "memory1",
        "content": "Test memory 1",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.7,
            "retrieval_count": 1,
        },
    }

    # Mock eval method
    im_store.redis.eval = mock.MagicMock(return_value=1)
    # Ensure Lua scripting is enabled
    im_store._lua_scripting_available = True

    # Call the internal method directly
    im_store._update_access_metadata("agent1", "memory1", memory_entry)

    # Verify eval was called with correct arguments
    im_store.redis.eval.assert_called_once()

    # Get the args from the call
    eval_args = im_store.redis.eval.call_args[0]

    # Verify the Lua script is provided (first argument)
    assert "local key = KEYS[1]" in eval_args[0]

    # Verify correct number of keys
    assert eval_args[1] == 2  # Number of keys should be 2

    # Verify the keys are correct
    assert eval_args[2] == f"{im_store._key_prefix}:agent1:memory:memory1"
    assert eval_args[3] == f"{im_store._key_prefix}:agent1:importance"

    # Verify the metadata is updated correctly
    metadata = json.loads(eval_args[4])  # First ARGV parameter is the metadata JSON
    assert metadata["retrieval_count"] == 2  # Should be incremented from 1 to 2
    assert "last_access_time" in metadata


def test_update_access_metadata_redis_error(im_store):
    """Test access metadata update when Redis raises an error."""
    memory_entry = {
        "memory_id": "memory1",
        "content": "Test memory 1",
        "timestamp": time.time(),
        "metadata": {"compression_level": 1, "importance_score": 0.7},
    }

    # Mock eval and make it raise an exception
    im_store.redis.eval = mock.MagicMock(side_effect=Exception("Redis error"))

    # Should not raise exception, just log warning
    im_store._update_access_metadata("agent1", "memory1", memory_entry)
    # No assertions needed as we're just testing it doesn't crash


def test_get_all(im_store):
    """Test retrieving all memories for an agent."""
    # Mock pipeline
    mock_pipe = mock.MagicMock()
    # Configure pipeline to return memory hash entries
    mock_pipe.execute.return_value = [
        {
            "memory_id": "memory1",
            "content": json.dumps("Test memory 1"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.7}),
            "compression_level": "1",
            "importance_score": "0.7",
        },
        {
            "memory_id": "memory2",
            "content": json.dumps("Test memory 2"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.5}),
            "compression_level": "1",
            "importance_score": "0.5",
        },
    ]
    im_store.redis.pipeline.return_value = mock_pipe

    # Mock update_access_metadata
    im_store._update_access_metadata = mock.MagicMock()

    memories = im_store.get_all(agent_id="agent1", limit=10)

    # Should return list of memories
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "memory1"
    assert memories[1]["memory_id"] == "memory2"

    # Verify zrange was called with correct parameters
    im_store.redis.zrange.assert_called_once()
    args, kwargs = im_store.redis.zrange.call_args
    assert args[0] == "test_agent_memory:im:agent1:memories"
    assert args[1] == 0
    assert args[2] == 9  # limit - 1
    assert kwargs["desc"] is True

    # Verify pipeline was created and executed
    im_store.redis.pipeline.assert_called_once()
    assert mock_pipe.hgetall.call_count == 2  # Now calling hgetall instead of get
    mock_pipe.execute.assert_called_once()

    # Verify _update_access_metadata was called for each memory
    assert im_store._update_access_metadata.call_count == 2


def test_get_size(im_store):
    """Test getting the approximate size of memories for an agent."""
    # Mock scan_iter to return a list of keys
    im_store.redis.scan_iter = mock.MagicMock(
        return_value=[
            "test_agent_memory:im:agent1:memory:memory1",
            "test_agent_memory:im:agent1:memory:memory2",
            "test_agent_memory:im:agent1:memory:memory3",
        ]
    )

    # Mock pipeline
    mock_pipe = mock.MagicMock()
    # Configure pipeline to return hash data of different sizes
    mock_pipe.execute.return_value = [
        {  # Memory 1 hash fields
            "memory_id": "memory1",
            "content": json.dumps("Test memory 1"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.7}),
            "compression_level": "1",
            "importance_score": "0.7",
        },
        {  # Memory 2 hash fields
            "memory_id": "memory2",
            "content": json.dumps("Test memory 2 with more content"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.5}),
            "compression_level": "1",
            "importance_score": "0.5",
        },
        {  # Memory 3 hash fields
            "memory_id": "memory3",
            "content": json.dumps("Test memory 3 with even more content"),
            "metadata": json.dumps({"compression_level": 1, "importance_score": 0.3}),
            "compression_level": "1",
            "importance_score": "0.3",
        },
    ]
    im_store.redis.pipeline.return_value = mock_pipe

    size = im_store.get_size(agent_id="agent1")

    # Should return the sum of the sizes of all memory hash fields
    assert size > 0

    # Verify scan_iter was called with correct pattern
    im_store.redis.scan_iter.assert_called_once_with(
        match="test_agent_memory:im:agent1:memory:*"
    )

    # Verify pipeline was created and executed
    im_store.redis.pipeline.assert_called_once()
    assert mock_pipe.hgetall.call_count == 3  # Now calling hgetall instead of get
    mock_pipe.execute.assert_called_once()


def test_check_vector_search_available(im_store):
    """Test checking if Redis vector search is available."""
    # Mock Redis execute_command to return modules with search
    im_store.redis.execute_command.return_value = [[b"name", b"search", b"ver", 20400]]

    result = im_store._check_vector_search_available()
    assert result is True

    # Test when search module is not available
    im_store.redis.execute_command.return_value = [
        [b"name", b"other_module", b"ver", 10000]
    ]

    result = im_store._check_vector_search_available()
    assert result is False

    # Test when command raises exception
    im_store.redis.execute_command.side_effect = Exception("Command failed")

    result = im_store._check_vector_search_available()
    assert result is False


def test_create_vector_index(im_store):
    """Test creating vector index."""
    # First mock that index doesn't exist
    im_store.redis.execute_command.side_effect = [
        Exception("Index not found"),  # FT.INFO fails
        "OK",  # FT.CREATE succeeds
    ]

    im_store._create_vector_index()

    # Verify first call was to check index existence
    call_args = im_store.redis.execute_command.call_args_list[0]
    assert call_args[0][0] == f"FT.INFO {im_store._key_prefix}_vector_idx"

    # Verify second call was to create index
    call_args = im_store.redis.execute_command.call_args_list[1]
    assert call_args[0][0] == "FT.CREATE"
    assert call_args[0][1] == f"{im_store._key_prefix}_vector_idx"

    # Now test when index already exists
    im_store.redis.execute_command.reset_mock()
    im_store.redis.execute_command.side_effect = None
    im_store.redis.execute_command.return_value = "Index info"

    im_store._create_vector_index()

    # Should only check if index exists, not create
    assert im_store.redis.execute_command.call_count == 1
    call_args = im_store.redis.execute_command.call_args
    assert call_args[0][0] == f"FT.INFO {im_store._key_prefix}_vector_idx"


def test_search_similar_with_vector_search(im_store):
    """Test searching similar vectors with Redis vector search."""
    # Set up mock response for vector search
    mock_search_results = [
        2,  # Number of results
        b"test_agent_memory:im:agent1:memory:memory1",
        [
            [b"$", b'{"memory_id":"memory1","content":"Test memory 1"}'],
            [b"__vector_score", b"0.8"],
        ],
        b"test_agent_memory:im:agent1:memory:memory2",
        [
            [b"$", b'{"memory_id":"memory2","content":"Test memory 2"}'],
            [b"__vector_score", b"0.6"],
        ],
    ]

    # Enable vector search for this test
    im_store._vector_search_available = True
    im_store.redis.execute_command.return_value = mock_search_results

    # Set up _update_access_metadata mock to prevent side effects
    im_store._update_access_metadata = mock.MagicMock()

    query_embedding = [0.1, 0.2, 0.3]  # Sample embedding
    results = im_store.search_similar("agent1", query_embedding, k=2)

    # Verify Redis execute_command was called with correct arguments
    im_store.redis.execute_command.assert_called_once()
    call_args = im_store.redis.execute_command.call_args
    assert call_args[0][0] == "FT.SEARCH"
    assert call_args[0][1] == f"{im_store._key_prefix}_vector_idx"

    # Verify results
    assert len(results) == 2
    assert results[0]["memory_id"] == "memory1"
    assert results[0]["similarity_score"] == 0.8
    assert results[1]["memory_id"] == "memory2"
    assert results[1]["similarity_score"] == 0.6


def test_search_similar_fallback(im_store):
    """Test fallback to Python implementation when Redis vector search fails."""
    # Enable vector search but make it fail
    im_store._vector_search_available = True
    im_store.redis.execute_command.side_effect = Exception("Redis search failed")

    # Set up mock memories for fallback implementation
    memories = [
        {
            "memory_id": "memory1",
            "content": "Test memory 1",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"compression_level": 1},
        },
        {
            "memory_id": "memory2",
            "content": "Test memory 2",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {"compression_level": 1},
        },
    ]

    # Mock get_all to return our test memories
    im_store.get_all = mock.MagicMock(return_value=memories)

    # Set up _cosine_similarity mock to return predetermined values
    im_store._cosine_similarity = mock.MagicMock(side_effect=[0.8, 0.5])

    query_embedding = [0.1, 0.2, 0.3]  # Sample embedding
    results = im_store.search_similar("agent1", query_embedding, k=2)

    # Verify Redis execute_command was called
    im_store.redis.execute_command.assert_called_once()

    # Verify get_all was called for fallback
    im_store.get_all.assert_called_once_with("agent1")

    # Verify cosine_similarity was called for each memory
    assert im_store._cosine_similarity.call_count == 2

    # Verify results
    assert len(results) == 2
    assert results[0]["memory_id"] == "memory1"
    assert results[0]["similarity_score"] == 0.8
    assert results[1]["memory_id"] == "memory2"
    assert results[1]["similarity_score"] == 0.5


def test_search_by_attributes_redis(im_store):
    """Test searching by attributes with Redis search."""
    # Set up mock response for attribute search
    mock_search_results = [
        1,  # Number of results
        b"test_agent_memory:im:agent1:memory:memory1",
        [
            [
                b"$",
                b'{"memory_id":"memory1","content":{"location":"New York","status":true}}',
            ]
        ],
    ]

    # Enable vector search for this test
    im_store._vector_search_available = True
    im_store.redis.execute_command.return_value = mock_search_results

    # Set up _update_access_metadata mock
    im_store._update_access_metadata = mock.MagicMock()

    # Mock the get method to return a properly formatted memory entry
    im_store.redis.get = mock.MagicMock(
        return_value=json.dumps(
            {
                "memory_id": "memory1",
                "content": {"location": "New York", "status": True},
                "metadata": {"compression_level": 1},
            }
        )
    )

    # Define attributes to search for
    attributes = {"location": "New York", "status": True}
    results = im_store._search_by_attributes_redis("agent1", attributes)

    # Verify Redis execute_command was called with correct arguments
    im_store.redis.execute_command.assert_called_once()
    call_args = im_store.redis.execute_command.call_args
    assert call_args[0][0] == "FT.SEARCH"

    # Verify results
    assert len(results) == 1
    assert results[0]["memory_id"] == "memory1"
    assert results[0]["content"]["location"] == "New York"
    assert results[0]["content"]["status"] is True


def test_search_by_step_range_redis(im_store):
    """Test searching by step range with Redis search."""
    # Set up mock response for step range search
    mock_search_results = [
        2,  # Number of results
        b"test_agent_memory:im:agent1:memory:memory1",
        [[b"$", b'{"memory_id":"memory1","step_number":5,"content":"Step 5"}']],
        b"test_agent_memory:im:agent1:memory:memory2",
        [[b"$", b'{"memory_id":"memory2","step_number":6,"content":"Step 6"}']],
    ]

    # Enable vector search for this test
    im_store._vector_search_available = True
    im_store.redis.execute_command.return_value = mock_search_results

    # Set up _update_access_metadata mock
    im_store._update_access_metadata = mock.MagicMock()

    # Mock the get method to return properly formatted memory entries
    def get_side_effect(key):
        if "memory1" in key:
            return json.dumps(
                {
                    "memory_id": "memory1",
                    "step_number": 5,
                    "content": "Step 5",
                    "metadata": {"compression_level": 1},
                }
            )
        elif "memory2" in key:
            return json.dumps(
                {
                    "memory_id": "memory2",
                    "step_number": 6,
                    "content": "Step 6",
                    "metadata": {"compression_level": 1},
                }
            )
        return None

    im_store.redis.get = mock.MagicMock(side_effect=get_side_effect)

    # Search for steps 5-10
    results = im_store._search_by_step_range_redis("agent1", 5, 10)

    # Verify Redis execute_command was called with correct arguments
    im_store.redis.execute_command.assert_called_once()
    call_args = im_store.redis.execute_command.call_args
    assert call_args[0][0] == "FT.SEARCH"
    assert "@step_number:[5 10]" in call_args[0][2]

    # Verify results
    assert len(results) == 2
    assert results[0]["memory_id"] == "memory1"
    assert results[0]["step_number"] == 5
    assert results[1]["memory_id"] == "memory2"
    assert results[1]["step_number"] == 6


def test_cosine_similarity():
    """Test the cosine similarity calculation."""
    # Create store instance directly for testing internal method
    config = RedisIMConfig(
        host="localhost",
        port=6379,
        db=1,
        namespace="test_agent_memory:im",
    )
    store = RedisIMStore(config)

    # Test with parallel vectors (should be 1.0)
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    similarity = store._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6

    # Test with orthogonal vectors (should be 0.0)
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = store._cosine_similarity(vec1, vec2)
    assert abs(similarity) < 1e-6

    # Test with opposite vectors (should be -1.0)
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [-1.0, -2.0, -3.0]
    similarity = store._cosine_similarity(vec1, vec2)
    assert abs(similarity + 1.0) < 1e-6

    # Test with empty vectors (should be 0.0)
    vec1 = []
    vec2 = [1.0, 2.0, 3.0]
    similarity = store._cosine_similarity(vec1, vec2)
    assert similarity == 0.0

    # Test with zero-magnitude vector (should be 0.0)
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 2.0, 3.0]
    similarity = store._cosine_similarity(vec1, vec2)
    assert similarity == 0.0
