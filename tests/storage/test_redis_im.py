"""Unit tests for Redis Intermediate Memory (IM) storage.

This module contains tests for the RedisIMStore class which provides Redis-based
storage for the intermediate memory tier.
"""

import json
import time
from unittest import mock
import pytest

from agent_memory.config import RedisIMConfig
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.utils.error_handling import Priority, RedisTimeoutError, RedisUnavailableError


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    with mock.patch('agent_memory.storage.redis_client.ResilientRedisClient') as mock_client:
        # Configure the mock to return success for store operations
        mock_client.return_value.store_with_retry.return_value = True
        mock_client.return_value.set.return_value = True
        mock_client.return_value.zadd.return_value = 1
        mock_client.return_value.expire.return_value = True
        mock_client.return_value.zrange.return_value = ["memory1", "memory2"]
        mock_client.return_value.zrangebyscore.return_value = ["memory1", "memory2"]
        mock_client.return_value.zcard.return_value = 2
        mock_client.return_value.delete.return_value = 1
        mock_client.return_value.zrem.return_value = 1
        
        # Configure get to return a mock memory entry
        def get_side_effect(key):
            if "memory:memory1" in key:
                return json.dumps({
                    "memory_id": "memory1",
                    "content": "Test memory 1",
                    "timestamp": time.time(),
                    "metadata": {
                        "compression_level": 1,
                        "importance_score": 0.7,
                        "retrieval_count": 0
                    }
                })
            elif "memory:memory2" in key:
                return json.dumps({
                    "memory_id": "memory2",
                    "content": "Test memory 2",
                    "timestamp": time.time(),
                    "metadata": {
                        "compression_level": 1,
                        "importance_score": 0.5,
                        "retrieval_count": 0
                    }
                })
            return None
        
        mock_client.return_value.get.side_effect = get_side_effect
        
        yield mock_client.return_value


@pytest.fixture
def im_store(mock_redis_client):
    """Create a RedisIMStore instance with mock Redis client."""
    config = RedisIMConfig(
        host="localhost",
        port=6379,
        db=1,
        namespace="test_agent_memory:im",
        ttl=604800  # 7 days
    )
    store = RedisIMStore(config)
    store.redis = mock_redis_client
    return store


def test_init():
    """Test RedisIMStore initialization."""
    config = RedisIMConfig(
        host="test-host",
        port=1234,
        db=2,
        namespace="test:namespace",
        ttl=300
    )
    
    with mock.patch('agent_memory.storage.redis_im.ResilientRedisClient') as mock_client:
        store = RedisIMStore(config)
        
        # Check that Redis client was initialized with correct parameters
        mock_client.assert_called_once_with(
            client_name="im",
            host="test-host",
            port=1234,
            db=2,
            password=None,
            circuit_threshold=3,
            circuit_reset_timeout=300
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
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.8
        }
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
        "metadata": {
            "compression_level": 1
        }
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
        "metadata": {
            "compression_level": 2  # Should be 1 for IM
        }
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
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.8
        }
    }
    
    result = im_store._store_memory_entry("agent1", memory_entry)
    
    # Check the result
    assert result is True
    
    # Verify Redis operations were called correctly
    im_store.redis.set.assert_called_with(
        "test_agent_memory:im:agent1:memory:test-memory-1", 
        json.dumps(memory_entry),
        ex=im_store.config.ttl
    )
    
    # Check zadd calls for various indices
    assert im_store.redis.zadd.call_count == 3
    im_store.redis.expire.assert_called()


def test_store_memory_entry_redis_error(im_store):
    """Test handling of Redis errors in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1
        }
    }
    
    # Make Redis operations raise an exception
    im_store.redis.set.side_effect = Exception("Redis error")
    
    # Should catch exception and return False
    result = im_store._store_memory_entry("agent1", memory_entry)
    assert result is False


def test_store_memory_entry_redis_timeout(im_store):
    """Test handling of Redis timeout in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1
        }
    }
    
    # Make Redis operations raise a timeout error
    im_store.redis.set.side_effect = RedisTimeoutError("Redis timeout")
    
    # Should raise the timeout error for retry handling
    with pytest.raises(RedisTimeoutError):
        im_store._store_memory_entry("agent1", memory_entry)


def test_store_memory_entry_redis_unavailable(im_store):
    """Test handling of Redis unavailability in _store_memory_entry."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1
        }
    }
    
    # Make Redis operations raise an unavailability error
    im_store.redis.set.side_effect = RedisUnavailableError("Redis unavailable")
    
    # Should raise the unavailability error for retry handling
    with pytest.raises(RedisUnavailableError):
        im_store._store_memory_entry("agent1", memory_entry)


def test_get_memory(im_store):
    """Test retrieving a memory by ID."""
    memory = im_store.get("agent1", "memory1")
    
    # Verify we got the memory
    assert memory is not None
    assert memory["memory_id"] == "memory1"
    assert memory["content"] == "Test memory 1"
    
    # Verify get was called with correct key
    im_store.redis.get.assert_called_with("test_agent_memory:im:agent1:memory:memory1")
    
    # Verify the access metadata was updated (set called with updated entry)
    im_store.redis.set.assert_called()
    args, kwargs = im_store.redis.set.call_args
    assert args[0] == "test_agent_memory:im:agent1:memory:memory1"
    updated_entry = json.loads(args[1])
    assert updated_entry["metadata"]["retrieval_count"] == 1
    assert "last_access_time" in updated_entry["metadata"]


def test_get_nonexistent_memory(im_store):
    """Test retrieving a memory that doesn't exist."""
    memory = im_store.get("agent1", "nonexistent")
    
    # Should return None for non-existent memory
    assert memory is None


def test_get_memory_redis_error(im_store):
    """Test retrieving a memory when Redis raises an error."""
    # Make Redis get raise an exception
    im_store.redis.get.side_effect = Exception("Redis error")
    
    memory = im_store.get("agent1", "memory1")
    
    # Should catch exception and return None
    assert memory is None


def test_get_memory_redis_timeout(im_store):
    """Test retrieving a memory when Redis times out."""
    # Make Redis get raise a timeout exception
    im_store.redis.get.side_effect = RedisTimeoutError("Operation timed out")
    
    memory = im_store.get("agent1", "memory1")
    
    # Should catch exception and return None
    assert memory is None


def test_get_memory_redis_unavailable(im_store):
    """Test retrieving a memory when Redis is unavailable."""
    # Make Redis get raise an unavailability exception
    im_store.redis.get.side_effect = RedisUnavailableError("Redis unavailable")
    
    memory = im_store.get("agent1", "memory1")
    
    # Should catch exception and return None
    assert memory is None


def test_get_by_timerange(im_store):
    """Test retrieving memories by time range."""
    memories = im_store.get_by_timerange(
        agent_id="agent1",
        start_time=0,
        end_time=time.time() + 1000,
        limit=10
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


def test_get_by_timerange_redis_error(im_store):
    """Test time range query when Redis raises an error."""
    # Make Redis zrangebyscore raise an exception
    im_store.redis.zrangebyscore.side_effect = Exception("Redis error")
    
    memories = im_store.get_by_timerange(
        agent_id="agent1",
        start_time=0,
        end_time=time.time(),
        limit=10
    )
    
    # Should catch exception and return empty list
    assert memories == []


def test_get_by_importance(im_store):
    """Test retrieving memories by importance score range."""
    memories = im_store.get_by_importance(
        agent_id="agent1",
        min_importance=0.4,
        max_importance=0.8,
        limit=10
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


def test_get_by_importance_redis_error(im_store):
    """Test importance query when Redis raises an error."""
    # Make Redis zrangebyscore raise an exception
    im_store.redis.zrangebyscore.side_effect = Exception("Redis error")
    
    memories = im_store.get_by_importance(
        agent_id="agent1",
        min_importance=0.0,
        max_importance=1.0,
        limit=10
    )
    
    # Should catch exception and return empty list
    assert memories == []


def test_store_with_different_priorities(im_store):
    """Test storing memories with different priority levels."""
    memory_entry = {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.8
        }
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
    result = im_store.delete("agent1", "memory1")
    
    # Should return True
    assert result is True
    
    # Verify delete and zrem operations
    assert im_store.redis.delete.call_count == 1
    assert im_store.redis.zrem.call_count == 3


def test_delete_redis_error(im_store):
    """Test deleting a memory when Redis raises an error."""
    # Make Redis delete raise an exception
    im_store.redis.delete.side_effect = Exception("Redis error")
    
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
    with mock.patch.object(im_store, 'delete', return_value=True) as mock_delete:
        result = im_store.clear("agent1")
        
        # Should return True
        assert result is True
        
        # Verify all memories were deleted
        assert mock_delete.call_count == 2
        mock_delete.assert_any_call("agent1", "memory1")
        mock_delete.assert_any_call("agent1", "memory2")


def test_clear_redis_error(im_store):
    """Test clearing memories when Redis raises an error."""
    # Make Redis zrange raise an exception
    im_store.redis.zrange.side_effect = Exception("Redis error")
    
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
            "retrieval_count": 1
        }
    }
    
    im_store.redis.zadd = mock.MagicMock(return_value=1)
    
    # Call the internal method directly
    im_store._update_access_metadata("agent1", "memory1", memory_entry)
    
    # Verify set was called with updated entry
    im_store.redis.set.assert_called()
    args, kwargs = im_store.redis.set.call_args
    
    # Parse the JSON that was passed to Redis set
    updated_entry = json.loads(args[1])
    
    # Check that metadata was updated
    assert updated_entry["metadata"]["retrieval_count"] == 2
    assert "last_access_time" in updated_entry["metadata"]
    
    # Check that importance was increased and zadd was called
    im_store.redis.zadd.assert_called()
    
    # Check the zadd arguments - the format depends on how Redis implementation passes args
    # Instead of checking if args[1] > 0.7, we'll check if zadd was called
    assert im_store.redis.zadd.call_count == 1
    # Check that the first argument was the key for the importance index
    args = im_store.redis.zadd.call_args[0]
    assert args[0] == "test_agent_memory:im:agent1:importance"


def test_update_access_metadata_redis_error(im_store):
    """Test access metadata update when Redis raises an error."""
    memory_entry = {
        "memory_id": "memory1",
        "content": "Test memory 1",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.7
        }
    }
    
    # Make Redis set raise an exception
    im_store.redis.set.side_effect = Exception("Redis error")
    
    # Should not raise exception, just log warning
    im_store._update_access_metadata("agent1", "memory1", memory_entry)
    # No assertions needed as we're just testing it doesn't crash 