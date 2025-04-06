"""Unit tests for Redis Short-Term Memory (STM) storage.

This module contains tests for the RedisSTMStore class which provides Redis-based
storage for the short-term memory tier.
"""

import json
import time
from unittest import mock
import pytest

from agent_memory.config import RedisSTMConfig
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.utils.error_handling import Priority, RedisTimeoutError, RedisUnavailableError


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_client = mock.MagicMock()
    
    # Configure the mock to return success for store operations
    mock_client.store_with_retry.return_value = True
    mock_client.set.return_value = True
    mock_client.zadd.return_value = 1
    mock_client.expire.return_value = True
    mock_client.zrange.return_value = ["memory1", "memory2"]
    
    # Define a side effect function for zrangebyscore
    def zrangebyscore_side_effect(*args, **kwargs):
        if "importance" in args[0] and kwargs.get('withscores', False):
            # For importance, return tuples with IDs and scores
            return [("memory1", 0.7), ("memory2", 0.5)]
        else:
            # For other calls like timeline, return just IDs
            return ["memory1", "memory2"]
    
    mock_client.zrangebyscore.side_effect = zrangebyscore_side_effect
    mock_client.zcard.return_value = 2
    mock_client.delete.return_value = 1
    mock_client.zrem.return_value = 1
    mock_client.ping.return_value = True
    
    # Configure get to return a mock memory entry
    def get_side_effect(key):
        if "memory:memory1" in key:
            return json.dumps({
                "memory_id": "memory1",
                "content": "Test memory 1",
                "timestamp": time.time(),
                "metadata": {
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
                    "importance_score": 0.5,
                    "retrieval_count": 0
                }
            })
        return None
    
    mock_client.get.side_effect = get_side_effect
    return mock_client


@pytest.fixture
def stm_store(mock_redis_client):
    """Create a RedisSTMStore with a mock Redis client."""
    with mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client') as mock_factory:
        mock_factory.return_value = mock_redis_client
        
        config = RedisSTMConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            namespace="test-stm",
            ttl=3600
        )
        store = RedisSTMStore(config)
        yield store


def test_init():
    """Test initialization of RedisSTMStore."""
    config = RedisSTMConfig(
        host="redis-host",
        port=6380,
        db=1,
        password="password",
        namespace="agent-stm",
        ttl=7200
    )
    
    with mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client') as mock_factory:
        store = RedisSTMStore(config)
        
        # Verify create_client was called with correct parameters
        mock_factory.assert_called_once_with(
            client_name="stm",
            use_mock=config.use_mock,
            host="redis-host",
            port=6380,
            db=1,
            password="password",
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )
        
        assert store.config is config
        assert store._key_prefix == "agent-stm"


def test_store_success(stm_store):
    """Test storing a memory entry successfully."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"importance_score": 0.8}
    }
    
    result = stm_store.store(agent_id, memory_entry)
    
    # Verify store_with_retry was called correctly
    stm_store.redis.store_with_retry.assert_called_once_with(
        agent_id=agent_id,
        state_data=memory_entry,
        store_func=stm_store._store_memory_entry,
        priority=Priority.NORMAL
    )
    assert result is True


def test_store_missing_memory_id(stm_store):
    """Test storing a memory entry without a memory_id."""
    agent_id = "test-agent"
    memory_entry = {
        "content": "This is a test memory",
        "timestamp": time.time(),
    }
    
    result = stm_store.store(agent_id, memory_entry)
    
    # Verify store_with_retry was not called
    stm_store.redis.store_with_retry.assert_not_called()
    assert result is False


def test_store_memory_entry(stm_store):
    """Test the internal _store_memory_entry method."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
        "timestamp": 1234567890.0,
        "metadata": {"importance_score": 0.8},
        "embeddings": {"full_vector": [0.1, 0.2, 0.3]}
    }
    
    result = stm_store._store_memory_entry(agent_id, memory_entry)
    
    # Verify Redis operations were performed correctly
    # Store memory entry
    stm_store.redis.set.assert_any_call(
        "test-stm:test-agent:memory:test-memory",
        json.dumps(memory_entry),
        ex=3600
    )
    
    # Add to memories set
    stm_store.redis.zadd.assert_any_call(
        "test-stm:test-agent:memories",
        {"test-memory": 1234567890.0}
    )
    
    # Set TTL on memories set
    stm_store.redis.expire.assert_any_call(
        "test-stm:test-agent:memories",
        3600
    )
    
    # Add to timeline index
    stm_store.redis.zadd.assert_any_call(
        "test-stm:test-agent:timeline",
        {"test-memory": 1234567890.0}
    )
    
    # Add to importance index
    stm_store.redis.zadd.assert_any_call(
        "test-stm:test-agent:importance",
        {"test-memory": 0.8}
    )
    
    # Store vector
    stm_store.redis.set.assert_any_call(
        "test-stm:test-agent:vector:test-memory",
        json.dumps([0.1, 0.2, 0.3]),
        ex=3600
    )
    
    assert result is True


def test_store_memory_entry_redis_error(stm_store):
    """Test _store_memory_entry with Redis error."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
    }
    
    # Simulate Redis error
    stm_store.redis.set.side_effect = Exception("Redis error")
    
    result = stm_store._store_memory_entry(agent_id, memory_entry)
    assert result is False
    
    # Reset side effect for subsequent tests
    stm_store.redis.set.side_effect = None


def test_store_memory_entry_redis_timeout(stm_store):
    """Test _store_memory_entry with Redis timeout."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
    }
    
    # Simulate Redis timeout
    stm_store.redis.set.side_effect = RedisTimeoutError("Redis timeout")
    
    # Should propagate the error for retry mechanism
    with pytest.raises(RedisTimeoutError):
        stm_store._store_memory_entry(agent_id, memory_entry)
    
    # Reset side effect for subsequent tests
    stm_store.redis.set.side_effect = None


def test_store_memory_entry_redis_unavailable(stm_store):
    """Test _store_memory_entry with Redis unavailable."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
    }
    
    # Simulate Redis unavailable
    stm_store.redis.set.side_effect = RedisUnavailableError("Redis unavailable")
    
    # Should propagate the error for retry mechanism
    with pytest.raises(RedisUnavailableError):
        stm_store._store_memory_entry(agent_id, memory_entry)
    
    # Reset side effect for subsequent tests
    stm_store.redis.set.side_effect = None


def test_get_memory(stm_store):
    """Test retrieving a memory entry."""
    agent_id = "test-agent"
    memory_id = "memory1"
    
    memory = stm_store.get(agent_id, memory_id)
    
    # Verify Redis get was called correctly
    stm_store.redis.get.assert_called_with("test-stm:test-agent:memory:memory1")
    assert memory is not None
    assert memory["memory_id"] == "memory1"


def test_get_nonexistent_memory(stm_store):
    """Test retrieving a nonexistent memory entry."""
    agent_id = "test-agent"
    memory_id = "nonexistent"
    
    # Configure mock to return None for this key
    stm_store.redis.get.side_effect = lambda key: None if "nonexistent" in key else "mock data"
    
    memory = stm_store.get(agent_id, memory_id)
    
    # Verify Redis get was called correctly
    stm_store.redis.get.assert_called_with("test-stm:test-agent:memory:nonexistent")
    assert memory is None
    
    # Reset side effect
    stm_store.redis.get.side_effect = None


def test_get_memory_redis_error(stm_store):
    """Test retrieving a memory entry with Redis error."""
    agent_id = "test-agent"
    memory_id = "memory1"
    
    # Simulate Redis error
    stm_store.redis.get.side_effect = Exception("Redis error")
    
    memory = stm_store.get(agent_id, memory_id)
    
    # Should handle error and return None
    assert memory is None
    
    # Reset side effect
    stm_store.redis.get.side_effect = None


def test_get_memory_redis_timeout(stm_store):
    """Test retrieving a memory entry with Redis timeout."""
    agent_id = "test-agent"
    memory_id = "memory1"
    
    # Simulate Redis timeout
    stm_store.redis.get.side_effect = RedisTimeoutError("Redis timeout")
    
    memory = stm_store.get(agent_id, memory_id)
    
    # Should handle error and return None
    assert memory is None
    
    # Reset side effect
    stm_store.redis.get.side_effect = None


def test_get_memory_redis_unavailable(stm_store):
    """Test retrieving a memory entry with Redis unavailable."""
    agent_id = "test-agent"
    memory_id = "memory1"
    
    # Simulate Redis unavailable
    stm_store.redis.get.side_effect = RedisUnavailableError("Redis unavailable")
    
    memory = stm_store.get(agent_id, memory_id)
    
    # Should handle error and return None
    assert memory is None
    
    # Reset side effect
    stm_store.redis.get.side_effect = None


def test_update_access_metadata(stm_store):
    """Test updating access metadata for a memory entry."""
    agent_id = "test-agent"
    memory_id = "test-memory"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {
            "importance_score": 0.5,
            "retrieval_count": 2
        }
    }
    
    with mock.patch('time.time', return_value=1234567890.0):
        # Reset the mock before calling the method
        stm_store.redis.set.reset_mock()
        stm_store.redis.zadd.reset_mock()
        
        # Call the method
        stm_store._update_access_metadata(agent_id, memory_id, memory_entry)
        
        # Verify metadata was updated correctly in the passed memory_entry
        assert memory_entry["metadata"]["last_access_time"] == 1234567890.0
        assert memory_entry["metadata"]["retrieval_count"] == 3
        
        # We need to verify the set call with the updated memory entry
        # But we can't use assert_called_once_with due to JSON serialization differences
        # Instead, check that set was called once and verify the arguments separately
        assert stm_store.redis.set.call_count == 1
        call_args = stm_store.redis.set.call_args[0]
        
        # Verify the key is correct
        assert call_args[0] == "test-stm:test-agent:memory:test-memory"
        
        # Verify the TTL is correct
        assert stm_store.redis.set.call_args[1]["ex"] == 3600
        
        # Parse the JSON to compare the actual memory entry
        actual_memory = json.loads(call_args[1])
        assert actual_memory["metadata"]["retrieval_count"] == 3
        assert actual_memory["metadata"]["last_access_time"] == 1234567890.0
        
        # Check that zadd was called with the importance key
        # The implementation will call zadd if retrieval_count > 1, which it is in our test
        if stm_store.redis.zadd.call_count > 0:
            calls = stm_store.redis.zadd.call_args_list
            found_importance_call = False
            for call in calls:
                if f"{stm_store._key_prefix}:{agent_id}:importance" == call[0][0]:
                    found_importance_call = True
                    # The importance score logic is implementation specific,
                    # so we only check that the memory ID is in the call
                    assert memory_id in call[0][1]
                    break
            # If any zadd was called, one of them should be for importance
            if calls:
                assert found_importance_call


def test_update_access_metadata_redis_error(stm_store):
    """Test updating access metadata with Redis error."""
    agent_id = "test-agent"
    memory_id = "test-memory"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
        "metadata": {}
    }
    
    # Simulate Redis error
    stm_store.redis.set.side_effect = Exception("Redis error")
    
    # Should handle error gracefully
    stm_store._update_access_metadata(agent_id, memory_id, memory_entry)
    
    # Verify metadata was still updated in memory
    assert "retrieval_count" in memory_entry["metadata"]
    
    # Reset side effect
    stm_store.redis.set.side_effect = None


def test_get_by_timerange(stm_store):
    """Test retrieving memories within a time range."""
    agent_id = "test-agent"
    start_time = 1000000000.0
    end_time = 2000000000.0
    
    memories = stm_store.get_by_timerange(agent_id, start_time, end_time)
    
    # Verify Redis zrangebyscore was called correctly
    stm_store.redis.zrangebyscore.assert_called_once_with(
        "test-stm:test-agent:timeline",
        min=start_time,
        max=end_time,
        start=0,
        num=100
    )
    
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "memory1"
    assert memories[1]["memory_id"] == "memory2"


def test_get_by_timerange_redis_error(stm_store):
    """Test retrieving memories within a time range with Redis error."""
    agent_id = "test-agent"
    start_time = 1000000000.0
    end_time = 2000000000.0
    
    # Simulate Redis error
    stm_store.redis.zrangebyscore.side_effect = Exception("Redis error")
    
    memories = stm_store.get_by_timerange(agent_id, start_time, end_time)
    
    # Should handle error and return empty list
    assert memories == []
    
    # Reset side effect
    stm_store.redis.zrangebyscore.side_effect = None


def test_get_by_importance(stm_store):
    """Test retrieving memories by importance score."""
    agent_id = "test-agent"
    min_importance = 0.5
    max_importance = 1.0
    
    memories = stm_store.get_by_importance(agent_id, min_importance, max_importance)
    
    # Verify Redis zrangebyscore was called correctly
    stm_store.redis.zrangebyscore.assert_called_once_with(
        "test-stm:test-agent:importance",
        min=min_importance,
        max=max_importance,
        withscores=True
    )
    
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "memory1"
    assert memories[1]["memory_id"] == "memory2"


def test_get_by_importance_redis_error(stm_store):
    """Test retrieving memories by importance score with Redis error."""
    agent_id = "test-agent"
    min_importance = 0.5
    max_importance = 1.0
    
    # Simulate Redis error
    stm_store.redis.zrangebyscore.side_effect = Exception("Redis error")
    
    memories = stm_store.get_by_importance(agent_id, min_importance, max_importance)
    
    # Should handle error and return empty list
    assert memories == []
    
    # Reset side effect
    stm_store.redis.zrangebyscore.side_effect = None


def test_delete(stm_store):
    """Test deleting a memory entry."""
    agent_id = "test-agent"
    memory_id = "test-memory"
    
    # Reset delete mock to clear any previous calls
    stm_store.redis.delete.reset_mock()
    
    result = stm_store.delete(agent_id, memory_id)
    
    # Verify Redis operations were performed correctly
    # Check that delete was called at least once with the memory key
    delete_calls = [call[0][0] for call in stm_store.redis.delete.call_args_list]
    assert "test-stm:test-agent:memory:test-memory" in delete_calls
    
    # Remove from indices
    stm_store.redis.zrem.assert_any_call("test-stm:test-agent:memories", memory_id)
    stm_store.redis.zrem.assert_any_call("test-stm:test-agent:timeline", memory_id)
    stm_store.redis.zrem.assert_any_call("test-stm:test-agent:importance", memory_id)
    
    # Check that delete was called with the vector key
    assert "test-stm:test-agent:vector:test-memory" in delete_calls
    
    assert result is True


def test_delete_redis_error(stm_store):
    """Test deleting a memory entry with Redis error."""
    agent_id = "test-agent"
    memory_id = "test-memory"
    
    # Simulate Redis error
    stm_store.redis.delete.side_effect = Exception("Redis error")
    
    result = stm_store.delete(agent_id, memory_id)
    
    # Should handle error and return False
    assert result is False
    
    # Reset side effect
    stm_store.redis.delete.side_effect = None


def test_count(stm_store):
    """Test counting memories for an agent."""
    agent_id = "test-agent"
    
    count = stm_store.count(agent_id)
    
    # Verify Redis zcard was called correctly
    stm_store.redis.zcard.assert_called_once_with("test-stm:test-agent:memories")
    
    assert count == 2


def test_count_redis_error(stm_store):
    """Test counting memories with Redis error."""
    agent_id = "test-agent"
    
    # Simulate Redis error
    stm_store.redis.zcard.side_effect = Exception("Redis error")
    
    count = stm_store.count(agent_id)
    
    # Should handle error and return 0
    assert count == 0
    
    # Reset side effect
    stm_store.redis.zcard.side_effect = None


def test_clear(stm_store):
    """Test clearing all memories for an agent."""
    agent_id = "test-agent"
    
    # Reset mocks to clear any previous calls
    stm_store.redis.delete.reset_mock()
    
    result = stm_store.clear(agent_id)
    
    # Verify Redis operations were performed correctly
    # Get memory IDs
    stm_store.redis.zrange.assert_called_once_with("test-stm:test-agent:memories", 0, -1)
    
    # Check delete calls for memory entries
    delete_calls = [call[0] for call in stm_store.redis.delete.call_args_list]
    
    # In the implementation, it calls delete on each memory, then does a batch delete on indices
    # So we need to check if either approach is used
    indices_deleted = False
    
    # Check if indices were deleted individually
    individual_indices = [
        ("test-stm:test-agent:memories",),
        ("test-stm:test-agent:timeline",),
        ("test-stm:test-agent:importance",)
    ]
    
    # Check if indices were deleted in a batch
    batch_indices = [("test-stm:test-agent:memories", "test-stm:test-agent:timeline", "test-stm:test-agent:importance")]
    
    # Check if either individual or batch deletion was used
    for call_args in delete_calls:
        if call_args in individual_indices or call_args in batch_indices:
            indices_deleted = True
            break
    
    assert indices_deleted, "Index keys were not deleted properly"
    
    assert result is True


def test_clear_redis_error(stm_store):
    """Test clearing memories with Redis error."""
    agent_id = "test-agent"
    
    # Simulate Redis error
    stm_store.redis.zrange.side_effect = Exception("Redis error")
    
    result = stm_store.clear(agent_id)
    
    # Should handle error and return False
    assert result is False
    
    # Reset side effect
    stm_store.redis.zrange.side_effect = None


def test_check_health(stm_store):
    """Test health check."""
    health = stm_store.check_health()
    
    # Verify Redis ping was called
    stm_store.redis.ping.assert_called_once()
    
    assert health["status"] == "healthy"
    # The implementation might return either a details structure or direct fields
    if "details" in health:
        assert health["details"]["redis_connection"] is True
    else:
        # Default implementation returns latency_ms and client fields
        assert "latency_ms" in health
        assert "client" in health


def test_check_health_redis_error(stm_store):
    """Test health check with Redis error."""
    # Simulate Redis error
    stm_store.redis.ping.side_effect = Exception("Redis error")
    
    health = stm_store.check_health()
    
    # Should handle error and report unhealthy
    assert health["status"] == "unhealthy"
    # The implementation might return either a details structure or direct fields
    if "details" in health:
        assert health["details"]["redis_connection"] is False
    else:
        assert "error" in health
    
    # Reset side effect
    stm_store.redis.ping.side_effect = None


def test_store_with_different_priorities(stm_store):
    """Test storing a memory entry with different priorities."""
    agent_id = "test-agent"
    memory_entry = {
        "memory_id": "test-memory",
        "content": "This is a test memory",
        "timestamp": time.time(),
        "metadata": {"importance_score": 0.8}
    }
    
    # Test with high priority
    result_high = stm_store.store(agent_id, memory_entry, Priority.HIGH)
    stm_store.redis.store_with_retry.assert_called_with(
        agent_id=agent_id,
        state_data=memory_entry,
        store_func=stm_store._store_memory_entry,
        priority=Priority.HIGH
    )
    assert result_high is True
    
    # Reset mock
    stm_store.redis.store_with_retry.reset_mock()
    
    # Test with low priority
    result_low = stm_store.store(agent_id, memory_entry, Priority.LOW)
    stm_store.redis.store_with_retry.assert_called_with(
        agent_id=agent_id,
        state_data=memory_entry,
        store_func=stm_store._store_memory_entry,
        priority=Priority.LOW
    )
    assert result_low is True 