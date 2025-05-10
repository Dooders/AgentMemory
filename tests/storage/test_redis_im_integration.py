"""Integration tests for Redis Intermediate Memory (IM) storage using MockRedis.

This module contains integration tests for the RedisIMStore class using MockRedis,
which eliminates the requirement for a real Redis instance to run.
"""

import json
import time
import uuid
import unittest.mock

import pytest
import redis

from memory.config import RedisIMConfig
from memory.storage.redis_im import RedisIMStore
from memory.storage.mockredis import MockRedis

# Create a global MockRedis instance to be used by all tests
mock_redis = MockRedis(decode_responses=True)

# Setup module-level patching to ensure all Redis instances are our mock
@pytest.fixture(scope="module", autouse=True)
def patch_redis():
    """Patch redis.Redis to use our MockRedis instance."""
    with unittest.mock.patch('redis.Redis', return_value=mock_redis):
        yield


def get_redis_config():
    """Get Redis config for integration tests."""
    # Use a high DB number to avoid conflicting with other DBs
    # and a unique namespace to avoid conflicts with other tests
    test_namespace = f"test_im_{uuid.uuid4().hex[:8]}"
    return RedisIMConfig(
        host="localhost",
        port=6379,
        db=15,  # Use DB 15 for tests
        namespace=test_namespace,
        ttl=60,  # Short TTL for tests
        use_mock=True,  # Set to use MockRedis instead of real Redis
        test_mode=True  # Enable test mode to prevent importance score updates
    )


@pytest.fixture
def im_store():
    """Create a RedisIMStore instance for testing."""
    config = get_redis_config()
    
    # Create the RedisIMStore - it will use our patched MockRedis
    store = RedisIMStore(config)
    
    # Clear any existing test data
    pattern = f"{config.namespace}:*"
    redis_client = store.redis.client
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
    
    yield store
    
    # Clean up test data
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)


def create_test_memory(memory_id=None):
    """Create a test memory entry."""
    return {
        "memory_id": memory_id or f"test-{uuid.uuid4().hex[:8]}",
        "content": "This is a test memory for integration testing",
        "timestamp": time.time(),
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.7,
            "retrieval_count": 0,
            "source": "integration_test",
            "creation_time": time.time(),
            "last_access_time": time.time(),
        },
    }


def test_store_and_retrieve(im_store):
    """Test storing and retrieving a memory with MockRedis."""
    agent_id = "test-agent"
    memory = create_test_memory()
    memory_id = memory["memory_id"]

    # Store the memory
    result = im_store.store(agent_id, memory)
    assert result is True

    # Verify the memory was stored in Redis as a hash
    key = f"{im_store._key_prefix}:{agent_id}:memory:{memory_id}"
    
    # Use im_store.redis to access the actual Redis client used by the store
    redis_client = im_store.redis.client
    
    # Check if the key exists
    assert redis_client.exists(key) == 1
    
    # Get the hash fields
    hash_fields = redis_client.hgetall(key)
    assert "memory_id" in hash_fields
    assert hash_fields["memory_id"] == memory_id
    assert "content" in hash_fields

    # Verify the memory was added to the indices
    memories_key = f"{im_store._key_prefix}:{agent_id}:memories"
    timeline_key = f"{im_store._key_prefix}:{agent_id}:timeline"
    importance_key = f"{im_store._key_prefix}:{agent_id}:importance"

    assert redis_client.zscore(memories_key, memory_id) is not None
    assert redis_client.zscore(timeline_key, memory_id) is not None
    assert redis_client.zscore(importance_key, memory_id) is not None

    # Retrieve the memory
    retrieved = im_store.get(agent_id, memory_id)
    assert retrieved is not None
    assert retrieved["memory_id"] == memory_id
    assert retrieved["content"] == memory["content"]

    # Verify access metadata was updated - check the hash fields
    updated_hash = redis_client.hgetall(key)
    assert "retrieval_count" in updated_hash
    assert int(updated_hash["retrieval_count"]) >= 1
    assert "last_access_time" in updated_hash

    # Also verify the full metadata JSON field was updated
    metadata_json = updated_hash.get("metadata")
    if metadata_json:
        metadata = json.loads(metadata_json)
        assert metadata["retrieval_count"] >= 1
        assert "last_access_time" in metadata


def test_timerange_query(im_store):
    """Test querying by time range with MockRedis."""
    agent_id = "test-agent"

    # Create test memories with different timestamps
    now = time.time()
    memory1 = create_test_memory()
    memory1["timestamp"] = now - 1000  # Older memory

    memory2 = create_test_memory()
    memory2["timestamp"] = now - 500  # Middle memory

    memory3 = create_test_memory()
    memory3["timestamp"] = now  # Recent memory

    # Store the memories
    im_store.store(agent_id, memory1)
    im_store.store(agent_id, memory2)
    im_store.store(agent_id, memory3)

    # Query for recent memories
    recent_memories = im_store.get_by_timerange(
        agent_id=agent_id, start_time=now - 600, end_time=now + 100, limit=10
    )

    # Should return the two most recent memories
    assert len(recent_memories) == 2
    memory_ids = [m["memory_id"] for m in recent_memories]
    assert memory2["memory_id"] in memory_ids
    assert memory3["memory_id"] in memory_ids
    assert memory1["memory_id"] not in memory_ids


def test_importance_query(im_store):
    """Test querying by importance with MockRedis."""
    agent_id = "test-agent"

    # Create test memories with different importance scores
    memory1 = create_test_memory()
    memory1["metadata"]["importance_score"] = 0.3  # Low importance

    memory2 = create_test_memory()
    memory2["metadata"]["importance_score"] = 0.6  # Medium importance

    memory3 = create_test_memory()
    memory3["metadata"]["importance_score"] = 0.9  # High importance

    # Store the memories
    im_store.store(agent_id, memory1)
    im_store.store(agent_id, memory2)
    im_store.store(agent_id, memory3)

    # Query for high importance memories
    important_memories = im_store.get_by_importance(
        agent_id=agent_id, min_importance=0.7, max_importance=1.0, limit=10
    )

    # Should return only the high importance memory
    assert len(important_memories) == 1
    assert important_memories[0]["memory_id"] == memory3["memory_id"]


def test_delete(im_store):
    """Test deleting a memory with MockRedis."""
    agent_id = "test-agent"
    memory = create_test_memory()
    memory_id = memory["memory_id"]

    # Store the memory
    im_store.store(agent_id, memory)

    # Get the Redis client used by the store
    redis_client = im_store.redis.client

    # Verify the memory was stored
    key = f"{im_store._key_prefix}:{agent_id}:memory:{memory_id}"
    assert redis_client.exists(key) == 1

    # Delete the memory
    result = im_store.delete(agent_id, memory_id)
    assert result is True

    # Verify the memory was removed
    assert redis_client.exists(key) == 0

    # Verify it was removed from indices
    memories_key = f"{im_store._key_prefix}:{agent_id}:memories"
    timeline_key = f"{im_store._key_prefix}:{agent_id}:timeline"
    importance_key = f"{im_store._key_prefix}:{agent_id}:importance"

    assert redis_client.zscore(memories_key, memory_id) is None
    assert redis_client.zscore(timeline_key, memory_id) is None
    assert redis_client.zscore(importance_key, memory_id) is None


def test_count_and_clear(im_store):
    """Test counting and clearing memories with MockRedis."""
    agent_id = "test-agent"

    # Store multiple memories
    for _ in range(5):
        memory = create_test_memory()
        im_store.store(agent_id, memory)

    # Count memories
    count = im_store.count(agent_id)
    assert count == 5

    # Clear memories
    result = im_store.clear(agent_id)
    assert result is True

    # Verify all memories were cleared
    count_after = im_store.count(agent_id)
    assert count_after == 0
