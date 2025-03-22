"""Integration tests for Redis Intermediate Memory (IM) storage.

This module contains integration tests for the RedisIMStore class which require
a real Redis instance to run. These tests are marked with the 'integration' marker
and can be skipped during normal test runs.

To run these tests:
pytest tests/storage/test_redis_im_integration.py -v

To skip these tests:
pytest tests/storage -k "not integration" -v
"""

import json
import time
import uuid
import pytest
import redis

from agent_memory.config import RedisIMConfig
from agent_memory.storage.redis_im import RedisIMStore


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


def get_redis_config():
    """Get Redis config for integration tests."""
    # Use a high DB number to avoid conflicting with other DBs
    # and a unique namespace to avoid conflicts with other tests
    test_namespace = f"test_im_{uuid.uuid4().hex[:8]}"
    return RedisIMConfig(
        host="localhost",  # Modify for your Redis host
        port=6379,         # Modify for your Redis port
        db=15,             # Use DB 15 for tests
        namespace=test_namespace,
        ttl=60             # Short TTL for tests
    )


@pytest.fixture(scope="module")
def redis_client():
    """Create a Redis client for test verification."""
    config = get_redis_config()
    client = redis.Redis(
        host=config.host,
        port=config.port,
        db=config.db,
        decode_responses=True
    )
    
    # Check if Redis is available
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis server not available")
    
    yield client
    
    # Clean up test data
    keys = client.keys(f"{config.namespace}:*")
    if keys:
        client.delete(*keys)
    client.close()


@pytest.fixture
def im_store(redis_client):
    """Create a RedisIMStore instance for testing."""
    config = get_redis_config()
    store = RedisIMStore(config)
    
    yield store
    
    # Clean up test data
    keys = redis_client.keys(f"{config.namespace}:*")
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
            "source": "integration_test"
        }
    }


def test_store_and_retrieve(im_store, redis_client):
    """Test storing and retrieving a memory with a real Redis instance."""
    agent_id = "test-agent"
    memory = create_test_memory()
    memory_id = memory["memory_id"]
    
    # Store the memory
    result = im_store.store(agent_id, memory)
    assert result is True
    
    # Verify the memory was stored in Redis
    key = f"{im_store._key_prefix}:{agent_id}:memory:{memory_id}"
    assert redis_client.exists(key) == 1
    
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
    
    # Verify access metadata was updated
    retrieved_json = redis_client.get(key)
    retrieved_data = json.loads(retrieved_json)
    assert retrieved_data["metadata"]["retrieval_count"] == 1
    assert "last_access_time" in retrieved_data["metadata"]


def test_timerange_query(im_store):
    """Test querying by time range with a real Redis instance."""
    agent_id = "test-agent"
    
    # Create test memories with different timestamps
    now = time.time()
    memory1 = create_test_memory()
    memory1["timestamp"] = now - 1000  # Older memory
    
    memory2 = create_test_memory()
    memory2["timestamp"] = now - 500   # Middle memory
    
    memory3 = create_test_memory()
    memory3["timestamp"] = now         # Recent memory
    
    # Store the memories
    im_store.store(agent_id, memory1)
    im_store.store(agent_id, memory2)
    im_store.store(agent_id, memory3)
    
    # Query for recent memories
    recent_memories = im_store.get_by_timerange(
        agent_id=agent_id,
        start_time=now - 600,
        end_time=now + 100,
        limit=10
    )
    
    # Should return the two most recent memories
    assert len(recent_memories) == 2
    memory_ids = [m["memory_id"] for m in recent_memories]
    assert memory2["memory_id"] in memory_ids
    assert memory3["memory_id"] in memory_ids
    assert memory1["memory_id"] not in memory_ids


def test_importance_query(im_store):
    """Test querying by importance with a real Redis instance."""
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
        agent_id=agent_id,
        min_importance=0.7,
        max_importance=1.0,
        limit=10
    )
    
    # Should return only the high importance memory
    assert len(important_memories) == 1
    assert important_memories[0]["memory_id"] == memory3["memory_id"]


def test_delete(im_store, redis_client):
    """Test deleting a memory with a real Redis instance."""
    agent_id = "test-agent"
    memory = create_test_memory()
    memory_id = memory["memory_id"]
    
    # Store the memory
    im_store.store(agent_id, memory)
    
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
    """Test counting and clearing memories with a real Redis instance."""
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
    
    # Verify count is now 0
    count = im_store.count(agent_id)
    assert count == 0 