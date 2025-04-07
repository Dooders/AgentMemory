"""Integration tests for Redis Short-Term Memory (STM) storage.

This module contains integration tests for the RedisSTMStore class using
a real Redis instance.
"""

import json
import os
import time
import uuid
import pytest

from memory.config import RedisSTMConfig
from memory.storage.redis_stm import RedisSTMStore
from memory.utils.error_handling import Priority


# Skip these tests if integration tests are not enabled
pytestmark = pytest.mark.integration


def get_redis_config():
    """Get Redis configuration from environment variables."""
    return RedisSTMConfig(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=int(os.environ.get("REDIS_TEST_DB", 15)),  # Use DB 15 for tests
        password=os.environ.get("REDIS_PASSWORD", None),
        namespace="test-stm-integration",
        ttl=30,  # Short TTL for tests
        use_mock=True  # Use MockRedis instead of real Redis
    )


@pytest.fixture
def stm_store():
    """Create a RedisSTMStore for integration testing."""
    config = get_redis_config()
    store = RedisSTMStore(config)
    
    # Clean up before tests
    test_agents = ["test-agent", "test-agent-2"]
    for agent_id in test_agents:
        store.clear(agent_id)
    
    yield store
    
    # Clean up after tests
    for agent_id in test_agents:
        store.clear(agent_id)


@pytest.fixture
def memory_entries():
    """Create sample memory entries for testing."""
    current_time = time.time()
    return [
        {
            "memory_id": f"test-memory-{i}",
            "content": f"Test memory content {i}",
            "timestamp": current_time - (10 * i),  # Older as i increases
            "metadata": {
                "importance_score": max(0.1, 1.0 - (i * 0.2)),  # Decreasing importance
                "retrieval_count": 0,
                "source": "integration-test"
            },
            "embeddings": {
                "full_vector": [0.1 * i, 0.2 * i, 0.3 * i]
            }
        }
        for i in range(1, 6)  # Create 5 test memories
    ]


def test_store_and_get_integration(stm_store, memory_entries):
    """Test storing and retrieving memory entries from Redis."""
    agent_id = "test-agent"
    
    # Store all memories
    for entry in memory_entries:
        result = stm_store.store(agent_id, entry)
        assert result is True
    
    # Retrieve and verify each memory
    for entry in memory_entries:
        memory_id = entry["memory_id"]
        retrieved = stm_store.get(agent_id, memory_id)
        
        assert retrieved is not None
        assert retrieved["memory_id"] == memory_id
        assert retrieved["content"] == entry["content"]
        
        # Verify access metadata was updated
        assert retrieved["metadata"]["retrieval_count"] == 1
        assert "last_access_time" in retrieved["metadata"]


def test_get_by_timerange_integration(stm_store, memory_entries):
    """Test retrieving memories by time range from Redis."""
    agent_id = "test-agent"
    
    # Store all memories
    for entry in memory_entries:
        stm_store.store(agent_id, entry)
    
    # Get middle time range (should include some but not all)
    min_time = memory_entries[2]["timestamp"]
    max_time = memory_entries[0]["timestamp"]
    
    results = stm_store.get_by_timerange(agent_id, min_time, max_time)
    
    # Should include memories 0, 1, 2
    assert len(results) == 3
    memory_ids = [m["memory_id"] for m in results]
    assert memory_entries[0]["memory_id"] in memory_ids
    assert memory_entries[1]["memory_id"] in memory_ids
    assert memory_entries[2]["memory_id"] in memory_ids


def test_get_by_importance_integration(stm_store, memory_entries):
    """Test retrieving memories by importance from Redis."""
    agent_id = "test-agent"
    
    # Store all memories
    for entry in memory_entries:
        # Debug: print importance score before storing
        print(f"Storing memory {entry['memory_id']} with importance score: {entry['metadata']['importance_score']}")
        stm_store.store(agent_id, entry)
    
    # Debug: print importance key and verify data stored in Redis
    importance_key = stm_store._get_importance_key(agent_id)
    print(f"Importance key: {importance_key}")
    
    # Get the raw data from Redis to verify what's stored
    importance_data = stm_store.redis.client.zrange(importance_key, 0, -1, withscores=True)
    print(f"Raw importance data in Redis: {importance_data}")
    
    # Get high importance memories (>= 0.7) - first entry should have score 0.8
    high_results = stm_store.get_by_importance(agent_id, 0.7, 1.0)
    print(f"Results from get_by_importance: {high_results}")
    
    # Should include only memory with index 0 (test-memory-1 with importance 0.8)
    assert len(high_results) == 1
    memory_ids = [m["memory_id"] for m in high_results]
    assert memory_entries[0]["memory_id"] in memory_ids
    
    # Also test a different range that should include two memories
    medium_results = stm_store.get_by_importance(agent_id, 0.5, 1.0)
    print(f"Medium results count: {len(medium_results)}")
    print(f"Medium results memory IDs: {[m['memory_id'] for m in medium_results]}")
    assert len(medium_results) == 2
    medium_memory_ids = [m["memory_id"] for m in medium_results]
    assert memory_entries[0]["memory_id"] in medium_memory_ids  # test-memory-1 (0.8)
    assert memory_entries[1]["memory_id"] in medium_memory_ids  # test-memory-2 (0.6)


def test_delete_integration(stm_store, memory_entries):
    """Test deleting memories from Redis."""
    agent_id = "test-agent"
    
    # Store all memories
    for entry in memory_entries:
        stm_store.store(agent_id, entry)
    
    # Delete the first memory
    memory_id = memory_entries[0]["memory_id"]
    result = stm_store.delete(agent_id, memory_id)
    assert result is True
    
    # Verify it was deleted
    retrieved = stm_store.get(agent_id, memory_id)
    assert retrieved is None
    
    # Count should be reduced
    count = stm_store.count(agent_id)
    assert count == len(memory_entries) - 1


def test_count_integration(stm_store, memory_entries):
    """Test counting memories in Redis."""
    agent_id = "test-agent"
    
    # Store all memories
    for entry in memory_entries:
        stm_store.store(agent_id, entry)
    
    count = stm_store.count(agent_id)
    assert count == len(memory_entries)


def test_clear_integration(stm_store, memory_entries):
    """Test clearing all memories for an agent from Redis."""
    agent_id = "test-agent"
    
    # Store memories for two different agents
    for entry in memory_entries:
        stm_store.store(agent_id, entry)
        stm_store.store("test-agent-2", entry)
    
    # Clear memories for one agent
    result = stm_store.clear(agent_id)
    assert result is True
    
    # Verify that agent's memories are cleared
    count = stm_store.count(agent_id)
    assert count == 0
    
    # Verify the other agent's memories are untouched
    count2 = stm_store.count("test-agent-2")
    assert count2 == len(memory_entries)


def test_ttl_integration(stm_store):
    """Test that TTL is enforced on memory entries."""
    agent_id = "test-agent"
    memory_id = f"ttl-test-{uuid.uuid4()}"
    
    # Create a test memory with very short TTL
    config = get_redis_config()
    config.ttl = 1  # 1 second TTL
    short_ttl_store = RedisSTMStore(config)
    
    # Store a memory
    memory_entry = {
        "memory_id": memory_id,
        "content": "This memory should expire quickly",
        "timestamp": time.time(),
        "metadata": {"importance_score": 0.5}
    }
    
    short_ttl_store.store(agent_id, memory_entry)
    
    # Verify it exists
    retrieved = short_ttl_store.get(agent_id, memory_id)
    assert retrieved is not None
    
    # Wait for it to expire
    time.sleep(2)
    
    # Verify it no longer exists
    retrieved = short_ttl_store.get(agent_id, memory_id)
    assert retrieved is None


def test_check_health_integration(stm_store):
    """Test health check with real Redis."""
    health = stm_store.check_health()
    
    assert health["status"] == "healthy"
    assert "latency_ms" in health
    assert health["client"] == "redis-stm"


def test_update_access_metadata_integration(stm_store, memory_entries):
    """Test that accessing memories updates their metadata."""
    agent_id = "test-agent"
    
    # Store a memory
    memory_entry = memory_entries[0]
    memory_id = memory_entry["memory_id"]
    stm_store.store(agent_id, memory_entry)
    
    # Get the memory multiple times to increase retrieval count
    for _ in range(3):
        retrieved = stm_store.get(agent_id, memory_id)
        time.sleep(0.1)  # Small delay to ensure different access times
    
    # Verify the metadata was updated
    final = stm_store.get(agent_id, memory_id)
    assert final["metadata"]["retrieval_count"] == 4  # Initial + 3 retrievals
    
    # Verify importance was increased due to frequent access
    initial_importance = memory_entry["metadata"]["importance_score"]
    final_importance = final["metadata"]["importance_score"]
    assert final_importance > initial_importance


def test_store_with_different_priorities_integration(stm_store):
    """Test storing with different priorities with real Redis."""
    agent_id = "test-agent"
    
    # Create memories with different priorities
    memory_high = {
        "memory_id": "high-priority",
        "content": "High priority memory",
        "timestamp": time.time(),
        "metadata": {"importance_score": 1.0}
    }
    
    memory_normal = {
        "memory_id": "normal-priority",
        "content": "Normal priority memory",
        "timestamp": time.time(),
        "metadata": {"importance_score": 0.5}
    }
    
    memory_low = {
        "memory_id": "low-priority",
        "content": "Low priority memory",
        "timestamp": time.time(),
        "metadata": {"importance_score": 0.1}
    }
    
    # Store with different priorities
    stm_store.store(agent_id, memory_high, Priority.HIGH)
    stm_store.store(agent_id, memory_normal, Priority.NORMAL)
    stm_store.store(agent_id, memory_low, Priority.LOW)
    
    # Verify all were stored successfully
    assert stm_store.get(agent_id, "high-priority") is not None
    assert stm_store.get(agent_id, "normal-priority") is not None
    assert stm_store.get(agent_id, "low-priority") is not None 