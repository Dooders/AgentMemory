"""Tests for MockRedis integration in the agent memory system."""

import pytest
import time
import uuid

from memory.config import RedisSTMConfig, RedisIMConfig
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.redis_im import RedisIMStore


def create_test_memory():
    """Create a test memory entry for STM."""
    memory_id = str(uuid.uuid4())
    return {
        "memory_id": memory_id,
        "agent_id": "test-agent",
        "timestamp": time.time(),
        "content": "This is a test memory",
        "metadata": {
            "importance_score": 0.5,
            "retrieval_count": 0,
            "creation_time": time.time(),
            "compression_level": 0,  # Level 0 for STM
        },
    }


def create_im_test_memory():
    """Create a test memory entry for IM with compression level 1."""
    memory_id = str(uuid.uuid4())
    return {
        "memory_id": memory_id,
        "agent_id": "test-agent",
        "timestamp": time.time(),
        "content": "This is a test memory",
        "metadata": {
            "importance_score": 0.5,
            "retrieval_count": 0,
            "creation_time": time.time(),
            "compression_level": 1,  # Level 1 for IM
        },
    }


class TestMockRedisIntegration:
    """Test the integration of MockRedis in the agent memory system."""

    def test_stm_store_with_mockredis(self):
        """Test that RedisSTMStore works with MockRedis."""
        # Create configuration with use_mock=True
        config = RedisSTMConfig(
            namespace="test-mock-stm",
            ttl=3600,
            use_mock=True,
        )

        # Create the store with MockRedis
        store = RedisSTMStore(config)

        # Create and store a memory
        memory = create_test_memory()
        agent_id = "test-agent"
        success = store.store(agent_id, memory)
        assert success, "Failed to store memory in MockRedis"

        # Retrieve the memory
        memory_id = memory["memory_id"]
        retrieved = store.get(agent_id, memory_id)
        assert retrieved is not None, "Failed to retrieve memory from MockRedis"
        assert retrieved["memory_id"] == memory_id, "Retrieved incorrect memory"
        assert retrieved["content"] == memory["content"], "Memory content doesn't match"

        # Test get_all
        all_memories = store.get_all(agent_id)
        assert len(all_memories) == 1, "Expected one memory in get_all"

        # Test delete
        delete_success = store.delete(agent_id, memory_id)
        assert delete_success, "Failed to delete memory"
        
        # Verify deletion
        assert store.get(agent_id, memory_id) is None, "Memory still exists after deletion"

        # Test clear
        store.store(agent_id, create_test_memory())
        store.store(agent_id, create_test_memory())
        assert store.count(agent_id) == 2, "Expected two memories after adding"
        
        clear_success = store.clear(agent_id)
        assert clear_success, "Failed to clear memories"
        assert store.count(agent_id) == 0, "Memories still exist after clearing"

    def test_im_store_with_mockredis(self):
        """Test that RedisIMStore works with MockRedis."""
        # Create configuration with use_mock=True
        config = RedisIMConfig(
            namespace="test-mock-im",
            ttl=3600,
            use_mock=True,
        )

        # Create the store with MockRedis
        store = RedisIMStore(config)

        # Create and store a memory
        memory = create_im_test_memory()
        agent_id = "test-agent"
        success = store.store(agent_id, memory)
        assert success, "Failed to store memory in MockRedis"

        # Retrieve the memory
        memory_id = memory["memory_id"]
        retrieved = store.get(agent_id, memory_id)
        assert retrieved is not None, "Failed to retrieve memory from MockRedis"
        assert retrieved["memory_id"] == memory_id, "Retrieved incorrect memory"
        assert retrieved["content"] == memory["content"], "Memory content doesn't match"

        # Test get_all
        all_memories = store.get_all(agent_id)
        assert len(all_memories) == 1, "Expected one memory in get_all"

        # Test delete
        delete_success = store.delete(agent_id, memory_id)
        assert delete_success, "Failed to delete memory"
        
        # Verify deletion
        assert store.get(agent_id, memory_id) is None, "Memory still exists after deletion"

        # Test clear
        store.store(agent_id, create_im_test_memory())
        store.store(agent_id, create_im_test_memory())
        assert store.count(agent_id) == 2, "Expected two memories after adding"
        
        clear_success = store.clear(agent_id)
        assert clear_success, "Failed to clear memories"
        assert store.count(agent_id) == 0, "Memories still exist after clearing" 