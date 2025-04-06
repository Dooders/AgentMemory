"""Example demonstrating the use of MockRedis in the agent memory system.

This example shows how to configure and use the MockRedis implementation
for local development and testing without a real Redis server.
"""

import logging
import time
import uuid
from typing import Any, Dict, List

from agent_memory.config import RedisSTMConfig
from agent_memory.storage.mockredis import MockRedis
from agent_memory.storage.redis_stm import RedisSTMStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompatibleMockRedis(MockRedis):
    """A MockRedis subclass that ignores unsupported parameters."""

    def set(self, key, value, ex=None, px=None, nx=None, xx=None):
        """Override set to ignore unsupported parameters."""
        # Only use ex parameter, ignore others
        super().set(key, value, ex=ex)
        return True

    def zadd(
        self, name, mapping=None, nx=False, xx=False, ch=False, incr=False, **kwargs
    ):
        """Override zadd to ignore unsupported parameters."""
        # Ignore nx, xx, ch, incr parameters
        return super().zadd(name, mapping, **kwargs)

    def zrangebyscore(
        self,
        name,
        min_score,
        max_score,
        withscores=False,
        start=0,
        num=None,
        score_cast_func=float,
    ):
        """Override zrangebyscore to ignore unsupported parameters."""
        # For importance retrieval, we need to return proper object format
        result = super().zrangebyscore(
            name, min_score, max_score, withscores=withscores, start=start, num=num
        )

        # When withscores=True, convert flat list to list of tuples if needed
        if withscores and result and isinstance(result[0], str):
            # Convert [member1, score1, member2, score2, ...] to [(member1, score1), (member2, score2), ...]
            tuples = []
            for i in range(0, len(result), 2):
                if i + 1 < len(result):
                    tuples.append((result[i], result[i + 1]))
            return tuples

        return result

    def eval(self, script, numkeys, *keys_and_args):
        """Mock implementation of Lua script evaluation."""
        # For delete operations, we need to manually implement the delete logic
        # This is a simplified version that assumes specific delete script patterns

        # If this is a delete operation (based on script content)
        if "redis.call('del'" in script or 'redis.call("del"' in script:
            # Extract the key to delete
            if len(keys_and_args) > 0:
                key = keys_and_args[0]
                # Perform delete operation
                self.delete(key)
                return 1

        # For get_by_importance, we need to handle hmget operations
        if "redis.call('hmget'" in script or 'redis.call("hmget"' in script:
            # This is a very simplified implementation - in a real scenario,
            # you'd need to parse the script to determine what it's doing
            return {"importance_score": 0.5}

        # Default fallback
        return 1

    # Implement any missing methods or overrides needed for delete operations
    def execute_command(self, *args, **kwargs):
        """Handle generic commands not specifically implemented."""
        command = args[0].lower() if args else ""

        if command == "del":
            # Handle delete command
            if len(args) > 1:
                key = args[1]
                self.delete(key)
                return 1

        # Default implementation for unhandled commands
        return 1


def create_memory_entry(content: str) -> Dict:
    """Create a simple memory entry with the given content.

    Args:
        content: The content for the memory entry

    Returns:
        A memory entry dictionary
    """
    memory_id = str(uuid.uuid4())
    return {
        "memory_id": memory_id,
        "agent_id": "test-agent",
        "timestamp": time.time(),
        "content": content,
        "metadata": {
            "importance_score": 0.5,
            "retrieval_count": 0,
            "creation_time": time.time(),
            "compression_level": 0,
        },
    }


def main():
    """Run the MockRedis example."""
    logger.info("Starting MockRedis example")

    # Create configuration
    config = RedisSTMConfig(
        host="localhost",
        port=6379,
        db=0,
        namespace="mock-stm",
        ttl=3600,  # 1 hour TTL
    )

    # Create the Redis STM store
    store = RedisSTMStore(config)

    # Replace Redis client with our compatible MockRedis implementation
    mock_redis = CompatibleMockRedis()
    store.redis.client = mock_redis
    logger.info("Created RedisSTMStore with compatible MockRedis")

    # Store some memory entries
    memories = []
    for i in range(5):
        memory = create_memory_entry(f"This is memory {i}")
        memories.append(memory)
        store.store("test-agent", memory)
        logger.info(f"Stored memory {i}: {memory['memory_id']}")

    # Retrieve the memories
    for memory in memories:
        memory_id = memory["memory_id"]
        retrieved = store.get("test-agent", memory_id)
        if retrieved:
            logger.info(f"Retrieved memory: {retrieved['content']}")
        else:
            logger.error(f"Failed to retrieve memory: {memory_id}")

    # Get memories by timerange
    now = time.time()
    one_hour_ago = now - 3600
    timerange_memories = store.get_by_timerange("test-agent", one_hour_ago, now)
    logger.info(f"Retrieved {len(timerange_memories)} memories by timerange")

    # Get memories by importance
    important_memories = store.get_by_importance("test-agent", min_importance=0.4)
    logger.info(f"Retrieved {len(important_memories)} important memories")

    # Delete individual memories
    for memory in memories:
        memory_id = memory["memory_id"]
        deleted = store.delete("test-agent", memory_id)
        logger.info(f"Deleted memory {memory_id}: {deleted}")

    # Check if all memories are deleted
    remaining_memories = store.get_by_timerange("test-agent", one_hour_ago, now)
    logger.info(f"Remaining memories: {len(remaining_memories)}")

    logger.info("MockRedis example completed successfully")


if __name__ == "__main__":
    main()
