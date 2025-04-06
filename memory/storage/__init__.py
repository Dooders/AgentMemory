"""Agent Memory Storage Module

This module provides storage implementations for the agent memory system,
supporting different memory tiers with various storage backends. It includes
robust error handling, resilience features, and efficient data access patterns.

Key components:

1. RedisClient: Resilient Redis client with circuit breaker patterns, automatic
   retries, and error handling for robust Redis operations.

2. AsyncRedisClient: Asynchronous version of the Redis client for non-blocking
   operations in asynchronous applications.

3. RedisFactory: Factory for creating Redis clients with the option to use
   either real Redis or MockRedis for testing and local development.

4. RedisIM: Redis-based Immediate Memory (IM) storage for high-speed, short-lived
   memories with automatic expiration.

5. RedisSTM: Redis-based Short-Term Memory (STM) storage for medium-term
   memories with comprehensive indexing and search capabilities.

6. SQLiteLTM: SQLite-based Long-Term Memory (LTM) storage for persistent,
   highly-compressed memories with efficient querying.

7. MockRedis: Redis mock implementation for testing and local development
   without requiring a real Redis instance.

This module works with the agent memory system to store memories efficiently
across different tiers based on their importance, recency, and relevance to
the agent's ongoing tasks.

Usage example:
```python
from memory.storage.redis_factory import RedisFactory
from memory.storage.redis_stm import RedisSTMStore
from memory.config import RedisSTMConfig

# Configure STM storage
stm_config = RedisSTMConfig(
    namespace="agent_memory",
    host="localhost",
    port=6379,
    db=0
)

# Create STM store
stm_store = RedisSTMStore(config=stm_config)

# Store a memory
memory_entry = {
    "memory_id": "mem_12345",
    "agent_id": "agent_1",
    "timestamp": 1617293982.5,
    "content": {"text": "The user requested help with Python."},
    "metadata": {
        "importance_score": 0.8,
        "creation_time": 1617293982.5,
        "last_access_time": 1617293982.5
    }
}

stm_store.store(agent_id="agent_1", memory_entry=memory_entry)

# Retrieve a memory
retrieved_memory = stm_store.get(agent_id="agent_1", memory_id="mem_12345")
```
"""

from memory.storage.async_redis_client import AsyncResilientRedisClient
from memory.storage.redis_client import ResilientRedisClient
from memory.storage.redis_factory import RedisFactory
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore

__all__ = [
    "AsyncResilientRedisClient",
    "ResilientRedisClient",
    "RedisFactory",
    "RedisIMStore",
    "RedisSTMStore",
    "SQLiteLTMStore",
]
