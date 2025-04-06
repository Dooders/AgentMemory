"""Agent Memory MockRedis Module

This module provides a Redis mock implementation for testing and development
without requiring an actual Redis server. It simulates Redis functionality with 
an in-memory implementation that follows the Redis API.

Key components:

1. MockRedis: The core class that simulates a Redis server with in-memory storage,
   supporting most common Redis operations like GET, SET, HSET, and ZADD.

2. Pipeline: Implements Redis pipelining to batch multiple commands for more
   efficient execution, just like in a real Redis instance.

3. PubSub: Provides publish-subscribe pattern functionality, allowing simulated
   message publishing and subscription.

This mock implementation is particularly useful for:
- Unit testing Redis-dependent code without external dependencies
- Local development without setting up Redis
- CI/CD environments where Redis might not be available

The implementation covers standard Redis data structures (strings, hashes, lists,
sorted sets) and common Redis features (expiration, transactions, pub/sub).

Usage example:
```python
from agent_memory.storage.mockredis import MockRedis

# Create a mock Redis instance
redis = MockRedis()

# Use it just like a normal Redis client
redis.set("key", "value")
value = redis.get("key")

# Hash operations
redis.hset("user:1", "name", "Alice")
name = redis.hget("user:1", "name")

# List operations
redis.lpush("queue", "job1", "job2")
job = redis.lpop("queue")
```
"""

from .core import MockRedis
from .pipeline import Pipeline
from .pubsub import PubSub
from .core import RedisError, ConnectionError

__all__ = [
    "MockRedis",
    "Pipeline",
    "PubSub",
    "RedisError",
    "ConnectionError"
]
