"""Redis-based Intermediate Memory (IM) storage for agent memory system.

This module provides a Redis-based implementation of the Intermediate Memory
storage tier with TTL-based expiration and level 1 compression.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from agent_memory.config import RedisIMConfig
from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)

logger = logging.getLogger(__name__)


class RedisIMStore:
    """Redis-based storage for Intermediate Memory (IM).

    This class provides storage operations for the medium-resolution,
    intermediate-lived agent memory tier using Redis as the backing store.

    Attributes:
        config: Configuration for IM Redis storage
        redis: Resilient Redis client instance
        _key_prefix: Prefix for Redis keys
    """

    def __init__(self, config: RedisIMConfig):
        """Initialize the Redis IM store.

        Args:
            config: Configuration for IM Redis storage
        """
        self.config = config
        self._key_prefix = config.namespace

        # Create resilient Redis client
        self.redis = ResilientRedisClient(
            client_name="im",
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )

        logger.info("Initialized RedisIMStore with namespace %s", self._key_prefix)

    def store(
        self,
        agent_id: str,
        memory_entry: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """Store a memory entry in IM.

        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store (should be level 1 compressed)
            priority: Priority level for this operation

        Returns:
            True if the operation succeeded, False otherwise
        """
        memory_id = memory_entry.get("memory_id")
        if not memory_id:
            logger.error("Cannot store memory entry without memory_id")
            return False

        # Verify compression level
        compression_level = memory_entry.get("metadata", {}).get("compression_level")
        if compression_level != 1:
            logger.error(
                "Invalid compression level for IM storage: %s. Expected level 1.",
                compression_level,
            )
            return False

        # Use store_with_retry for resilient storage
        return self.redis.store_with_retry(
            agent_id=agent_id,
            state_data=memory_entry,
            store_func=self._store_memory_entry,
            priority=priority,
        )

    def _store_memory_entry(self, agent_id: str, memory_entry: Dict[str, Any]) -> bool:
        """Internal method to store a memory entry.

        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store

        Returns:
            True if the operation succeeded, False otherwise

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        memory_id = memory_entry["memory_id"]
        timestamp = memory_entry.get("timestamp", time.time())

        try:
            # Use pipeline to batch all Redis commands
            pipe = self.redis.pipeline()

            # Store the full memory entry
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            pipe.set(key, json.dumps(memory_entry), ex=self.config.ttl)

            # Add to agent's memory list
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            pipe.zadd(agent_memories_key, {memory_id: timestamp})
            pipe.expire(agent_memories_key, self.config.ttl)

            # Add to timeline index for chronological retrieval
            timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
            pipe.zadd(timeline_key, {memory_id: timestamp})
            pipe.expire(timeline_key, self.config.ttl)

            # Add to importance index for importance-based retrieval
            importance = memory_entry.get("metadata", {}).get("importance_score", 0.0)
            importance_key = f"{self._key_prefix}:{agent_id}:importance"
            pipe.zadd(importance_key, {memory_id: importance})
            pipe.expire(importance_key, self.config.ttl)

            # Execute all commands as a single atomic operation
            pipe.execute()

            return True

        except (RedisUnavailableError, RedisTimeoutError) as e:
            # Let these propagate up for retry handling
            raise
        except Exception as e:
            logger.error("Failed to store memory entry %s: %s", memory_id, str(e))
            return False

    def get(self, agent_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to retrieve

        Returns:
            Memory entry if found, None otherwise
        """
        try:
            # Construct the key using agent_id and memory_id
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            data = self.redis.get(key)

            if not data:
                return None

            memory_entry = json.loads(data)
            self._update_access_metadata(agent_id, memory_id, memory_entry)
            return memory_entry

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memory %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to retrieve memory entry %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return None

    def get_by_timerange(
        self, agent_id: str, start_time: float, end_time: float, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a time range.

        Args:
            agent_id: ID of the agent
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            limit: Maximum number of memories to return

        Returns:
            List of memory entries within the time range
        """
        timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
        try:
            # Get memory IDs within time range
            memory_ids = self.redis.zrangebyscore(
                timeline_key, min=start_time, max=end_time, start=0, num=limit
            )

            if not memory_ids:
                return []

            # Batch retrieve memory entries using pipeline
            pipe = self.redis.pipeline()
            memory_keys = []

            for memory_id in memory_ids:
                # Handle memory_id if it's bytes
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")

                key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
                memory_keys.append(memory_id)
                pipe.get(key)

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, data in enumerate(results):
                if data:
                    memory_entry = json.loads(data)
                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories

        except Exception as e:
            logger.error("Failed to retrieve memories by time range: %s", str(e))
            return []

    def get_by_importance(
        self,
        agent_id: str,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories by importance score range.

        Args:
            agent_id: ID of the agent
            min_importance: Minimum importance score (0.0-1.0)
            max_importance: Maximum importance score (0.0-1.0)
            limit: Maximum number of memories to return

        Returns:
            List of memory entries within the importance range
        """
        importance_key = f"{self._key_prefix}:{agent_id}:importance"
        try:
            # Get memory IDs within importance range
            memory_ids = self.redis.zrangebyscore(
                importance_key,
                min=min_importance,
                max=max_importance,
                start=0,
                num=limit,
            )

            if not memory_ids:
                return []

            # Batch retrieve memory entries using pipeline
            pipe = self.redis.pipeline()
            memory_keys = []

            for memory_id in memory_ids:
                # Handle memory_id if it's bytes
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")

                key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
                memory_keys.append(memory_id)
                pipe.get(key)

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, data in enumerate(results):
                if data:
                    memory_entry = json.loads(data)
                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories

        except Exception as e:
            logger.error("Failed to retrieve memories by importance: %s", str(e))
            return []

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        try:
            # Remove from all indices
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
            importance_key = f"{self._key_prefix}:{agent_id}:importance"

            # Use pipeline for atomic deletion from all indices
            pipe = self.redis.pipeline()
            pipe.delete(key)
            pipe.zrem(agent_memories_key, memory_id)
            pipe.zrem(timeline_key, memory_id)
            pipe.zrem(importance_key, memory_id)
            pipe.execute()

            return True

        except Exception as e:
            logger.error("Failed to delete memory entry %s: %s", memory_id, str(e))
            return False

    def count(self, agent_id: str) -> int:
        """Get the number of memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of memories stored for the agent
        """
        agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
        try:
            return self.redis.zcard(agent_memories_key)
        except Exception as e:
            logger.error(
                "Failed to get memory count for agent %s: %s", agent_id, str(e)
            )
            return 0

    def clear(self, agent_id: str) -> bool:
        """Clear all memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            True if the operation succeeded, False otherwise
        """
        try:
            # Get all memory IDs
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            memory_ids = self.redis.zrange(agent_memories_key, 0, -1)

            if not memory_ids:
                return True

            # Build keys to delete
            keys_to_delete = []
            for memory_id in memory_ids:
                # Handle memory_id if it's bytes
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")
                keys_to_delete.append(
                    f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
                )

            # Add index keys
            keys_to_delete.extend(
                [
                    f"{self._key_prefix}:{agent_id}:memories",
                    f"{self._key_prefix}:{agent_id}:timeline",
                    f"{self._key_prefix}:{agent_id}:importance",
                ]
            )

            # Use pipeline to delete all keys in a single operation
            pipe = self.redis.pipeline()
            if keys_to_delete:
                pipe.delete(*keys_to_delete)
            pipe.execute()

            return True
        except Exception as e:
            logger.error("Error clearing memories for agent %s: %s", agent_id, str(e))
            return False

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the Redis store.

        Returns:
            Dictionary containing health metrics
        """
        try:
            ping_result = self.redis.ping()
            return {
                "status": "healthy" if ping_result else "degraded",
                "message": (
                    "Redis connection successful"
                    if ping_result
                    else "Redis ping failed"
                ),
                "latency_ms": self.redis.get_latency(),
                "client": "redis-im",
            }
        except (RedisTimeoutError, RedisUnavailableError, Exception) as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "error": str(e),
                "client": "redis-im",
            }

    def _update_access_metadata(
        self, agent_id: str, memory_id: str, memory_entry: Dict[str, Any]
    ) -> None:
        """Update access metadata for a memory entry.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory
            memory_entry: Memory entry to update
        """
        try:
            # Update access time and retrieval count
            metadata = memory_entry.get("metadata", {})
            retrieval_count = metadata.get("retrieval_count", 0) + 1
            access_time = time.time()

            metadata.update(
                {"last_access_time": access_time, "retrieval_count": retrieval_count}
            )
            memory_entry["metadata"] = metadata

            # Use pipeline for atomic updates
            pipe = self.redis.pipeline()

            # Store updated entry
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            pipe.set(key, json.dumps(memory_entry), ex=self.config.ttl)

            # Update importance based on access patterns
            # Increase importance for frequently accessed memories
            if retrieval_count > 1:
                importance = metadata.get("importance_score", 0.0)
                access_factor = min(retrieval_count / 10.0, 1.0)  # Cap at 1.0
                new_importance = importance + (access_factor * 0.1)  # Slight boost

                importance_key = f"{self._key_prefix}:{agent_id}:importance"
                pipe.zadd(importance_key, {memory_id: new_importance})

                # Update in memory entry
                metadata["importance_score"] = new_importance

            pipe.execute()

        except Exception as e:
            # Non-critical operation, just log
            logger.warning(
                "Failed to update access metadata for memory %s: %s", memory_id, str(e)
            )

    def get_size(self, agent_id: str) -> int:
        """Get the approximate size in bytes of all memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Approximate size in bytes
        """
        try:
            # Get all memory keys for this agent
            pattern = f"{self._key_prefix}:{agent_id}:memory:*"
            memory_keys = list(self.redis.scan_iter(match=pattern))

            if not memory_keys:
                return 0

            # Use pipeline to get the memory sizes in batches
            total_size = 0
            batch_size = 100  # Process keys in batches to avoid large pipelines

            for i in range(0, len(memory_keys), batch_size):
                batch_keys = memory_keys[i : i + batch_size]
                pipe = self.redis.pipeline()

                for key in batch_keys:
                    pipe.get(key)

                values = pipe.execute()

                # Sum up the sizes of non-empty values
                for value in values:
                    if value:
                        total_size += len(value)

            return total_size
        except Exception as e:
            logger.error("Error calculating memory size: %s", e)
            return 0

    def get_all(self, agent_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all memories for an agent.

        Args:
            agent_id: ID of the agent
            limit: Maximum number of memories to return

        Returns:
            List of memory entries
        """
        try:
            # Get all memory IDs sorted by recency
            memories_key = f"{self._key_prefix}:{agent_id}:memories"
            memory_ids = self.redis.zrange(
                memories_key, 0, limit - 1, desc=True  # Most recent first
            )

            if not memory_ids:
                return []

            # Batch retrieve memory entries using pipeline
            pipe = self.redis.pipeline()
            memory_keys = []

            for memory_id in memory_ids:
                # Handle memory_id if it's bytes (could happen with some Redis clients)
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")

                key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
                memory_keys.append(memory_id)
                pipe.get(key)

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, data in enumerate(results):
                if data:
                    memory_entry = json.loads(data)
                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories
        except Exception as e:
            logger.error("Error retrieving all memories: %s", e)
            return []

    def search_similar(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories with similar embeddings.

        Args:
            agent_id: Unique identifier for the agent
            query_embedding: The vector embedding to use for similarity search
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity score
        """
        try:
            memories = self.get_all(agent_id)

            # Filter by memory type if specified
            if memory_type:
                memories = [m for m in memories if m.get("memory_type") == memory_type]

            # Filter memories without embeddings
            memories = [m for m in memories if "embedding" in m]

            # Return empty list if no memories with embeddings
            if not memories:
                return []

            # Calculate similarity scores
            for memory in memories:
                # Calculate cosine similarity if the memory has an embedding
                memory_embedding = memory.get("embedding", [])
                if memory_embedding:
                    similarity = self._cosine_similarity(
                        query_embedding, memory_embedding
                    )
                    memory["similarity_score"] = float(similarity)
                else:
                    memory["similarity_score"] = 0.0

            # Sort by similarity score (descending)
            memories.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            # Return top k results
            return memories[:k]

        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity as a float between -1 and 1
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        try:
            # Convert to numpy for vector operations
            a_array = np.array(a)
            b_array = np.array(b)

            # Calculate norm products
            norm_a = np.linalg.norm(a_array)
            norm_b = np.linalg.norm(b_array)

            # Prevent division by zero
            if norm_a == 0 or norm_b == 0:
                return 0.0

            # Calculate cosine similarity
            return float(np.dot(a_array, b_array) / (norm_a * norm_b))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def search_by_attributes(
        self,
        agent_id: str,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories matching specific attributes.

        Args:
            agent_id: Unique identifier for the agent
            attributes: Dictionary of attribute keys and values to match
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries with matching attributes
        """
        try:
            # Get all memories
            memories = self.get_all(agent_id)

            # Filter by memory type if specified
            if memory_type:
                memories = [m for m in memories if m.get("memory_type") == memory_type]

            # Filter by attributes
            results = []
            for memory in memories:
                if self._matches_attributes(memory, attributes):
                    results.append(memory)

            return results

        except Exception as e:
            logger.error(f"Error in search_by_attributes: {e}")
            return []

    def _matches_attributes(
        self, memory: Dict[str, Any], attributes: Dict[str, Any]
    ) -> bool:
        """Check if a memory matches the specified attributes.

        Args:
            memory: Memory entry to check
            attributes: Dictionary of attribute keys and values to match

        Returns:
            True if the memory matches all attributes, False otherwise
        """
        for attr_path, attr_value in attributes.items():
            # Handle nested attributes using dot notation (e.g., "position.location")
            parts = attr_path.split(".")

            # Start from the memory content
            current = memory.get("content", {})

            # Navigate through the nested structure
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    return False
                current = current[part]

            # Check the final attribute value
            last_part = parts[-1]
            if last_part not in current or current[last_part] != attr_value:
                return False

        return True

    def search_by_step_range(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories within a specific step range.

        Args:
            agent_id: Unique identifier for the agent
            start_step: Beginning of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries with step numbers in the range
        """
        try:
            # Get all memories
            memories = self.get_all(agent_id)

            # Filter by memory type if specified
            if memory_type:
                memories = [m for m in memories if m.get("memory_type") == memory_type]

            # Filter by step range
            results = []
            for memory in memories:
                step_number = memory.get("step_number")
                # Only include memories with step numbers in the requested range
                if step_number is not None and start_step <= step_number <= end_step:
                    results.append(memory)

            # Sort by step number
            results.sort(key=lambda x: x.get("step_number", 0))

            return results

        except Exception as e:
            logger.error(f"Error in search_by_step_range: {e}")
            return []
