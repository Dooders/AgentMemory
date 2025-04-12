"""Redis-based Short-Term Memory (STM) storage for agent memory system.

This module provides a Redis-based implementation of the Short-Term Memory
storage tier with comprehensive error handling and recovery mechanisms.

Key features:
- High-resolution memory storage with full vector embeddings
- Efficient storage and retrieval by time, importance, and similarity
- Built-in resilience with circuit breaking and automatic retries
- Memory filtering and searching by attributes and content
- Step-based memory access for workflow-oriented reasoning

This component serves as the intermediate storage layer in the memory hierarchy,
sitting between immediate memory and long-term memory.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np

from memory.config import RedisSTMConfig
from memory.utils.error_handling import (
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)

from .redis_client import ResilientRedisClient
from .redis_factory import RedisFactory

logger = logging.getLogger(__name__)


# TypedDict definitions for complex structures
class MemoryMetadata(TypedDict, total=False):
    """Metadata information for a memory entry."""

    compression_level: int  # Always 0 for STM
    importance_score: float
    retrieval_count: int
    creation_time: float
    last_access_time: float


class MemoryEmbeddings(TypedDict, total=False):
    """Vector embeddings for a memory entry."""

    full_vector: List[float]  # Full resolution embedding


class MemoryEntry(TypedDict, total=False):
    """Structure of a memory entry in the short-term memory store."""

    memory_id: str
    agent_id: str
    timestamp: float
    content: Any  # Can be any structured data
    metadata: MemoryMetadata
    embeddings: MemoryEmbeddings  # Vector embeddings for similarity search
    memory_type: Optional[str]  # Optional type classification
    step_number: Optional[int]  # Optional step number for step-based retrieval


class RedisSTMStore:
    """Redis-based storage for Short-Term Memory (STM).

    This class provides storage operations for the high-resolution,
    short-lived agent memory tier using Redis as the backing store.

    Attributes:
        config: Configuration for STM Redis storage
        redis: Resilient Redis client instance
        _key_prefix: Prefix for Redis keys
    """

    def __init__(self, config: RedisSTMConfig):
        """Initialize the Redis STM store.

        Args:
            config: Configuration for STM Redis storage
        """
        self.config = config
        self._key_prefix = config.namespace

        # Create resilient Redis client using the factory
        self.redis = RedisFactory.create_client(
            client_name="stm",
            use_mock=config.use_mock,
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )

        logger.info("Initialized RedisSTMStore with namespace %s", self._key_prefix)

    # Helper methods for key construction
    def _get_memory_key(self, agent_id: str, memory_id: str) -> str:
        """Construct the Redis key for a memory entry.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory

        Returns:
            Redis key string
        """
        return f"{self._key_prefix}:{agent_id}:memory:{memory_id}"

    def _get_agent_memories_key(self, agent_id: str) -> str:
        """Construct the Redis key for an agent's memories list.

        Args:
            agent_id: ID of the agent

        Returns:
            Redis key string
        """
        return f"{self._key_prefix}:{agent_id}:memories"

    def _get_timeline_key(self, agent_id: str) -> str:
        """Construct the Redis key for an agent's timeline index.

        Args:
            agent_id: ID of the agent

        Returns:
            Redis key string
        """
        return f"{self._key_prefix}:{agent_id}:timeline"

    def _get_importance_key(self, agent_id: str) -> str:
        """Construct the Redis key for an agent's importance index.

        Args:
            agent_id: ID of the agent

        Returns:
            Redis key string
        """
        return f"{self._key_prefix}:{agent_id}:importance"

    def _get_vector_key(self, agent_id: str, memory_id: str) -> str:
        """Construct the Redis key for a memory's vector embedding.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory

        Returns:
            Redis key string
        """
        return f"{self._key_prefix}:{agent_id}:vector:{memory_id}"

    def _get_agent_prefix(self, agent_id: str) -> str:
        """Construct the Redis key prefix for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Redis key prefix string
        """
        return f"{self._key_prefix}:{agent_id}"

    def store(
        self,
        agent_id: str,
        memory_entry: MemoryEntry,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """Store a memory entry in STM.

        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store
            priority: Priority level for this operation

        Returns:
            True if the operation succeeded, False otherwise
        """
        memory_id = memory_entry.get("memory_id")
        if not memory_id:
            logger.error("Cannot store memory entry without memory_id")
            return False

        # Use store_with_retry for resilient storage
        return self.redis.store_with_retry(
            agent_id=agent_id,
            state_data=memory_entry,
            store_func=self._store_memory_entry,
            priority=priority,
        )

    def _store_memory_entry(self, agent_id: str, memory_entry: MemoryEntry) -> bool:
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
            # Store the full memory entry
            key = self._get_memory_key(agent_id, memory_id)
            self.redis.set(key, json.dumps(memory_entry), ex=self.config.ttl)

            # Add to agent's memory list
            agent_memories_key = self._get_agent_memories_key(agent_id)
            self.redis.zadd(agent_memories_key, {memory_id: timestamp})

            # Set TTL on the sorted set
            self.redis.expire(agent_memories_key, self.config.ttl)

            # Add to timeline index for chronological retrieval
            timeline_key = self._get_timeline_key(agent_id)
            self.redis.zadd(timeline_key, {memory_id: timestamp})
            self.redis.expire(timeline_key, self.config.ttl)

            # Add to importance index for importance-based retrieval
            importance = memory_entry.get("metadata", {}).get("importance_score", 0.0)
            importance_key = self._get_importance_key(agent_id)
            self.redis.zadd(importance_key, {memory_id: importance})
            self.redis.expire(importance_key, self.config.ttl)

            # If it has embeddings, store for vector search
            embeddings = memory_entry.get("embeddings", {})
            if embeddings and "full_vector" in embeddings:
                vector_key = self._get_vector_key(agent_id, memory_id)
                self.redis.set(
                    vector_key,
                    json.dumps(embeddings["full_vector"]),
                    ex=self.config.ttl,
                )

            logger.debug("Stored memory %s for agent %s in STM", memory_id, agent_id)
            return True

        except (RedisUnavailableError, RedisTimeoutError) as e:
            # These exceptions are caught by store_with_retry
            raise e
        except Exception as e:
            logger.error(
                "Unexpected error storing memory %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return False

    def get(self, agent_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to retrieve

        Returns:
            Memory entry or None if not found
        """
        try:
            # Construct the key using agent_id and memory_id
            key = self._get_memory_key(agent_id, memory_id)
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
                "Failed to retrieve memory %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return None

    def _update_access_metadata(
        self, agent_id: str, memory_id: str, memory_entry: MemoryEntry
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

            # Store updated entry
            key = self._get_memory_key(agent_id, memory_id)
            self.redis.set(key, json.dumps(memory_entry), ex=self.config.ttl)

            # Update importance based on access patterns
            # Increase importance for frequently accessed memories
            if retrieval_count > 1:
                importance = metadata.get("importance_score", 0.0)
                access_factor = min(retrieval_count / 10.0, 1.0)  # Cap at 1.0
                new_importance = importance + (access_factor * 0.1)  # Slight boost

                importance_key = self._get_importance_key(agent_id)
                self.redis.zadd(importance_key, {memory_id: new_importance})

                # Update in memory entry
                metadata["importance_score"] = new_importance

        except Exception as e:
            # Non-critical operation, just log
            logger.warning(
                "Failed to update access metadata for memory %s: %s", memory_id, str(e)
            )

    def get_by_timerange(
        self, agent_id: str, start_time: float, end_time: float, limit: int = 100
    ) -> List[MemoryEntry]:
        """Retrieve memories within a time range.

        Args:
            agent_id: ID of the agent
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            limit: Maximum number of results

        Returns:
            List of memory entries
        """
        try:
            timeline_key = self._get_timeline_key(agent_id)
            memory_ids = self.redis.zrangebyscore(
                timeline_key, min=start_time, max=end_time, start=0, num=limit
            )

            results = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    results.append(memory)

            return results

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memories by timerange for agent %s: %s",
                agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by timerange: %s", str(e)
            )
            return []

    def get_by_importance(
        self,
        agent_id: str,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """Retrieve memories by importance score.

        Args:
            agent_id: ID of the agent
            min_importance: Minimum importance score (inclusive)
            max_importance: Maximum importance score (inclusive)
            limit: Maximum number of results

        Returns:
            List of memory entries
        """
        try:
            importance_key = self._get_importance_key(agent_id)
            # Get memory IDs in range with their scores
            memory_id_scores = self.redis.zrangebyscore(
                importance_key, min=min_importance, max=max_importance, withscores=True
            )

            # Sort by score in descending order (higher importance first)
            memory_id_scores = sorted(
                memory_id_scores, key=lambda x: x[1], reverse=True
            )

            # Limit the number of results
            memory_id_scores = memory_id_scores[:limit]

            # Get just the memory IDs (without scores)
            memory_ids = [item[0] for item in memory_id_scores]

            results = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    results.append(memory)

            return results

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memories by importance for agent %s: %s",
                agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by importance: %s", str(e)
            )
            return []

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to delete

        Returns:
            True if the memory was deleted, False otherwise
        """
        try:
            # Delete the memory entry
            key = self._get_memory_key(agent_id, memory_id)
            deleted = self.redis.delete(key) > 0

            if deleted:
                # Remove from indexes
                self.redis.zrem(self._get_agent_memories_key(agent_id), memory_id)
                self.redis.zrem(self._get_timeline_key(agent_id), memory_id)
                self.redis.zrem(self._get_importance_key(agent_id), memory_id)

                # Remove vector if it exists
                self.redis.delete(self._get_vector_key(agent_id, memory_id))

                logger.debug(
                    "Deleted memory %s for agent %s from STM", memory_id, agent_id
                )

            return deleted

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to delete memory %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return False
        except Exception as e:
            logger.error("Unexpected error deleting memory %s: %s", memory_id, str(e))
            return False

    def count(self, agent_id: str) -> int:
        """Get the number of memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of memories
        """
        try:
            key = self._get_agent_memories_key(agent_id)
            return self.redis.zcard(key)

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to count memories for agent %s: %s", agent_id, str(e)
            )
            return 0
        except Exception as e:
            logger.error("Unexpected error counting memories: %s", str(e))
            return 0

    def clear(self, agent_id: str) -> bool:
        """Clear all memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all memory IDs
            key = self._get_agent_memories_key(agent_id)
            memory_ids = self.redis.zrange(key, 0, -1)

            # Delete each memory
            for memory_id in memory_ids:
                self.delete(agent_id, memory_id)

            # Delete indexes
            self.redis.delete(
                self._get_agent_memories_key(agent_id),
                self._get_timeline_key(agent_id),
                self._get_importance_key(agent_id),
            )

            logger.info(
                "Cleared all %d memories for agent %s from STM",
                len(memory_ids),
                agent_id,
            )
            return True

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to clear memories for agent %s: %s", agent_id, str(e)
            )
            return False
        except Exception as e:
            logger.error("Unexpected error clearing memories: %s", str(e))
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
                "client": "redis-stm",
            }
        except (RedisTimeoutError, RedisUnavailableError, Exception) as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "error": str(e),
                "client": "redis-stm",
            }

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

            # Get memory size by dumping each key
            total_size = 0
            for key in memory_keys:
                try:
                    # Get the memory entry JSON size
                    value = self.redis.get(key)
                    if value:
                        total_size += len(value)
                except Exception as e:
                    # Skip keys that cause errors (wrong type, etc.)
                    logger.debug("Skipping key %s: %s", key, e)
                    continue

            return total_size
        except Exception as e:
            logger.error("Error calculating memory size: %s", e)
            return 0

    def get_all(self, agent_id: str, limit: int = 1000) -> List[MemoryEntry]:
        """Get all memories for an agent.

        Args:
            agent_id: ID of the agent
            limit: Maximum number of memories to return

        Returns:
            List of memory entries
        """
        try:
            # Get all memory IDs sorted by recency
            memories_key = self._get_agent_memories_key(agent_id)
            memory_ids = self.redis.zrange(
                memories_key, 0, limit - 1, desc=True  # Most recent first
            )

            # Get each memory
            results = []
            for memory_id in memory_ids:
                # Handle memory_id if it's bytes
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")

                memory = self.get(agent_id, memory_id)
                if memory:
                    results.append(memory)

            return results
        except Exception as e:
            logger.error("Error retrieving all memories: %s", e)
            return []

    def search_similar(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
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
    ) -> List[MemoryEntry]:
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
        self, memory: MemoryEntry, attributes: Dict[str, Any]
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
    ) -> List[MemoryEntry]:
        """Search for memories within a specific step range.

        Args:
            agent_id: ID of the agent
            start_step: Start of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries within the step range
        """
        # Get all memories for the agent
        memories = self.get_all(agent_id)

        # Filter by step range
        results = []
        for memory in memories:
            step = memory.get("step_number")
            if step is not None and start_step <= step <= end_step:
                if (
                    memory_type is None
                    or memory.get("metadata", {}).get("memory_type") == memory_type
                ):
                    results.append(memory)

        return results

    def search_by_content(
        self,
        agent_id: str,
        content_query: Union[str, Dict[str, Any]],
        k: int = 5,
    ) -> List[MemoryEntry]:
        """Search for memories based on content text/attributes.

        Args:
            agent_id: ID of the agent
            content_query: String or dict to search for in memory contents
            k: Maximum number of results to return

        Returns:
            List of memory entries matching the content query
        """
        # Get all memories for the agent
        memories = self.get_all(agent_id)
        results = []

        # Process string query
        query_text = ""
        if isinstance(content_query, str):
            query_text = content_query.lower()
        elif isinstance(content_query, dict) and "text" in content_query:
            query_text = content_query["text"].lower()

        for memory in memories:
            relevance_score = 0.0

            # Get memory content as string for text search
            memory_content = json.dumps(memory.get("content", {})).lower()

            # Simple relevance scoring - if query text is in content
            if query_text and query_text in memory_content:
                # More specific matches get higher scores
                relevance_score = len(query_text) / len(memory_content) * 10

                # Bonus for exact matches
                if query_text == memory_content:
                    relevance_score += 1.0

                # Add relevance score to memory
                memory["relevance_score"] = relevance_score
                results.append(memory)

            # If query is a dict, also match by specific attributes
            elif isinstance(content_query, dict) and not query_text:
                for key, value in content_query.items():
                    if key == "text":
                        continue  # Already handled above

                    content = memory.get("content", {})
                    if isinstance(content, dict):
                        # Direct attribute match
                        if key in content and content[key] == value:
                            relevance_score += 0.5

                        # Nested attribute match (only one level deep for simplicity)
                        for content_key, content_value in content.items():
                            if isinstance(content_value, dict) and key in content_value:
                                if content_value[key] == value:
                                    relevance_score += 0.3

                if relevance_score > 0:
                    memory["relevance_score"] = relevance_score
                    results.append(memory)

        # Sort by relevance and limit results
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:k]
