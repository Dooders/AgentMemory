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

        # Check for Redis vector search capabilities
        self._vector_search_available = self._check_vector_search_available()
        if self._vector_search_available:
            logger.info("Redis vector search capabilities detected")
            # Create vector index if it doesn't exist
            self._create_vector_index()
        else:
            logger.info("Redis vector search capabilities not detected, using fallback method")

        logger.info("Initialized RedisIMStore with namespace %s", self._key_prefix)

    def _check_vector_search_available(self) -> bool:
        """Check if Redis vector search capabilities are available.

        Returns:
            True if vector search is available, False otherwise
        """
        try:
            # Check for Redis Stack / RediSearch module
            modules = self.redis.execute_command("MODULE LIST")
            return any(module[1] == b'search' for module in modules)
        except Exception as e:
            logger.warning(f"Failed to check Redis modules: {e}")
            return False

    def _create_vector_index(self) -> None:
        """Create vector index for embeddings if it doesn't exist."""
        try:
            # Define index key name
            index_name = f"{self._key_prefix}_vector_idx"
            
            # Check if index already exists
            try:
                self.redis.execute_command(f"FT.INFO {index_name}")
                logger.info(f"Vector index {index_name} already exists")
                return
            except Exception:
                # Index doesn't exist, continue to create it
                pass
                
            # Create vector index for agent memory embeddings
            # Using FLAT index for simplicity but can be changed to HNSW for better performance
            create_cmd = [
                "FT.CREATE", index_name,
                "ON", "JSON",
                "PREFIX", 1, f"{self._key_prefix}:",
                "SCHEMA",
                "$.embedding", "AS", "embedding", "VECTOR", "FLAT", 
                "6", "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE",
                # Add additional fields for attribute and step-range searches
                "$.memory_type", "AS", "memory_type", "TAG",
                "$.step_number", "AS", "step_number", "NUMERIC",
                "$.content.*", "AS", "content", "TEXT",
                "$.metadata.importance_score", "AS", "importance_score", "NUMERIC"
            ]
            
            self.redis.execute_command(*create_cmd)
            logger.info(f"Created vector index {index_name}")
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

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
            Dictionary containing health metrics for monitoring dashboards
        """
        health_data = {
            "client": "redis-im",
            "timestamp": time.time(),
            "metrics": {},
        }
        
        try:
            # Basic connectivity check
            ping_start = time.time()
            ping_result = self.redis.ping()
            ping_latency = (time.time() - ping_start) * 1000  # Convert to ms
            
            # Get Redis info for key metrics
            info = self.redis.info()
            
            # Memory metrics
            memory_metrics = {
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "N/A"),
                "used_memory_rss_human": info.get("used_memory_rss_human", "N/A"),
                "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
            }
            
            # Performance metrics
            performance_metrics = {
                "connected_clients": info.get("connected_clients", 0),
                "connected_slaves": info.get("connected_slaves", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
                "latency_ms": ping_latency,
                "circuit_breaker_open": self.redis.is_circuit_open(),
            }
            
            # Get keyspace stats for our database
            db_key = f"db{self.config.db}"
            keyspace_metrics = {}
            if db_key in info:
                keyspace_metrics = info[db_key]
            
            # IM specific metrics
            im_metrics = {
                "vector_search_available": self._vector_search_available,
            }
            
            # Add all metrics to response
            health_data["metrics"] = {
                "memory": memory_metrics,
                "performance": performance_metrics,
                "keyspace": keyspace_metrics,
                "im": im_metrics,
            }
            
            # Set overall status
            health_data["status"] = "healthy" if ping_result else "degraded"
            health_data["message"] = (
                "Redis connection successful"
                if ping_result
                else "Redis ping failed"
            )
            
            return health_data
            
        except (RedisTimeoutError, RedisUnavailableError, Exception) as e:
            health_data.update({
                "status": "unhealthy",
                "message": str(e),
                "error": str(e),
            })
            return health_data
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate Redis cache hit rate from info statistics.
        
        Args:
            info: Redis INFO command result dictionary
            
        Returns:
            Cache hit rate as a percentage (0-100)
        """
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        
        if hits + misses == 0:
            return 0.0
            
        return (hits / (hits + misses)) * 100
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for integration with monitoring dashboards.
        
        This method provides a standard format that can be used with monitoring tools
        like Prometheus, Datadog, etc. through appropriate exporters.
        
        Returns:
            Dictionary containing detailed metrics in a format suitable for exporters
        """
        # Get basic health data first
        monitoring_data = self.check_health()
        
        try:
            # Add additional metrics specific for monitoring systems
            
            # Memory usage for this agent namespace
            namespace_pattern = f"{self._key_prefix}:*"
            keys_count = len(list(self.redis.scan_iter(match=namespace_pattern, count=1000)))
            
            # Memory statistics by key type (if Redis >=4.0)
            memory_stats = {}
            if self.redis.info().get("redis_version", "0.0.0") >= "4.0.0":
                try:
                    memory_stats = self.redis.execute_command("MEMORY STATS")
                except Exception as e:
                    logger.warning(f"Failed to get memory stats: {e}")
            
            # Add to monitoring data
            monitoring_data["metrics"]["namespace"] = {
                "keys_count": keys_count,
                "memory_stats": memory_stats,
            }
            
            # Add server-related metrics that might be useful for monitoring
            monitoring_data["metrics"]["server"] = {
                "uptime_in_seconds": self.redis.info().get("uptime_in_seconds", 0),
                "redis_version": self.redis.info().get("redis_version", "unknown"),
            }
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Error getting monitoring data: {e}")
            monitoring_data["metrics"]["error"] = str(e)
            return monitoring_data

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
        if self._vector_search_available:
            try:
                return self._search_similar_redis_vector(agent_id, query_embedding, k, memory_type)
            except Exception as e:
                logger.warning(f"Redis vector search failed, falling back to Python implementation: {e}")
                # Fall back to Python implementation
                return self._search_similar_python(agent_id, query_embedding, k, memory_type)
        else:
            # Use Python implementation
            return self._search_similar_python(agent_id, query_embedding, k, memory_type)

    def _search_similar_redis_vector(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using Redis vector search.

        Args:
            agent_id: Unique identifier for the agent
            query_embedding: The vector embedding to use for similarity search
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity score
        """
        index_name = f"{self._key_prefix}_vector_idx"
        
        # Prepare the query
        query = f"@embedding:[VECTOR_RANGE $K $vec]=>{{}}"
        
        # Add memory_type filter if specified
        filter_args = []
        if memory_type:
            query = f"@memory_type:{{{memory_type}}} {query}"
        
        # Prepare the embedding for Redis
        vector_str = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Execute the search
        results = self.redis.execute_command(
            "FT.SEARCH", index_name, 
            query,
            "PARAMS", 2, "K", k, "vec", vector_str,
            "RETURN", 1, ".",
            "LIMIT", 0, k
        )
        
        if not results or results[0] == 0:
            return []
            
        # Parse results (format depends on Redis version)
        memories = []
        for i in range(1, len(results), 2):
            key = results[i]
            if isinstance(key, bytes):
                key = key.decode('utf-8')
                
            # Extract memory data
            data = self.redis.get(key)
            if data:
                memory_entry = json.loads(data)
                # Extract agent_id and memory_id from the key
                parts = key.split(":")
                if len(parts) >= 4 and parts[-2] == "memory":
                    memory_id = parts[-1]
                    self._update_access_metadata(agent_id, memory_id, memory_entry)
                    # Add similarity score
                    if i+1 < len(results) and isinstance(results[i+1], list):
                        for field in results[i+1]:
                            if isinstance(field, list) and len(field) >= 2 and field[0] == b'__vector_score':
                                memory_entry["similarity_score"] = float(field[1])
                    memories.append(memory_entry)
                    
        return memories

    def _search_similar_python(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories with similar embeddings using Python implementation.

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
        if self._vector_search_available:
            try:
                return self._search_by_attributes_redis(agent_id, attributes, memory_type)
            except Exception as e:
                logger.warning(f"Redis attribute search failed, falling back to Python implementation: {e}")
                # Fall back to Python implementation
                return self._search_by_attributes_python(agent_id, attributes, memory_type)
        else:
            # Use Python implementation
            return self._search_by_attributes_python(agent_id, attributes, memory_type)

    def _search_by_attributes_redis(
        self, 
        agent_id: str, 
        attributes: Dict[str, Any], 
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for memories by attributes using Redis search.
        
        Args:
            agent_id: Unique identifier for the agent
            attributes: Dictionary of attribute keys and values to match
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries with matching attributes
        """
        index_name = f"{self._key_prefix}_vector_idx"
        
        # Build query for Redis search
        query_parts = []
        
        # Add agent ID filter - we'll filter by pattern prefix
        agent_prefix = f"{self._key_prefix}:{agent_id}:memory:"
        
        # Add memory type filter if specified
        if memory_type:
            query_parts.append(f"@memory_type:{{{memory_type}}}")
        
        # Add attribute filters
        for attr_path, attr_value in attributes.items():
            # Convert the attribute path to the correct format for RediSearch
            redis_path = f"@content.{attr_path}"
            
            if isinstance(attr_value, (int, float)):
                # Numeric value
                query_parts.append(f"{redis_path}:[{attr_value} {attr_value}]")
            elif isinstance(attr_value, str):
                # String value - escape any special characters
                escaped_value = attr_value.replace('"', '\\"')
                query_parts.append(f'{redis_path}:"{escaped_value}"')
            elif isinstance(attr_value, bool):
                # Boolean value
                query_parts.append(f'{redis_path}:{str(attr_value).lower()}')
        
        # Combine query parts
        query = " ".join(query_parts) if query_parts else "*"
        
        try:
            # Execute search with prefix filter for agent_id
            results = self.redis.execute_command(
                "FT.SEARCH", index_name,
                query,
                "LIMIT", 0, 1000,  # Reasonable limit
                "FILTER", "PREFLEN", len(agent_prefix),
                "PREFIX", 1, agent_prefix
            )
            
            if not results or results[0] == 0:
                return []
                
            # Process results
            memories = []
            for i in range(1, len(results), 2):
                key = results[i]
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                    
                # Extract memory data
                data = self.redis.get(key)
                if data:
                    memory_entry = json.loads(data)
                    # Extract memory_id from the key
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[-2] == "memory":
                        memory_id = parts[-1]
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)
                        
            return memories
            
        except Exception as e:
            logger.error(f"Error in Redis attribute search: {e}")
            raise
    
    def _search_by_attributes_python(
        self,
        agent_id: str,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories matching specific attributes using Python implementation.

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
        if self._vector_search_available:
            try:
                return self._search_by_step_range_redis(agent_id, start_step, end_step, memory_type)
            except Exception as e:
                logger.warning(f"Redis step range search failed, falling back to Python implementation: {e}")
                # Fall back to Python implementation
                return self._search_by_step_range_python(agent_id, start_step, end_step, memory_type)
        else:
            # Use Python implementation
            return self._search_by_step_range_python(agent_id, start_step, end_step, memory_type)

    def _search_by_step_range_redis(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories within a step range using Redis search.
        
        Args:
            agent_id: Unique identifier for the agent
            start_step: Beginning of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries within the step range
        """
        index_name = f"{self._key_prefix}_vector_idx"
        
        # Build query for Redis search
        query_parts = []
        
        # Add agent ID filter - we'll filter by pattern prefix
        agent_prefix = f"{self._key_prefix}:{agent_id}:memory:"
        
        # Add step range filter
        query_parts.append(f"@step_number:[{start_step} {end_step}]")
        
        # Add memory type filter if specified
        if memory_type:
            query_parts.append(f"@memory_type:{{{memory_type}}}")
        
        # Combine query parts
        query = " ".join(query_parts)
        
        try:
            # Execute search with prefix filter for agent_id
            results = self.redis.execute_command(
                "FT.SEARCH", index_name,
                query,
                "LIMIT", 0, 1000,  # Reasonable limit
                "SORTBY", "step_number", "ASC",
                "FILTER", "PREFLEN", len(agent_prefix),
                "PREFIX", 1, agent_prefix
            )
            
            if not results or results[0] == 0:
                return []
                
            # Process results
            memories = []
            for i in range(1, len(results), 2):
                key = results[i]
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                    
                # Extract memory data
                data = self.redis.get(key)
                if data:
                    memory_entry = json.loads(data)
                    # Extract memory_id from the key
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[-2] == "memory":
                        memory_id = parts[-1]
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)
                        
            return memories
            
        except Exception as e:
            logger.error(f"Error in Redis step range search: {e}")
            raise
    
    def _search_by_step_range_python(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories within a specific step range using Python implementation.

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
