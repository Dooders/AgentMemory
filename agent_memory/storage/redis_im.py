"""Redis-based Intermediate Memory (IM) storage for agent memory system.

This module provides a Redis-based implementation of the Intermediate Memory
storage tier with TTL-based expiration and level 1 compression.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import redis

from agent_memory.config import RedisIMConfig
from agent_memory.storage.models import IMMemoryEntry
from agent_memory.storage.redis_base import RedisMemoryStore
from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    CircuitOpenError,
    MemoryError,
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)

logger = logging.getLogger(__name__)


# TypedDict definitions for complex structures
class MemoryMetadata(TypedDict, total=False):
    """Metadata information for a memory entry."""

    compression_level: int
    importance_score: float
    retrieval_count: int
    creation_time: float
    last_access_time: float


class MemoryEntry(TypedDict, total=False):
    """Structure of a memory entry in the intermediate memory store."""

    memory_id: str
    agent_id: str
    timestamp: float
    content: Any  # Can be any structured data
    metadata: MemoryMetadata
    embedding: List[float]  # Vector embedding for similarity search
    memory_type: Optional[str]  # Optional type classification
    step_number: Optional[int]  # Optional step number for step-based retrieval


class RedisIMStore(RedisMemoryStore[IMMemoryEntry]):
    """Redis-based storage for Intermediate Memory (IM).

    This class provides storage operations for the medium-resolution,
    intermediate-lived agent memory tier using Redis as the backing store.

    Attributes:
        config: Configuration for IM Redis storage
        redis: Resilient Redis client instance

    Raises:
        RedisUnavailableError: If Redis is unavailable
        RedisTimeoutError: If operation times out
        RedisError: If Redis returns an error
    """

    def __init__(self, agent_id: str, config: RedisIMConfig):
        """Initialize the Redis IM store.

        Args:
            agent_id: ID of the agent
            config: Configuration for IM Redis storage
        """
        # Create resilient Redis client
        redis_client = ResilientRedisClient(
            client_name="im",
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )
        
        # Initialize the base class
        super().__init__(
            store_type="IM",
            agent_id=agent_id,
            redis_client=redis_client,
            key_prefix=config.namespace
        )
        
        # Store the config for IM-specific settings
        self.config = config
        
        # Check if Redis vector search capabilities are available
        self._vector_search_available = self._check_vector_search_available()
        if self._vector_search_available:
            logger.info("Redis vector search capabilities detected")
            # Create vector index if it doesn't exist
            self._create_vector_index()
        else:
            logger.info(
                "Redis vector search capabilities not detected, using fallback method"
            )

        # Check if Lua scripting is fully supported
        self._lua_scripting_available = self._check_lua_scripting()
        if self._lua_scripting_available:
            logger.info("Redis Lua scripting fully supported")
        else:
            logger.info(
                "Redis Lua scripting not fully supported, using fallback pipeline methods"
            )
        
        logger.info(f"Initialized RedisIMStore for agent {agent_id}")

    # Helper methods for key construction
    def _get_memory_key(self, memory_id: str) -> str:
        """Construct the Redis key for a memory entry.

        Args:
            memory_id: ID of the memory

        Returns:
            Redis key string
        """
        return f"{self.key_prefix}:{self.agent_id}:memory:{memory_id}"

    def _get_timestamp_key(self) -> str:
        """Get the Redis key for timestamp index.
        
        Returns:
            Redis key for timestamp sorted set
        """
        return f"{self.key_prefix}:{self.agent_id}:timeline"

    def _get_importance_key(self) -> str:
        """Get the Redis key for importance index.
        
        Returns:
            Redis key for importance score sorted set
        """
        return f"{self.key_prefix}:{self.agent_id}:importance"

    def _get_memory_type_key(self, memory_type: str) -> str:
        """Get the Redis key for memory type index.
        
        Args:
            memory_type: Type of memory
            
        Returns:
            Redis key for memory type set
        """
        return f"{self.key_prefix}:{self.agent_id}:type:{memory_type}"

    def _get_step_key(self) -> str:
        """Get the Redis key for step index.
        
        Returns:
            Redis key for step sorted set
        """
        return f"{self.key_prefix}:{self.agent_id}:step_idx"

    def _get_vector_key(self, memory_id: str) -> str:
        """Get the Redis key for a memory's vector embedding.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Redis key for the vector embedding
        """
        return f"{self.key_prefix}:{self.agent_id}:vector:{memory_id}"

    def _check_vector_search_available(self) -> bool:
        """Check if Redis vector search capabilities are available.

        Returns:
            True if vector search is available, False otherwise
        """
        try:
            # Check for Redis Stack / RediSearch module
            modules = self.client.execute_command("MODULE LIST")
            return any(module[1] == b"search" for module in modules)
        except Exception as e:
            logger.warning(f"Failed to check Redis modules: {e}")
            return False

    def _create_vector_index(self) -> None:
        """Create vector index for embeddings if it doesn't exist."""
        try:
            # Define index key name
            index_name = f"{self.key_prefix}_vector_idx"

            # Check if index already exists
            try:
                self.client.execute_command(f"FT.INFO {index_name}")
                logger.info(f"Vector index {index_name} already exists")
                return
            except redis.ResponseError as e:
                # If index doesn't exist, we'll create it
                if "unknown index name" not in str(e).lower():
                    logger.error(f"Error checking vector index: {str(e)}")
                    return

            # Define vector dimension based on config
            vector_dim = self.config.embedding_dim

            # Create index
            create_cmd = [
                "FT.CREATE", index_name,
                "ON", "HASH",
                "PREFIX", 1, f"{self.key_prefix}:{self.agent_id}:memory:",
                "SCHEMA",
                "vector", "VECTOR", "FLOAT32", vector_dim, "DISTANCE_METRIC", "COSINE"
            ]

            self.client.execute_command(*create_cmd)
            logger.info(f"Created vector index {index_name}")

        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            self._vector_search_available = False

    def _check_lua_scripting(self) -> bool:
        """Check if Lua scripting is fully supported.

        Returns:
            True if Lua scripting is fully supported, False otherwise
        """
        try:
            # Try a simple Lua script
            self.client.eval("return 1", 0)
            return True
        except Exception as e:
            logger.warning(f"Lua scripting not fully supported: {e}")
            return False
    
    def store(self, memory_entry: IMMemoryEntry, priority: Priority = Priority.NORMAL) -> bool:
        """Store a memory entry in IM.

        Args:
            memory_entry: Memory entry to store
            priority: Priority level for this operation

        Returns:
            True if the operation succeeded, False otherwise
            
        Raises:
            MemoryError: If the store operation fails
        """
        try:
            memory_id = memory_entry.get("memory_id")
            if not memory_id:
                logger.error("Cannot store memory entry without memory_id")
                return False
            
            # Get Redis keys
            memory_key = self._get_memory_key(memory_id)
            timestamp_key = self._get_timestamp_key()
            importance_key = self._get_importance_key()
            
            # Set defaults and ensure metadata exists
            if "metadata" not in memory_entry:
                memory_entry["metadata"] = {}
            
            # IM-specific defaults
            memory_entry["metadata"]["compression_level"] = 1  # Level 1 compression for IM
            
            # Ensure we have creation time
            if "creation_time" not in memory_entry["metadata"]:
                memory_entry["metadata"]["creation_time"] = float(time.time())
            
            # Set default access time if not present
            if "last_access_time" not in memory_entry["metadata"]:
                memory_entry["metadata"]["last_access_time"] = float(time.time())
            
            # Set default retrieval count if not present
            if "retrieval_count" not in memory_entry["metadata"]:
                memory_entry["metadata"]["retrieval_count"] = 0
            
            # Serialize memory entry
            serialized_memory = self._serialize_memory(memory_entry)
            
            # Get importance score (default to 0.5 if not present)
            importance_score = memory_entry["metadata"].get("importance_score", 0.5)
            
            # Calculate TTL based on importance
            ttl = self._calculate_ttl(importance_score)
            
            # Use different storage methods depending on capabilities
            if self._vector_search_available and "embeddings" in memory_entry:
                return self._store_with_vector_search(memory_entry, memory_key, ttl)
            else:
                return self._store_with_standard_redis(memory_entry, memory_key, ttl)
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory storage: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory storage: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
    
    def _store_with_vector_search(self, memory_entry: IMMemoryEntry, memory_key: str, ttl: int) -> bool:
        """Store a memory entry using Redis vector search.
        
        Args:
            memory_entry: Memory entry to store
            memory_key: Redis key for the memory entry
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory_id = memory_entry["memory_id"]
            timestamp_key = self._get_timestamp_key()
            importance_key = self._get_importance_key()
            
            # Create hash fields
            hash_fields = {}
            
            # Add basic fields
            hash_fields["memory_id"] = memory_id
            hash_fields["agent_id"] = self.agent_id
            hash_fields["timestamp"] = str(memory_entry.get("timestamp", time.time()))
            
            # Add content as JSON
            hash_fields["content"] = json.dumps(memory_entry.get("content", {}))
            
            # Add metadata as JSON
            hash_fields["metadata"] = json.dumps(memory_entry.get("metadata", {}))
            
            # Add memory type if present
            if memory_entry.get("memory_type"):
                hash_fields["memory_type"] = memory_entry["memory_type"]
            
            # Add step number if present
            if memory_entry.get("step_number") is not None:
                hash_fields["step_number"] = str(memory_entry["step_number"])
            
            # Add vector if present
            if "embeddings" in memory_entry and "compressed_vector" in memory_entry["embeddings"]:
                # Convert to bytes for Redis
                vector = memory_entry["embeddings"]["compressed_vector"]
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                hash_fields["vector"] = vector_bytes
            
            # Create pipeline
            pipeline = self.client.pipeline()
            
            # Store hash
            pipeline.hset(memory_key, mapping=hash_fields)
            pipeline.expire(memory_key, ttl)
            
            # Update indices
            timestamp = float(hash_fields["timestamp"])
            pipeline.zadd(timestamp_key, {memory_id: timestamp})
            
            # Add to importance index
            importance_score = memory_entry["metadata"].get("importance_score", 0.5)
            pipeline.zadd(importance_key, {memory_id: importance_score})
            
            # Add to memory type index if present
            if memory_entry.get("memory_type"):
                type_key = self._get_memory_type_key(memory_entry["memory_type"])
                pipeline.sadd(type_key, memory_id)
            
            # Add to step index if present
            if memory_entry.get("step_number") is not None:
                step_key = self._get_step_key()
                pipeline.zadd(step_key, {memory_id: memory_entry["step_number"]})
            
            # Execute pipeline
            pipeline.execute()
            
            logger.debug(f"Stored memory {memory_id} in IM with vector search")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory with vector search: {str(e)}")
            # Fall back to standard storage
            return self._store_with_standard_redis(memory_entry, memory_key, ttl)
    
    def _store_with_standard_redis(self, memory_entry: IMMemoryEntry, memory_key: str, ttl: int) -> bool:
        """Store a memory entry using standard Redis operations.
        
        Args:
            memory_entry: Memory entry to store
            memory_key: Redis key for the memory entry
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory_id = memory_entry["memory_id"]
            timestamp_key = self._get_timestamp_key()
            importance_key = self._get_importance_key()
            
            # Serialize memory entry
            serialized_memory = self._serialize_memory(memory_entry)
            
            # Create pipeline
            pipeline = self.client.pipeline()
            
            # Store memory
            pipeline.set(memory_key, serialized_memory, ex=ttl)
            
            # Update timestamp index
            timestamp = memory_entry.get("timestamp", time.time())
            pipeline.zadd(timestamp_key, {memory_id: timestamp})
            
            # Update importance index
            importance_score = memory_entry["metadata"].get("importance_score", 0.5)
            pipeline.zadd(importance_key, {memory_id: importance_score})
            
            # Add to memory type index if present
            if memory_entry.get("memory_type"):
                type_key = self._get_memory_type_key(memory_entry["memory_type"])
                pipeline.sadd(type_key, memory_id)
            
            # Add to step index if present
            if memory_entry.get("step_number") is not None:
                step_key = self._get_step_key()
                pipeline.zadd(step_key, {memory_id: memory_entry["step_number"]})
            
            # Store vector embedding if present
            if "embeddings" in memory_entry and "compressed_vector" in memory_entry["embeddings"]:
                vector_key = self._get_vector_key(memory_id)
                vector = memory_entry["embeddings"]["compressed_vector"]
                vector_json = json.dumps(vector)
                pipeline.set(vector_key, vector_json, ex=ttl)
            
            # Execute pipeline
            pipeline.execute()
            
            logger.debug(f"Stored memory {memory_id} in IM with standard Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory with standard Redis: {str(e)}")
            raise
    
    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[IMMemoryEntry]:
        """Search memories by vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries ordered by similarity
            
        Raises:
            MemoryError: If the search fails
        """
        try:
            # Use vector search if available, otherwise fall back to Python implementation
            if self._vector_search_available:
                return self._search_similar_redis_vector(
                    query_embedding, 
                    limit=k, 
                    memory_type=memory_type
                )
            else:
                return self._search_similar_python(
                    query_embedding, 
                    k=k, 
                    memory_type=memory_type
                )
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
    
    def _search_similar_redis_vector(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        memory_type: Optional[str] = None
    ) -> List[IMMemoryEntry]:
        """Search for similar vectors using Redis vector search.
        
        Args:
            query_embedding: Query vector embedding
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries ordered by similarity
        """
        try:
            # Prepare query embedding
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build query
            index_name = f"{self.key_prefix}_vector_idx"
            base_query = f"*=>[KNN {limit} @vector $query_vector AS score"
            
            # Add type filter if requested
            if memory_type:
                base_query += f" @memory_type:{memory_type}"
            
            base_query += "]"
            
            # Execute vector search
            result = self.client.execute_command(
                "FT.SEARCH",
                index_name,
                base_query,
                "PARAMS", "2",
                "query_vector", query_vector,
                "DIALECT", "2",
                "LIMIT", "0", str(limit),
                "SORTBY", "score", "DESC",
            )
            
            # Process results
            if not result or result[0] == 0:  # No results
                return []
            
            # Extract matching entries
            memories = []
            for i in range(1, len(result), 2):  # Skip count and iterate through document-properties pairs
                key = result[i]  # Document ID (memory key)
                memory_id = key.split(":")[-1]  # Extract memory_id from key
                
                # Get complete memory
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.warning(f"Redis vector search failed: {str(e)}")
            # Fall back to Python implementation
            return self._search_similar_python(query_embedding, limit, memory_type)
    
    def _search_similar_python(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[IMMemoryEntry]:
        """Search for similar vectors using Python implementation.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries ordered by similarity
        """
        try:
            # Get all memories of the specified type (or all memories)
            if memory_type:
                type_key = self._get_memory_type_key(memory_type)
                memory_ids = self.client.smembers(type_key)
            else:
                timestamp_key = self._get_timestamp_key()
                memory_ids = self.client.zrange(timestamp_key, 0, -1)
            
            if not memory_ids:
                return []
            
            # Calculate similarity for each memory
            memories_with_scores = []
            
            for memory_id in memory_ids:
                # Get vector embedding
                vector_key = self._get_vector_key(memory_id)
                vector_json = self.client.get(vector_key)
                
                if not vector_json:
                    continue
                
                try:
                    vector = json.loads(vector_json)
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, vector)
                    
                    # Get the memory
                    memory = self.get(memory_id)
                    if memory:
                        memories_with_scores.append((memory, similarity))
                except Exception as e:
                    logger.warning(f"Error processing vector for memory {memory_id}: {str(e)}")
                    continue
            
            # Sort by similarity (descending)
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return [memory for memory, _ in memories_with_scores[:k]]
            
        except Exception as e:
            logger.error(f"Error in Python similarity search: {str(e)}")
            raise
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")
        
        # Convert to numpy arrays for efficient calculation
        a_array = np.array(a)
        b_array = np.array(b)
        
        # Calculate dot product
        dot_product = np.dot(a_array, b_array)
        
        # Calculate magnitudes
        a_magnitude = np.linalg.norm(a_array)
        b_magnitude = np.linalg.norm(b_array)
        
        # Calculate similarity
        if a_magnitude == 0 or b_magnitude == 0:
            return 0.0
        
        similarity = dot_product / (a_magnitude * b_magnitude)
        
        # Handle numerical errors that might push similarity outside [-1, 1]
        return max(-1.0, min(1.0, similarity))
    
    def _calculate_ttl(self, importance_score: float) -> int:
        """Calculate TTL based on importance score.
        
        Higher importance = longer TTL.
        
        Args:
            importance_score: Importance score (0.0-1.0)
            
        Returns:
            TTL in seconds
        """
        # Base TTL from config
        base_ttl = self.config.ttl
        
        # Scale TTL by importance (max 3x base TTL)
        ttl = int(base_ttl * (1 + 2 * importance_score))
        
        return ttl
