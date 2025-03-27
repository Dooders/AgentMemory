"""Redis-based Short-Term Memory (STM) storage for agent memory system.

This module provides a Redis-based implementation of the Short-Term Memory
storage tier with comprehensive error handling and recovery mechanisms.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from agent_memory.config import RedisSTMConfig
from agent_memory.storage.models import MemoryEntry, STMMemoryEntry
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


class RedisSTMStore(RedisMemoryStore[STMMemoryEntry]):
    """Redis-based storage for Short-Term Memory (STM).

    This class provides storage operations for the high-resolution,
    short-lived agent memory tier using Redis as the backing store.

    Attributes:
        config: Configuration for STM Redis storage
        key_prefix: Prefix for Redis keys
        client: Redis client for database operations
    """

    def __init__(self, agent_id: str, config: RedisSTMConfig):
        """Initialize the Redis STM store.

        Args:
            agent_id: ID of the agent
            config: Configuration for STM Redis storage
        """
        # Create resilient Redis client
        redis_client = ResilientRedisClient(
            client_name="stm",
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
            store_type="STM",
            agent_id=agent_id,
            redis_client=redis_client,
            key_prefix=config.namespace
        )
        
        # Store the config for STM-specific settings
        self.config = config
        
        logger.info(f"Initialized RedisSTMStore for agent {agent_id}")
    
    def store(self, memory_entry: STMMemoryEntry, priority: Priority = Priority.NORMAL) -> bool:
        """Store a memory entry in STM.

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
            
            # STM-specific defaults
            memory_entry["metadata"]["compression_level"] = 0  # No compression for STM
            
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
            
            # Store vector embedding if present
            vector_key = None
            if "embeddings" in memory_entry and "full_vector" in memory_entry["embeddings"]:
                vector_key = self._get_vector_key(memory_id)
                vector = memory_entry["embeddings"]["full_vector"]
                vector_json = json.dumps(vector)
            
            # Create a pipeline for atomic operations
            pipeline = self.client.pipeline()
            
            # Store the memory with TTL based on importance
            ttl = self._calculate_ttl(importance_score)
            pipeline.set(memory_key, serialized_memory, ex=ttl)
            
            # Update indices
            timestamp = memory_entry.get("timestamp", float(time.time()))
            pipeline.zadd(timestamp_key, {memory_id: timestamp})
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
            if vector_key and "vector_json" in locals():
                pipeline.set(vector_key, vector_json, ex=ttl)
            
            # Execute all commands
            pipeline.execute()
            
            logger.debug(f"Stored memory {memory_id} in STM")
            return True
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory storage: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory storage: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
    
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
        
        # Scale TTL by importance (max 2x base TTL)
        ttl = int(base_ttl * (1 + importance_score))
        
        return ttl
    
    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[STMMemoryEntry]:
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
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
    
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
    
    def search_by_attributes(
        self,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None
    ) -> List[STMMemoryEntry]:
        """Search memories by attribute matching.
        
        Args:
            attributes: Attributes to match
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries matching the attributes
            
        Raises:
            MemoryError: If the search fails
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
            
            matching_memories = []
            
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if not memory:
                    continue
                
                # Check if memory matches attributes
                if self._matches_attributes(memory, attributes):
                    matching_memories.append(memory)
            
            return matching_memories
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during attribute search: {str(e)}")
            raise MemoryError(f"Failed to search by attributes: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during attribute search: {str(e)}")
            raise MemoryError(f"Failed to search by attributes: {str(e)}")
    
    def get_all(self, limit: int = 1000) -> List[STMMemoryEntry]:
        """Get all memories up to limit.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entries
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            timestamp_key = self._get_timestamp_key()
            
            # Get memory IDs sorted by timestamp (newest first)
            memory_ids = self.client.zrevrange(timestamp_key, 0, limit - 1)
            
            # Retrieve each memory
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during retrieval of all memories: {str(e)}")
            raise MemoryError(f"Failed to retrieve all memories: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during retrieval of all memories: {str(e)}")
            raise MemoryError(f"Failed to retrieve all memories: {str(e)}")
