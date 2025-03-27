"""Redis base class for memory store implementations.

This module provides a base class for Redis-based memory store implementations,
abstracting common Redis operations and patterns.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from agent_memory.storage.base import BaseMemoryStore
from agent_memory.storage.models import MemoryEntry
from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    CircuitOpenError,
    MemoryError,
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
)

logger = logging.getLogger(__name__)

# Type variable for memory entry
M = TypeVar('M', bound=Dict[str, Any])


class RedisMemoryStore(BaseMemoryStore[M]):
    """Base class for Redis-based memory stores.
    
    This class implements common Redis operations and patterns
    for memory storage, providing a foundation for STM and IM
    implementations.
    
    Attributes:
        store_type: Type of memory store (STM, IM)
        agent_id: ID of the agent
        client: Redis client for database operations
        key_prefix: Prefix for Redis keys
    """
    
    def __init__(
        self,
        store_type: str,
        agent_id: str,
        redis_client: ResilientRedisClient,
        key_prefix: str
    ):
        """Initialize the Redis memory store.
        
        Args:
            store_type: Type of memory store (STM, IM)
            agent_id: ID of the agent
            redis_client: Redis client for database operations
            key_prefix: Prefix for Redis keys
        """
        super().__init__(store_type)
        self.agent_id = agent_id
        self.client = redis_client
        self.key_prefix = key_prefix
        
        logger.info(f"Initialized {store_type} Redis memory store for agent {agent_id}")
    
    def _get_memory_key(self, memory_id: str) -> str:
        """Get the Redis key for a memory entry.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Redis key for the memory entry
        """
        return f"{self.key_prefix}:{self.agent_id}:memory:{memory_id}"
    
    def _get_timestamp_key(self) -> str:
        """Get the Redis key for timestamp index.
        
        Returns:
            Redis key for timestamp sorted set
        """
        return f"{self.key_prefix}:{self.agent_id}:timestamp_idx"
    
    def _get_importance_key(self) -> str:
        """Get the Redis key for importance index.
        
        Returns:
            Redis key for importance score sorted set
        """
        return f"{self.key_prefix}:{self.agent_id}:importance_idx"
    
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
    
    def _serialize_memory(self, memory_entry: M) -> str:
        """Serialize a memory entry to JSON string.
        
        Args:
            memory_entry: Memory entry to serialize
            
        Returns:
            JSON string representation of the memory entry
        
        Raises:
            MemoryError: If serialization fails
        """
        try:
            return json.dumps(memory_entry)
        except Exception as e:
            logger.error(f"Failed to serialize memory: {str(e)}")
            raise MemoryError(f"Failed to serialize memory: {str(e)}")
    
    def _deserialize_memory(self, serialized_memory: str) -> M:
        """Deserialize a JSON string to memory entry.
        
        Args:
            serialized_memory: JSON string to deserialize
            
        Returns:
            Memory entry deserialized from JSON
            
        Raises:
            MemoryError: If deserialization fails
        """
        try:
            return cast(M, json.loads(serialized_memory))
        except Exception as e:
            logger.error(f"Failed to deserialize memory: {str(e)}")
            raise MemoryError(f"Failed to deserialize memory: {str(e)}")
    
    def get(self, memory_id: str) -> Optional[M]:
        """Retrieve a memory entry by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry or None if not found
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            memory_key = self._get_memory_key(memory_id)
            serialized_memory = self.client.get(memory_key)
            
            if not serialized_memory:
                return None
                
            memory_entry = self._deserialize_memory(serialized_memory)
            self._update_access_metadata(memory_id, memory_entry)
            
            # Update the memory with the new metadata
            self.client.set(memory_key, self._serialize_memory(memory_entry))
            
            return memory_entry
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memory: {str(e)}")
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if the memory was deleted, False otherwise
            
        Raises:
            MemoryError: If deletion fails
        """
        try:
            # Get the memory first to get metadata for index removal
            memory = self.get(memory_id)
            if not memory:
                return False
                
            # Get all the keys to delete
            memory_key = self._get_memory_key(memory_id)
            vector_key = self._get_vector_key(memory_id)
            
            # Remove from indexes
            timestamp_key = self._get_timestamp_key()
            importance_key = self._get_importance_key()
            
            # Create a pipeline for atomic operations
            pipeline = self.client.pipeline()
            
            # Remove from main storage
            pipeline.delete(memory_key)
            pipeline.delete(vector_key)
            
            # Remove from indexes
            pipeline.zrem(timestamp_key, memory_id)
            pipeline.zrem(importance_key, memory_id)
            
            # Remove from type index if exists
            if memory.get("memory_type"):
                type_key = self._get_memory_type_key(memory["memory_type"])
                pipeline.srem(type_key, memory_id)
            
            # Remove from step index if exists
            if memory.get("step_number") is not None:
                step_key = self._get_step_key()
                pipeline.zrem(step_key, memory_id)
            
            # Execute all commands
            pipeline.execute()
            
            return True
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory deletion: {str(e)}")
            raise MemoryError(f"Failed to delete memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory deletion: {str(e)}")
            raise MemoryError(f"Failed to delete memory: {str(e)}")
    
    def count(self) -> int:
        """Count memories.
        
        Returns:
            Number of memories in the store
            
        Raises:
            MemoryError: If count operation fails
        """
        try:
            # Use the timestamp index to count all memories
            timestamp_key = self._get_timestamp_key()
            return self.client.zcard(timestamp_key)
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory count: {str(e)}")
            raise MemoryError(f"Failed to count memories: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory count: {str(e)}")
            raise MemoryError(f"Failed to count memories: {str(e)}")
    
    def clear(self) -> bool:
        """Clear all memories.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemoryError: If clear operation fails
        """
        try:
            # Get all keys matching the prefix pattern
            pattern = f"{self.key_prefix}:{self.agent_id}:*"
            keys = self.client.scan_iter(match=pattern, count=100)
            
            if not keys:
                return True
                
            # Delete all keys in batches
            batch_size = 100
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i+batch_size]
                self.client.delete(*batch)
            
            return True
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during memory clear: {str(e)}")
            raise MemoryError(f"Failed to clear memories: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory clear: {str(e)}")
            raise MemoryError(f"Failed to clear memories: {str(e)}")
    
    def get_by_timerange(
        self, start_time: float, end_time: float, limit: int = 100
    ) -> List[M]:
        """Get memories in a time range.
        
        Args:
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entries in the time range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            timestamp_key = self._get_timestamp_key()
            
            # Get memory IDs in the time range (Redis stores timestamps as scores)
            memory_ids = self.client.zrangebyscore(
                name=timestamp_key,
                min=start_time,
                max=end_time,
                start=0,
                num=limit
            )
            
            # Retrieve each memory
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during timerange retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by timerange: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during timerange retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by timerange: {str(e)}")
    
    def get_by_importance(
        self, min_importance: float = 0.0, max_importance: float = 1.0, limit: int = 100
    ) -> List[M]:
        """Get memories by importance score.
        
        Args:
            min_importance: Minimum importance score
            max_importance: Maximum importance score
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entries in the importance range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            importance_key = self._get_importance_key()
            
            # Get memory IDs in the importance range, highest importance first
            memory_ids = self.client.zrangebyscore(
                name=importance_key,
                min=min_importance,
                max=max_importance,
                start=0,
                num=limit
            )
            
            # Retrieve each memory
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during importance retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by importance: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during importance retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by importance: {str(e)}")
            
    def search_by_step_range(
        self,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None
    ) -> List[M]:
        """Get memories in a step range.
        
        Args:
            start_step: Start step number
            end_step: End step number
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries in the step range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            step_key = self._get_step_key()
            
            # Get memory IDs in the step range
            memory_ids = self.client.zrangebyscore(
                name=step_key,
                min=start_step,
                max=end_step,
                start=0,
                num=1000  # Temporary high limit for filtering
            )
            
            # Retrieve each memory
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory and (memory_type is None or memory.get("memory_type") == memory_type):
                    memories.append(memory)
                    if len(memories) >= 100:  # Default limit
                        break
            
            return memories
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during step range retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by step range: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during step range retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by step range: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the memory store.
        
        Returns:
            Dictionary with health status information
            
        Raises:
            MemoryError: If health check fails
        """
        try:
            # Check Redis connection
            ping_success = self.client.ping()
            latency = self.client.get_latency()
            
            # Count memories
            memory_count = self.count()
            
            return {
                "status": "healthy" if ping_success else "unhealthy",
                "store_type": self.store_type,
                "latency_ms": round(latency * 1000, 2),
                "memory_count": memory_count,
                "timestamp": time.time()
            }
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during health check: {str(e)}")
            return {
                "status": "unhealthy",
                "store_type": self.store_type,
                "error": str(e),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Unexpected error during health check: {str(e)}")
            return {
                "status": "error",
                "store_type": self.store_type,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_size(self) -> int:
        """Get the size of the memory store in bytes.
        
        Estimates the size by getting memory usage information from Redis.
        
        Returns:
            Estimated memory usage in bytes
            
        Raises:
            MemoryError: If size calculation fails
        """
        try:
            # Get all keys for this agent
            key_pattern = f"{self.key_prefix}:{self.agent_id}:*"
            keys = self.client.keys(key_pattern)
            
            if not keys:
                return 0
                
            # Get memory usage for each key
            total_size = 0
            pipeline = self.client.pipeline()
            
            for key in keys:
                pipeline.memory_usage(key)
            
            # Execute pipeline and sum sizes
            sizes = pipeline.execute()
            for size in sizes:
                if size is not None:
                    total_size += size
            
            return total_size
            
        except (RedisUnavailableError, RedisTimeoutError, CircuitOpenError) as e:
            logger.error(f"Redis error during size calculation: {str(e)}")
            raise MemoryError(f"Failed to get memory store size: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during size calculation: {str(e)}")
            raise MemoryError(f"Failed to get memory store size: {str(e)}")
    
    def search_by_attributes(
        self,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None
    ) -> List[M]:
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