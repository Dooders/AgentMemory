"""Redis-based Immediate Memory (IM) storage for agent memory system.

This module provides a Redis-based implementation of the Immediate Memory
storage tier with TTL-based expiration and level 1 compression.

Key features:
- Ultra-fast access for frequently accessed memories
- Automatic TTL-based memory expiration
- Optimized for high-throughput and low-latency
- Vector similarity search with Redis Stack extensions
- Memory filtering by time, importance, and attributes

This component serves as the fastest-access storage layer in the memory hierarchy,
providing rapid recall of recent and important information.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import redis

from memory.config import RedisIMConfig
from memory.storage.redis_client import ResilientRedisClient
from memory.storage.redis_factory import RedisFactory
from memory.utils.checksums import generate_checksum, validate_checksum
from memory.utils.error_handling import (
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


class RedisIMStore:
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

    def __init__(self, config: RedisIMConfig):
        """Initialize the Redis IM store.

        Args:
            config: Configuration for IM Redis storage
        """
        self.config = config
        self._key_prefix = config.namespace

        # Create resilient Redis client using the factory
        self.redis = RedisFactory.create_client(
            client_name="im",
            use_mock=config.use_mock,
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300,
        )

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

        logger.info("Initialized RedisIMStore with namespace %s", self._key_prefix)

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

    def _get_agent_prefix(self, agent_id: str) -> str:
        """Construct the Redis key prefix for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Redis key prefix string
        """
        return f"{self._key_prefix}:{agent_id}"

    def _check_vector_search_available(self) -> bool:
        """Check if Redis vector search capabilities are available.

        Returns:
            True if vector search is available, False otherwise
        """
        try:
            # Check for Redis Stack / RediSearch module
            modules = self.redis.execute_command("MODULE LIST")
            return any(module[1] == b"search" for module in modules)
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
                "FT.CREATE",
                index_name,
                "ON",
                "JSON",
                "PREFIX",
                1,
                f"{self._key_prefix}:",
                "SCHEMA",
                "$.embedding",
                "AS",
                "embedding",
                "VECTOR",
                "FLAT",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                "1536",
                "DISTANCE_METRIC",
                "COSINE",
                # Add additional fields for attribute and step-range searches
                "$.memory_type",
                "AS",
                "memory_type",
                "TAG",
                "$.step_number",
                "AS",
                "step_number",
                "NUMERIC",
                "$.content.*",
                "AS",
                "content",
                "TEXT",
                "$.metadata.importance_score",
                "AS",
                "importance_score",
                "NUMERIC",
            ]

            self.redis.execute_command(*create_cmd)
            logger.info(f"Created vector index {index_name}")
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

    def _check_lua_scripting(self) -> bool:
        """Check if Redis server fully supports Lua scripting.

        Returns:
            True if Lua scripting is fully supported, False otherwise
        """
        try:
            # Try a simple Lua script to check if scripting is available
            test_result = self.redis.eval("return 1", 0)
            return test_result == 1
        except Exception as e:
            logger.warning(f"Lua scripting check failed: {e}")
            return False

    def store(
        self,
        agent_id: str,
        memory_entry: MemoryEntry,
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
            # Prepare hash values
            hash_values = {}

            # Basic fields
            hash_values["memory_id"] = memory_id
            hash_values["timestamp"] = str(timestamp)

            # Store content as JSON if it's a dict/list
            content = memory_entry.get("content", {})
            if isinstance(content, (dict, list)):
                hash_values["content"] = json.dumps(content)
            else:
                hash_values["content"] = str(content)

            # Generate checksum for content if not already in metadata
            metadata = memory_entry.get("metadata", {})
            if "checksum" not in metadata:
                if not isinstance(content, (dict, list)):
                    content = str(
                        content
                    )  # Convert non-dict/non-list content to string
                metadata["checksum"] = generate_checksum(content)

            # Store metadata as JSON
            hash_values["metadata"] = json.dumps(metadata)

            # Store specific metadata fields directly for easier access
            hash_values["compression_level"] = str(metadata.get("compression_level", 1))
            hash_values["importance_score"] = str(metadata.get("importance_score", 0.0))
            hash_values["retrieval_count"] = str(metadata.get("retrieval_count", 0))

            # Construct keys
            key = self._get_memory_key(agent_id, memory_id)
            agent_memories_key = self._get_agent_memories_key(agent_id)
            timeline_key = self._get_timeline_key(agent_id)
            importance_key = self._get_importance_key(agent_id)
            importance = float(metadata.get("importance_score", 0.0))

            if self._lua_scripting_available:
                try:
                    # Define Lua script for atomic memory storage and index updates
                    store_script = """
                        local memory_key = KEYS[1]
                        local memories_key = KEYS[2]
                        local timeline_key = KEYS[3]
                        local importance_key = KEYS[4]
                        
                        local hash_json = ARGV[1]
                        local memory_id = ARGV[2]
                        local timestamp = ARGV[3]
                        local importance = ARGV[4]
                        local ttl = ARGV[5]
                        
                        -- Parse hash values
                        local hash_values = cjson.decode(hash_json)
                        
                        -- Store hash
                        for k, v in pairs(hash_values) do
                            redis.call('HSET', memory_key, k, v)
                        end
                        
                        -- Add to indices
                        redis.call('ZADD', memories_key, timestamp, memory_id)
                        redis.call('ZADD', timeline_key, timestamp, memory_id)
                        redis.call('ZADD', importance_key, importance, memory_id)
                        
                        -- Set TTL on all keys
                        redis.call('EXPIRE', memory_key, ttl)
                        redis.call('EXPIRE', memories_key, ttl)
                        redis.call('EXPIRE', timeline_key, ttl)
                        redis.call('EXPIRE', importance_key, ttl)
                        
                        return 1
                    """

                    # Execute the Lua script
                    logger.debug("About to execute Redis eval")
                    result = self.redis.eval(
                        store_script,
                        4,  # Number of keys
                        key,
                        agent_memories_key,
                        timeline_key,
                        importance_key,
                        json.dumps(hash_values),
                        memory_id,
                        str(timestamp),
                        str(importance),
                        str(self.config.ttl),
                    )
                    logger.debug("Redis eval completed successfully")
                    return result == 1
                except (RedisUnavailableError, RedisTimeoutError) as e:
                    logger.debug(
                        f"Caught Redis error in inner block: {type(e).__name__}"
                    )
                    raise  # propagate these errors
                except redis.RedisError as e:
                    logger.exception(
                        "Redis error when storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
                except json.JSONDecodeError as e:
                    logger.exception(
                        "JSON encoding error when storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
                except Exception as e:
                    logger.exception(
                        "Unexpected error storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
            else:
                try:
                    pipe = self.redis.pipeline()
                    pipe.hset(key, mapping=hash_values)
                    pipe.expire(key, self.config.ttl)
                    pipe.zadd(agent_memories_key, {memory_id: timestamp})
                    pipe.expire(agent_memories_key, self.config.ttl)
                    pipe.zadd(timeline_key, {memory_id: timestamp})
                    pipe.expire(timeline_key, self.config.ttl)
                    pipe.zadd(importance_key, {memory_id: importance})
                    pipe.expire(importance_key, self.config.ttl)
                    results = pipe.execute()
                    if results is None or any(r is None or r is False for r in results):
                        logger.error(
                            "One or more pipeline commands failed for memory %s",
                            memory_id,
                        )
                        return False
                    stored = self.redis.hgetall(key)
                    if not stored or len(stored) == 0:
                        logger.error(
                            "Memory entry %s was not properly stored in Redis",
                            memory_id,
                        )
                        return False
                    if not self.redis.zscore(agent_memories_key, memory_id):
                        logger.error(
                            "Memory entry %s was not properly indexed in memories set",
                            memory_id,
                        )
                        return False
                    if not self.redis.zscore(timeline_key, memory_id):
                        logger.error(
                            "Memory entry %s was not properly indexed in timeline set",
                            memory_id,
                        )
                        return False
                    if not self.redis.zscore(importance_key, memory_id):
                        logger.error(
                            "Memory entry %s was not properly indexed in importance set",
                            memory_id,
                        )
                        return False
                    return True
                except (RedisUnavailableError, RedisTimeoutError) as e:
                    logger.debug(f"Caught Redis error in pipeline block: {type(e).__name__}")
                    raise  # propagate these errors
                except redis.RedisError as e:
                    logger.exception(
                        "Redis error when storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
                except json.JSONDecodeError as e:
                    logger.exception(
                        "JSON encoding error when storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
                except Exception as e:
                    logger.exception(
                        "Unexpected error storing memory entry %s: %s",
                        memory_id,
                        str(e),
                    )
                    return False
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.debug(f"Caught Redis error in outer block: {type(e).__name__}")
            logger.exception(
                "Redis unavailable/timeout when storing memory entry %s: %s",
                memory_id,
                str(e),
            )
            raise  # propagate these errors
        except redis.RedisError as e:
            logger.exception(
                "Redis error when storing memory entry %s: %s",
                memory_id,
                str(e),
            )
            return False
        except json.JSONDecodeError as e:
            logger.exception(
                "JSON encoding error when storing memory entry %s: %s",
                memory_id,
                str(e),
            )
            return False
        except Exception as e:
            logger.exception(
                "Unexpected error storing memory entry %s: %s",
                memory_id,
                str(e),
            )
            return False

    def get(
        self, agent_id: str, memory_id: str, skip_validation: bool = True
    ) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to retrieve
            skip_validation: If True, skip checksum validation

        Returns:
            Memory entry if found, None otherwise
        """
        try:
            # Construct the key using agent_id and memory_id
            key = self._get_memory_key(agent_id, memory_id)

            # Get all hash fields
            hash_data = self.redis.hgetall(key)

            if not hash_data:
                return None

            # Convert hash data back to memory entry dict
            memory_entry = self._hash_to_memory_entry(hash_data)

            # Validate checksum if present and validation not skipped
            metadata = memory_entry.get("metadata", {})
            if "checksum" in metadata and not skip_validation:
                is_valid = validate_checksum(memory_entry)
                if not is_valid:
                    logger.warning(
                        "Checksum validation failed for memory %s", memory_id
                    )
                    # Add integrity flag to metadata
                    metadata["integrity_verified"] = False
                    memory_entry["metadata"] = metadata
                else:
                    metadata["integrity_verified"] = True
                    memory_entry["metadata"] = metadata
            elif "checksum" in metadata and skip_validation:
                # Mark as not verified when skipping validation
                metadata["integrity_verified"] = None  # None means "not checked"
                memory_entry["metadata"] = metadata

            # Update access metadata
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
        except redis.RedisError as e:
            logger.exception(
                "Redis error retrieving memory %s for agent %s", memory_id, agent_id
            )
            return None
        except json.JSONDecodeError as e:
            logger.exception(
                "JSON decoding error for memory %s for agent %s", memory_id, agent_id
            )
            return None
        except Exception as e:
            logger.exception(
                "Unexpected error retrieving memory %s for agent %s",
                memory_id,
                agent_id,
            )
            return None

    def _hash_to_memory_entry(self, hash_data: Dict[str, Any]) -> MemoryEntry:
        """Convert Redis hash data to memory entry dictionary.

        Args:
            hash_data: Raw hash data from Redis

        Returns:
            Reconstructed memory entry dictionary
        """
        # Convert bytes to strings if needed
        if any(isinstance(v, bytes) for v in hash_data.values()):
            hash_data = {
                k: v.decode("utf-8") if isinstance(v, bytes) else v
                for k, v in hash_data.items()
            }

        memory_entry = {"memory_id": hash_data.get("memory_id")}

        # Parse timestamp
        if "timestamp" in hash_data:
            memory_entry["timestamp"] = float(hash_data["timestamp"])

        # Parse content
        if "content" in hash_data:
            try:
                memory_entry["content"] = json.loads(hash_data["content"])
            except (json.JSONDecodeError, TypeError):
                memory_entry["content"] = hash_data["content"]

        # Parse metadata
        if "metadata" in hash_data:
            try:
                memory_entry["metadata"] = json.loads(hash_data["metadata"])
            except (json.JSONDecodeError, TypeError):
                # Fallback to constructing from individual fields
                memory_entry["metadata"] = {
                    "compression_level": int(hash_data.get("compression_level", 1)),
                    "importance_score": float(hash_data.get("importance_score", 0.0)),
                    "retrieval_count": int(hash_data.get("retrieval_count", 0)),
                }
                if "creation_time" in hash_data:
                    memory_entry["metadata"]["creation_time"] = float(
                        hash_data["creation_time"]
                    )

        # Parse embedding if it exists
        if "embedding" in hash_data:
            try:
                memory_entry["embedding"] = json.loads(hash_data["embedding"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse step_number if it exists
        if "step_number" in hash_data:
            memory_entry["step_number"] = int(hash_data["step_number"])

        # Add memory_type if it exists
        if "memory_type" in hash_data:
            memory_entry["memory_type"] = hash_data["memory_type"]

        return memory_entry

    def get_by_timerange(
        self,
        agent_id: str,
        start_time: float,
        end_time: float,
        limit: int = 100,
        skip_validation: bool = True,
    ) -> List[MemoryEntry]:
        """Retrieve memories within a time range.

        Args:
            agent_id: ID of the agent
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            limit: Maximum number of memories to return
            skip_validation: If True, skip checksum validation

        Returns:
            List of memory entries within the time range
        """
        timeline_key = self._get_timeline_key(agent_id)
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

                key = self._get_memory_key(agent_id, memory_id)
                memory_keys.append(memory_id)
                pipe.hgetall(key)  # Get all hash fields instead of JSON string

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, hash_data in enumerate(results):
                if hash_data:
                    memory_entry = self._hash_to_memory_entry(hash_data)

                    # Handle validation if needed
                    metadata = memory_entry.get("metadata", {})
                    if "checksum" in metadata and not skip_validation:
                        is_valid = validate_checksum(memory_entry)
                        if not is_valid:
                            logger.warning(
                                "Checksum validation failed for memory %s",
                                memory_keys[i],
                            )
                            metadata["integrity_verified"] = False
                        else:
                            metadata["integrity_verified"] = True
                        memory_entry["metadata"] = metadata
                    elif "checksum" in metadata and skip_validation:
                        metadata["integrity_verified"] = None  # Not checked
                        memory_entry["metadata"] = metadata

                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable/timeout when retrieving memories by time range: %s",
                str(e),
            )
            return []
        except redis.RedisError as e:
            logger.exception(
                "Redis error when retrieving memories by time range for agent %s",
                agent_id,
            )
            return []
        except json.JSONDecodeError as e:
            logger.exception(
                "JSON decoding error when processing memories by time range for agent %s",
                agent_id,
            )
            return []
        except Exception as e:
            logger.exception(
                "Unexpected error retrieving memories by time range for agent %s",
                agent_id,
            )
            return []

    def get_by_importance(
        self,
        agent_id: str,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 100,
        skip_validation: bool = True,
    ) -> List[MemoryEntry]:
        """Retrieve memories by importance score range.

        Args:
            agent_id: ID of the agent
            min_importance: Minimum importance score (0.0-1.0)
            max_importance: Maximum importance score (0.0-1.0)
            limit: Maximum number of memories to return
            skip_validation: If True, skip checksum validation

        Returns:
            List of memory entries within the importance range
        """
        importance_key = self._get_importance_key(agent_id)
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

                key = self._get_memory_key(agent_id, memory_id)
                memory_keys.append(memory_id)
                pipe.hgetall(key)  # Get all hash fields instead of JSON string

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, hash_data in enumerate(results):
                if hash_data:
                    memory_entry = self._hash_to_memory_entry(hash_data)

                    # Handle validation if needed
                    metadata = memory_entry.get("metadata", {})
                    if "checksum" in metadata and not skip_validation:
                        is_valid = validate_checksum(memory_entry)
                        if not is_valid:
                            logger.warning(
                                "Checksum validation failed for memory %s",
                                memory_keys[i],
                            )
                            metadata["integrity_verified"] = False
                        else:
                            metadata["integrity_verified"] = True
                        memory_entry["metadata"] = metadata
                    elif "checksum" in metadata and skip_validation:
                        metadata["integrity_verified"] = None  # Not checked
                        memory_entry["metadata"] = metadata

                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable/timeout when retrieving memories by importance: %s",
                str(e),
            )
            return []
        except redis.RedisError as e:
            logger.exception(
                "Redis error when retrieving memories by importance for agent %s",
                agent_id,
            )
            return []
        except json.JSONDecodeError as e:
            logger.exception(
                "JSON decoding error when processing memories by importance for agent %s",
                agent_id,
            )
            return []
        except Exception as e:
            logger.exception(
                "Unexpected error retrieving memories by importance for agent %s",
                agent_id,
            )
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
            # Set up keys
            key = self._get_memory_key(agent_id, memory_id)
            agent_memories_key = self._get_agent_memories_key(agent_id)
            timeline_key = self._get_timeline_key(agent_id)
            importance_key = self._get_importance_key(agent_id)

            if self._lua_scripting_available:
                # Define Lua script for atomic deletion from all indices
                delete_script = """
                    local memory_key = KEYS[1]
                    local memories_key = KEYS[2]
                    local timeline_key = KEYS[3]
                    local importance_key = KEYS[4]
                    local memory_id = ARGV[1]
                    
                    -- Delete the memory hash
                    local exists = redis.call('EXISTS', memory_key)
                    if exists == 0 then
                        return 0
                    end
                    
                    redis.call('DEL', memory_key)
                    
                    -- Remove from all indices
                    redis.call('ZREM', memories_key, memory_id)
                    redis.call('ZREM', timeline_key, memory_id)
                    redis.call('ZREM', importance_key, memory_id)
                    
                    return 1
                """

                # Execute the Lua script
                result = self.redis.eval(
                    delete_script,
                    4,  # Number of keys
                    key,
                    agent_memories_key,
                    timeline_key,
                    importance_key,
                    memory_id,
                )
                return result == 1
            else:
                # Fallback to pipeline implementation for compatibility
                pipe = self.redis.pipeline()

                # Check if memory exists first
                pipe.exists(key)
                pipe.delete(key)
                pipe.zrem(agent_memories_key, memory_id)
                pipe.zrem(timeline_key, memory_id)
                pipe.zrem(importance_key, memory_id)

                results = pipe.execute()
                # First result is from EXISTS check
                return results[0] == 1

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable/timeout when deleting memory %s for agent %s: %s",
                memory_id,
                agent_id,
                str(e),
            )
            return False
        except redis.RedisError as e:
            logger.exception(
                "Redis error when deleting memory %s for agent %s", memory_id, agent_id
            )
            return False
        except Exception as e:
            logger.exception(
                "Unexpected error deleting memory %s for agent %s", memory_id, agent_id
            )
            return False

    def count(self, agent_id: str) -> int:
        """Get the number of memories for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of memories stored for the agent
        """
        agent_memories_key = self._get_agent_memories_key(agent_id)
        try:
            return self.redis.zcard(agent_memories_key)
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable/timeout when counting memories for agent %s: %s",
                agent_id,
                str(e),
            )
            return 0
        except redis.RedisError as e:
            logger.exception(
                "Redis error when counting memories for agent %s", agent_id
            )
            return 0
        except Exception as e:
            logger.exception(
                "Unexpected error counting memories for agent %s", agent_id
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
            # Set up keys and pattern
            agent_pattern = self._get_agent_prefix(agent_id)
            agent_memories_key = self._get_agent_memories_key(agent_id)
            timeline_key = self._get_timeline_key(agent_id)
            importance_key = self._get_importance_key(agent_id)

            if self._lua_scripting_available:
                # Define Lua script for efficiently clearing all agent memories
                clear_script = """
                    local memories_key = KEYS[1]
                    local timeline_key = KEYS[2]
                    local importance_key = KEYS[3]
                    local agent_pattern = ARGV[1]
                    
                    -- Get all memory IDs
                    local memory_ids = redis.call('ZRANGE', memories_key, 0, -1)
                    
                    -- Delete all memory keys
                    local deleted = 0
                    for i, memory_id in ipairs(memory_ids) do
                        local memory_key = agent_pattern .. ':memory:' .. memory_id
                        deleted = deleted + redis.call('DEL', memory_key)
                    end
                    
                    -- Delete all index keys
                    deleted = deleted + redis.call('DEL', memories_key)
                    deleted = deleted + redis.call('DEL', timeline_key)
                    deleted = deleted + redis.call('DEL', importance_key)
                    
                    return deleted
                """

                # Execute the Lua script
                result = self.redis.eval(
                    clear_script,
                    3,  # Number of keys
                    agent_memories_key,
                    timeline_key,
                    importance_key,
                    agent_pattern,
                )
                return result > 0
            else:
                # Fallback to pipeline implementation for compatibility
                pipe = self.redis.pipeline()

                # Get all memory IDs
                memory_ids = self.redis.zrange(agent_memories_key, 0, -1)
                if not memory_ids:
                    return True

                # Build keys to delete
                keys_to_delete = []
                for memory_id in memory_ids:
                    # Handle memory_id if it's bytes
                    if isinstance(memory_id, bytes):
                        memory_id = memory_id.decode("utf-8")
                    keys_to_delete.append(self._get_memory_key(agent_id, memory_id))

                # Add index keys
                keys_to_delete.extend(
                    [
                        agent_memories_key,
                        timeline_key,
                        importance_key,
                    ]
                )

                # Delete all keys in a single operation
                if keys_to_delete:
                    pipe.delete(*keys_to_delete)
                    results = pipe.execute()
                    return results[0] > 0
                return True

        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable/timeout when clearing memories for agent %s: %s",
                agent_id,
                str(e),
            )
            return False
        except redis.RedisError as e:
            logger.exception(
                "Redis error when clearing memories for agent %s", agent_id
            )
            return False
        except Exception as e:
            logger.exception(
                "Unexpected error clearing memories for agent %s", agent_id
            )
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
                "Redis connection successful" if ping_result else "Redis ping failed"
            )

            return health_data

        except (RedisTimeoutError, RedisUnavailableError, Exception) as e:
            health_data.update(
                {
                    "status": "unhealthy",
                    "message": str(e),
                    "error": str(e),
                }
            )
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
            keys_count = len(
                list(self.redis.scan_iter(match=namespace_pattern, count=1000))
            )

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
        self, agent_id: str, memory_id: str, memory_entry: MemoryEntry
    ) -> None:
        """Update access metadata for a memory entry.

        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory
            memory_entry: Memory entry to update
        """
        try:
            # Update access time and retrieval count in memory_entry dict first
            metadata = memory_entry.get("metadata", {})
            retrieval_count = metadata.get("retrieval_count", 0) + 1
            access_time = time.time()

            metadata.update(
                {"last_access_time": access_time, "retrieval_count": retrieval_count}
            )
            memory_entry["metadata"] = metadata

            # Calculate new importance if retrieval count warrants it and not in test mode
            new_importance = 0.0
            if not self.config.test_mode and retrieval_count > 1:
                importance = metadata.get("importance_score", 0.0)
                access_factor = min(retrieval_count / 10.0, 1.0)  # Cap at 1.0
                new_importance = importance + (access_factor * 0.1)  # Slight boost
                metadata["importance_score"] = new_importance

            # Set up keys
            key = self._get_memory_key(agent_id, memory_id)
            importance_key = self._get_importance_key(agent_id)

            if self._lua_scripting_available:
                # Define the Lua script for atomic metadata updates
                update_script = """
                    local key = KEYS[1]
                    local importance_key = KEYS[2]
                    
                    local metadata = ARGV[1]
                    local retrieval_count = ARGV[2]
                    local access_time = ARGV[3]
                    local ttl = ARGV[4]
                    local memory_id = ARGV[5]
                    local new_importance = ARGV[6]
                    
                    -- Update hash fields
                    redis.call('HSET', key, 'metadata', metadata)
                    redis.call('HSET', key, 'retrieval_count', retrieval_count)
                    redis.call('HSET', key, 'last_access_time', access_time)
                    
                    -- Always refresh TTL
                    redis.call('EXPIRE', key, ttl)
                    
                    -- Update importance if needed
                    if tonumber(new_importance) > 0 then
                        redis.call('HSET', key, 'importance_score', new_importance)
                        redis.call('ZADD', importance_key, new_importance, memory_id)
                        redis.call('EXPIRE', importance_key, ttl)
                    end
                    
                    return 1
                """

                # Execute the Lua script
                result = self.redis.eval(
                    update_script,
                    2,  # Number of keys
                    key,
                    importance_key,
                    json.dumps(metadata),
                    str(retrieval_count),
                    str(access_time),
                    str(self.config.ttl),
                    memory_id,
                    str(new_importance),
                )
                if result != 1:
                    logger.warning(
                        "Failed to update access metadata for memory %s: Lua script returned %s",
                        memory_id,
                        result,
                    )
            else:
                # Fallback to pipeline implementation for compatibility
                pipe = self.redis.pipeline()

                # Update the metadata, retrieval count, and last access time
                pipe.hset(key, "metadata", json.dumps(metadata))
                pipe.hset(key, "retrieval_count", str(retrieval_count))
                pipe.hset(key, "last_access_time", str(access_time))

                # Refresh TTL
                pipe.expire(key, self.config.ttl)

                # Update importance if it has changed
                if new_importance > 0:
                    pipe.hset(key, "importance_score", str(new_importance))
                    pipe.zadd(importance_key, {memory_id: new_importance})
                    pipe.expire(importance_key, self.config.ttl)

                try:
                    pipe.execute()
                except Exception as e:
                    logger.warning(
                        "Failed to update access metadata for memory %s: %s",
                        memory_id,
                        str(e),
                    )

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

            # Use pipeline to get hash fields in batches
            total_size = 0
            batch_size = 100  # Process keys in batches to avoid large pipelines

            for i in range(0, len(memory_keys), batch_size):
                batch_keys = memory_keys[i : i + batch_size]
                pipe = self.redis.pipeline()

                for key in batch_keys:
                    pipe.hgetall(key)  # Get all hash fields

                hash_results = pipe.execute()

                # Sum up the sizes considering both keys and values in each hash
                for hash_data in hash_results:
                    if hash_data:
                        # Add size of keys and values
                        for field, value in hash_data.items():
                            # Account for field name
                            if isinstance(field, bytes):
                                total_size += len(field)
                            else:
                                total_size += len(str(field).encode("utf-8"))

                            # Account for field value
                            if isinstance(value, bytes):
                                total_size += len(value)
                            else:
                                total_size += len(str(value).encode("utf-8"))

            return total_size
        except Exception as e:
            logger.exception("Error retrieving memory size for agent %s", agent_id)
            return 0

    def get_all(
        self, agent_id: str, limit: int = 1000, skip_validation: bool = True
    ) -> List[MemoryEntry]:
        """Get all memories for an agent.

        Args:
            agent_id: ID of the agent
            limit: Maximum number of memories to return
            skip_validation: If True, skip checksum validation

        Returns:
            List of memory entries
        """
        try:
            # Get all memory IDs sorted by recency
            memories_key = self._get_agent_memories_key(agent_id)
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

                key = self._get_memory_key(agent_id, memory_id)
                memory_keys.append(memory_id)
                pipe.hgetall(key)  # Get all hash fields instead of JSON string

            # Execute pipeline and process results
            results = pipe.execute()
            memories = []

            for i, hash_data in enumerate(results):
                if hash_data:
                    memory_entry = self._hash_to_memory_entry(hash_data)

                    # Handle validation if needed
                    metadata = memory_entry.get("metadata", {})
                    if "checksum" in metadata and not skip_validation:
                        is_valid = validate_checksum(memory_entry)
                        if not is_valid:
                            logger.warning(
                                "Checksum validation failed for memory %s",
                                memory_keys[i],
                            )
                            metadata["integrity_verified"] = False
                        else:
                            metadata["integrity_verified"] = True
                        memory_entry["metadata"] = metadata
                    elif "checksum" in metadata and skip_validation:
                        metadata["integrity_verified"] = None  # Not checked
                        memory_entry["metadata"] = metadata

                    self._update_access_metadata(agent_id, memory_keys[i], memory_entry)
                    memories.append(memory_entry)

            return memories
        except Exception as e:
            logger.exception("Error retrieving all memories for agent %s", agent_id)
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
        if self._vector_search_available:
            try:
                return self._search_similar_redis_vector(
                    agent_id, query_embedding, k, memory_type
                )
            except Exception as e:
                logger.warning(
                    f"Redis vector search failed, falling back to Python implementation: {e}"
                )
                # Fall back to Python implementation
                return self._search_similar_python(
                    agent_id, query_embedding, k, memory_type
                )
        else:
            # Use Python implementation
            return self._search_similar_python(
                agent_id, query_embedding, k, memory_type
            )

    def _search_similar_redis_vector(
        self,
        agent_id: str,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Search for memories by vector similarity in Redis.

        Args:
            agent_id: Unique identifier for the agent
            query_embedding: Vector to compare for similarity
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score to include in results
            memory_type: Optional memory type filter

        Returns:
            List of memory entries ordered by similarity to the query
        """
        index_name = f"{self._key_prefix}_vector_idx"

        # Convert embedding to blob for hybrid query
        embedding_str = ",".join([str(x) for x in query_embedding])

        # Build Redis query
        query_parts = []

        # Add memory type filter if specified
        if memory_type:
            query_parts.append(f"@memory_type:{{{memory_type}}}")

        # Add final query string
        query = "*"
        if query_parts:
            query = " ".join(query_parts)

        # Add agent ID filter using prefix
        agent_prefix = f"{self._key_prefix}:{agent_id}:memory:"

        try:
            # Execute vector search
            results = self.redis.execute_command(
                "FT.SEARCH",
                index_name,
                query,
                "LIMIT",
                0,
                limit,
                "SORTBY",
                "__embedding_score",
                "DESC",
                "FILTER",
                "PREFLEN",
                len(agent_prefix),
                "PREFIX",
                1,
                agent_prefix,
                "PARAMS",
                2,
                "embedding_query",
                embedding_str,
                "DIALECT",
                2,
            )

            if not results or results[0] == 0:
                return []

            # Process results
            memories = []
            doc_scores = {}

            # Extract document scores from the response
            for i in range(1, len(results), 2):
                key = results[i]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                # First try to parse the $ field in the response (test format)
                doc_data = {}
                score = 0.0

                for field_value_pair in results[i + 1]:
                    if (
                        isinstance(field_value_pair, list)
                        and len(field_value_pair) >= 2
                    ):
                        field = field_value_pair[0]
                        value = field_value_pair[1]

                        # Decode byte strings if needed
                        if isinstance(field, bytes):
                            field = field.decode("utf-8")
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")

                        # Handle the $ field which contains the full JSON document
                        if field == "$":
                            try:
                                doc_data = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    f"Failed to parse JSON data in search result: {value}"
                                )

                        # Extract vector score
                        elif field in ["__embedding_score", "__vector_score"]:
                            try:
                                score = float(value)
                            except (ValueError, TypeError):
                                logger.warning(f"Failed to parse score value: {value}")

                # Store the score for this document
                doc_scores[key] = score

                # If we got valid data from the $ field, create a memory entry
                if doc_data:
                    # Extract memory_id from the document or the key
                    memory_id = doc_data.get("memory_id")
                    if not memory_id and key:
                        parts = key.split(":")
                        if len(parts) >= 4 and parts[-2] == "memory":
                            memory_id = parts[-1]

                    if memory_id:
                        # Create a memory entry from the document data
                        memory_entry = doc_data.copy()
                        # Use the score we extracted earlier
                        memory_entry["similarity_score"] = score
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)
                    continue  # Skip to next result

                # If we couldn't get data from $ field, try regular methods
                # Extract memory data using hgetall
                hash_data = self.redis.hgetall(key)
                if hash_data:
                    # Convert hash data to memory entry dict
                    memory_entry = self._hash_to_memory_entry(hash_data)

                    # Extract memory_id from the key
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[-2] == "memory":
                        memory_id = parts[-1]
                        # Fallback to get if hgetall didn't work (for tests)
                        if not memory_entry and hasattr(self.redis, "get"):
                            json_data = self.redis.get(key)
                            if json_data:
                                try:
                                    memory_entry = json.loads(json_data)
                                except (json.JSONDecodeError, TypeError):
                                    # Skip this entry if we can't parse it
                                    continue

                        if memory_entry:
                            self._update_access_metadata(
                                agent_id, memory_id, memory_entry
                            )
                            memories.append(memory_entry)

            # Sort by similarity score (highest first)
            memories.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            return memories[:limit]

        except Exception as e:
            logger.error(f"Error in Redis vector search: {e}")
            raise

    def _search_similar_python(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
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
            logger.exception("Error calculating cosine similarity between vectors")
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
        if self._vector_search_available:
            try:
                return self._search_by_attributes_redis(
                    agent_id, attributes, memory_type
                )
            except Exception as e:
                logger.warning(
                    f"Redis attribute search failed, falling back to Python implementation: {e}"
                )
                # Fall back to Python implementation
                return self._search_by_attributes_python(
                    agent_id, attributes, memory_type
                )
        else:
            # Use Python implementation
            return self._search_by_attributes_python(agent_id, attributes, memory_type)

    def _search_by_attributes_redis(
        self,
        agent_id: str,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
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
                query_parts.append(f"{redis_path}:{str(attr_value).lower()}")

        # Combine query parts
        query = " ".join(query_parts) if query_parts else "*"

        try:
            # Execute search with prefix filter for agent_id
            results = self.redis.execute_command(
                "FT.SEARCH",
                index_name,
                query,
                "LIMIT",
                0,
                1000,  # Reasonable limit
                "FILTER",
                "PREFLEN",
                len(agent_prefix),
                "PREFIX",
                1,
                agent_prefix,
            )

            if not results or results[0] == 0:
                return []

            # Process results
            memories = []
            for i in range(1, len(results), 2):
                key = results[i]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                # First try to parse the $ field in the response (test format)
                doc_data = {}
                for field_value_pair in results[i + 1]:
                    if (
                        isinstance(field_value_pair, list)
                        and len(field_value_pair) >= 2
                    ):
                        field = field_value_pair[0]
                        value = field_value_pair[1]

                        # Decode byte strings if needed
                        if isinstance(field, bytes):
                            field = field.decode("utf-8")
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")

                        # Handle the $ field which contains the full JSON document
                        if field == "$":
                            try:
                                doc_data = json.loads(value)
                                break  # Found what we need, exit loop
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    f"Failed to parse JSON data in search result: {value}"
                                )

                # If we got valid data from the $ field, create a memory entry
                if doc_data:
                    # Extract memory_id from the document or the key
                    memory_id = doc_data.get("memory_id")
                    if not memory_id and key:
                        parts = key.split(":")
                        if len(parts) >= 4 and parts[-2] == "memory":
                            memory_id = parts[-1]

                    if memory_id:
                        # Create a memory entry from the document data
                        memory_entry = doc_data.copy()
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)
                    continue  # Skip to next result

                # If we couldn't get data from $ field, try regular methods
                # Extract memory data using hgetall
                hash_data = self.redis.hgetall(key)
                if hash_data:
                    # Convert hash data to memory entry dict
                    memory_entry = self._hash_to_memory_entry(hash_data)

                    # Extract memory_id from the key
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[-2] == "memory":
                        memory_id = parts[-1]
                        # Fallback to get if hgetall didn't work (for tests)
                        if not memory_entry and hasattr(self.redis, "get"):
                            json_data = self.redis.get(key)
                            if json_data:
                                try:
                                    memory_entry = json.loads(json_data)
                                except (json.JSONDecodeError, TypeError):
                                    # Skip this entry if we can't parse it
                                    continue

                        if memory_entry:
                            self._update_access_metadata(
                                agent_id, memory_id, memory_entry
                            )
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
    ) -> List[MemoryEntry]:
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
        if self._vector_search_available:
            try:
                return self._search_by_step_range_redis(
                    agent_id, start_step, end_step, memory_type
                )
            except Exception as e:
                logger.warning(
                    f"Redis step range search failed, falling back to Python implementation: {e}"
                )
                # Fall back to optimized Python implementation

        # Even without vector search capabilities, we can use Redis sorted sets or SCAN with pattern matching
        try:
            # Create a timeline key if using a sorted set approach
            timeline_key = self._get_timeline_key(agent_id)

            # Check if we have a sorted set timeline index
            if self.redis.exists(timeline_key):
                # Use Redis ZRANGEBYSCORE to get memory IDs in the step range
                memory_ids = self.redis.zrangebyscore(
                    timeline_key, min=start_step, max=end_step
                )

                results = []
                for memory_id in memory_ids:
                    if isinstance(memory_id, bytes):
                        memory_id = memory_id.decode("utf-8")

                    # Get the memory data
                    memory_key = self._get_memory_key(agent_id, memory_id)
                    memory_data = self.redis.get(memory_key)

                    if memory_data:
                        try:
                            memory = json.loads(memory_data)
                            # Apply memory type filter if needed
                            if (
                                memory_type is None
                                or memory.get("memory_type") == memory_type
                            ):
                                # Update access metadata and add to results
                                self._update_access_metadata(
                                    agent_id, memory_id, memory
                                )
                                results.append(memory)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse memory data for ID {memory_id}"
                            )

                return results

            # If no timeline index exists, use SCAN with pattern matching and step filtering
            pattern = f"{self._key_prefix}:{agent_id}:memory:*"
            cursor = 0
            results = []

            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=100)

                if not keys:
                    if cursor == 0:
                        break
                    continue

                # Get memory data for each key using pipeline for efficiency
                pipe = self.redis.pipeline()
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    pipe.get(key)

                memory_data_list = pipe.execute()

                # Process memory data
                for i, memory_data in enumerate(memory_data_list):
                    if not memory_data:
                        continue

                    try:
                        memory = json.loads(memory_data)
                        step = memory.get("step_number")

                        # Filter by step range
                        if step is not None and start_step <= step <= end_step:
                            # Apply memory type filter if needed
                            if (
                                memory_type is None
                                or memory.get("memory_type") == memory_type
                            ):
                                # Extract memory ID from key
                                key = keys[i]
                                if isinstance(key, bytes):
                                    key = key.decode("utf-8")
                                memory_id = key.split(":")[-1]

                                # Update access metadata and add to results
                                self._update_access_metadata(
                                    agent_id, memory_id, memory
                                )
                                results.append(memory)
                    except json.JSONDecodeError:
                        continue

                if cursor == 0:
                    break

            return results

        except Exception as e:
            # If Redis operations fail, fall back to retrieving all and filtering
            logger.warning(
                f"Optimized step range search failed, using full fallback: {e}"
            )
            memories = self.get_all(agent_id)

            # Filter by step range
            results = []
            for memory in memories:
                step = memory.get("step_number")
                if step is not None and start_step <= step <= end_step:
                    if memory_type is None or memory.get("memory_type") == memory_type:
                        results.append(memory)

            return results

    def _search_by_step_range_redis(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Search for memories by step range using Redis search.

        Args:
            agent_id: Unique identifier for the agent
            start_step: Start of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries with matching step range
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
                "FT.SEARCH",
                index_name,
                query,
                "LIMIT",
                0,
                1000,  # Reasonable limit
                "FILTER",
                "PREFLEN",
                len(agent_prefix),
                "PREFIX",
                1,
                agent_prefix,
            )

            if not results or results[0] == 0:
                return []

            # Process results
            memories = []
            for i in range(1, len(results), 2):
                key = results[i]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                # First try to parse the $ field in the response (test format)
                doc_data = {}
                for field_value_pair in results[i + 1]:
                    if (
                        isinstance(field_value_pair, list)
                        and len(field_value_pair) >= 2
                    ):
                        field = field_value_pair[0]
                        value = field_value_pair[1]

                        # Decode byte strings if needed
                        if isinstance(field, bytes):
                            field = field.decode("utf-8")
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")

                        # Handle the $ field which contains the full JSON document
                        if field == "$":
                            try:
                                doc_data = json.loads(value)
                                break  # Found what we need, exit loop
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    f"Failed to parse JSON data in search result: {value}"
                                )

                # If we got valid data from the $ field, create a memory entry
                if doc_data:
                    # Extract memory_id from the document or the key
                    memory_id = doc_data.get("memory_id")
                    if not memory_id and key:
                        parts = key.split(":")
                        if len(parts) >= 4 and parts[-2] == "memory":
                            memory_id = parts[-1]

                    if memory_id:
                        # Create a memory entry from the document data
                        memory_entry = doc_data.copy()
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)
                    continue  # Skip to next result

                # If we couldn't get data from $ field, try regular methods
                # Extract memory data using the key
                memory_entry = None
                try:
                    json_data = self.redis.get(key)
                    if json_data:
                        memory_entry = json.loads(json_data)
                except (json.JSONDecodeError, TypeError, redis.RedisError) as e:
                    logger.warning(f"Error parsing memory data for key {key}: {e}")

                # Extract memory_id from the key
                if memory_entry:
                    parts = key.split(":")
                    if len(parts) >= 4 and parts[-2] == "memory":
                        memory_id = parts[-1]
                        self._update_access_metadata(agent_id, memory_id, memory_entry)
                        memories.append(memory_entry)

            return memories

        except Exception as e:
            logger.error(f"Error in Redis step range search: {e}")
            raise

    def search_by_content(
        self,
        agent_id: str,
        content_query: Union[str, Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
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
