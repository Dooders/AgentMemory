"""Utility functions for the agent memory system.

This package provides various utility functions and classes used throughout
the agent memory system for serialization, Redis operations, and error handling.
"""

from memory.utils.checksums import (
    add_checksum_to_memory,
    generate_checksum,
    validate_checksum,
)
from memory.utils.error_handling import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    EmbeddingGenerationError,
    IMError,
    LTMError,
    MemoryError,
    MemoryTransitionError,
    Priority,
    RecoveryQueue,
    RedisTimeoutError,
    RedisUnavailableError,
    RetryableOperation,
    RetryPolicy,
    SQLitePermanentError,
    SQLiteTemporaryError,
    STMError,
    StoreOperation,
    TransactionError,
)
from memory.utils.redis_utils import (
    RedisBatchProcessor,
    RedisConnectionManager,
    bytes_to_vector,
    deserialize_memory_entry,
    deserialize_vector,
    get_redis_connection_manager,
    get_redis_info,
    redis_create_index,
    redis_drop_index,
    redis_key_exists,
    redis_memory_scan,
    serialize_memory_entry,
    serialize_vector,
    vector_to_bytes,
)
from memory.utils.serialization import (
    MemorySerializer,
    deserialize_memory,
    from_json,
    serialize_memory,
    to_json,
)

__all__ = [
    # Checksum utilities
    "add_checksum_to_memory",
    "generate_checksum",
    "validate_checksum",
    # Serialization
    "MemorySerializer",
    "serialize_memory",
    "deserialize_memory",
    "to_json",
    "from_json",
    # Redis utilities
    "serialize_memory_entry",
    "deserialize_memory_entry",
    "serialize_vector",
    "deserialize_vector",
    "vector_to_bytes",
    "bytes_to_vector",
    "RedisConnectionManager",
    "RedisBatchProcessor",
    "redis_key_exists",
    "redis_memory_scan",
    "redis_create_index",
    "redis_drop_index",
    "get_redis_info",
    "get_redis_connection_manager",
    # Error handling
    "MemoryError",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "EmbeddingGenerationError",
    "IMError",
    "LTMError",
    "MemoryTransitionError",
    "Priority",
    "RecoveryQueue",
    "RedisTimeoutError",
    "RedisUnavailableError",
    "RetryPolicy",
    "RetryableOperation",
    "SQLitePermanentError",
    "SQLiteTemporaryError",
    "STMError",
    "StoreOperation",
    "TransactionError",
]
