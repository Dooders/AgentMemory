"""Redis client with error handling for agent memory system.

This module provides a Redis client wrapper with circuit breaker and
retry functionality for resilient Redis operations.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import redis

from agent_memory.utils.error_handling import (
    CircuitBreaker,
    CircuitOpenError,
    Priority,
    RecoveryQueue,
    RedisTimeoutError,
    RedisUnavailableError,
    RetryPolicy,
    StoreOperation,
)

logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar("T")


def resilient_operation(operation_name: str):
    """Decorator to wrap Redis operations with circuit breaker pattern.

    Args:
        operation_name: Name of the operation for logging

    Returns:
        Decorator function that wraps methods with circuit breaker
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self, *args, **kwargs) -> T:
            return self._execute_with_circuit_breaker(
                operation_name, lambda: func(self, *args, **kwargs)
            )

        return wrapper

    return decorator


def exponential_backoff(
    attempt: int, base_delay: float = 0.5, max_delay: float = 30.0
) -> float:
    """Calculate delay for exponential backoff strategy.

    Args:
        attempt: The current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds for the current attempt
    """
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)


class ResilientRedisClient:
    """Redis client with circuit breaker and retry functionality.

    This class wraps Redis operations with circuit breaker pattern
    to prevent cascading failures and retry mechanisms for transient
    errors.

    Attributes:
        client_name: Name of this Redis client instance
        connection_params: Redis connection parameters
        client: Redis client instance
        circuit_breaker: Circuit breaker for Redis operations
        recovery_queue: Queue for retrying failed operations
        retry_policy: Policy for automatic retries
        critical_retry_attempts: Number of immediate retries for critical operations
        retry_base_delay: Base delay for exponential backoff (seconds)
        retry_max_delay: Maximum delay for exponential backoff (seconds)
    """

    def __init__(
        self,
        client_name: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_threshold: int = 3,
        circuit_reset_timeout: int = 300,
        critical_retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 30.0,
        max_connections: int = 10,
        health_check_interval: int = 30,
    ):
        """Initialize the Redis client.

        Args:
            client_name: Name of this Redis client instance
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            socket_timeout: Timeout for Redis operations
            socket_connect_timeout: Timeout for connection attempts
            retry_policy: Policy for automatic retries
            circuit_threshold: Failures before circuit breaker opens
            circuit_reset_timeout: Seconds before circuit breaker resets
            critical_retry_attempts: Number of immediate retries for critical operations
            retry_base_delay: Base delay for exponential backoff (seconds)
            retry_max_delay: Maximum delay for exponential backoff (seconds)
            max_connections: Maximum number of connections in the pool
            health_check_interval: Seconds between health checks of idle connections

        Raises:
            RedisUnavailableError: If initial Redis client creation fails
        """
        self.client_name = client_name

        # Configure connection pool settings
        connection_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True,
            max_connections=max_connections,
            health_check_interval=health_check_interval,
        )

        self.connection_params = {
            "connection_pool": connection_pool,
        }

        # Create Redis client
        self.client = self._create_redis_client()

        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"redis-{client_name}",
            failure_threshold=circuit_threshold,
            reset_timeout=circuit_reset_timeout,
        )

        # Create retry policy
        self.retry_policy = retry_policy or RetryPolicy()

        # Create recovery queue
        self.recovery_queue = RecoveryQueue(retry_policy=self.retry_policy)

        # Configure retry settings
        self.critical_retry_attempts = critical_retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

        logger.info(
            f"Initialized Redis client {client_name} (host={host}, port={port}, db={db}, pool_size={max_connections})"
        )

    def _create_redis_client(self) -> redis.Redis:
        """Create a new Redis client instance.

        Returns:
            Redis client instance

        Raises:
            RedisUnavailableError: If Redis client creation fails
        """
        try:
            client = redis.Redis(**self.connection_params)
            return client
        except Exception as e:
            logger.exception("Failed to create Redis client")
            raise RedisUnavailableError(
                f"Failed to create Redis client: {str(e)}"
            ) from e

    def close(self) -> None:
        """Close the Redis client connection pool.

        This method should be called when the client is no longer needed
        to properly release resources and connections.
        """
        if hasattr(self.client, "connection_pool"):
            self.client.connection_pool.disconnect()
            logger.info(f"Closed Redis client {self.client_name} connections")

    def _execute_with_circuit_breaker(
        self, operation_name: str, operation: Callable[[], T]
    ) -> T:
        """Execute operation with circuit breaker pattern.

        Args:
            operation_name: Name of the operation
            operation: Function to execute

        Returns:
            Result of the operation

        Raises:
            CircuitOpenError: If the circuit is open
            RedisUnavailableError: If Redis is unavailable (connection error)
            RedisTimeoutError: If operation times out
            Exception: Other exceptions from the Redis operation
        """
        try:
            return self.circuit_breaker.execute(operation)
        except Exception as e:
            if isinstance(e, redis.exceptions.ConnectionError):
                logger.exception(f"Redis connection error in {operation_name}")
                raise RedisUnavailableError(f"Redis unavailable: {str(e)}") from e
            elif isinstance(e, redis.exceptions.TimeoutError):
                logger.exception(f"Redis timeout in {operation_name}")
                raise RedisTimeoutError(f"Redis operation timed out: {str(e)}") from e
            elif isinstance(e, CircuitOpenError):
                logger.warning(f"Circuit breaker open for {operation_name}")
                # Let CircuitOpenError pass through instead of converting to RedisUnavailableError
                raise
            else:
                logger.exception(f"Redis error in {operation_name}")
                raise e

    @resilient_operation("ping")
    def ping(self) -> bool:
        """Ping Redis server to check connection.

        Returns:
            True if successful, raises exception otherwise

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.ping()

    def get_latency(self) -> float:
        """Measure Redis server latency.

        Returns:
            Latency in milliseconds or -1 if an error occurred

        Raises:
            No exceptions are raised; returns -1 on error
        """
        try:
            start_time = time.time()
            self.ping()
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception:
            return -1  # Return -1 to indicate an error

    @resilient_operation("get")
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Value or None if not found

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.get(key)

    @resilient_operation("set")
    def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set key-value pair in Redis.

        Args:
            key: Redis key
            value: Value to set
            ex: Expiry in seconds
            px: Expiry in milliseconds
            nx: Only set if key does not exist
            xx: Only set if key exists

        Returns:
            True if successful

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    @resilient_operation("delete")
    def delete(self, *keys: str) -> int:
        """Delete keys from Redis.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.delete(*keys)

    @resilient_operation("exists")
    def exists(self, *keys: str) -> int:
        """Check if keys exist in Redis.

        Args:
            keys: Keys to check

        Returns:
            Number of keys that exist

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.exists(*keys)

    @resilient_operation("expire")
    def expire(self, key: str, time: int) -> bool:
        """Set expiry on key.

        Args:
            key: Redis key
            time: Expiry in seconds

        Returns:
            True if successful

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.expire(key, time)

    @resilient_operation("hset")
    def hset(self, name: str, key: str, value: str) -> int:
        """Set field in hash.

        Args:
            name: Hash name
            key: Field name
            value: Field value

        Returns:
            1 if new field, 0 if field existed

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.hset(name, key, value)

    @resilient_operation("hget")
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get field from hash.

        Args:
            name: Hash name
            key: Field name

        Returns:
            Field value or None if not found

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.hget(name, key)

    @resilient_operation("hgetall")
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all fields from hash.

        Args:
            name: Hash name

        Returns:
            Dictionary of field names to values

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.hgetall(name)

    @resilient_operation("hset_dict")
    def hset_dict(self, name: str, mapping: Dict[str, str]) -> int:
        """Set multiple fields in hash.

        This is the recommended replacement for the deprecated hmset.

        Args:
            name: Hash name
            mapping: Dictionary of field names to values

        Returns:
            Number of fields that were added (not updated)

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.hset(name, mapping=mapping)

    @resilient_operation("hdel")
    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash.

        Args:
            name: Hash name
            keys: Field names to delete

        Returns:
            Number of fields deleted

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.hdel(name, *keys)

    @resilient_operation("zadd")
    def zadd(
        self,
        name: str,
        mapping: Dict[str, float],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        incr: bool = False,
    ) -> int:
        """Add members to sorted set.

        Args:
            name: Sorted set name
            mapping: Dictionary of member names to scores
            nx: Only add new elements
            xx: Only update existing elements
            ch: Return number of changed elements
            incr: Increment score instead of replacing

        Returns:
            Number of elements added or updated

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.zadd(name, mapping, nx=nx, xx=xx, ch=ch, incr=incr)

    @resilient_operation("zrange")
    def zrange(
        self,
        name: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[Any], Any] = float,
    ) -> List[Any]:
        """Get range of members from sorted set.

        Args:
            name: Sorted set name
            start: Start index
            end: End index
            desc: Order by descending score
            withscores: Include scores in result
            score_cast_func: Function to cast scores

        Returns:
            List of members or (member, score) tuples

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.zrange(
            name,
            start,
            end,
            desc=desc,
            withscores=withscores,
            score_cast_func=score_cast_func,
        )

    @resilient_operation("zrangebyscore")
    def zrangebyscore(
        self,
        name: str,
        min: float,
        max: float,
        start: Optional[int] = None,
        num: Optional[int] = None,
        withscores: bool = False,
        score_cast_func: Callable[[Any], Any] = float,
    ) -> List[Any]:
        """Get members from sorted set with scores in range.

        Args:
            name: Sorted set name
            min: Minimum score
            max: Maximum score
            start: Start offset
            num: Number of elements
            withscores: Include scores in result
            score_cast_func: Function to cast scores

        Returns:
            List of members or (member, score) tuples

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.zrangebyscore(
            name,
            min,
            max,
            start=start,
            num=num,
            withscores=withscores,
            score_cast_func=score_cast_func,
        )

    @resilient_operation("zrem")
    def zrem(self, name: str, *values: str) -> int:
        """Remove members from sorted set.

        Args:
            name: Sorted set name
            values: Members to remove

        Returns:
            Number of members removed

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.zrem(name, *values)

    @resilient_operation("zcard")
    def zcard(self, name: str) -> int:
        """Get the number of members in a sorted set.

        Args:
            name: Name of sorted set

        Returns:
            Cardinality of set

        Raises:
            RedisTimeoutError: If operation times out
            RedisUnavailableError: If Redis is unavailable
        """
        return self.client.zcard(name)

    @resilient_operation("scan_iter")
    def scan_iter(self, match: Optional[str] = None, count: int = 10) -> list:
        """Iterates over keys in the database matching the pattern.

        Args:
            match: Pattern to match
            count: Number of keys to return at a time

        Returns:
            List of matching keys

        Raises:
            RedisTimeoutError: If operation times out
            RedisUnavailableError: If Redis is unavailable
        """
        # Manual implementation using scan instead of scan_iter
        # as we need to handle the cursor ourselves
        keys = []
        cursor = 0
        while True:
            cursor, chunk = self.client.scan(cursor, match=match, count=count)
            keys.extend(chunk)
            if cursor == 0:
                break
        return keys

    def store_with_retry(
        self,
        agent_id: str,
        state_data: Dict[str, Any],
        store_func: Callable[[str, Dict[str, Any]], bool],
        priority: Priority = Priority.NORMAL,
        retry_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
    ) -> bool:
        """Store data with automatic retry on failure.

        This method attempts to store data and enqueues a retry
        operation if it fails.

        Args:
            agent_id: ID of the agent
            state_data: Data to store
            store_func: Function that performs the actual storage
            priority: Priority level for the operation
            retry_attempts: Override default retry attempts for this operation
            base_delay: Override default base delay for this operation
            max_delay: Override default max delay for this operation

        Returns:
            True if the operation succeeded, False if enqueued for retry or failed

        Raises:
            No exceptions are propagated; all exceptions are caught and handled internally
        """
        operation_id = str(uuid.uuid4())

        try:
            # Try immediate store
            success = store_func(agent_id, state_data)
            return success
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.exception(f"Redis operation failed for agent {agent_id}")

            if priority == Priority.CRITICAL:
                # For critical states, we need to retry immediately with exponential backoff
                attempts = retry_attempts or self.critical_retry_attempts
                base = base_delay or self.retry_base_delay
                max_d = max_delay or self.retry_max_delay

                for attempt in range(attempts):
                    try:
                        delay = exponential_backoff(attempt, base, max_d)
                        logger.info(
                            f"Immediate retry {attempt + 1} for critical data after {delay:.1f} seconds"
                        )
                        time.sleep(delay)
                        return store_func(agent_id, state_data)
                    except (RedisUnavailableError, RedisTimeoutError) as e:
                        logger.exception(f"Immediate retry {attempt + 1} failed")

                logger.error(
                    f"Critical data store failed after {attempts} immediate retries for agent {agent_id}"
                )
                return False
            elif priority == Priority.HIGH or priority == Priority.NORMAL:
                # For high/normal priority, enqueue for background retry
                operation = StoreOperation(
                    operation_id=operation_id,
                    agent_id=agent_id,
                    state_data=state_data,
                    store_function=store_func,
                )
                # Priority conversion: 4 - priority.value gives inverse priority,
                # where HIGH(2) becomes 2 and NORMAL(1) becomes 3
                # Lower queue priority number = higher actual priority
                self.recovery_queue.enqueue(
                    operation,
                    priority=4 - priority.value,  # Lower value = higher priority
                )
                logger.info(
                    f"Enqueued {priority.name} priority store operation for agent {agent_id}"
                )
                return False
            else:
                # For low priority, just log and continue
                logger.info(
                    f"Low priority store operation failed for agent {agent_id}: {str(e)}"
                )
                return False

    @resilient_operation("hmset")
    def hmset(self, name: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in hash.

        Warning: This method is deprecated in redis-py.
        Use hset_dict() instead.

        Args:
            name: Hash name
            mapping: Dictionary of field names to values

        Returns:
            True if successful

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        # For backwards compatibility, always return True on success
        self.client.hset(name, mapping=mapping)
        return True

    @resilient_operation("pipeline")
    def pipeline(self) -> redis.client.Pipeline:
        """Create a Redis pipeline for batching commands.

        Returns:
            Redis pipeline object for batching commands

        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        return self.client.pipeline()
