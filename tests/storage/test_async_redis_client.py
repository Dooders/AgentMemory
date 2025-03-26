"""Unit tests for AsyncResilientRedisClient.

Tests the asynchronous Redis client with circuit breaker and retry functionality.
Includes tests for normal operations, error scenarios, and recovery mechanisms.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import redis.asyncio as redis

from agent_memory.storage.async_redis_client import AsyncResilientRedisClient
from agent_memory.utils.error_handling import (
    AsyncCircuitBreaker,
    AsyncStoreOperation,
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
    RetryPolicy,
)


@pytest_asyncio.fixture
async def mock_redis():
    with patch("redis.asyncio.Redis") as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock, mock_instance


@pytest_asyncio.fixture
async def async_client(mock_redis):
    mock_factory, mock_instance = mock_redis

    # Create client with test configuration
    client = AsyncResilientRedisClient(
        client_name="test-client",
        host="localhost",
        port=6379,
        db=0,
        password=None,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
        circuit_threshold=2,
        circuit_reset_timeout=5,
    )

    # Initialize the client
    await client.init()

    # Replace the actual circuit breaker with a mock
    client.circuit_breaker = AsyncMock(spec=AsyncCircuitBreaker)
    client.circuit_breaker.execute.side_effect = lambda x: x()

    # Replace the recovery queue with a mock
    client.recovery_queue = MagicMock()

    yield client

    # Cleanup
    await client.close()


@pytest.mark.asyncio
async def test_init():
    """Test client initialization."""
    # Create a new client to test initialization
    with patch("redis.asyncio.Redis") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance

        client = AsyncResilientRedisClient(client_name="test-init")
        await client.init()

        # Verify client creation
        mock_redis.assert_called_once()
        assert client.client_name == "test-init"
        assert isinstance(client.circuit_breaker, AsyncCircuitBreaker)
        assert isinstance(client.retry_policy, RetryPolicy)

        await client.close()


@pytest.mark.asyncio
async def test_create_redis_client():
    """Test Redis client creation."""
    with patch("redis.asyncio.Redis") as mock_redis:
        mock_redis.side_effect = Exception("Connection failed")

        # Should raise RedisUnavailableError
        client = AsyncResilientRedisClient(client_name="test-fail")
        with pytest.raises(RedisUnavailableError):
            await client._create_redis_client()


# Basic Redis operations tests
@pytest.mark.asyncio
async def test_ping(async_client, mock_redis):
    """Test ping operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.ping.return_value = True

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.ping()
    intermediate = await coroutine
    result = await intermediate

    assert result is True
    async_client.circuit_breaker.execute.assert_called_once()
    mock_redis_instance.ping.assert_called_once()


@pytest.mark.asyncio
async def test_get(async_client, mock_redis):
    """Test get operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.get.return_value = "value"

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.get("key")
    intermediate = await coroutine
    result = await intermediate

    assert result == "value"
    mock_redis_instance.get.assert_called_once_with("key")


@pytest.mark.asyncio
async def test_set(async_client, mock_redis):
    """Test set operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.set.return_value = True

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.set("key", "value", ex=60)
    intermediate = await coroutine
    result = await intermediate

    assert result is True
    mock_redis_instance.set.assert_called_once_with(
        "key", "value", ex=60, px=None, nx=False, xx=False
    )


@pytest.mark.asyncio
async def test_delete(async_client, mock_redis):
    """Test delete operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.delete.return_value = 1

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.delete("key1", "key2")
    intermediate = await coroutine
    result = await intermediate

    assert result == 1
    mock_redis_instance.delete.assert_called_once_with("key1", "key2")


@pytest.mark.asyncio
async def test_exists(async_client, mock_redis):
    """Test exists operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.exists.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.exists("key1", "key2")
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.exists.assert_called_once_with("key1", "key2")


@pytest.mark.asyncio
async def test_expire(async_client, mock_redis):
    """Test expire operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.expire.return_value = True

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.expire("key", 300)
    intermediate = await coroutine
    result = await intermediate

    assert result is True
    mock_redis_instance.expire.assert_called_once_with("key", 300)


# Hash operations tests
@pytest.mark.asyncio
async def test_hset(async_client, mock_redis):
    """Test hset operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.hset.return_value = 1

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.hset("hash", "field", "value")
    intermediate = await coroutine
    result = await intermediate

    assert result == 1
    mock_redis_instance.hset.assert_called_once_with("hash", "field", "value")


@pytest.mark.asyncio
async def test_hget(async_client, mock_redis):
    """Test hget operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.hget.return_value = "value"

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.hget("hash", "field")
    intermediate = await coroutine
    result = await intermediate

    assert result == "value"
    mock_redis_instance.hget.assert_called_once_with("hash", "field")


@pytest.mark.asyncio
async def test_hgetall(async_client, mock_redis):
    """Test hgetall operation."""
    _, mock_redis_instance = mock_redis
    expected = {"field1": "value1", "field2": "value2"}
    mock_redis_instance.hgetall.return_value = expected

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.hgetall("hash")
    intermediate = await coroutine
    result = await intermediate

    assert result == expected
    mock_redis_instance.hgetall.assert_called_once_with("hash")


@pytest.mark.asyncio
async def test_hset_dict(async_client, mock_redis):
    """Test hset_dict operation."""
    _, mock_redis_instance = mock_redis
    mapping = {"field1": "value1", "field2": "value2"}
    mock_redis_instance.hset.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.hset_dict("hash", mapping)
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.hset.assert_called_once_with("hash", mapping=mapping)


@pytest.mark.asyncio
async def test_hdel(async_client, mock_redis):
    """Test hdel operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.hdel.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.hdel("hash", "field1", "field2")
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.hdel.assert_called_once_with("hash", "field1", "field2")


# Sorted set operations tests
@pytest.mark.asyncio
async def test_zadd(async_client, mock_redis):
    """Test zadd operation."""
    _, mock_redis_instance = mock_redis
    mapping = {"member1": 1.0, "member2": 2.0}
    mock_redis_instance.zadd.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.zadd("zset", mapping)
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.zadd.assert_called_once_with(
        "zset", mapping, nx=False, xx=False, ch=False, incr=False
    )


@pytest.mark.asyncio
async def test_zrange(async_client, mock_redis):
    """Test zrange operation."""
    _, mock_redis_instance = mock_redis
    expected = ["member1", "member2"]
    mock_redis_instance.zrange.return_value = expected

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.zrange("zset", 0, -1)
    intermediate = await coroutine
    result = await intermediate

    assert result == expected
    mock_redis_instance.zrange.assert_called_once_with(
        "zset", 0, -1, desc=False, withscores=False, score_cast_func=float
    )


@pytest.mark.asyncio
async def test_zrangebyscore(async_client, mock_redis):
    """Test zrangebyscore operation."""
    _, mock_redis_instance = mock_redis
    expected = ["member1", "member2"]
    mock_redis_instance.zrangebyscore.return_value = expected

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.zrangebyscore("zset", 0, 100)
    intermediate = await coroutine
    result = await intermediate

    assert result == expected
    mock_redis_instance.zrangebyscore.assert_called_once_with(
        "zset", 0, 100, start=None, num=None, withscores=False, score_cast_func=float
    )


@pytest.mark.asyncio
async def test_zrem(async_client, mock_redis):
    """Test zrem operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.zrem.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.zrem("zset", "member1", "member2")
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.zrem.assert_called_once_with("zset", "member1", "member2")


@pytest.mark.asyncio
async def test_zcard(async_client, mock_redis):
    """Test zcard operation."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.zcard.return_value = 2

    # Need to await twice - once for the decorator, once for the circuit breaker
    coroutine = async_client.zcard("zset")
    intermediate = await coroutine
    result = await intermediate

    assert result == 2
    mock_redis_instance.zcard.assert_called_once_with("zset")


# Error handling tests
@pytest.mark.asyncio
async def test_connection_error(async_client):
    """Test handling of connection errors."""
    # Mock client.circuit_breaker to pass through the function and let it raise the error
    async_client.circuit_breaker.execute.side_effect = redis.ConnectionError(
        "Connection refused"
    )

    # Should raise RedisUnavailableError
    with pytest.raises(RedisUnavailableError):
        await async_client.get("key")


@pytest.mark.asyncio
async def test_timeout_error(async_client):
    """Test handling of timeout errors."""
    # Mock client.circuit_breaker to pass through the function and let it raise the error
    async_client.circuit_breaker.execute.side_effect = asyncio.TimeoutError(
        "Operation timed out"
    )

    # Should raise RedisTimeoutError
    with pytest.raises(RedisTimeoutError):
        await async_client.get("key")


@pytest.mark.asyncio
async def test_other_error(async_client):
    """Test handling of other errors."""
    # Mock client.circuit_breaker to pass through the function and let it raise the error
    error = redis.ResponseError("Invalid command")
    async_client.circuit_breaker.execute.side_effect = error

    # Should pass through the original error
    with pytest.raises(redis.ResponseError):
        await async_client.get("key")


# Circuit breaker tests
@pytest.mark.asyncio
async def test_circuit_breaker_execution(async_client):
    """Test that operations go through the circuit breaker."""
    # Reset mock
    async_client.circuit_breaker = AsyncMock(spec=AsyncCircuitBreaker)
    async_client.circuit_breaker.execute.return_value = "value"

    # Call Redis operation
    result = await async_client.get("key")

    # Verify circuit breaker was used
    async_client.circuit_breaker.execute.assert_called_once()
    assert result == "value"


# Store with retry tests
@pytest.mark.asyncio
async def test_store_with_retry_success(async_client):
    """Test successful store operation."""
    store_func = AsyncMock(return_value=True)
    agent_id = "agent1"
    state_data = {"key": "value"}

    result = await async_client.store_with_retry(agent_id, state_data, store_func)

    assert result is True
    store_func.assert_called_once_with(agent_id, state_data)
    # Ensure recovery queue was not used
    async_client.recovery_queue.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_store_with_retry_failure_normal_priority(async_client):
    """Test store operation failure with normal priority."""
    store_func = AsyncMock(side_effect=RedisUnavailableError("Test error"))
    agent_id = "agent1"
    state_data = {"key": "value"}

    with patch("uuid.uuid4", return_value="test-uuid"):
        result = await async_client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.NORMAL
        )

    assert result is False
    store_func.assert_called_once_with(agent_id, state_data)
    # Verify operation was enqueued for retry
    async_client.recovery_queue.enqueue.assert_called_once()

    # Verify correct operation was enqueued
    args, kwargs = async_client.recovery_queue.enqueue.call_args
    operation = args[0]
    assert isinstance(operation, AsyncStoreOperation)
    assert operation.agent_id == agent_id
    assert operation.state_data == state_data
    assert kwargs["priority"] == 3  # 4 - NORMAL(1) = 3


@pytest.mark.asyncio
async def test_store_with_retry_failure_high_priority(async_client):
    """Test store operation failure with high priority."""
    store_func = AsyncMock(side_effect=RedisUnavailableError("Test error"))
    agent_id = "agent1"
    state_data = {"key": "value"}

    with patch("uuid.uuid4", return_value="test-uuid"):
        result = await async_client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.HIGH
        )

    assert result is False
    store_func.assert_called_once_with(agent_id, state_data)
    # Verify operation was enqueued for retry
    async_client.recovery_queue.enqueue.assert_called_once()

    # Verify correct operation was enqueued with high priority
    args, kwargs = async_client.recovery_queue.enqueue.call_args
    assert kwargs["priority"] == 2  # 4 - HIGH(2) = 2


@pytest.mark.asyncio
async def test_store_with_retry_failure_low_priority(async_client):
    """Test store operation failure with low priority."""
    store_func = AsyncMock(side_effect=RedisUnavailableError("Test error"))
    agent_id = "agent1"
    state_data = {"key": "value"}

    result = await async_client.store_with_retry(
        agent_id, state_data, store_func, priority=Priority.LOW
    )

    assert result is False
    store_func.assert_called_once_with(agent_id, state_data)
    # Verify operation was NOT enqueued for retry (low priority)
    async_client.recovery_queue.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_store_with_retry_failure_critical_priority(async_client):
    """Test store operation failure with critical priority."""
    # Create a side effect that fails first, then succeeds
    store_func = AsyncMock()
    store_func.side_effect = [
        RedisUnavailableError("Test error"),  # First call fails
        True,  # Second call succeeds
    ]

    agent_id = "agent1"
    state_data = {"key": "value"}

    with patch(
        "agent_memory.storage.async_redis_client.async_exponential_backoff", AsyncMock()
    ) as mock_backoff:
        result = await async_client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.CRITICAL
        )

    assert result is True
    # Verify immediate retry was attempted
    assert store_func.call_count == 2
    # Verify backoff was called once between retries
    mock_backoff.assert_called_once()


@pytest.mark.asyncio
async def test_store_with_retry_failure_critical_all_retries_fail(async_client):
    """Test critical priority store with all retries failing."""
    # Create a side effect that always fails
    store_func = AsyncMock(side_effect=RedisUnavailableError("Test error"))

    agent_id = "agent1"
    state_data = {"key": "value"}

    # Override the retry attempts to 2 for faster test
    async_client.critical_retry_attempts = 2

    with patch(
        "agent_memory.storage.async_redis_client.async_exponential_backoff", AsyncMock()
    ):
        result = await async_client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.CRITICAL
        )

    assert result is False
    # Verify all retries were attempted (1 initial + 2 retries = 3)
    assert store_func.call_count == 3
    # Verify operation was NOT enqueued (all immediate retries were attempted)
    async_client.recovery_queue.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_double_await(async_client, mock_redis):
    """Test the execute_with_double_await helper method."""
    _, mock_redis_instance = mock_redis
    mock_redis_instance.get.return_value = "value"

    # Use the helper method to handle the double await
    result = await AsyncResilientRedisClient.execute_with_double_await(
        async_client.get("key")
    )

    assert result == "value"
    mock_redis_instance.get.assert_called_once_with("key")
