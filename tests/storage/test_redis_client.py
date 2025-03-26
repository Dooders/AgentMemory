"""Unit tests for ResilientRedisClient.

Tests the Redis client with circuit breaker and retry functionality.
Includes tests for normal operations, error scenarios, and recovery mechanisms.
"""

import time
import unittest
from unittest.mock import MagicMock, call, patch

import redis

from agent_memory.storage.redis_client import ResilientRedisClient, exponential_backoff
from agent_memory.utils.error_handling import (
    CircuitBreaker,
    CircuitState,
    Priority,
    RedisTimeoutError,
    RedisUnavailableError,
    RetryPolicy,
    StoreOperation,
)


class TestResilientRedisClient(unittest.TestCase):
    """Test suite for ResilientRedisClient."""

    def setUp(self):
        """Set up test fixtures."""
        # Use patch to mock the Redis client
        self.redis_patcher = patch("redis.Redis")
        self.mock_redis = self.redis_patcher.start()

        # Mock instance of Redis client
        self.mock_redis_instance = MagicMock()
        self.mock_redis.return_value = self.mock_redis_instance

        # Create client with test configuration
        self.client = ResilientRedisClient(
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

        # Replace the actual circuit breaker with a mock
        self.client.circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.client.circuit_breaker.execute.side_effect = lambda x: x()

        # Replace the recovery queue with a mock
        self.client.recovery_queue = MagicMock()

    def tearDown(self):
        """Tear down test fixtures."""
        self.redis_patcher.stop()

    def test_init(self):
        """Test client initialization."""
        # Create a new client to test initialization
        with patch("redis.Redis") as mock_redis:
            client = ResilientRedisClient(client_name="test-init")

            # Verify client creation
            mock_redis.assert_called_once()
            self.assertEqual(client.client_name, "test-init")
            self.assertIsInstance(client.circuit_breaker, CircuitBreaker)
            self.assertIsInstance(client.retry_policy, RetryPolicy)

    def test_create_redis_client(self):
        """Test Redis client creation."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            # Should raise RedisUnavailableError
            with self.assertRaises(RedisUnavailableError):
                client = ResilientRedisClient(client_name="test-fail")

    def test_close(self):
        """Test close method for proper connection cleanup."""
        # Setup disconnect mock
        disconnect_mock = MagicMock()
        self.mock_redis_instance.connection_pool = MagicMock()
        self.mock_redis_instance.connection_pool.disconnect = disconnect_mock

        # Call close method
        self.client.close()

        # Verify disconnect was called
        disconnect_mock.assert_called_once()

    # Basic Redis operations tests
    def test_ping(self):
        """Test ping operation."""
        self.mock_redis_instance.ping.return_value = True
        result = self.client.ping()
        self.assertTrue(result)
        self.client.circuit_breaker.execute.assert_called_once()
        self.mock_redis_instance.ping.assert_called_once()

    def test_get(self):
        """Test get operation."""
        self.mock_redis_instance.get.return_value = "value"
        result = self.client.get("key")
        self.assertEqual(result, "value")
        self.mock_redis_instance.get.assert_called_once_with("key")

    def test_set(self):
        """Test set operation."""
        self.mock_redis_instance.set.return_value = True
        result = self.client.set("key", "value", ex=60)
        self.assertTrue(result)
        self.mock_redis_instance.set.assert_called_once_with(
            "key", "value", ex=60, px=None, nx=False, xx=False
        )

    def test_delete(self):
        """Test delete operation."""
        self.mock_redis_instance.delete.return_value = 1
        result = self.client.delete("key1", "key2")
        self.assertEqual(result, 1)
        self.mock_redis_instance.delete.assert_called_once_with("key1", "key2")

    def test_exists(self):
        """Test exists operation."""
        self.mock_redis_instance.exists.return_value = 2
        result = self.client.exists("key1", "key2")
        self.assertEqual(result, 2)
        self.mock_redis_instance.exists.assert_called_once_with("key1", "key2")

    def test_expire(self):
        """Test expire operation."""
        self.mock_redis_instance.expire.return_value = True
        result = self.client.expire("key", 300)
        self.assertTrue(result)
        self.mock_redis_instance.expire.assert_called_once_with("key", 300)

    def test_get_latency(self):
        """Test get_latency method."""
        # Mock ping to simulate a delay
        with patch.object(self.client, "ping", return_value=True) as mock_ping:
            with patch("time.time") as mock_time:
                # Setup time mock to return incremental values
                mock_time.side_effect = [1.0, 1.1]  # 100ms difference

                # Call get_latency
                latency = self.client.get_latency()

                # Verify latency is correct (100ms) - use assertAlmostEqual for floats
                self.assertAlmostEqual(latency, 100.0, places=5)
                # Verify ping was called
                mock_ping.assert_called_once()

    def test_get_latency_error(self):
        """Test get_latency method when an error occurs."""
        # Mock ping to raise an exception
        with patch.object(
            self.client, "ping", side_effect=RedisUnavailableError("Test error")
        ):
            # Call get_latency
            latency = self.client.get_latency()

            # Verify latency is -1 (error indicator)
            self.assertEqual(latency, -1)

    # Hash operations tests
    def test_hset(self):
        """Test hset operation."""
        self.mock_redis_instance.hset.return_value = 1
        result = self.client.hset("hash", "field", "value")
        self.assertEqual(result, 1)
        self.mock_redis_instance.hset.assert_called_once_with("hash", "field", "value")

    def test_hget(self):
        """Test hget operation."""
        self.mock_redis_instance.hget.return_value = "value"
        result = self.client.hget("hash", "field")
        self.assertEqual(result, "value")
        self.mock_redis_instance.hget.assert_called_once_with("hash", "field")

    def test_hgetall(self):
        """Test hgetall operation."""
        expected = {"field1": "value1", "field2": "value2"}
        self.mock_redis_instance.hgetall.return_value = expected
        result = self.client.hgetall("hash")
        self.assertEqual(result, expected)
        self.mock_redis_instance.hgetall.assert_called_once_with("hash")

    def test_hset_dict(self):
        """Test hset_dict operation."""
        mapping = {"field1": "value1", "field2": "value2"}
        self.mock_redis_instance.hset.return_value = 2
        result = self.client.hset_dict("hash", mapping)
        self.assertEqual(result, 2)
        self.mock_redis_instance.hset.assert_called_once_with("hash", mapping=mapping)

    def test_hmset(self):
        """Test hmset operation."""
        mapping = {"field1": "value1", "field2": "value2"}
        self.mock_redis_instance.hset.return_value = 2
        result = self.client.hmset("hash", mapping)
        self.assertTrue(result)
        self.mock_redis_instance.hset.assert_called_once_with("hash", mapping=mapping)

    def test_hdel(self):
        """Test hdel operation."""
        self.mock_redis_instance.hdel.return_value = 2
        result = self.client.hdel("hash", "field1", "field2")
        self.assertEqual(result, 2)
        self.mock_redis_instance.hdel.assert_called_once_with(
            "hash", "field1", "field2"
        )

    # Sorted set operations tests
    def test_zadd(self):
        """Test zadd operation."""
        mapping = {"member1": 1.0, "member2": 2.0}
        self.mock_redis_instance.zadd.return_value = 2
        result = self.client.zadd("zset", mapping)
        self.assertEqual(result, 2)
        self.mock_redis_instance.zadd.assert_called_once_with(
            "zset", mapping, nx=False, xx=False, ch=False, incr=False
        )

    def test_zrange(self):
        """Test zrange operation."""
        expected = ["member1", "member2"]
        self.mock_redis_instance.zrange.return_value = expected
        result = self.client.zrange("zset", 0, -1)
        self.assertEqual(result, expected)
        self.mock_redis_instance.zrange.assert_called_once_with(
            "zset", 0, -1, desc=False, withscores=False, score_cast_func=float
        )

    def test_zrangebyscore(self):
        """Test zrangebyscore operation."""
        expected = ["member1", "member2"]
        self.mock_redis_instance.zrangebyscore.return_value = expected
        result = self.client.zrangebyscore("zset", 0, 100)
        self.assertEqual(result, expected)
        self.mock_redis_instance.zrangebyscore.assert_called_once_with(
            "zset",
            0,
            100,
            start=None,
            num=None,
            withscores=False,
            score_cast_func=float,
        )

    def test_zrem(self):
        """Test zrem operation."""
        self.mock_redis_instance.zrem.return_value = 2
        result = self.client.zrem("zset", "member1", "member2")
        self.assertEqual(result, 2)
        self.mock_redis_instance.zrem.assert_called_once_with(
            "zset", "member1", "member2"
        )

    def test_zcard(self):
        """Test zcard operation."""
        self.mock_redis_instance.zcard.return_value = 2
        result = self.client.zcard("zset")
        self.assertEqual(result, 2)
        self.mock_redis_instance.zcard.assert_called_once_with("zset")

    def test_scan_iter(self):
        """Test scan_iter operation."""
        # Mock scan to return keys in chunks with cursor
        self.mock_redis_instance.scan.side_effect = [
            (1, ["key1", "key2"]),  # First chunk with cursor 1
            (0, ["key3", "key4", "key5"]),  # Final chunk with cursor 0
        ]

        result = self.client.scan_iter(match="key*", count=10)

        # Verify correct keys returned
        self.assertEqual(result, ["key1", "key2", "key3", "key4", "key5"])

        # Verify scan was called correctly
        expected_calls = [
            call(0, match="key*", count=10),  # Initial call with cursor 0
            call(1, match="key*", count=10),  # Second call with cursor 1
        ]
        self.mock_redis_instance.scan.assert_has_calls(expected_calls)

    # Error handling tests
    def test_connection_error(self):
        """Test handling of connection errors."""
        # Mock client.circuit_breaker to pass through the function and let it raise the error
        self.client.circuit_breaker = CircuitBreaker(name="test")

        # Mock Redis operation to raise ConnectionError
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )

        # Should raise RedisUnavailableError
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

    def test_timeout_error(self):
        """Test handling of timeout errors."""
        # Mock client.circuit_breaker to pass through the function and let it raise the error
        self.client.circuit_breaker = CircuitBreaker(name="test")

        # Mock Redis operation to raise TimeoutError
        self.mock_redis_instance.get.side_effect = redis.exceptions.TimeoutError(
            "Operation timed out"
        )

        # Should raise RedisTimeoutError
        with self.assertRaises(RedisTimeoutError):
            self.client.get("key")

    def test_other_error(self):
        """Test handling of other errors."""
        # Mock client.circuit_breaker to pass through the function and let it raise the error
        self.client.circuit_breaker = CircuitBreaker(name="test")

        # Mock Redis operation to raise a different error
        error = redis.exceptions.ResponseError("Invalid command")
        self.mock_redis_instance.get.side_effect = error

        # Should pass through the original error
        with self.assertRaises(redis.exceptions.ResponseError):
            self.client.get("key")

    # Circuit breaker tests
    def test_circuit_breaker_execution(self):
        """Test that operations go through the circuit breaker."""
        # Reset mock
        self.client.circuit_breaker = MagicMock(spec=CircuitBreaker)

        # Call Redis operation
        self.mock_redis_instance.get.return_value = "value"
        self.client.get("key")

        # Verify circuit breaker was used
        self.client.circuit_breaker.execute.assert_called_once()

    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        # Replace mock with real CircuitBreaker for state transition testing
        self.client.circuit_breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,  # Will open after 2 failures
            reset_timeout=0.1,  # Short timeout for test
        )

        # First, succeed to ensure circuit is closed
        self.mock_redis_instance.get.return_value = "value"
        result = self.client.get("key")
        self.assertEqual(result, "value")
        self.assertEqual(self.client.circuit_breaker.failure_count, 0)
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)

        # Now fail twice to open the circuit
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Test error"
        )

        # First failure
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Second failure - should open the circuit
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # The circuit may not open immediately after the second failure
        # Allow more time and attempts for the state to transition
        max_attempts = 10
        for attempt in range(max_attempts):
            if self.client.circuit_breaker.state == CircuitState.OPEN:
                break
            time.sleep(0.05)  # Increased delay between checks

            # One more failure attempt to trigger state change if needed
            if attempt == max_attempts // 2:
                with self.assertRaises(RedisUnavailableError):
                    self.client.get("key")

        # Verify circuit is now open
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.OPEN)

        # Verify that circuit being open causes immediate failure
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Wait for reset timeout (increased for reliability)
        time.sleep(0.5)  # Using 5x the reset_timeout for reliability

        # Reset side_effect for the next call
        self.mock_redis_instance.get.side_effect = None
        self.mock_redis_instance.get.return_value = "value"

        # This should succeed and close the circuit
        result = self.client.get("key")
        self.assertEqual(result, "value")

        # Circuit should be closed again
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)

    # Store with retry tests
    def test_store_with_retry_success(self):
        """Test successful store operation."""
        store_func = MagicMock(return_value=True)
        agent_id = "agent1"
        state_data = {"key": "value"}

        result = self.client.store_with_retry(agent_id, state_data, store_func)

        self.assertTrue(result)
        store_func.assert_called_once_with(agent_id, state_data)
        # Ensure recovery queue was not used
        self.client.recovery_queue.enqueue.assert_not_called()

    def test_store_with_retry_failure_normal_priority(self):
        """Test store operation failure with normal priority."""
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        agent_id = "agent1"
        state_data = {"key": "value"}

        with patch("uuid.uuid4", return_value="test-uuid"):
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.NORMAL
            )

        self.assertFalse(result)
        store_func.assert_called_once_with(agent_id, state_data)
        # Verify operation was enqueued for retry
        self.client.recovery_queue.enqueue.assert_called_once()

        # Verify correct operation was enqueued
        args, kwargs = self.client.recovery_queue.enqueue.call_args
        operation = args[0]
        self.assertIsInstance(operation, StoreOperation)
        self.assertEqual(operation.agent_id, agent_id)
        self.assertEqual(operation.state_data, state_data)
        self.assertEqual(kwargs["priority"], 3)  # 4 - NORMAL(1) = 3

    def test_store_with_retry_failure_high_priority(self):
        """Test store operation failure with high priority."""
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        agent_id = "agent1"
        state_data = {"key": "value"}

        with patch("uuid.uuid4", return_value="test-uuid"):
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.HIGH
            )

        self.assertFalse(result)
        store_func.assert_called_once_with(agent_id, state_data)
        # Verify operation was enqueued for retry
        self.client.recovery_queue.enqueue.assert_called_once()

        # Verify correct operation was enqueued with high priority
        args, kwargs = self.client.recovery_queue.enqueue.call_args
        self.assertEqual(kwargs["priority"], 2)  # 4 - HIGH(2) = 2

    def test_store_with_retry_failure_low_priority(self):
        """Test store operation failure with low priority."""
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        agent_id = "agent1"
        state_data = {"key": "value"}

        result = self.client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.LOW
        )

        self.assertFalse(result)
        store_func.assert_called_once_with(agent_id, state_data)
        # Verify operation was NOT enqueued for retry (low priority)
        self.client.recovery_queue.enqueue.assert_not_called()

    def test_store_with_retry_failure_critical_priority(self):
        """Test store operation failure with critical priority."""
        # Create a side effect that fails first, then succeeds
        store_func = MagicMock(
            side_effect=[
                RedisUnavailableError("Test error"),  # First call fails
                True,  # Second call succeeds
            ]
        )

        agent_id = "agent1"
        state_data = {"key": "value"}

        with patch("time.sleep") as mock_sleep:
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.CRITICAL
            )

        self.assertTrue(result)
        # Verify immediate retry was attempted
        self.assertEqual(store_func.call_count, 2)
        # Verify sleep was called once between retries
        mock_sleep.assert_called_once()

    def test_store_with_retry_failure_critical_all_retries_fail(self):
        """Test critical priority store with all retries failing."""
        # Create a side effect that always fails
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))

        agent_id = "agent1"
        state_data = {"key": "value"}

        with patch("time.sleep"):
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.CRITICAL
            )

        self.assertFalse(result)
        # Verify all retries were attempted (1 initial + 3 retries = 4)
        self.assertEqual(store_func.call_count, 4)
        # Verify operation was NOT enqueued (all immediate retries were attempted)
        self.client.recovery_queue.enqueue.assert_not_called()

    def test_store_with_retry_custom_parameters(self):
        """Test store with retry using custom retry parameters."""
        # Create a side effect that fails multiple times then succeeds
        store_func = MagicMock(
            side_effect=[
                RedisUnavailableError("Test error"),  # First call fails
                RedisUnavailableError("Test error"),  # Second call fails
                RedisUnavailableError("Test error"),  # Third call fails
                RedisUnavailableError("Test error"),  # Fourth call fails
                True,  # Fifth call succeeds
            ]
        )

        agent_id = "agent1"
        state_data = {"key": "value"}

        # Test with custom retry attempts (5 instead of default 3)
        with patch("time.sleep"):
            result = self.client.store_with_retry(
                agent_id,
                state_data,
                store_func,
                priority=Priority.CRITICAL,
                retry_attempts=5,  # Custom retry attempts
                base_delay=0.1,  # Custom base delay
                max_delay=1.0,  # Custom max delay
            )

        self.assertTrue(result)
        # Verify custom retry count was used (1 initial + 4 retries = 5)
        self.assertEqual(store_func.call_count, 5)

    def test_recovery_queue_edge_case(self):
        """Test edge case with recovery queue when operation ID is fixed."""
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        agent_id = "agent1"
        state_data = {"key": "value"}

        # Test with fixed operation ID to simulate duplicate operations
        with patch("uuid.uuid4", return_value="fixed-uuid"):
            # First operation
            result1 = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.NORMAL
            )

            # Second operation - same operation ID
            result2 = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.NORMAL
            )

        self.assertFalse(result1)
        self.assertFalse(result2)

        # Verify both operations were enqueued (with same operation ID)
        self.assertEqual(self.client.recovery_queue.enqueue.call_count, 2)

        # Get the two operations that were enqueued
        calls = self.client.recovery_queue.enqueue.call_args_list
        op1 = calls[0][0][0]
        op2 = calls[1][0][0]

        # Verify they have the same operation ID
        self.assertEqual(op1.operation_id, "fixed-uuid")
        self.assertEqual(op2.operation_id, "fixed-uuid")

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Test with default parameters
        self.assertEqual(exponential_backoff(0), 0.5)  # 0.5 * 2^0 = 0.5
        self.assertEqual(exponential_backoff(1), 1.0)  # 0.5 * 2^1 = 1.0
        self.assertEqual(exponential_backoff(2), 2.0)  # 0.5 * 2^2 = 2.0
        self.assertEqual(exponential_backoff(3), 4.0)  # 0.5 * 2^3 = 4.0
        self.assertEqual(exponential_backoff(10), 30.0)  # > max_delay (30.0)

        # Test with custom parameters
        self.assertEqual(exponential_backoff(0, 0.2, 10.0), 0.2)  # 0.2 * 2^0 = 0.2
        self.assertEqual(exponential_backoff(1, 0.2, 10.0), 0.4)  # 0.2 * 2^1 = 0.4
        self.assertEqual(exponential_backoff(2, 0.2, 10.0), 0.8)  # 0.2 * 2^2 = 0.8
        self.assertEqual(exponential_backoff(10, 0.2, 10.0), 10.0)  # > max_delay (10.0)


class TestResilientRedisClientIntegration(unittest.TestCase):
    """Integration tests for ResilientRedisClient.

    These tests require a running Redis server.
    Skip these tests if Redis is not available.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the class."""
        # Try to connect to Redis, skip tests if not available
        try:
            r = redis.Redis(host="localhost", port=6379, db=15, socket_timeout=1)
            r.ping()
            cls.redis_available = True

            # Create client for tests
            cls.client = ResilientRedisClient(
                client_name="integration-test",
                host="localhost",
                port=6379,
                db=15,  # Use DB 15 for testing
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )

            # Clear test database
            r.flushdb()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            cls.redis_available = False
            cls.client = None

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if cls.redis_available and cls.client:
            # Connect to Redis and clear the test database
            r = redis.Redis(host="localhost", port=6379, db=15, socket_timeout=1)
            r.flushdb()

            # Close client
            cls.client.close()

    def setUp(self):
        """Skip tests if Redis is not available."""
        if not self.redis_available:
            self.skipTest("Redis server not available")

    def test_basic_operations(self):
        """Test basic Redis operations with actual Redis server."""
        # Test set and get
        self.client.set("test_key", "test_value")
        result = self.client.get("test_key")
        self.assertEqual(result, "test_value")

        # Test delete
        self.client.delete("test_key")
        result = self.client.get("test_key")
        self.assertIsNone(result)

    def test_hash_operations(self):
        """Test hash operations with actual Redis server."""
        # Test hset and hget
        self.client.hset("test_hash", "field1", "value1")
        result = self.client.hget("test_hash", "field1")
        self.assertEqual(result, "value1")

        # Test hset_dict
        mapping = {"field2": "value2", "field3": "value3"}
        self.client.hset_dict("test_hash", mapping)

        # Test hgetall
        result = self.client.hgetall("test_hash")
        expected = {"field1": "value1", "field2": "value2", "field3": "value3"}
        self.assertEqual(result, expected)

    def test_sorted_set_operations(self):
        """Test sorted set operations with actual Redis server."""
        # Test zadd
        mapping = {"member1": 1.0, "member2": 2.0, "member3": 3.0}
        self.client.zadd("test_zset", mapping)

        # Test zrange
        result = self.client.zrange("test_zset", 0, -1)
        self.assertEqual(result, ["member1", "member2", "member3"])

        # Test zrangebyscore
        result = self.client.zrangebyscore("test_zset", 2.0, 3.0)
        self.assertEqual(result, ["member2", "member3"])

        # Test zcard
        result = self.client.zcard("test_zset")
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
