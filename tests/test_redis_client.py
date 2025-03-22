"""Unit tests for ResilientRedisClient.

Tests the Redis client with circuit breaker and retry functionality.
Includes tests for normal operations, error scenarios, and recovery mechanisms.
"""

import unittest
from unittest.mock import MagicMock, patch, call
import time
import redis

from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    CircuitBreaker,
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
        self.redis_patcher = patch('redis.Redis')
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
        with patch('redis.Redis') as mock_redis:
            client = ResilientRedisClient(client_name="test-init")
            
            # Verify client creation
            mock_redis.assert_called_once()
            self.assertEqual(client.client_name, "test-init")
            self.assertIsInstance(client.circuit_breaker, CircuitBreaker)
            self.assertIsInstance(client.retry_policy, RetryPolicy)

    def test_create_redis_client(self):
        """Test Redis client creation."""
        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            # Should raise RedisUnavailableError
            with self.assertRaises(RedisUnavailableError):
                client = ResilientRedisClient(client_name="test-fail")

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
        self.mock_redis_instance.hdel.assert_called_once_with("hash", "field1", "field2")

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
            "zset", 0, 100, start=None, num=None, withscores=False, score_cast_func=float
        )

    def test_zrem(self):
        """Test zrem operation."""
        self.mock_redis_instance.zrem.return_value = 2
        result = self.client.zrem("zset", "member1", "member2")
        self.assertEqual(result, 2)
        self.mock_redis_instance.zrem.assert_called_once_with("zset", "member1", "member2")

    def test_zcard(self):
        """Test zcard operation."""
        self.mock_redis_instance.zcard.return_value = 2
        result = self.client.zcard("zset")
        self.assertEqual(result, 2)
        self.mock_redis_instance.zcard.assert_called_once_with("zset")

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
        
        with patch('uuid.uuid4', return_value="test-uuid"):
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
        
        with patch('uuid.uuid4', return_value="test-uuid"):
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
        store_func = MagicMock(side_effect=[
            RedisUnavailableError("Test error"),  # First call fails
            True,                                 # Second call succeeds
        ])
        
        agent_id = "agent1"
        state_data = {"key": "value"}
        
        with patch('time.sleep') as mock_sleep:
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
        
        with patch('time.sleep'):
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.CRITICAL
            )
        
        self.assertFalse(result)
        # Verify all retries were attempted (1 initial + 3 retries = 4)
        self.assertEqual(store_func.call_count, 4)
        # Verify operation was NOT enqueued (all immediate retries were attempted)
        self.client.recovery_queue.enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main() 