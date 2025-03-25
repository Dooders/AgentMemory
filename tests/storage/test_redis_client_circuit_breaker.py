"""Circuit breaker tests for ResilientRedisClient.

These tests focus on the circuit breaker functionality in the Redis client,
simulating various failure scenarios and testing how the circuit breaker
handles them.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import redis

from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RedisUnavailableError,
)


class TestResilientRedisClientCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality in ResilientRedisClient."""

    def setUp(self):
        """Set up test fixtures."""
        # Use patch to mock the Redis client
        self.redis_patcher = patch("redis.Redis")
        self.mock_redis = self.redis_patcher.start()

        # Mock instance of Redis client
        self.mock_redis_instance = MagicMock()
        self.mock_redis.return_value = self.mock_redis_instance

        # Create client with a low circuit breaker threshold for testing
        self.client = ResilientRedisClient(
            client_name="circuit-test",
            circuit_threshold=2,  # Open after 2 failures
            circuit_reset_timeout=1,  # Reset after 1 second (for faster testing)
        )

        # Use real circuit breaker for these tests
        self.original_circuit_breaker = self.client.circuit_breaker

        # Mock Redis client to simulate failures
        self.client.client = self.mock_redis_instance

    def tearDown(self):
        """Tear down test fixtures."""
        self.redis_patcher.stop()

    def test_circuit_breaker_closed_to_open(self):
        """Test circuit transitions from closed to open after failures."""
        # Initial state should be closed
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)

        # Simulate two connection failures in a row
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )

        # First failure
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Circuit should still be closed after first failure
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.client.circuit_breaker.failure_count, 1)

        # Second failure
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Circuit should be open after second failure
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.OPEN)
        self.assertEqual(self.client.circuit_breaker.failure_count, 2)

    def test_circuit_open_blocks_operations(self):
        """Test that open circuit blocks operations."""
        # Manually set circuit to open state
        self.client.circuit_breaker.state = CircuitState.OPEN
        self.client.circuit_breaker.failure_count = 2
        self.client.circuit_breaker.last_failure_time = time.time()

        # Try to execute operation
        with self.assertRaises(CircuitOpenError):
            self.client.get("key")

        # Redis operation should not be called
        self.mock_redis_instance.get.assert_not_called()

    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions from open to half-open after timeout."""
        # Set circuit breaker to open state with last failure time in the past
        self.client.circuit_breaker.state = CircuitState.OPEN
        self.client.circuit_breaker.failure_count = 2
        self.client.circuit_breaker.last_failure_time = time.time() - 2  # 2 seconds ago

        # Mock Redis to return success
        self.mock_redis_instance.get.return_value = "value"

        # Execute operation
        result = self.client.get("key")

        # Operation should succeed and circuit should be closed
        self.assertEqual(result, "value")
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.client.circuit_breaker.failure_count, 0)

    def test_circuit_remains_open_on_half_open_failure(self):
        """Test that circuit remains open if half-open request fails."""
        # Set circuit breaker to open state with last failure time in the past
        self.client.circuit_breaker.state = CircuitState.OPEN
        self.client.circuit_breaker.failure_count = 2
        self.client.circuit_breaker.last_failure_time = time.time() - 2  # 2 seconds ago

        # Mock Redis to fail
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )

        # Execute operation
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Circuit should remain open with failure count reset to 1
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.OPEN)
        self.assertEqual(self.client.circuit_breaker.failure_count, 1)

    def test_circuit_multiple_operations(self):
        """Test circuit breaker with multiple operations sequence."""
        # Start with a closed circuit
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)

        # Step 1: First operation fails
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key1")

        # Circuit still closed, failure count = 1
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.client.circuit_breaker.failure_count, 1)

        # Step 2: Second operation succeeds
        self.mock_redis_instance.get.side_effect = None
        self.mock_redis_instance.get.return_value = "value"
        result = self.client.get("key2")

        # Circuit closed, failure count reset to 0
        self.assertEqual(result, "value")
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.client.circuit_breaker.failure_count, 0)

        # Step 3: Two more operations fail
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key3")
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key4")

        # Circuit now open
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.OPEN)

        # Step 4: Operation while circuit is open
        with self.assertRaises(CircuitOpenError):
            self.client.get("key5")

        # Step 5: Wait for timeout
        last_failure = self.client.circuit_breaker.last_failure_time
        self.client.circuit_breaker.last_failure_time = (
            last_failure - 2
        )  # 2 seconds ago

        # Step 6: Operation succeeds after timeout
        self.mock_redis_instance.get.side_effect = None
        self.mock_redis_instance.get.return_value = "new value"
        result = self.client.get("key6")

        # Circuit should be closed again
        self.assertEqual(result, "new value")
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.client.circuit_breaker.failure_count, 0)

    def test_different_operations_share_circuit_breaker(self):
        """Test that different Redis operations share the same circuit breaker."""
        # Initial state is closed
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.CLOSED)

        # Make get() fail
        self.mock_redis_instance.get.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )
        with self.assertRaises(RedisUnavailableError):
            self.client.get("key")

        # Failure count should be 1
        self.assertEqual(self.client.circuit_breaker.failure_count, 1)

        # Make set() fail
        self.mock_redis_instance.set.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )
        with self.assertRaises(RedisUnavailableError):
            self.client.set("key", "value")

        # Circuit should now be open since we reached threshold
        self.assertEqual(self.client.circuit_breaker.state, CircuitState.OPEN)
        self.assertEqual(self.client.circuit_breaker.failure_count, 2)

        # All operations should be blocked
        with self.assertRaises(CircuitOpenError):
            self.client.delete("key")
        with self.assertRaises(CircuitOpenError):
            self.client.hset("hash", "field", "value")


if __name__ == "__main__":
    unittest.main()
