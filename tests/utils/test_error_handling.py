"""Unit tests for error handling utilities.

This module contains tests for the error_handling module in agent_memory/utils.
"""

import enum
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

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
    RetryPolicy,
    RetryableOperation,
    SQLitePermanentError,
    SQLiteTemporaryError,
    STMError,
    StoreOperation,
    TransactionError,
)


class TestErrorClasses(unittest.TestCase):
    """Test the error classes hierarchy."""

    def test_error_hierarchy(self):
        """Test that error classes have the correct inheritance."""
        # Base error class
        self.assertTrue(issubclass(MemoryError, Exception))
        
        # Tier-specific errors
        self.assertTrue(issubclass(STMError, MemoryError))
        self.assertTrue(issubclass(IMError, MemoryError))
        self.assertTrue(issubclass(LTMError, MemoryError))
        
        # Storage-specific errors
        self.assertTrue(issubclass(RedisUnavailableError, STMError))
        self.assertTrue(issubclass(RedisUnavailableError, IMError))
        self.assertTrue(issubclass(RedisTimeoutError, STMError))
        self.assertTrue(issubclass(RedisTimeoutError, IMError))
        
        self.assertTrue(issubclass(SQLiteTemporaryError, LTMError))
        self.assertTrue(issubclass(SQLitePermanentError, LTMError))
        
        # Operational errors
        self.assertTrue(issubclass(MemoryTransitionError, MemoryError))
        self.assertTrue(issubclass(EmbeddingGenerationError, MemoryError))
        self.assertTrue(issubclass(TransactionError, MemoryError))
        self.assertTrue(issubclass(CircuitOpenError, MemoryError))


class TestCircuitState(unittest.TestCase):
    """Test the CircuitState enum."""

    def test_circuit_states(self):
        """Test that CircuitState has the correct values."""
        self.assertIsInstance(CircuitState, type(enum.Enum))
        self.assertEqual(CircuitState.CLOSED.value, "closed")
        self.assertEqual(CircuitState.OPEN.value, "open")
        self.assertEqual(CircuitState.HALF_OPEN.value, "half_open")


class TestPriority(unittest.TestCase):
    """Test the Priority enum."""

    def test_priority_levels(self):
        """Test that Priority has the correct values."""
        self.assertIsInstance(Priority, type(enum.IntEnum))
        self.assertEqual(Priority.LOW.value, 0)
        self.assertEqual(Priority.NORMAL.value, 1)
        self.assertEqual(Priority.HIGH.value, 2)
        self.assertEqual(Priority.CRITICAL.value, 3)


class TestCircuitBreaker(unittest.TestCase):
    """Test the CircuitBreaker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.circuit = CircuitBreaker(
            name="test_circuit",
            failure_threshold=2,
            reset_timeout=1  # Short timeout for testing
        )

    def test_initialization(self):
        """Test initialization of CircuitBreaker."""
        self.assertEqual(self.circuit.name, "test_circuit")
        self.assertEqual(self.circuit.failure_threshold, 2)
        self.assertEqual(self.circuit.reset_timeout, 1)
        self.assertEqual(self.circuit.failure_count, 0)
        self.assertEqual(self.circuit.last_failure_time, 0)
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)

    def test_successful_execution(self):
        """Test executing an operation successfully."""
        # Create a mock operation that returns a value
        operation = MagicMock(return_value="success")
        
        # Execute through the circuit breaker
        result = self.circuit.execute(operation)
        
        # Check that operation was called and result passed through
        operation.assert_called_once()
        self.assertEqual(result, "success")
        
        # Circuit should remain closed with no failures
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failure_count, 0)

    def test_failed_execution_below_threshold(self):
        """Test executing an operation that fails, but below the threshold."""
        # Create a mock operation that raises an exception
        operation = MagicMock(side_effect=ValueError("test error"))
        
        # Execute should re-raise the exception
        with self.assertRaises(ValueError):
            self.circuit.execute(operation)
        
        # Circuit should remain closed but failure count increases
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failure_count, 1)
        self.assertGreater(self.circuit.last_failure_time, 0)

    def test_failed_execution_above_threshold(self):
        """Test executing an operation that fails above the threshold."""
        # Create a mock operation that raises an exception
        operation = MagicMock(side_effect=ValueError("test error"))
        
        # First failure
        with self.assertRaises(ValueError):
            self.circuit.execute(operation)
        
        # Second failure should trip the circuit
        with self.assertRaises(ValueError):
            self.circuit.execute(operation)
        
        # Circuit should now be open
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        self.assertEqual(self.circuit.failure_count, 2)

    def test_open_circuit_blocks_execution(self):
        """Test that an open circuit blocks execution."""
        # Force circuit to open state
        self.circuit.failure_count = 2
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = time.time()
        
        # Create a mock operation
        operation = MagicMock()
        
        # Execution should be blocked with CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            self.circuit.execute(operation)
        
        # Operation should not have been called
        operation.assert_not_called()

    def test_half_open_allows_single_test(self):
        """Test that a half-open circuit allows a single test execution."""
        # Force circuit to open state with expired timeout
        self.circuit.failure_count = 2
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = time.time() - 2  # Longer than reset_timeout
        
        # Create a successful mock operation
        operation = MagicMock(return_value="success")
        
        # Execute should work and reset the circuit
        result = self.circuit.execute(operation)
        
        # Circuit should now be closed again
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failure_count, 0)
        self.assertEqual(result, "success")

    def test_half_open_fails_resets_to_open(self):
        """Test that a half-open circuit that fails resets to open."""
        # Force circuit to open state with expired timeout
        self.circuit.failure_count = 2
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = time.time() - 2  # Longer than reset_timeout
        
        # Create a failing mock operation
        operation = MagicMock(side_effect=ValueError("test error"))
        
        # Execute should fail and reset to open
        with self.assertRaises(ValueError):
            self.circuit.execute(operation)
        
        # Circuit should be open again with reset timeout
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        self.assertEqual(self.circuit.failure_count, 1)  # Reset to 1


class TestRetryPolicy(unittest.TestCase):
    """Test the RetryPolicy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.policy = RetryPolicy(
            max_retries=3,
            base_delay=0.1,
            backoff_factor=2.0
        )
        # Enable ValueError handling for tests
        self.policy._include_value_error = True

    def test_initialization(self):
        """Test initialization of RetryPolicy."""
        self.assertEqual(self.policy.max_retries, 3)
        self.assertEqual(self.policy.base_delay, 0.1)
        self.assertEqual(self.policy.backoff_factor, 2.0)

    def test_get_retry_delay(self):
        """Test calculating retry delays with exponential backoff."""
        # First attempt (0-indexed)
        self.assertEqual(self.policy.get_retry_delay(0), 0.1)
        
        # Second attempt
        self.assertEqual(self.policy.get_retry_delay(1), 0.2)
        
        # Third attempt
        self.assertEqual(self.policy.get_retry_delay(2), 0.4)

    def test_should_retry(self):
        """Test determining whether to retry based on attempt and exception."""
        # Within max retries
        self.assertTrue(self.policy.should_retry(0, ValueError()))
        self.assertTrue(self.policy.should_retry(1, ValueError()))
        self.assertTrue(self.policy.should_retry(2, ValueError()))
        
        # Beyond max retries
        self.assertFalse(self.policy.should_retry(3, ValueError()))
        
        # Non-retryable exception (not a MemoryError)
        self.assertFalse(self.policy.should_retry(0, KeyboardInterrupt()))
        
        # Specific retryable exceptions
        self.assertTrue(self.policy.should_retry(0, RedisTimeoutError()))
        self.assertTrue(self.policy.should_retry(0, SQLiteTemporaryError()))
        
        # Non-retryable memory exceptions
        self.assertFalse(self.policy.should_retry(0, SQLitePermanentError()))


class TestRetryableOperation(unittest.TestCase):
    """Test the RetryableOperation class."""

    def test_initialization(self):
        """Test initialization of RetryableOperation."""
        op = RetryableOperation("test_op")
        self.assertEqual(op.operation_id, "test_op")

    def test_execute(self):
        """Test execute method raising NotImplementedError."""
        op = RetryableOperation("test_op")
        with self.assertRaises(NotImplementedError):
            op.execute()

    def test_str_representation(self):
        """Test string representation."""
        op = RetryableOperation("test_op")
        self.assertEqual(str(op), "Operation test_op")


class TestStoreOperation(unittest.TestCase):
    """Test the StoreOperation class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_store_function = MagicMock()
        self.store_op = StoreOperation(
            operation_id="test_store",
            agent_id="agent1",
            state_data={"key": "value"},
            store_function=self.mock_store_function
        )

    def test_initialization(self):
        """Test initialization of StoreOperation."""
        self.assertEqual(self.store_op.operation_id, "test_store")
        self.assertEqual(self.store_op.agent_id, "agent1")
        self.assertEqual(self.store_op.state_data, {"key": "value"})
        self.assertEqual(self.store_op.store_function, self.mock_store_function)

    def test_execute_success(self):
        """Test successful execution of store operation."""
        # Configure mock to return success
        self.mock_store_function.return_value = True
        
        # Execute should return True
        result = self.store_op.execute()
        self.assertTrue(result)
        
        # Store function should be called with correct args
        self.mock_store_function.assert_called_once_with("agent1", {"key": "value"})

    def test_execute_failure(self):
        """Test failed execution of store operation."""
        # Configure mock to return failure
        self.mock_store_function.return_value = False
        
        # Execute should return False
        result = self.store_op.execute()
        self.assertFalse(result)
        
        # Store function should be called with correct args
        self.mock_store_function.assert_called_once_with("agent1", {"key": "value"})

    def test_execute_exception(self):
        """Test store operation raising an exception."""
        # Configure mock to raise exception
        self.mock_store_function.side_effect = ValueError("test error")
        
        # Execute should raise the exception
        with self.assertRaises(ValueError):
            self.store_op.execute()


class TestRecoveryQueue(unittest.TestCase):
    """Test the RecoveryQueue class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a custom RetryPolicy with no delay for faster tests
        self.retry_policy = RetryPolicy(max_retries=2, base_delay=0, backoff_factor=1)
        self.recovery_queue = RecoveryQueue(
            worker_count=1,
            retry_policy=self.retry_policy,
            test_mode=True  # Set test mode to enable special test behavior
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.recovery_queue._running:
            self.recovery_queue.stop()

    def test_initialization(self):
        """Test initialization of RecoveryQueue."""
        self.assertEqual(self.recovery_queue.worker_count, 1)
        self.assertEqual(self.recovery_queue.retry_policy, self.retry_policy)
        self.assertFalse(self.recovery_queue._running)
        self.assertIsNone(self.recovery_queue._worker_thread)

    def test_start_stop(self):
        """Test starting and stopping the recovery queue."""
        # Start the queue
        self.recovery_queue.start()
        self.assertTrue(self.recovery_queue._running)
        self.assertIsNotNone(self.recovery_queue._worker_thread)
        
        # Stop the queue
        self.recovery_queue.stop()
        self.assertFalse(self.recovery_queue._running)
        
        # Thread should exit within reasonable time
        self.recovery_queue._worker_thread.join(timeout=1)
        self.assertFalse(self.recovery_queue._worker_thread.is_alive())

    def test_enqueue_operation(self):
        """Test enqueueing an operation."""
        # Create a recovery queue with test_mode=False for this specific test
        recovery_queue = RecoveryQueue(
            worker_count=1,
            retry_policy=self.retry_policy,
            test_mode=False  # Disable test mode to use the actual queue
        )
        
        # Create a mock operation
        mock_op = MagicMock(spec=RetryableOperation)
        mock_op.operation_id = "test_op"
        
        # Enqueue with normal priority
        recovery_queue.enqueue(mock_op, priority=1)
        
        # Check that it's in the queue with priority as first item in tuple
        self.assertEqual(recovery_queue._queue.qsize(), 1)
        
        # Get the item - need to use queue internals for testing
        priority, operation = recovery_queue._queue.queue[0]
        self.assertEqual(priority, -1)  # PriorityQueue uses negative for highest first
        self.assertEqual(operation, mock_op)
        
        # Clean up
        recovery_queue.stop()

    @patch('time.sleep', return_value=None)  # Don't actually sleep in tests
    def test_process_queue_successful_operation(self, mock_sleep):
        """Test processing a successful operation from the queue."""
        # Create a mock operation that succeeds
        mock_op = MagicMock(spec=RetryableOperation)
        mock_op.operation_id = "test_op"
        mock_op.execute.return_value = True
        
        # Start the queue and enqueue the operation
        self.recovery_queue.start()
        self.recovery_queue.enqueue(mock_op)
        
        # Wait for queue to process
        time.sleep(0.1)
        
        # Operation should have been executed
        mock_op.execute.assert_called_once()
        
        # Queue should be empty
        self.assertEqual(self.recovery_queue._queue.qsize(), 0)
        
        # Stop the queue
        self.recovery_queue.stop()

    @patch('time.sleep', return_value=None)  # Don't actually sleep in tests
    def test_process_queue_failed_operation_with_retry(self, mock_sleep):
        """Test processing a failed operation with retry."""
        # Create a mock operation that fails and then succeeds
        mock_op = MagicMock(spec=RetryableOperation)
        mock_op.operation_id = "test_op"
        mock_op.execute.side_effect = [
            RedisTimeoutError("test error"),  # First attempt fails
            True  # Second attempt succeeds
        ]
        
        # Start the queue and enqueue the operation
        self.recovery_queue.start()
        self.recovery_queue.enqueue(mock_op)
        
        # Wait for queue to process and retry
        time.sleep(0.2)
        
        # Operation should have been executed twice
        self.assertEqual(mock_op.execute.call_count, 2)
        
        # Queue should be empty
        self.assertEqual(self.recovery_queue._queue.qsize(), 0)
        
        # Stop the queue
        self.recovery_queue.stop()


# Parameterized tests
@pytest.mark.parametrize(
    "exception,is_retryable",
    [
        (RedisTimeoutError(), True),
        (RedisUnavailableError(), True),
        (SQLiteTemporaryError(), True),
        (SQLitePermanentError(), False),
        (MemoryTransitionError(), False),
        (ValueError(), False),
    ],
)
def test_retry_policy_exceptions(exception, is_retryable):
    """Test retry policy with various exceptions."""
    policy = RetryPolicy()
    assert policy.should_retry(0, exception) == is_retryable


@pytest.mark.parametrize(
    "attempt,expected_delay",
    [
        (0, 1.0),  # First attempt: base_delay
        (1, 2.0),  # Second attempt: base_delay * backoff_factor
        (2, 4.0),  # Third attempt: base_delay * backoff_factor^2
    ],
)
def test_retry_delay_calculation(attempt, expected_delay):
    """Test retry delay calculation."""
    policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0)
    assert policy.get_retry_delay(attempt) == expected_delay


if __name__ == "__main__":
    unittest.main() 