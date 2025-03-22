"""Recovery queue tests for ResilientRedisClient.

These tests focus on the recovery queue functionality in the Redis client,
testing how failed operations are enqueued and retried.
"""

import unittest
from unittest.mock import MagicMock, patch, call
import time
import uuid

from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import (
    Priority,
    RecoveryQueue,
    RedisUnavailableError,
    RetryPolicy,
    StoreOperation,
)


class TestRedisRecoveryQueue(unittest.TestCase):
    """Test recovery queue functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the Redis client
        self.redis_patcher = patch('redis.Redis')
        self.mock_redis = self.redis_patcher.start()
        
        # Create a retry policy with predictable behavior for testing
        self.retry_policy = RetryPolicy(
            max_retries=2,
            base_delay=0.1,
            backoff_factor=2.0,
        )
        
        # Create a mock recovery queue
        self.mock_recovery_queue = MagicMock(spec=RecoveryQueue)
        
        # Create client with mocked components
        self.client = ResilientRedisClient(client_name="recovery-test")
        self.client.recovery_queue = self.mock_recovery_queue
        self.client.retry_policy = self.retry_policy

    def tearDown(self):
        """Tear down test fixtures."""
        self.redis_patcher.stop()

    def test_store_with_retry_enqueues_operation(self):
        """Test that failed operations are enqueued for retry."""
        # Mock store function that fails
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        
        # Test data
        agent_id = "test-agent"
        state_data = {"status": "active"}
        
        # UUID for predictable testing
        test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        with patch('uuid.uuid4', return_value=test_uuid):
            # Call store_with_retry
            result = self.client.store_with_retry(
                agent_id, state_data, store_func, priority=Priority.NORMAL
            )
        
        # Should return False since operation failed
        self.assertFalse(result)
        
        # Verify operation was enqueued
        self.mock_recovery_queue.enqueue.assert_called_once()
        
        # Check the enqueued operation
        args, kwargs = self.mock_recovery_queue.enqueue.call_args
        operation = args[0]
        
        # Verify it's a StoreOperation with correct parameters
        self.assertIsInstance(operation, StoreOperation)
        self.assertEqual(operation.operation_id, str(test_uuid))
        self.assertEqual(operation.agent_id, agent_id)
        self.assertEqual(operation.state_data, state_data)
        
        # Check priority calculation based on actual implementation
        self.assertEqual(kwargs["priority"], 3)

    def test_priority_affects_queue_position(self):
        """Test that priority affects queue position."""
        # Create store functions that fail
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        
        # Test data
        agent_id = "test-agent"
        state_data = {"status": "active"}
        
        # Reset the mock before test
        self.mock_recovery_queue.reset_mock()
        
        # Test NORMAL priority
        self.client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.NORMAL
        )
        
        # Test HIGH priority
        self.client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.HIGH
        )
        
        # Test LOW priority - should not enqueue
        self.client.store_with_retry(
            agent_id, state_data, store_func, priority=Priority.LOW
        )
        
        # Check enqueue calls
        enqueue_calls = self.mock_recovery_queue.enqueue.call_args_list
        
        # Verify we have only 2 calls (NORMAL and HIGH), not 3
        # LOW priority doesn't use the recovery queue
        self.assertEqual(len(enqueue_calls), 2)
        
        # Verify the priorities match what we observed
        # From redis_client.py, we can see the priority is calculated as 4 - priority.value
        # NORMAL (1) -> 4 - 1 = 3
        # HIGH (2) -> 4 - 2 = 2
        self.assertEqual(enqueue_calls[0][1]["priority"], 3)  # NORMAL priority
        self.assertEqual(enqueue_calls[1][1]["priority"], 2)  # HIGH priority

    @patch('agent_memory.storage.redis_client.RecoveryQueue')
    def test_recovery_queue_integration(self, mock_recovery_queue_class):
        """Test integration with the real RecoveryQueue."""
        # Create a mock queue instance
        mock_queue_instance = MagicMock(spec=RecoveryQueue)
        mock_recovery_queue_class.return_value = mock_queue_instance
        
        # Create a new instance with the patched RecoveryQueue
        with patch('agent_memory.storage.redis_client.RecoveryQueue', mock_recovery_queue_class):
            client = ResilientRedisClient(
                client_name="recovery-test-real",
                retry_policy=self.retry_policy,
            )
            
            # Now the mock should be called during initialization
            mock_recovery_queue_class.assert_called_once()
            
            # Test that the client uses the queue for retries
            store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
            agent_id = "test-agent"
            state_data = {"status": "active"}
            
            client.store_with_retry(agent_id, state_data, store_func, priority=Priority.NORMAL)
            
            # Verify enqueue was called on the queue instance
            mock_queue_instance.enqueue.assert_called_once()

    def test_store_operation_execution(self):
        """Test execution of a StoreOperation."""
        # Create a store function that succeeds
        store_func = MagicMock(return_value=True)
        
        # Create a StoreOperation
        agent_id = "test-agent"
        state_data = {"status": "active"}
        operation = StoreOperation(
            operation_id="test-op",
            agent_id=agent_id,
            state_data=state_data,
            store_function=store_func,
        )
        
        # Execute the operation
        result = operation.execute()
        
        # Verify it calls the store function with correct arguments
        self.assertTrue(result)
        store_func.assert_called_once_with(agent_id, state_data)
        
        # Test with a failing store function
        store_func = MagicMock(side_effect=RedisUnavailableError("Test error"))
        operation = StoreOperation(
            operation_id="test-op-fail",
            agent_id=agent_id,
            state_data=state_data,
            store_function=store_func,
        )
        
        # Execute should return False on failure
        with self.assertRaises(RedisUnavailableError):
            operation.execute()


if __name__ == "__main__":
    unittest.main() 