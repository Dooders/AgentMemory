"""Unit tests for Redis utilities.

This module contains tests for the redis_utils module in agent_memory/utils.
"""

import struct
import unittest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pytest
import redis

from agent_memory.utils.redis_utils import (
    serialize_memory_entry,
    deserialize_memory_entry,
    serialize_vector,
    deserialize_vector,
    vector_to_bytes,
    bytes_to_vector,
    RedisConnectionManager,
    RedisBatchProcessor,
    redis_key_exists,
    redis_memory_scan,
    redis_create_index,
    redis_drop_index,
    get_redis_info,
    get_redis_connection_manager,
)


class TestRedisSerializationFunctions(unittest.TestCase):
    """Test the Redis serialization/deserialization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_memory = {
            "id": "mem_123",
            "agent_id": "agent_1",
            "content": "This is a test memory",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": {
                "priority": 1,
            },
        }
        self.test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_serialize_deserialize_memory_entry(self):
        """Test serializing and deserializing memory entries."""
        # Serialize
        serialized = serialize_memory_entry(self.test_memory)
        self.assertIsInstance(serialized, str)
        
        # Deserialize
        deserialized = deserialize_memory_entry(serialized)
        
        # Check fields
        self.assertEqual(deserialized["id"], self.test_memory["id"])
        self.assertEqual(deserialized["agent_id"], self.test_memory["agent_id"])
        self.assertEqual(deserialized["content"], self.test_memory["content"])
        self.assertEqual(deserialized["embedding"], self.test_memory["embedding"])
        self.assertEqual(deserialized["metadata"], self.test_memory["metadata"])

    def test_serialize_deserialize_vector(self):
        """Test serializing and deserializing vectors for Redis."""
        # Serialize
        serialized = serialize_vector(self.test_vector)
        self.assertIsInstance(serialized, str)
        
        # Deserialize
        deserialized = deserialize_vector(serialized)
        self.assertEqual(deserialized, self.test_vector)

    def test_vector_to_bytes(self):
        """Test converting vectors to binary format."""
        binary = vector_to_bytes(self.test_vector)
        self.assertIsInstance(binary, bytes)
        
        # Check length (5 floats * 4 bytes per float = 20 bytes)
        self.assertEqual(len(binary), len(self.test_vector) * 4)

    def test_bytes_to_vector(self):
        """Test converting binary format back to vectors."""
        # Convert to binary
        binary = vector_to_bytes(self.test_vector)
        
        # Convert back to vector
        vector = bytes_to_vector(binary)
        
        # Check values (float comparison needs small epsilon)
        for i, val in enumerate(self.test_vector):
            self.assertAlmostEqual(vector[i], val, places=6)

    def test_bytes_to_vector_empty(self):
        """Test converting empty binary data."""
        vector = bytes_to_vector(b"")
        self.assertEqual(vector, [])

    def test_vector_conversions_roundtrip(self):
        """Test roundtrip conversion of vectors to bytes and back."""
        # Various test vectors
        test_vectors = [
            [0.0, 0.0, 0.0],  # Zeros
            [1.0, 1.0, 1.0],  # Ones
            [-1.0, 0.0, 1.0],  # Mixed values
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Decimal values
            [1e-10, 1e10],  # Very small and large values
        ]
        
        for vec in test_vectors:
            binary = vector_to_bytes(vec)
            result = bytes_to_vector(binary)
            
            # Check each value (floating point comparison)
            for i, val in enumerate(vec):
                self.assertAlmostEqual(result[i], val, places=6)


@patch('redis.Redis')
class TestRedisConnectionManager(unittest.TestCase):
    """Test the RedisConnectionManager class."""
    
    def test_singleton_instance(self, mock_redis):
        """Test that RedisConnectionManager is a singleton."""
        # Get instance twice
        manager1 = RedisConnectionManager.get_instance()
        manager2 = RedisConnectionManager.get_instance()
        
        # Should be the same object
        self.assertIs(manager1, manager2)
    
    def test_get_connection_default(self, mock_redis):
        """Test getting a connection with default parameters."""
        manager = RedisConnectionManager()
        connection = manager.get_connection()
        
        # Redis should be created with default parameters
        mock_redis.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            decode_responses=True
        )
        
        # Connection should be stored in connections dict
        self.assertIn("localhost:6379:0:False", manager.connections)
        self.assertEqual(manager.connections["localhost:6379:0:False"], connection)
    
    def test_get_connection_custom(self, mock_redis):
        """Test getting a connection with custom parameters."""
        manager = RedisConnectionManager()
        connection = manager.get_connection(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            decode_responses=False
        )
        
        # Redis should be created with custom parameters
        mock_redis.assert_called_once_with(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            decode_responses=False
        )
        
        # Connection should be stored in connections dict
        self.assertIn("redis.example.com:6380:1:False", manager.connections)
        self.assertEqual(manager.connections["redis.example.com:6380:1:False"], connection)
    
    def test_get_connection_reuse(self, mock_redis):
        """Test that connections are reused."""
        manager = RedisConnectionManager()
        
        # Get connection first time
        connection1 = manager.get_connection(host="test", port=1234, db=0)
        
        # Reset mock to verify it's not called again
        mock_redis.reset_mock()
        
        # Get connection again with same parameters
        connection2 = manager.get_connection(host="test", port=1234, db=0)
        
        # Should return same connection without creating a new one
        self.assertEqual(connection1, connection2)
        mock_redis.assert_not_called()
    
    def test_close_all(self, mock_redis):
        """Test closing all connections."""
        manager = RedisConnectionManager()
        
        # Create some mock connections
        conn1 = MagicMock()
        conn2 = MagicMock()
        
        # Add them to the manager
        manager.connections = {
            "conn1": conn1,
            "conn2": conn2
        }
        
        # Close all
        manager.close_all()
        
        # Each connection should have close() called
        conn1.close.assert_called_once()
        conn2.close.assert_called_once()
        
        # Connections dict should be empty
        self.assertEqual(manager.connections, {})


@patch('redis.Redis')
class TestRedisBatchProcessor(unittest.TestCase):
    """Test the RedisBatchProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_redis = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_redis.pipeline.return_value = self.mock_pipeline
        
        # Make the pipeline a context manager
        self.mock_pipeline.__enter__.return_value = self.mock_pipeline
        self.mock_pipeline.__exit__.return_value = None
    
    def test_init(self, _):
        """Test initialization."""
        processor = RedisBatchProcessor(
            self.mock_redis,
            max_batch_size=200,
            auto_execute=False
        )
        
        self.assertEqual(processor.redis_client, self.mock_redis)
        self.assertEqual(processor.max_batch_size, 200)
        self.assertEqual(processor.auto_execute, False)
        self.assertEqual(processor.commands, [])
    
    def test_add_command(self, _):
        """Test adding a command to the batch."""
        processor = RedisBatchProcessor(self.mock_redis, auto_execute=False)
        
        # Add a command
        result = processor.add_command("set", "key", "value")
        
        # Should return self for chaining
        self.assertEqual(result, processor)
        
        # Command should be added to the list
        self.assertEqual(len(processor.commands), 1)
        self.assertEqual(processor.commands[0], ("set", ("key", "value"), {}))

    def test_auto_execute(self, _):
        """Test auto-execution when batch size is reached."""
        processor = RedisBatchProcessor(
            self.mock_redis,
            max_batch_size=2,
            auto_execute=True
        )
        
        # Add commands to trigger auto-execute
        processor.add_command("set", "key1", "value1")
        self.mock_pipeline.execute.assert_not_called()
        
        # This should trigger execution
        processor.add_command("set", "key2", "value2")
        
        # Pipeline execute should have been called
        self.mock_pipeline.execute.assert_called_once()
        
        # Commands should be cleared
        self.assertEqual(processor.commands, [])
    
    def test_execute(self, _):
        """Test manual execution of the batch."""
        processor = RedisBatchProcessor(
            self.mock_redis,
            auto_execute=False
        )
        
        # Add some commands
        processor.add_command("set", "key1", "value1")
        processor.add_command("get", "key2")
        
        # Set up mock results
        self.mock_pipeline.execute.return_value = ["OK", "value2"]
        
        # Execute the batch
        results = processor.execute()
        
        # Check results
        self.assertEqual(results, ["OK", "value2"])
        
        # Check that pipeline was used correctly
        self.mock_redis.pipeline.assert_called_once()
        self.mock_pipeline.set.assert_called_once_with("key1", "value1")
        self.mock_pipeline.get.assert_called_once_with("key2")
        self.mock_pipeline.execute.assert_called_once()
        
        # Commands should be cleared
        self.assertEqual(processor.commands, [])


@patch('redis.Redis')
class TestRedisUtilityFunctions(unittest.TestCase):
    """Test the Redis utility functions."""
    
    def test_redis_key_exists(self, _):
        """Test checking if a key exists in Redis."""
        mock_client = MagicMock()
        mock_client.exists.return_value = 1
        
        result = redis_key_exists(mock_client, "test_key")
        
        mock_client.exists.assert_called_once_with("test_key")
        self.assertTrue(result)
        
        # Test non-existent key
        mock_client.exists.return_value = 0
        result = redis_key_exists(mock_client, "missing_key")
        self.assertFalse(result)
    
    def test_redis_memory_scan(self, _):
        """Test scanning for memory entries in Redis."""
        mock_client = MagicMock()
        
        # Set up mock scan_iter to return keys
        mock_client.scan_iter.return_value = ["key1", "key2"]
        
        # Set up mock get to return memory data
        memory_data = {
            "id": "mem_123",
            "content": "Test memory"
        }
        mock_client.get.side_effect = [
            serialize_memory_entry(memory_data),
            serialize_memory_entry(memory_data)
        ]
        
        # Call the function
        results = list(redis_memory_scan(mock_client, "mem:*"))
        
        # Check the results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["id"], "mem_123")
            self.assertEqual(result["content"], "Test memory")
        
        # Check that the Redis client was called correctly
        mock_client.scan_iter.assert_called_once_with(match="mem:*", count=100)
        self.assertEqual(mock_client.get.call_count, 2)
    
    @patch('agent_memory.utils.redis_utils.logger')
    def test_redis_create_index(self, mock_logger, _):
        """Test creating a Redis index."""
        mock_client = MagicMock()
        
        # Set up mock ft command to indicate success
        mock_client.ft.return_value.create_index.return_value = True
        
        # Call the function
        schema = {
            "text": "TEXT",
            "tag": "TAG",
            "vector": "VECTOR"
        }
        result = redis_create_index(mock_client, "idx", "prefix:", schema)
        
        # Check the result
        self.assertTrue(result)
        
        # Check Redis calls
        mock_client.ft.assert_called_once_with()
        mock_client.ft().create_index.assert_called_once()
        
        # Test with exception
        mock_client.ft().create_index.side_effect = redis.ResponseError("Index already exists")
        result = redis_create_index(mock_client, "idx", "prefix:", schema)
        
        # Should log warning but return True
        self.assertTrue(result)
        mock_logger.warning.assert_called_once()
    
    def test_redis_drop_index(self, _):
        """Test dropping a Redis index."""
        mock_client = MagicMock()
        
        # Set up mock ft command to indicate success
        mock_client.ft.return_value.dropindex.return_value = True
        
        # Call the function
        result = redis_drop_index(mock_client, "idx")
        
        # Check the result
        self.assertTrue(result)
        
        # Check Redis calls
        mock_client.ft.assert_called_once_with()
        mock_client.ft().dropindex.assert_called_once_with("idx")
        
        # Test with exception
        mock_client.ft().dropindex.side_effect = redis.ResponseError("Unknown index name")
        result = redis_drop_index(mock_client, "missing_idx")
        
        # Should return False for errors
        self.assertFalse(result)
    
    def test_get_redis_info(self, _):
        """Test getting Redis server info."""
        mock_client = MagicMock()
        
        # Set up mock info to return server data
        mock_info = {
            "redis_version": "6.2.0",
            "used_memory": "1048576",
            "connected_clients": "10"
        }
        mock_client.info.return_value = mock_info
        
        # Call the function
        result = get_redis_info(mock_client)
        
        # Check the result
        self.assertEqual(result["version"], "6.2.0")
        self.assertEqual(result["memory"]["used"], 1048576)
        self.assertEqual(result["clients"], 10)
        
        # Check Redis calls
        mock_client.info.assert_called_once()
    
    def test_get_redis_connection_manager(self, _):
        """Test getting the Redis connection manager."""
        # Call the function
        result = get_redis_connection_manager()
        
        # Should return singleton instance
        self.assertIsInstance(result, RedisConnectionManager)
        self.assertIs(result, RedisConnectionManager.get_instance())


if __name__ == "__main__":
    unittest.main() 