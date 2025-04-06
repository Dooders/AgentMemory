"""Edge case tests for Redis Short-Term Memory (STM) storage.

This module contains tests that verify RedisSTMStore behavior with edge cases,
unusual inputs, and error conditions.
"""

import json
import time
from unittest import mock
import pytest

from agent_memory.config import RedisSTMConfig
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.utils.error_handling import Priority, RedisTimeoutError, RedisUnavailableError


class TestRedisSTMEdgeCases:
    """Edge case tests for RedisSTMStore."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures for each test."""
        # Create the mock Redis client
        self.mock_redis = mock.MagicMock()
        
        # Mock the RedisFactory.create_client method instead of ResilientRedisClient
        self.redis_factory_patcher = mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client')
        self.mock_redis_factory = self.redis_factory_patcher.start()
        self.mock_redis_factory.return_value = self.mock_redis
        
        # Set up default successful responses for Redis methods
        self.mock_redis.get.return_value = None
        self.mock_redis.set.return_value = True
        self.mock_redis.delete.return_value = 1
        self.mock_redis.zrem.return_value = 1
        self.mock_redis.zadd.return_value = 1
        self.mock_redis.zcard.return_value = 0
        self.mock_redis.zrange.return_value = []
        self.mock_redis.zrangebyscore.return_value = []
        self.mock_redis.expire.return_value = True
        
        # Mock the store_with_retry method directly
        self.mock_redis.store_with_retry = mock.MagicMock(return_value=True)
        
        # Create a Redis STM store with test configuration
        self.config = RedisSTMConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            namespace="test-stm-edge",
            ttl=3600
        )
        self.stm_store = RedisSTMStore(self.config)
        
        yield
        
        # Stop patcher after the test
        self.redis_factory_patcher.stop()
    
    def test_store_empty_memory(self):
        """Test storing a nearly empty memory entry."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "empty-memory",
            # Missing content and other fields
        }
        
        # Should still store despite missing fields
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify correct calls were made
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_store_large_memory(self):
        """Test storing a very large memory entry."""
        agent_id = "test-agent"
        
        # Create a memory with large content
        large_content = "x" * 1024 * 1024  # 1MB string
        memory_entry = {
            "memory_id": "large-memory",
            "content": large_content,
            "timestamp": time.time(),
            "metadata": {"importance_score": 0.5}
        }
        
        # Should handle large content
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_store_memory_with_unusual_fields(self):
        """Test storing a memory with unusual additional fields."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "unusual-memory",
            "content": "Test content",
            "timestamp": time.time(),
            "metadata": {"importance_score": 0.5},
            # Unusual fields
            "custom_field": "custom value",
            "nested": {
                "deep": {
                    "deeper": [1, 2, 3]
                }
            },
            "unicode_field": "你好世界",  # Hello world in Chinese
            "emoji": "🚀🔥👍",
            "binary_like": "\x00\x01\x02\x03"
        }
        
        # Should store without issues
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_store_memory_with_boolean_id(self):
        """Test storing a memory with a boolean memory_id."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": True,  # Unusual ID type
            "content": "Test content",
            "timestamp": time.time()
        }
        
        # Should work as memory_id will be converted to string
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_get_with_unusual_ids(self):
        """Test retrieving memories with unusual IDs."""
        agent_id = "test-agent"
        
        # Setup mock to return data for any key
        memory_data = json.dumps({
            "memory_id": "test",
            "content": "Test content"
        })
        self.mock_redis.get.return_value = memory_data
        
        # Test with various unusual IDs
        unusual_ids = [
            "",  # Empty string
            "memory with spaces",
            "memory:with:colons",
            "memory/with/slashes",
            "memory\nwith\nnewlines",
            "memory\twith\ttabs",
            "memory\"with\"quotes",
            "memory'with'apostrophes",
            "memory\\with\\backslashes",
            "memory@#$%^&*()_+-=[]{}|;':<>,.?/",  # Special characters
            "你好世界",  # Unicode
            "🚀🔥👍"  # Emojis
        ]
        
        for memory_id in unusual_ids:
            # Should handle unusual IDs
            memory = self.stm_store.get(agent_id, memory_id)
            assert memory is not None
    
    def test_empty_agent_id(self):
        """Test operations with an empty agent ID."""
        empty_agent = ""
        memory_entry = {
            "memory_id": "test-memory",
            "content": "Test content",
            "timestamp": time.time()
        }
        
        # Store should work with empty agent ID
        result = self.stm_store.store(empty_agent, memory_entry)
        assert result is True
        
        # Set up mock to return data for the empty agent ID
        memory_data = json.dumps({
            "memory_id": "test-memory",
            "content": "Test content"
        })
        self.mock_redis.get.return_value = memory_data
        
        # Retrieve should work with empty agent ID
        memory = self.stm_store.get(empty_agent, "test-memory")
        assert memory is not None  # We've set mock_redis.get to return json data
    
    def test_negative_importance(self):
        """Test handling of negative importance scores."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "negative-importance",
            "content": "Test content",
            "timestamp": time.time(),
            "metadata": {"importance_score": -0.5}  # Negative importance
        }
        
        # Should handle negative importance
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
        
        # Reset the mock for the next test
        self.mock_redis.store_with_retry.reset_mock()
        
        # Get by importance should work with negative bounds
        memories = self.stm_store.get_by_importance(agent_id, -1.0, 0.0)
        assert isinstance(memories, list)  # Should return a list (even if empty)
    
    def test_extreme_timestamps(self):
        """Test handling of extreme timestamp values."""
        agent_id = "test-agent"
        
        # Test with very old timestamp
        old_memory = {
            "memory_id": "old-memory",
            "content": "Very old memory",
            "timestamp": 0  # Unix epoch start
        }
        result_old = self.stm_store.store(agent_id, old_memory)
        assert result_old is True
        
        # Reset mock between calls
        self.mock_redis.store_with_retry.reset_mock()
        
        # Test with future timestamp
        future_memory = {
            "memory_id": "future-memory",
            "content": "Future memory",
            "timestamp": 9999999999  # Far in the future
        }
        result_future = self.stm_store.store(agent_id, future_memory)
        assert result_future is True
        
        # Test retrieving memories with extreme time range
        memories = self.stm_store.get_by_timerange(agent_id, 0, 9999999999)
        assert isinstance(memories, list)  # Should return a list (even if empty)
    
    def test_non_standard_metadata(self):
        """Test handling of non-standard metadata."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "metadata-test",
            "content": "Test content",
            "timestamp": time.time(),
            "metadata": {
                "importance_score": 0.5,
                # Non-standard metadata
                "custom_score": 123,
                "tags": ["tag1", "tag2", "tag3"],
                "nested": {"key": "value"},
                "null_value": None,
                "boolean": True
            }
        }
        
        # Should handle complex metadata
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_json_decode_error(self):
        """Test handling of JSON decode errors."""
        agent_id = "test-agent"
        memory_id = "corrupt-memory"
        
        # Setup mock to return invalid JSON
        self.mock_redis.get.return_value = "{invalid json"
        
        # Should handle JSON decode error
        memory = self.stm_store.get(agent_id, memory_id)
        assert memory is None
    
    def test_circuit_breaker_functionality(self):
        """Test that circuit breaker pattern is used in Redis client."""
        # The RedisSTMStore should use RedisFactory.create_client with circuit breaker settings
        with mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client') as mock_factory:
            # Create configuration
            config = RedisSTMConfig(
                host="localhost",
                port=6379,
                db=0,
                password=None,
                namespace="test-stm",
                ttl=3600
            )
            
            # Create store which should use our mocked factory
            _ = RedisSTMStore(config)
            
            # Verify create_client was called with correct parameters
            mock_factory.assert_called_once_with(
                client_name="stm",
                use_mock=config.use_mock,
                host="localhost",
                port=6379,
                db=0,
                password=None,
                circuit_threshold=3,
                circuit_reset_timeout=300
            )
    
    def test_null_values_in_memory(self):
        """Test handling of null/None values in memory entries."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "null-values",
            "content": None,  # Null content
            "timestamp": time.time(),
            "metadata": None,  # Null metadata
            "embeddings": None  # Null embeddings
        }
        
        # Should handle null values
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once()
    
    def test_very_long_namespace(self):
        """Test using a very long namespace."""
        # Create a very long namespace
        long_namespace = "a" * 1000
        
        config = RedisSTMConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            namespace=long_namespace,
            ttl=3600
        )
        
        with mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client'):
            # Should initialize with long namespace
            store = RedisSTMStore(config)
            assert store._key_prefix == long_namespace
    
    def test_delete_nonexistent_memory(self):
        """Test deleting a memory that doesn't exist."""
        agent_id = "test-agent"
        memory_id = "nonexistent"
        
        # Configure delete to return 0 (indicating no keys were deleted)
        self.mock_redis.delete.return_value = 0
        
        # Should return False when memory doesn't exist
        result = self.stm_store.delete(agent_id, memory_id)
        assert result is False
    
    def test_very_high_ttl(self):
        """Test using a very high TTL value."""
        config = RedisSTMConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            namespace="test-stm",
            ttl=2147483647  # Close to max 32-bit integer
        )
        
        with mock.patch('agent_memory.storage.redis_stm.RedisFactory.create_client'):
            # Should initialize with high TTL
            store = RedisSTMStore(config)
            assert store.config.ttl == 2147483647
    
    def test_unicode_agent_id(self):
        """Test operations with a Unicode agent ID."""
        unicode_agent = "你好世界"  # Hello world in Chinese
        memory_entry = {
            "memory_id": "test-memory",
            "content": "Test content",
            "timestamp": time.time()
        }
        
        # Store should work with Unicode agent ID
        result = self.stm_store.store(unicode_agent, memory_entry)
        assert result is True
        
        # Setup mock to return data for get call
        memory_data = json.dumps({
            "memory_id": "test-memory",
            "content": "Test content"
        })
        self.mock_redis.get.return_value = memory_data
        
        # Retrieve should work with Unicode agent ID
        memory = self.stm_store.get(unicode_agent, "test-memory")
        assert memory is not None
    
    def test_embedded_null_characters(self):
        """Test handling of strings with embedded null characters."""
        agent_id = "test-agent"
        memory_entry = {
            "memory_id": "null-char-memory",
            "content": "content with \x00 null \x00 characters",
            "timestamp": time.time()
        }
        
        # Should handle null characters in strings
        result = self.stm_store.store(agent_id, memory_entry)
        assert result is True
        
        # Verify store_with_retry was called
        self.mock_redis.store_with_retry.assert_called_once() 