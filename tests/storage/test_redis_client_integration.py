"""Integration tests for ResilientRedisClient using MockRedis.

These tests use MockRedis instead of requiring a real Redis server.
"""

import unittest
import os
import time
import redis
import unittest.mock
import pytest

from memory.storage.redis_client import ResilientRedisClient
from memory.utils.error_handling import Priority
from memory.storage.mockredis import MockRedis

# Create a global MockRedis instance to be used by all tests
mock_redis = MockRedis()

# Setup module-level patching to ensure all Redis instances are our mock
@pytest.fixture(scope="module", autouse=True)
def patch_redis():
    """Patch redis.Redis to use our MockRedis instance."""
    with unittest.mock.patch('redis.Redis', return_value=mock_redis):
        yield


class TestResilientRedisClientIntegration(unittest.TestCase):
    """Integration tests for ResilientRedisClient with MockRedis."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        # No special setup needed with the patched MockRedis
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # No cleanup needed
        pass

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a ResilientRedisClient instance for testing
        self.client = ResilientRedisClient(
            client_name="integration-test",
            host="localhost",
            port=6379,
            db=0,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        
        # Ensure we have a clean state
        mock_redis.flushall()

    def tearDown(self):
        """Clean up after each test."""
        # No special teardown needed
        pass

    def test_basic_operations(self):
        """Test basic Redis operations."""
        # Test ping
        self.assertTrue(self.client.ping())
        
        # Test set and get - MockRedis returns None for set() instead of True
        self.client.set("test_key", "test_value")
        self.assertEqual(self.client.get("test_key"), "test_value")
        
        # Test delete
        self.assertEqual(self.client.delete("test_key"), 1)
        self.assertIsNone(self.client.get("test_key"))
        
        # Test exists
        self.client.set("test_key1", "value1")
        self.client.set("test_key2", "value2")
        self.assertEqual(self.client.exists("test_key1", "test_key2", "nonexistent"), 2)
        
        # Test expire
        self.client.set("expire_key", "value")
        self.assertTrue(self.client.expire("expire_key", 1))
        time.sleep(2)  # Wait for expiration
        self.assertIsNone(self.client.get("expire_key"))

    def test_hash_operations(self):
        """Test Redis hash operations."""
        # Test hset and hget
        self.assertEqual(self.client.hset("hash", "field1", "value1"), 1)
        self.assertEqual(self.client.hget("hash", "field1"), "value1")
        
        # Test hmset
        mapping = {"field2": "value2", "field3": "value3"}
        self.assertTrue(self.client.hmset("hash", mapping))
        
        # Test hgetall
        expected = {"field1": "value1", "field2": "value2", "field3": "value3"}
        self.assertEqual(self.client.hgetall("hash"), expected)
        
        # Test hdel
        self.assertEqual(self.client.hdel("hash", "field1", "field2"), 2)
        self.assertEqual(self.client.hgetall("hash"), {"field3": "value3"})

    def test_sorted_set_operations(self):
        """Test Redis sorted set operations."""
        # Test zadd
        mapping = {"member1": 1.0, "member2": 2.0, "member3": 3.0}
        self.assertEqual(self.client.zadd("zset", mapping), 3)
        
        # Test zrange
        self.assertEqual(
            self.client.zrange("zset", 0, -1), ["member1", "member2", "member3"]
        )
        
        # Test zrange with scores - MockRedis returns flat list instead of tuples
        result_with_scores = self.client.zrange("zset", 0, -1, withscores=True)
        
        # Convert the flat list to expected tuple format for the test
        converted_result = []
        for i in range(0, len(result_with_scores), 2):
            converted_result.append((result_with_scores[i], result_with_scores[i+1]))
            
        expected_with_scores = [
            ("member1", 1.0), ("member2", 2.0), ("member3", 3.0)
        ]
        self.assertEqual(converted_result, expected_with_scores)
        
        # Test zrangebyscore
        self.assertEqual(
            self.client.zrangebyscore("zset", 2.0, 3.0), ["member2", "member3"]
        )
        
        # Test zrem
        self.assertEqual(self.client.zrem("zset", "member1", "member2"), 2)
        self.assertEqual(self.client.zrange("zset", 0, -1), ["member3"])
        
        # Test zcard
        self.assertEqual(self.client.zcard("zset"), 1)

    def test_store_with_retry(self):
        """Test store with retry functionality."""
        # Define a store function
        def store_func(agent_id, data):
            key = f"agent:{agent_id}"
            self.client.set(key, str(data))
            return True
        
        # Test normal case
        agent_id = "test_agent"
        data = {"status": "active", "timestamp": time.time()}
        
        result = self.client.store_with_retry(agent_id, data, store_func)
        self.assertTrue(result)
        
        # Verify data was stored
        key = f"agent:{agent_id}"
        self.assertIsNotNone(self.client.get(key))


if __name__ == "__main__":
    unittest.main() 