"""Integration tests for ResilientRedisClient.

These tests require a running Redis server.
They will be skipped if Redis is not available.
"""

import unittest
import os
import time
import redis

from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.utils.error_handling import Priority


# Check if Redis is available
def is_redis_available():
    """Check if a Redis server is available."""
    try:
        redis_host = os.environ.get("REDIS_TEST_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_TEST_PORT", "6379"))
        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=1)
        return r.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        return False


@unittest.skipIf(not is_redis_available(), "Redis server not available")
class TestResilientRedisClientIntegration(unittest.TestCase):
    """Integration tests for ResilientRedisClient with a real Redis server."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        # Use environment variables or defaults for Redis connection
        cls.redis_host = os.environ.get("REDIS_TEST_HOST", "localhost")
        cls.redis_port = int(os.environ.get("REDIS_TEST_PORT", "6379"))
        cls.redis_db = int(os.environ.get("REDIS_TEST_DB", "15"))  # Use DB 15 for tests
        
        # Create client for test setup/teardown
        cls.setup_client = redis.Redis(
            host=cls.redis_host, port=cls.redis_port, db=cls.redis_db, decode_responses=True
        )
        
        # Ensure test DB is empty
        cls.setup_client.flushdb()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up test database
        cls.setup_client.flushdb()
        cls.setup_client.close()

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a ResilientRedisClient instance for testing
        self.client = ResilientRedisClient(
            client_name="integration-test",
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        
        # Ensure we have a clean state
        self.setup_client.flushdb()

    def tearDown(self):
        """Clean up after each test."""
        self.setup_client.flushdb()

    def test_basic_operations(self):
        """Test basic Redis operations."""
        # Test ping
        self.assertTrue(self.client.ping())
        
        # Test set and get
        self.assertTrue(self.client.set("test_key", "test_value"))
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
        
        # Test zrange with scores
        expected_with_scores = [
            ("member1", 1.0), ("member2", 2.0), ("member3", 3.0)
        ]
        self.assertEqual(
            self.client.zrange("zset", 0, -1, withscores=True), expected_with_scores
        )
        
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