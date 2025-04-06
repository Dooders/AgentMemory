import unittest
import time
from agent_memory.storage.mockredis import MockRedis

class TestMockRedis(unittest.TestCase):
    def setUp(self):
        self.redis = MockRedis()
        
    def tearDown(self):
        self.redis.flushall()
        
    def test_set_get(self):
        self.redis.set("key1", "value1")
        self.assertEqual(self.redis.get("key1"), "value1")
        
    def test_set_with_expiry(self):
        self.redis.set("key1", "value1", ex=1)
        self.assertEqual(self.redis.get("key1"), "value1")
        time.sleep(1.1)
        self.assertIsNone(self.redis.get("key1"))
        
    def test_delete(self):
        self.redis.set("key1", "value1")
        self.assertEqual(self.redis.get("key1"), "value1")
        self.redis.delete("key1")
        self.assertIsNone(self.redis.get("key1"))
        
    def test_exists(self):
        self.redis.set("key1", "value1")
        self.assertEqual(self.redis.exists("key1"), 1)
        self.assertEqual(self.redis.exists("key1", "key2"), 1)
        self.redis.set("key2", "value2")
        self.assertEqual(self.redis.exists("key1", "key2"), 2)
        
    def test_expire(self):
        self.redis.set("key1", "value1")
        self.redis.expire("key1", 1)
        self.assertEqual(self.redis.get("key1"), "value1")
        time.sleep(1.1)
        self.assertIsNone(self.redis.get("key1"))
        
    def test_hash_operations(self):
        # Test hset and hget
        self.redis.hset("hash1", "field1", "value1")
        self.assertEqual(self.redis.hget("hash1", "field1"), "value1")
        
        # Test hmset
        self.redis.hmset("hash1", {"field2": "value2", "field3": "value3"})
        self.assertEqual(self.redis.hget("hash1", "field2"), "value2")
        self.assertEqual(self.redis.hget("hash1", "field3"), "value3")
        
        # Test hdel
        self.redis.hdel("hash1", "field1")
        self.assertIsNone(self.redis.hget("hash1", "field1"))
        
    def test_list_operations(self):
        # Test lpush and rpush
        self.redis.lpush("list1", "value1", "value2")
        self.redis.rpush("list1", "value3", "value4")
        
        # Test lpop and rpop
        self.assertEqual(self.redis.lpop("list1"), "value2")
        self.assertEqual(self.redis.rpop("list1"), "value4")
        
    def test_sorted_set_operations(self):
        # Test zadd
        self.redis.zadd("zset1", {"a": 1, "b": 2, "c": 3})
        
        # Test zrange
        self.assertEqual(self.redis.zrange("zset1", 0, -1), ["a", "b", "c"])
        self.assertEqual(self.redis.zrange("zset1", 0, -1, desc=True), ["c", "b", "a"])
        
        # Test zrangebyscore
        self.assertEqual(self.redis.zrangebyscore("zset1", 2, 3), ["b", "c"])
        
        # Test zrem
        self.redis.zrem("zset1", "a")
        self.assertEqual(self.redis.zrange("zset1", 0, -1), ["b", "c"])
        
        # Test zcard
        self.assertEqual(self.redis.zcard("zset1"), 2) 