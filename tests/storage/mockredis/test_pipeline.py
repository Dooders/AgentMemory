import unittest
import time
from memory.storage.mockredis import MockRedis

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.redis = MockRedis()
        
    def tearDown(self):
        self.redis.flushall()
        
    def test_basic_pipeline(self):
        pipe = self.redis.pipeline()
        pipe.set("key1", "value1")
        pipe.get("key1")
        results = pipe.execute()
        self.assertEqual(results, [None, "value1"])
        
    def test_pipeline_with_multiple_commands(self):
        pipe = self.redis.pipeline()
        pipe.set("key1", "value1")
        pipe.set("key2", "value2")
        pipe.get("key1")
        pipe.get("key2")
        results = pipe.execute()
        self.assertEqual(results, [None, None, "value1", "value2"])
        
    def test_transaction_with_watch(self):
        # Set initial value
        self.redis.set("counter", "1")
        
        # Start transaction with watch
        pipe = self.redis.pipeline()
        pipe.watch("counter")
        pipe.multi()
        
        # Get current value and increment
        current = self.redis.get("counter")
        pipe.set("counter", str(int(current) + 1))
        
        # Execute transaction
        results = pipe.execute()
        self.assertEqual(results, [None])  # set returns None
        self.assertEqual(self.redis.get("counter"), "2")
        
    def test_transaction_aborted_when_watched_key_changes(self):
        # Set initial value
        self.redis.set("counter", "1")
        
        # Start transaction with watch
        pipe = self.redis.pipeline()
        pipe.watch("counter")
        
        # Simulate another client changing the value
        self.redis.set("counter", "999")
        
        # Try to execute transaction
        pipe.multi()
        pipe.set("counter", "2")
        results = pipe.execute()
        
        # Transaction should be aborted (returns None)
        self.assertIsNone(results)
        self.assertEqual(self.redis.get("counter"), "999")  # Value unchanged by our transaction
        
    def test_unwatch(self):
        # Set initial value
        self.redis.set("counter", "1")
        
        # Start transaction with watch
        pipe = self.redis.pipeline()
        pipe.watch("counter")
        
        # Unwatch before value changes
        pipe.unwatch()
        
        # Simulate another client changing the value
        self.redis.set("counter", "999")
        
        # Transaction should still execute even though watched key changed
        pipe.multi()
        pipe.set("counter", "2")
        results = pipe.execute()
        
        self.assertEqual(results, [None])
        self.assertEqual(self.redis.get("counter"), "2")  # Our value overwrites the other one
        
    def test_context_manager(self):
        # Test using pipeline as a context manager
        with self.redis.pipeline() as pipe:
            pipe.set("key1", "value1")
            pipe.get("key1")
            results = pipe.execute()
            
        self.assertEqual(results, [None, "value1"])
        self.assertEqual(self.redis.get("key1"), "value1")
        
    def test_hash_operations(self):
        pipe = self.redis.pipeline()
        pipe.hset("hash1", "field1", "value1")
        pipe.hget("hash1", "field1")
        pipe.hmset("hash1", {"field2": "value2", "field3": "value3"})
        pipe.hget("hash1", "field2")
        pipe.hdel("hash1", "field1")
        pipe.hget("hash1", "field1")
        
        results = pipe.execute()
        self.assertEqual(results[1], "value1")
        self.assertEqual(results[3], "value2")
        self.assertIsNone(results[5])  # field1 was deleted
        
    def test_sorted_set_operations(self):
        pipe = self.redis.pipeline()
        pipe.zadd("zset1", {"a": 1, "b": 2, "c": 3})
        pipe.zrange("zset1", 0, -1)
        pipe.zrangebyscore("zset1", 2, 3)
        pipe.zrem("zset1", "a")
        pipe.zrange("zset1", 0, -1)
        pipe.zcard("zset1")
        
        results = pipe.execute()
        self.assertEqual(results[1], ["a", "b", "c"])
        self.assertEqual(results[2], ["b", "c"])
        self.assertEqual(results[4], ["b", "c"])
        self.assertEqual(results[5], 2) 