import unittest
import time
import threading
from agent_memory.storage.mockredis import MockRedis

class TestPubSub(unittest.TestCase):
    def setUp(self):
        self.redis = MockRedis()
        
    def tearDown(self):
        self.redis.flushall()
        
    def test_basic_subscribe_and_publish(self):
        # Create pubsub object and subscribe to a channel
        pubsub = self.redis.pubsub()
        pubsub.subscribe("channel1")
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # First message should be subscribe confirmation
        message = pubsub.get_message(timeout=1)
        self.assertEqual(message["type"], "subscribe")
        self.assertEqual(message["channel"], "channel1")
        
        # Publish a message
        self.redis.publish("channel1", "hello world")
        
        # Allow time for message to be processed
        time.sleep(0.1)
        
        # Get the published message
        message = pubsub.get_message(timeout=1)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["data"], "hello world")
        self.assertEqual(message["channel"], "channel1")
        
        # Clean up
        pubsub.close()
        
    def test_multiple_channels(self):
        # Subscribe to multiple channels
        pubsub = self.redis.pubsub()
        pubsub.subscribe("channel1", "channel2")
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # Consume subscription confirmations
        pubsub.get_message(timeout=1)
        pubsub.get_message(timeout=1)
        
        # Publish to both channels
        self.redis.publish("channel1", "message 1")
        self.redis.publish("channel2", "message 2")
        
        # Allow time for messages to be processed
        time.sleep(0.1)
        
        # Get messages from both channels
        messages = []
        messages.append(pubsub.get_message(timeout=1))
        messages.append(pubsub.get_message(timeout=1))
        
        # Verify messages were received (order may vary)
        self.assertTrue(any(m["data"] == "message 1" and m["channel"] == "channel1" for m in messages))
        self.assertTrue(any(m["data"] == "message 2" and m["channel"] == "channel2" for m in messages))
        
        # Clean up
        pubsub.close()
        
    def test_pattern_subscribe(self):
        # Subscribe to a pattern
        pubsub = self.redis.pubsub()
        pubsub.psubscribe("channel*")
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # Consume subscription confirmation
        pubsub.get_message(timeout=1)
        
        # Publish to matching channels
        self.redis.publish("channel1", "message 1")
        self.redis.publish("channel2", "message 2")
        self.redis.publish("other", "message 3")  # Should not match
        
        # Allow time for messages to be processed
        time.sleep(0.1)
        
        # Get messages that match the pattern
        messages = []
        messages.append(pubsub.get_message(timeout=1))
        messages.append(pubsub.get_message(timeout=1))
        
        # Should have received two messages for the pattern
        self.assertEqual(len(messages), 2)
        self.assertTrue(all(m["type"] == "pmessage" for m in messages))
        self.assertTrue(all(m["pattern"] == "channel*" for m in messages))
        
        # Verify no more messages (the "other" channel shouldn't match)
        self.assertIsNone(pubsub.get_message(timeout=0.1))
        
        # Clean up
        pubsub.close()
        
    def test_unsubscribe(self):
        # Subscribe and then unsubscribe
        pubsub = self.redis.pubsub()
        pubsub.subscribe("channel1", "channel2")
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # Consume subscription confirmations
        pubsub.get_message(timeout=1)
        pubsub.get_message(timeout=1)
        
        # Unsubscribe from one channel
        pubsub.unsubscribe("channel1")
        
        # Allow time for unsubscribe to process
        time.sleep(0.1)
        
        # Consume unsubscribe confirmation
        message = pubsub.get_message(timeout=1)
        self.assertEqual(message["type"], "unsubscribe")
        
        # Publish to both channels
        self.redis.publish("channel1", "message 1")
        self.redis.publish("channel2", "message 2")
        
        # Allow time for messages to be processed
        time.sleep(0.1)
        
        # Should only receive message for channel2
        message = pubsub.get_message(timeout=1)
        self.assertEqual(message["channel"], "channel2")
        self.assertEqual(message["data"], "message 2")
        
        # There should be no more messages (channel1 unsubscribed)
        self.assertIsNone(pubsub.get_message(timeout=0.1))
        
        # Clean up
        pubsub.close()
        
    def test_callback(self):
        # Test callback functionality
        result = []
        
        def message_handler(message):
            result.append(message["data"])
        
        # Subscribe with callback
        pubsub = self.redis.pubsub()
        pubsub.subscribe(channel1=message_handler)
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # Consume subscription confirmation
        pubsub.get_message(timeout=1)
        
        # Publish a message
        self.redis.publish("channel1", "callback test")
        
        # Allow time for callback to be executed
        time.sleep(0.1)
        
        # Verify callback was executed
        self.assertEqual(result, ["callback test"])
        
        # Clean up
        pubsub.close()
        
    def test_listen(self):
        # Test the listen() generator
        pubsub = self.redis.pubsub()
        pubsub.subscribe("channel1")
        
        # Allow time for subscription setup
        time.sleep(0.1)
        
        # Consume subscription confirmation
        pubsub.get_message(timeout=1)
        
        # Start listening in a separate thread
        messages = []
        stop_event = threading.Event()
        
        def listener():
            for message in pubsub.listen():
                if message["type"] == "message":
                    messages.append(message["data"])
                    if message["data"] == "stop":
                        stop_event.set()
                        break
        
        thread = threading.Thread(target=listener)
        thread.daemon = True
        thread.start()
        
        # Publish some messages
        self.redis.publish("channel1", "message 1")
        self.redis.publish("channel1", "message 2")
        self.redis.publish("channel1", "stop")
        
        # Wait for listener to process the stop message
        stop_event.wait(timeout=2)
        
        # Verify all messages were received
        self.assertEqual(messages, ["message 1", "message 2", "stop"])
        
        # Clean up
        pubsub.close()
        thread.join(timeout=1) 