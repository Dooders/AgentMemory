import queue
import fnmatch
import threading
import time

class PubSub:
    def __init__(self, redis):
        self.redis = redis
        self.subscribed_channels = set()
        self.subscribed_patterns = set()
        self.running = True
        self.message_queue = queue.Queue()
        self.callbacks = {}
        self.pattern_callbacks = {}
        self._initialize_listener()

    def _initialize_listener(self):
        def listener():
            while self.running:
                try:
                    # Track channels we've already processed to avoid duplicates with patterns
                    processed_channels = set()
                    
                    # Check for channel messages first
                    for channel in list(self.subscribed_channels):
                        if channel in self.redis.pubsub_queues:
                            try:
                                msg = self.redis.pubsub_queues[channel].get(block=False)
                                message = {
                                    "type": "message",
                                    "pattern": None,
                                    "channel": channel,
                                    "data": msg
                                }
                                
                                # Add to message queue
                                self.message_queue.put(message)
                                
                                # Process any callbacks
                                if channel in self.callbacks:
                                    for callback in self.callbacks[channel]:
                                        try:
                                            callback(message)
                                        except Exception:
                                            # Don't let callback exceptions crash our thread
                                            pass
                                            
                                # Mark this channel as processed
                                processed_channels.add(channel)
                            except queue.Empty:
                                pass
                    
                    # Handle pattern messages - for all channels, not just the ones we're directly subscribed to
                    for pattern in list(self.subscribed_patterns):
                        # Check all active channels, even ones we're not directly subscribed to
                        for channel_name in list(self.redis.pubsub_queues.keys()):
                            # Skip channels we've already processed for direct subscriptions
                            if channel_name in processed_channels:
                                continue
                                
                            if fnmatch.fnmatch(channel_name, pattern):
                                try:
                                    # Get message from channel queue
                                    msg = self.redis.pubsub_queues[channel_name].get(block=False)
                                    message = {
                                        "type": "pmessage",
                                        "pattern": pattern,
                                        "channel": channel_name,
                                        "data": msg
                                    }
                                    
                                    # Add to message queue
                                    self.message_queue.put(message)
                                    
                                    # Process any pattern callbacks
                                    if pattern in self.pattern_callbacks:
                                        for callback in self.pattern_callbacks[pattern]:
                                            try:
                                                callback(message)
                                            except Exception:
                                                # Don't let callback exceptions crash our thread
                                                pass
                                                
                                    # Mark this channel as processed
                                    processed_channels.add(channel_name)
                                except queue.Empty:
                                    pass
                    
                    # Don't hog CPU
                    time.sleep(0.01)
                except Exception:
                    pass
        
        # Start background thread
        threading.Thread(target=listener, daemon=True).start()

    def subscribe(self, *channels, **kwargs):
        """
        Subscribe to channels
        Can also be used with channel=callback keyword arguments
        """
        result = []
        # Handle *channels argument
        for channel in channels:
            if channel not in self.redis.pubsub_queues:
                self.redis.pubsub_queues[channel] = queue.Queue()
            self.subscribed_channels.add(channel)
            message = {"type": "subscribe", "channel": channel, "data": 1}
            result.append(message)
            # Add subscription confirmation to message queue
            self.message_queue.put(message)
        
        # Handle channel=callback keyword arguments
        for channel, callback in kwargs.items():
            if channel not in self.redis.pubsub_queues:
                self.redis.pubsub_queues[channel] = queue.Queue()
            self.subscribed_channels.add(channel)
            self.callbacks.setdefault(channel, []).append(callback)
            message = {"type": "subscribe", "channel": channel, "data": 1}
            result.append(message)
            # Add subscription confirmation to message queue
            self.message_queue.put(message)
        
        return result

    def psubscribe(self, *patterns, **kwargs):
        """
        Subscribe to patterns
        Can also be used with pattern=callback keyword arguments
        """
        result = []
        # Handle *patterns argument
        for pattern in patterns:
            self.subscribed_patterns.add(pattern)
            message = {"type": "psubscribe", "pattern": pattern, "data": 1}
            result.append(message)
            # Add subscription confirmation to message queue
            self.message_queue.put(message)
        
        # Handle pattern=callback keyword arguments
        for pattern, callback in kwargs.items():
            self.subscribed_patterns.add(pattern)
            self.pattern_callbacks.setdefault(pattern, []).append(callback)
            message = {"type": "psubscribe", "pattern": pattern, "data": 1}
            result.append(message)
            # Add subscription confirmation to message queue
            self.message_queue.put(message)
        
        return result

    def unsubscribe(self, *channels):
        if not channels:  # If no channels specified, unsubscribe from all
            channels = list(self.subscribed_channels)
        
        result = []
        for channel in channels:
            self.subscribed_channels.discard(channel)
            if channel in self.callbacks:
                del self.callbacks[channel]
            message = {"type": "unsubscribe", "channel": channel, "data": 0}
            result.append(message)
            # Add unsubscribe confirmation to message queue
            self.message_queue.put(message)
        
        return result

    def punsubscribe(self, *patterns):
        if not patterns:  # If no patterns specified, unsubscribe from all
            patterns = list(self.subscribed_patterns)
        
        result = []
        for pattern in patterns:
            self.subscribed_patterns.discard(pattern)
            if pattern in self.pattern_callbacks:
                del self.pattern_callbacks[pattern]
            message = {"type": "punsubscribe", "pattern": pattern, "data": 0}
            result.append(message)
            # Add unsubscribe confirmation to message queue
            self.message_queue.put(message)
        
        return result

    def get_message(self, timeout=0):
        """
        Get a single message from the queue
        
        Args:
            timeout: Time to wait for a message in seconds, 0 = non-blocking
        
        Returns:
            A message, or None if no message was available
        """
        try:
            if timeout == 0:
                # Non-blocking
                return self.message_queue.get(block=False)
            else:
                # Blocking with timeout
                return self.message_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def listen(self):
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                yield message
            except queue.Empty:
                continue

    def close(self):
        self.running = False
        self.subscribed_channels.clear()
        self.subscribed_patterns.clear()
        self.callbacks.clear()
        self.pattern_callbacks.clear()
