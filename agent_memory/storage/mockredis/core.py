import time
import threading
from .pubsub import PubSub
from .pipeline import Pipeline
import queue

# Redis Error Classes for accurate error handling
class RedisError(Exception):
    """Base exception for Redis errors"""
    pass

class ConnectionError(RedisError):
    """Raised when there is a connection error"""
    pass

class CircuitBreakerError(ConnectionError):
    """Raised when the circuit breaker is open"""
    pass

class MockRedis:
    def __init__(self, circuit_breaker_enabled=False):
        self.store = {}
        self.expirations = {}
        self.pubsub_queues = {}
        self.lock = threading.Lock()
        self._start_cleaner()
        
        # Circuit breaker properties
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.circuit_state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.failure_threshold = 5
        self.recovery_timeout = 30
        self.last_failure_time = 0
        
        # Script cache for Lua scripts
        self._script_cache = {}
        
        # Map of commands to their implementation methods
        self.commands = {
            'set': self.set,
            'get': self.get,
            'delete': self.delete,
            'hset': self.hset,
            'hget': self.hget,
            'hmset': self.hmset,
            'hset_dict': self.hset_dict,
            'hdel': self.hdel,
            'lpush': self.lpush,
            'rpush': self.rpush,
            'lpop': self.lpop,
            'rpop': self.rpop,
            'zadd': self.zadd,
            'zrange': self.zrange,
            'zrangebyscore': self.zrangebyscore,
            'zrem': self.zrem,
            'zcard': self.zcard,
            'expire': self.expire,
            'exists': self.exists,
            'flushall': self.flushall,
            'eval': self.eval,
            'evalsha': self.evalsha,
            'script_load': self.script_load,
            'script_exists': self.script_exists,
            'script_flush': self.script_flush
        }

    def _start_cleaner(self):
        def cleaner():
            while True:
                now = time.time()
                with self.lock:
                    to_delete = [k for k, exp in self.expirations.items() if exp <= now]
                    for k in to_delete:
                        self.store.pop(k, None)
                        self.expirations.pop(k, None)
                time.sleep(1)
        threading.Thread(target=cleaner, daemon=True).start()

    def set(self, key, value, ex=None):
        with self.lock:
            self.store[key] = value
            if ex:
                self.expirations[key] = time.time() + ex

    def get(self, key):
        with self.lock:
            return self.store.get(key)

    def delete(self, key):
        with self.lock:
            self.store.pop(key, None)
            self.expirations.pop(key, None)

    def hset(self, name, key, value):
        with self.lock:
            self.store.setdefault(name, {})[key] = value

    def hget(self, name, key):
        with self.lock:
            return self.store.get(name, {}).get(key)

    def lpush(self, name, *values):
        with self.lock:
            self.store.setdefault(name, [])
            self.store[name] = list(reversed(values)) + self.store[name]

    def rpush(self, name, *values):
        with self.lock:
            self.store.setdefault(name, [])
            self.store[name].extend(values)

    def lpop(self, name):
        with self.lock:
            if name not in self.store or not self.store[name]:
                return None
            return self.store[name].pop(0)

    def rpop(self, name):
        with self.lock:
            if name not in self.store or not self.store[name]:
                return None
            return self.store[name].pop()

    def flushall(self):
        with self.lock:
            self.store.clear()
            self.expirations.clear()

    def publish(self, channel, message):
        # Create the channel queue if it doesn't exist
        if channel not in self.pubsub_queues:
            self.pubsub_queues[channel] = queue.Queue()
        
        # Add message to the channel queue
        self.pubsub_queues[channel].put(message)

    def pubsub(self):
        return PubSub(self)

    def pipeline(self):
        return Pipeline(self)

    # Sorted Set Operations
    def zadd(self, name, mapping=None, **kwargs):
        with self.lock:
            if mapping is None:
                mapping = kwargs
            
            if name not in self.store:
                self.store[name] = {}
                
            for member, score in mapping.items():
                self.store[name][member] = float(score)
                
            return len(mapping)
    
    def zrange(self, name, start, end, withscores=False, desc=False):
        with self.lock:
            if name not in self.store:
                return []
                
            sorted_items = sorted(self.store[name].items(), key=lambda x: (x[1], x[0]), reverse=desc)
            
            # Handle negative indices
            if start < 0:
                start = max(len(sorted_items) + start, 0)
            if end < 0:
                end = len(sorted_items) + end + 1
            else:
                end = min(end + 1, len(sorted_items))
                
            result = sorted_items[start:end]
            
            if withscores:
                return [item for pair in result for item in pair]
            else:
                return [item[0] for item in result]
    
    def zrangebyscore(self, name, min_score, max_score, withscores=False, start=0, num=None):
        with self.lock:
            if name not in self.store:
                return []
                
            # Handle infinity values
            if min_score == "-inf":
                min_score = float("-inf")
            else:
                min_score = float(min_score)
                
            if max_score == "+inf":
                max_score = float("inf")
            else:
                max_score = float(max_score)
                
            filtered_items = [
                (member, score) for member, score in self.store[name].items()
                if min_score <= score <= max_score
            ]
            
            sorted_items = sorted(filtered_items, key=lambda x: (x[1], x[0]))
            
            if num is not None:
                sorted_items = sorted_items[start:start + num]
            else:
                sorted_items = sorted_items[start:]
                
            if withscores:
                return [item for pair in sorted_items for item in pair]
            else:
                return [item[0] for item in sorted_items]
    
    def zrem(self, name, *values):
        with self.lock:
            if name not in self.store:
                return 0
                
            removed = 0
            for value in values:
                if value in self.store[name]:
                    del self.store[name][value]
                    removed += 1
                    
            return removed
    
    def zcard(self, name):
        with self.lock:
            if name not in self.store:
                return 0
            return len(self.store[name])
    
    # Additional Hash Operations
    def hset_dict(self, name, mapping):
        """
        Set multiple hash fields to multiple values using a mapping
        Alternative to hmset with dictionary input
        """
        with self.lock:
            if name not in self.store:
                self.store[name] = {}
            
            # Update hash with mapping
            self.store[name].update(mapping)
            return len(mapping)
            
    def hmset(self, name, mapping):
        with self.lock:
            if name not in self.store:
                self.store[name] = {}
            self.store[name].update(mapping)
            return True
    
    def hdel(self, name, *keys):
        with self.lock:
            if name not in self.store:
                return 0
            
            deleted = 0
            for key in keys:
                if key in self.store[name]:
                    del self.store[name][key]
                    deleted += 1
            
            return deleted
    
    # Additional String Operations
    def expire(self, name, time_seconds):
        with self.lock:
            if name in self.store:
                self.expirations[name] = time.time() + time_seconds
                return 1
            return 0
    
    def exists(self, *names):
        with self.lock:
            return sum(1 for name in names if name in self.store)
    
    # Scanning operations
    def scan_iter(self, match=None, count=None):
        with self.lock:
            import fnmatch
            
            if match:
                for key in self.store:
                    if fnmatch.fnmatch(key, match):
                        yield key
            else:
                for key in self.store:
                    yield key

    # Advanced Client Features
    def execute_command(self, command, *args, **kwargs):
        """
        Execute a Redis command directly.
        """
        with self.lock:
            command = command.lower()
            if command in self.commands:
                return self._circuit_breaker_wrapper(lambda: self.commands[command](*args, **kwargs))
            else:
                raise NotImplementedError(f"Command '{command}' not implemented in MockRedis")
    
    # Circuit Breaker Pattern Implementation
    def _check_circuit_state(self):
        """Check and potentially update the state of the circuit breaker"""
        if not self.circuit_breaker_enabled:
            return
            
        now = time.time()
        
        # If circuit is open, check if recovery timeout has passed
        if self.circuit_state == "open":
            if now - self.last_failure_time >= self.recovery_timeout:
                self.circuit_state = "half_open"
                self.failure_count = 0
        
        # If circuit is half-open, let a single request through to test
        # The actual test happens in the circuit_breaker_wrapper
    
    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        if not self.circuit_breaker_enabled:
            return
            
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_state = "open"
    
    def _record_success(self):
        """Record a success and potentially close the circuit"""
        if not self.circuit_breaker_enabled:
            return
            
        if self.circuit_state == "half_open":
            self.circuit_state = "closed"
            self.failure_count = 0
    
    def _circuit_breaker_wrapper(self, func):
        """Wrap a function call with circuit breaker logic"""
        if not self.circuit_breaker_enabled:
            return func()
            
        self._check_circuit_state()
        
        # If circuit is open, raise an exception
        if self.circuit_state == "open":
            raise CircuitBreakerError("Circuit is open, commands are not being executed")
        
        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e
    
    # Simulate connection failures for testing
    def simulate_connection_failure(self, duration=None):
        """
        Simulate a connection failure by opening the circuit
        
        Args:
            duration: If provided, automatically recover after this many seconds
        """
        self.circuit_breaker_enabled = True
        self.circuit_state = "open"
        self.last_failure_time = time.time()
        
        if duration:
            # Automatically close the circuit after duration
            def auto_recover():
                time.sleep(duration)
                self.circuit_state = "closed"
                self.failure_count = 0
            
            threading.Thread(target=auto_recover, daemon=True).start()
    
    def eval(self, script, numkeys, *keys_and_args):
        """
        Evaluate a Lua script. A very basic and limited implementation
        that only supports a few simple operations.
        
        Args:
            script: The Lua script string
            numkeys: Number of keys that the script will touch
            *keys_and_args: The keys and arguments to pass to the script
        """
        # Extract keys and arguments
        keys = keys_and_args[:numkeys]
        args = keys_and_args[numkeys:]
        
        # Extremely simplified Lua script support - only handles basic operations
        if "return redis.call('get'" in script:
            key = keys[0]
            return self.get(key)
        elif "return redis.call('set'" in script:
            key = keys[0]
            value = args[0]
            self.set(key, value)
            return True
        elif "return redis.call('hget'" in script:
            name = keys[0]
            field = args[0]
            return self.hget(name, field)
        elif "return redis.call('hset'" in script:
            name = keys[0]
            field = args[0]
            value = args[1]
            self.hset(name, field, value)
            return True
        elif "redis.call('del'" in script:
            for key in keys:
                self.delete(key)
            return True
        else:
            # Store script hash for future evalsha calls
            import hashlib
            sha1 = hashlib.sha1(script.encode()).hexdigest()
            self._script_cache[sha1] = script
            
            # For unsupported scripts, just return a simple value
            # In a real implementation, we would actually execute the Lua code
            return None
    
    def evalsha(self, sha1, numkeys, *keys_and_args):
        """
        Evaluate a script from the script cache using its SHA1 digest
        
        Args:
            sha1: The SHA1 digest of the script
            numkeys: Number of keys that the script will touch
            *keys_and_args: The keys and arguments to pass to the script
        """
        if not hasattr(self, '_script_cache'):
            self._script_cache = {}
            
        if sha1 not in self._script_cache:
            raise RedisError(f"NOSCRIPT No matching script. Please use EVAL.")
            
        script = self._script_cache[sha1]
        return self.eval(script, numkeys, *keys_and_args)
    
    def script_load(self, script):
        """
        Load a script into the script cache
        
        Args:
            script: The Lua script to load
            
        Returns:
            The SHA1 digest of the script
        """
        if not hasattr(self, '_script_cache'):
            self._script_cache = {}
            
        import hashlib
        sha1 = hashlib.sha1(script.encode()).hexdigest()
        self._script_cache[sha1] = script
        return sha1
    
    def script_exists(self, *sha1s):
        """
        Check if scripts exist in the script cache
        
        Args:
            *sha1s: The SHA1 digests to check
            
        Returns:
            A list of booleans indicating whether each script exists
        """
        if not hasattr(self, '_script_cache'):
            self._script_cache = {}
            
        return [sha1 in self._script_cache for sha1 in sha1s]
    
    def script_flush(self):
        """
        Flush the script cache
        """
        self._script_cache = {}
        return True
    
    @classmethod
    def from_url(cls, url):
        return cls()
