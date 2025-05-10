class Pipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []
        self.transaction_mode = False
        self.watched_keys = set()
        self.original_watched_values = {}
        self.reset_transaction_state()

    def reset_transaction_state(self):
        """Reset the transaction state"""
        self.transaction_mode = False
        self.watched_keys.clear()
        self.original_watched_values.clear()

    def multi(self):
        """Start a transaction block"""
        self.transaction_mode = True
        return self

    def watch(self, *keys):
        """
        Watch keys for changes during a transaction
        """
        if self.transaction_mode:
            raise ValueError("WATCH can only be called before MULTI")
            
        self.watched_keys.update(keys)
        
        # Store original values for comparison during exec
        for key in keys:
            self.original_watched_values[key] = self.redis.get(key)
            
        return self

    def unwatch(self):
        """
        Forget all watched keys
        """
        self.watched_keys.clear()
        self.original_watched_values.clear()
        return self

    def _check_watched_keys(self):
        """
        Check if any watched keys have changed
        Returns True if transaction should be aborted
        """
        for key in self.watched_keys:
            current_value = self.redis.get(key)
            if current_value != self.original_watched_values.get(key):
                return True
        return False

    def set(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.set(*args, **kwargs))
        return self

    def get(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.get(*args, **kwargs))
        return self

    def delete(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.delete(*args, **kwargs))
        return self

    def hset(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hset(*args, **kwargs))
        return self

    def hget(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hget(*args, **kwargs))
        return self

    def hgetall(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hgetall(*args, **kwargs))
        return self

    # Sorted Set Operations
    def zadd(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.zadd(*args, **kwargs))
        return self
    
    def zrange(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.zrange(*args, **kwargs))
        return self
    
    def zrangebyscore(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.zrangebyscore(*args, **kwargs))
        return self
    
    def zrem(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.zrem(*args, **kwargs))
        return self
    
    def zcard(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.zcard(*args, **kwargs))
        return self
    
    # Additional Hash Operations
    def hmset(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hmset(*args, **kwargs))
        return self
    
    def hset_dict(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hset_dict(*args, **kwargs))
        return self
    
    def hdel(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.hdel(*args, **kwargs))
        return self
    
    # Additional String Operations
    def expire(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.expire(*args, **kwargs))
        return self
    
    def exists(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.exists(*args, **kwargs))
        return self

    # Lua script operations
    def eval(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.eval(*args, **kwargs))
        return self
        
    def evalsha(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.evalsha(*args, **kwargs))
        return self
        
    def script_load(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.script_load(*args, **kwargs))
        return self
        
    def script_exists(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.script_exists(*args, **kwargs))
        return self
        
    def script_flush(self, *args, **kwargs):
        self.commands.append(lambda: self.redis.script_flush(*args, **kwargs))
        return self

    def execute(self):
        try:
            # Check if any watched keys have changed
            if self._check_watched_keys():
                # If keys changed, abort transaction and return None
                self.reset_transaction_state()
                self.commands.clear()
                return None
                
            # Execute all commands
            results = [cmd() for cmd in self.commands]
            
            # Reset state after execution
            self.commands = []
            self.reset_transaction_state()
            
            # Return True if all commands succeeded (no None or False results)
            return all(result is not None and result is not False for result in results)
        except Exception as e:
            # Clear commands on exception
            self.commands = []
            self.reset_transaction_state()
            raise e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_transaction_state()
        self.commands.clear()
        return False  # Don't suppress exceptions
