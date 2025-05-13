import logging

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
        def cmd():
            return self.redis.set(*args, **kwargs)
        cmd.__name__ = 'set'
        self.commands.append(cmd)
        return self

    def get(self, *args, **kwargs):
        def cmd():
            return self.redis.get(*args, **kwargs)
        cmd.__name__ = 'get'
        self.commands.append(cmd)
        return self

    def delete(self, *args, **kwargs):
        def cmd():
            return self.redis.delete(*args, **kwargs)
        cmd.__name__ = 'delete'
        self.commands.append(cmd)
        return self

    def hset(self, *args, **kwargs):
        def cmd():
            return self.redis.hset(*args, **kwargs)
        cmd.__name__ = 'hset'
        self.commands.append(cmd)
        return self

    def hget(self, *args, **kwargs):
        def cmd():
            return self.redis.hget(*args, **kwargs)
        cmd.__name__ = 'hget'
        self.commands.append(cmd)
        return self

    def hgetall(self, *args, **kwargs):
        def cmd():
            return self.redis.hgetall(*args, **kwargs)
        cmd.__name__ = 'hgetall'
        self.commands.append(cmd)
        return self

    # Sorted Set Operations
    def zadd(self, *args, **kwargs):
        def cmd():
            return self.redis.zadd(*args, **kwargs)
        cmd.__name__ = 'zadd'
        self.commands.append(cmd)
        return self

    def zrange(self, *args, **kwargs):
        def cmd():
            return self.redis.zrange(*args, **kwargs)
        cmd.__name__ = 'zrange'
        self.commands.append(cmd)
        return self

    def zrangebyscore(self, *args, **kwargs):
        def cmd():
            return self.redis.zrangebyscore(*args, **kwargs)
        cmd.__name__ = 'zrangebyscore'
        self.commands.append(cmd)
        return self

    def zrem(self, *args, **kwargs):
        def cmd():
            return self.redis.zrem(*args, **kwargs)
        cmd.__name__ = 'zrem'
        self.commands.append(cmd)
        return self

    def zcard(self, *args, **kwargs):
        def cmd():
            return self.redis.zcard(*args, **kwargs)
        cmd.__name__ = 'zcard'
        self.commands.append(cmd)
        return self

    def zscore(self, *args, **kwargs):
        def cmd():
            return self.redis.zscore(*args, **kwargs)
        cmd.__name__ = 'zscore'
        self.commands.append(cmd)
        return self

    # Additional Hash Operations
    def hmset(self, *args, **kwargs):
        def cmd():
            return self.redis.hmset(*args, **kwargs)
        cmd.__name__ = 'hmset'
        self.commands.append(cmd)
        return self

    def hset_dict(self, *args, **kwargs):
        def cmd():
            return self.redis.hset_dict(*args, **kwargs)
        cmd.__name__ = 'hset_dict'
        self.commands.append(cmd)
        return self

    def hdel(self, *args, **kwargs):
        def cmd():
            return self.redis.hdel(*args, **kwargs)
        cmd.__name__ = 'hdel'
        self.commands.append(cmd)
        return self

    # Additional String Operations
    def expire(self, *args, **kwargs):
        def cmd():
            return self.redis.expire(*args, **kwargs)
        cmd.__name__ = 'expire'
        self.commands.append(cmd)
        return self

    def exists(self, *args, **kwargs):
        def cmd():
            return self.redis.exists(*args, **kwargs)
        cmd.__name__ = 'exists'
        self.commands.append(cmd)
        return self

    def type(self, *args, **kwargs):
        def cmd():
            return self.redis.type(*args, **kwargs)
        cmd.__name__ = 'type'
        self.commands.append(cmd)
        return self

    # Lua script operations
    def eval(self, *args, **kwargs):
        def cmd():
            return self.redis.eval(*args, **kwargs)
        cmd.__name__ = 'eval'
        self.commands.append(cmd)
        return self

    def evalsha(self, *args, **kwargs):
        def cmd():
            return self.redis.evalsha(*args, **kwargs)
        cmd.__name__ = 'evalsha'
        self.commands.append(cmd)
        return self

    def script_load(self, *args, **kwargs):
        def cmd():
            return self.redis.script_load(*args, **kwargs)
        cmd.__name__ = 'script_load'
        self.commands.append(cmd)
        return self

    def script_exists(self, *args, **kwargs):
        def cmd():
            return self.redis.script_exists(*args, **kwargs)
        cmd.__name__ = 'script_exists'
        self.commands.append(cmd)
        return self

    def script_flush(self, *args, **kwargs):
        def cmd():
            return self.redis.script_flush(*args, **kwargs)
        cmd.__name__ = 'script_flush'
        self.commands.append(cmd)
        return self

    def execute(self):
        logger = logging.getLogger("mockredis.pipeline")
        try:
            # Check if any watched keys have changed
            if self._check_watched_keys():
                # If keys changed, abort transaction and return None
                self.reset_transaction_state()
                self.commands.clear()
                logger.debug("Pipeline aborted due to watched key change.")
                return None

            # Execute all commands
            results = []
            for cmd in self.commands:
                try:
                    result = cmd()
                    logger.debug(f"Pipeline command: {cmd.__name__}, result: {result}")
                    
                    # Handle specific command return values to match Redis behavior
                    if cmd.__name__ == 'hset':
                        # hset returns 1 for new field, 0 for existing field
                        results.append(1 if result == 1 else 0)
                    elif cmd.__name__ == 'expire':
                        # expire returns 1 if timeout was set, 0 if key doesn't exist
                        results.append(1 if result else 0)
                    elif cmd.__name__ == 'zadd':
                        # zadd returns number of elements added
                        results.append(result if result is not None else 0)
                    elif cmd.__name__ == 'delete':
                        # delete returns number of keys deleted
                        results.append(result if result is not None else 0)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Pipeline command {cmd.__name__} failed: {e}")
                    results.append(None)

            # Reset state after execution
            self.commands = []
            self.reset_transaction_state()

            # Return the list of results
            logger.debug(f"Pipeline results: {results}")
            return results
        except Exception as e:
            # Clear commands on exception
            self.commands = []
            self.reset_transaction_state()
            logger.error(f"Pipeline execution error: {e}")
            return None  # Return None instead of raising exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_transaction_state()
        self.commands.clear()
        return False  # Don't suppress exceptions
