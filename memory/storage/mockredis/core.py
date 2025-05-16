import json
import logging
import queue
import threading
import time
import random

from .pipeline import Pipeline
from .pubsub import PubSub

# Create logger
logger = logging.getLogger(__name__)


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
    def __init__(self, circuit_breaker_enabled=False, connection_pool=None, **kwargs):
        """Initialize MockRedis.

        Args:
            circuit_breaker_enabled: Whether to simulate circuit breaker behavior
            connection_pool: Ignored, for compatibility with redis.Redis
            **kwargs: Additional parameters (ignored, for compatibility)
        """
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
            "set": self.set,
            "get": self.get,
            "delete": self.delete,
            "hset": self.hset,
            "hget": self.hget,
            "hmset": self.hmset,
            "hset_dict": self.hset_dict,
            "hdel": self.hdel,
            "lpush": self.lpush,
            "rpush": self.rpush,
            "lpop": self.lpop,
            "rpop": self.rpop,
            "zadd": self.zadd,
            "zrange": self.zrange,
            "zrangebyscore": self.zrangebyscore,
            "zrem": self.zrem,
            "zcard": self.zcard,
            "expire": self.expire,
            "exists": self.exists,
            "flushall": self.flushall,
            "eval": self.eval,
            "evalsha": self.evalsha,
            "script_load": self.script_load,
            "script_exists": self.script_exists,
            "script_flush": self.script_flush,
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

    def set(self, key, value, ex=None, px=None, nx=None, xx=None):
        """Set key to value with optional expiration.

        Args:
            key: Key to set
            value: Value to set
            ex: Expire time in seconds
            px: Expire time in milliseconds
            nx: Only set if key does not exist
            xx: Only set if key exists
        """
        with self.lock:
            # Handle NX/XX options
            if nx and key in self.store:
                return None
            if xx and key not in self.store:
                return None

            # If value is a string that looks like JSON, convert it to a dict
            if isinstance(value, str) and value.strip().startswith("{"):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Ensure memory entries have proper structure
            if isinstance(value, dict) and "memory_id" in value:
                # Convert timestamp to int if it exists and is a string
                if "timestamp" in value and isinstance(value["timestamp"], str):
                    try:
                        value["timestamp"] = int(float(value["timestamp"]))
                    except (ValueError, TypeError):
                        pass

                # Convert step_number to int if it exists and is a string
                if "step_number" in value and isinstance(value["step_number"], str):
                    try:
                        value["step_number"] = int(value["step_number"])
                    except (ValueError, TypeError):
                        pass

                # Ensure metadata exists
                if "metadata" not in value:
                    value["metadata"] = {
                        "importance_score": 0.0,
                        "retrieval_count": 0,
                        "creation_time": time.time(),
                        "last_access_time": time.time(),
                    }

            # Set the value
            self.store[key] = value

            # Handle expiration
            if ex:
                self.expirations[key] = time.time() + ex
            elif px:
                self.expirations[key] = time.time() + (px / 1000.0)

            return None  # Return None to match redis-py behavior

    def get(self, key):
        with self.lock:
            value = self.store.get(key)

            # If this is a memory entry (dictionary), ensure timestamp is consistent
            if isinstance(value, dict) and ("memory_id" in value or "content" in value):
                # Ensure timestamp is an integer for consistent comparison
                if "timestamp" in value:
                    value["timestamp"] = self._normalize_numeric_value(
                        value["timestamp"]
                    )

                # Also normalize step_number if present
                if "step_number" in value:
                    value["step_number"] = self._normalize_numeric_value(
                        value["step_number"]
                    )

                # Check content for nested timestamp
                if "content" in value and isinstance(value["content"], dict):
                    if "timestamp" in value["content"]:
                        value["content"]["timestamp"] = self._normalize_numeric_value(
                            value["content"]["timestamp"]
                        )

                # Serialize to JSON string
                return json.dumps(value)

            return value

    def delete(self, *names):
        """Delete one or more keys and return the number of keys deleted."""
        deleted = 0
        with self.lock:
            for name in names:
                if name in self.store:
                    self.store.pop(name, None)
                    self.expirations.pop(name, None)
                    deleted += 1
        return deleted

    def hset(self, name, key=None, value=None, mapping=None):
        """Set key-value pair(s) in a hash.

        This method supports both:
        - hset(name, key, value) syntax
        - hset(name, mapping={key: value, ...}) syntax

        Args:
            name: Hash name
            key: Field name (when using key-value syntax)
            value: Field value (when using key-value syntax)
            mapping: Dictionary of field-value pairs (when using mapping syntax)

        Returns:
            Number of fields that were added (1 for new field, 0 for existing field)
        """
        with self.lock:
            # Create the hash if it doesn't exist
            if name not in self.store:
                self.store[name] = {}

            # Handle mapping form: hset(name, mapping={key: value, ...})
            if mapping is not None:
                count = 0
                for k, v in mapping.items():
                    # Handle common JSON fields by ensuring they're properly parsed
                    if k in ["content", "metadata", "embedding"] and isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Convert numeric fields to appropriate types
                    if k == "step_number" and isinstance(v, str):
                        try:
                            v = int(v)
                        except (ValueError, TypeError):
                            pass

                    if k == "timestamp" and isinstance(v, str):
                        try:
                            v = int(float(v))
                        except (ValueError, TypeError):
                            pass

                    # Count only newly added fields
                    if k not in self.store[name]:
                        count += 1
                    self.store[name][k] = v
                return count

            # Handle key-value form: hset(name, key, value)
            if key is not None:
                # Handle common JSON fields by ensuring they're properly parsed
                if key in ["content", "metadata", "embedding"] and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Convert numeric fields to appropriate types
                if key == "step_number" and isinstance(value, str):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass

                if key == "timestamp" and isinstance(value, str):
                    try:
                        value = int(float(value))
                    except (ValueError, TypeError):
                        pass

                # Return 1 for new field, 0 for existing field
                if key not in self.store[name]:
                    self.store[name][key] = value
                    return 1
                self.store[name][key] = value
                return 0

            return 0

    def hget(self, name, key):
        with self.lock:
            value = self.store.get(name, {}).get(key)

            # Serialize dictionaries and lists for JSON fields
            if key in ["content", "metadata", "embedding"] and isinstance(
                value, (dict, list)
            ):
                return json.dumps(value)

            # Convert numeric fields to strings
            if key in [
                "timestamp",
                "step_number",
                "retrieval_count",
                "importance_score",
            ] and not isinstance(value, str):
                return str(value)

            return value

    def hgetall(self, name):
        """Get all fields and values in a hash.

        Args:
            name: Hash name

        Returns:
            Dictionary of field-value pairs
        """
        with self.lock:
            # Return a copy of the hash to avoid modification issues
            result = dict(self.store.get(name, {}))

            # Ensure fields that should be JSON are serialized as strings
            for key in result:
                # Fields that should be JSON strings
                if key in ["content", "metadata", "embedding"]:
                    if isinstance(result[key], (dict, list)):
                        result[key] = json.dumps(result[key])

                # Convert numeric fields to strings to match Redis behavior
                elif key in [
                    "timestamp",
                    "step_number",
                    "retrieval_count",
                    "importance_score",
                ]:
                    if not isinstance(result[key], str):
                        result[key] = str(result[key])

            return result

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

            # Check if this is a timeline or time-based set
            is_timeline = "timeline" in name or "time" in name
            is_memory_set = ":memories:" in name

            # Special handling for timelines to ensure proper step matching
            if is_timeline or is_memory_set:
                for member, score in mapping.items():
                    # Normalize score to proper numeric value
                    score = self._normalize_numeric_value(score)

                    # Store in the sorted set - always as float for Redis compatibility
                    self.store[name][member] = float(score)

                    # For memory sets, also try to update timestamps in the memory objects
                    if is_memory_set:
                        # Extract memory_id from the member
                        member_str = (
                            member
                            if isinstance(member, str)
                            else member.decode("utf-8")
                        )
                        parts = member_str.split(":")
                        if len(parts) >= 1:
                            memory_id = parts[-1]

                            # Find memory in store
                            for key in self.store.keys():
                                if (
                                    isinstance(key, str)
                                    and ":memory:" in key
                                    and key.endswith(memory_id)
                                ):
                                    memory_data = self.store.get(key)
                                    if isinstance(memory_data, dict):
                                        # Set timestamp using normalized value
                                        memory_data["timestamp"] = score

                                        # Set step_number to match the score for step-based timelines
                                        if isinstance(score, int) or (
                                            isinstance(score, float)
                                            and score.is_integer()
                                        ):
                                            memory_data["step_number"] = int(score)

                                        # If there's content with timestamp, update that too
                                        if "content" in memory_data and isinstance(
                                            memory_data["content"], dict
                                        ):
                                            memory_data["content"]["timestamp"] = (
                                                memory_data["timestamp"]
                                            )
            else:
                # Standard sorted set behavior
                for member, score in mapping.items():
                    self.store[name][member] = float(score)

            return len(mapping)

    def zrange(
        self, name, start, end, withscores=False, desc=False, score_cast_func=float
    ):
        """Get range of members from sorted set with support for score_cast_func.

        Args:
            name: Sorted set name
            start: Start index
            end: End index
            desc: Whether to sort in descending order
            withscores: Whether to include scores in result
            score_cast_func: Function to convert score values (ignored in mock)

        Returns:
            List of members or member/score pairs
        """
        with self.lock:
            if name not in self.store:
                return []

            sorted_items = sorted(
                self.store[name].items(), key=lambda x: (x[1], x[0]), reverse=desc
            )

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

    def zrangebyscore(
        self,
        name,
        min_score,
        max_score,
        withscores=False,
        start=0,
        num=None,
        score_cast_func=float,
    ):
        """Get members with scores between min and max from sorted set.

        Args:
            name: Sorted set name
            min_score: Minimum score
            max_score: Maximum score
            withscores: Whether to include scores in the result
            start: Start offset for pagination
            num: Number of elements to return
            score_cast_func: Function to cast score values (ignored in mock)

        Returns:
            List of members or member/score pairs
        """
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

            # If this is a timeline or importance query, treat it specially
            is_timeline_query = "timeline" in name
            is_importance_query = "importance" in name
            
            # For importance queries, we directly check the score range
            if is_importance_query:
                filtered_items = [
                    (member, score)
                    for member, score in self.store[name].items()
                    if min_score <= float(score) <= max_score
                ]
                
                # For importance queries, sort by score in descending order (higher importance first)
                sorted_items = sorted(filtered_items, key=lambda x: (x[1], x[0]), reverse=True)
                
                if num is not None:
                    sorted_items = sorted_items[start : start + num]
                else:
                    sorted_items = sorted_items[start:]

                if withscores:
                    return [(pair[0], pair[1]) for pair in sorted_items]
                else:
                    return [item[0] for item in sorted_items]
            
            # Original implementation for timeline queries
            filtered_items = []
            if is_timeline_query:
                # A more robust way to match timeline queries
                for member, score in self.store[name].items():
                    # Store both the member and its score
                    member_str = (
                        member if isinstance(member, str) else member.decode("utf-8")
                    )

                    # Check if we can directly match the score
                    if min_score <= float(score) <= max_score:
                        filtered_items.append((member_str, float(score)))
                        continue

                    # Special handling for step_number-based queries
                    # Extract memory_id from the member
                    parts = member_str.split(":")
                    if len(parts) >= 1:
                        memory_id = parts[-1]

                        # Try to find any memory in the store with this ID
                        for key in self.store.keys():
                            if (
                                isinstance(key, str)
                                and memory_id in key
                                and ":memory:" in key
                            ):
                                memory_data = self.store.get(key)
                                if isinstance(memory_data, dict):
                                    # Check for step_number based matching
                                    step_number = memory_data.get("step_number")
                                    if step_number is not None:
                                        step_number = int(step_number)
                                        if min_score <= step_number <= max_score:
                                            filtered_items.append(
                                                (member_str, float(step_number))
                                            )
                                            break

                                    # Fallback to timestamp
                                    timestamp = memory_data.get("timestamp")
                                    if timestamp is not None:
                                        timestamp = float(timestamp)
                                        if min_score <= timestamp <= max_score:
                                            filtered_items.append(
                                                (member_str, float(timestamp))
                                            )
                                            break
            else:
                # General case - filter by score range
                filtered_items = [
                    (member, score)
                    for member, score in self.store[name].items()
                    if min_score <= float(score) <= max_score
                ]

            # Sort by score
            sorted_items = sorted(filtered_items, key=lambda x: (x[1], x[0]))

            # Handle pagination
            if num is not None:
                sorted_items = sorted_items[start : start + num]
            else:
                sorted_items = sorted_items[start:]

            if withscores:
                return [
                    (member, score_cast_func(score)) for member, score in sorted_items
                ]
            else:
                return [member for member, _ in sorted_items]

    def zscore(self, name, value):
        """Get the score of member in a sorted set.
        
        Args:
            name: Sorted set name
            value: Member to get score for
            
        Returns:
            Score of member as a float, or None if member or set doesn't exist
        """
        with self.lock:
            if name not in self.store:
                return None
                
            # Get the score if it exists
            score = self.store[name].get(value)
            
            if score is None:
                # Try to handle case conversion if needed
                if isinstance(value, str):
                    for member in self.store[name]:
                        if isinstance(member, str) and member.lower() == value.lower():
                            return float(self.store[name][member])
            
            return float(score) if score is not None else None

    def zrem(self, name, *values):
        """Remove members from a sorted set.

        Args:
            name: Sorted set name
            *values: Members to remove

        Returns:
            Number of members removed
        """
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
            return None  # Return None to match redis-py behavior

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
            count = 0
            for name in names:
                # Direct key check
                if name in self.store:
                    count += 1
                    continue

                # Handle memory ID pattern matching
                # Format typically looks like: agent:memory:<memory_id>
                if isinstance(name, str):
                    # Try to find the memory ID in any key pattern
                    for key in self.store.keys():
                        key_str = key if isinstance(key, str) else key.decode("utf-8")

                        # Simple case: direct substring match
                        if name in key_str:
                            count += 1
                            break

                        # More complex pattern matching for agent:memory:ID format
                        key_parts = key_str.split(":")
                        # Check if this is a memory key and the last part matches the memory ID
                        if (
                            len(key_parts) >= 3
                            and key_parts[-2] == "memory"
                            and key_parts[-1] == name
                        ):
                            count += 1
                            break

            return count

    def type(self, key):
        """Return the type of value stored at key.
        
        Args:
            key: Key to check type of
            
        Returns:
            String type name: "string", "hash", "list", "set", "zset" or "none" if key doesn't exist
        """
        with self.lock:
            if key not in self.store:
                return "none"
                
            value = self.store[key]
            
            if isinstance(value, dict):
                # Check if it's a sorted set (has float values)
                if value and all(isinstance(v, (int, float)) for v in value.values()):
                    return "zset"
                return "hash"
            elif isinstance(value, list):
                return "list"
            elif isinstance(value, set):
                return "set"
            else:
                return "string"

    # Scanning operations
    def scan_iter(self, match=None, count=None):
        """Iterate over keys matching the given pattern.

        Args:
            match: Pattern to match (using glob-style patterns)
            count: Number of keys to return in each batch (ignored in MockRedis)

        Returns:
            List of matching keys
        """
        if match:
            return self.keys(match)
        return list(self.store.keys())

    def scan(self, cursor=0, match=None, count=None):
        """Incrementally return keys matching a pattern.

        Args:
            cursor: Cursor position (ignored in MockRedis implementation)
            match: Pattern to match (using glob-style patterns)
            count: Number of keys to return in each batch (ignored in MockRedis)

        Returns:
            A tuple of (next_cursor, keys_list)
        """
        # MockRedis simplification: return all keys at once with cursor=0
        keys = self.scan_iter(match=match, count=count)
        return (0, keys)  # 0 cursor means no more keys

    # Advanced Client Features
    def execute_command(self, command, *args, **kwargs):
        """
        Execute a Redis command directly.
        """
        with self.lock:
            command = command.lower()

            # Special handling for specific commands that don't map directly to methods
            if command == "module list":
                # Return a list with some mock modules
                return [
                    [b"name", b"search", b"ver", 20204],  # Simulate RediSearch module
                    [b"name", b"json", b"ver", 20000],  # Simulate RedisJSON module
                ]

            # Handle RediSearch commands
            if command == "ft.info":
                # Mock response for ft.info
                index_name = args[0]
                logger.info(f"Mocking ft.info for index: {index_name}")
                
                # Return vector info if it's a vector index
                if "_vector_idx" in index_name:
                    return {
                        "index_name": index_name,
                        "index_options": ["VEC.HNSW"],
                        "index_definition": {
                            "key_type": "HASH",
                            "prefixes": [f"{index_name.split('_')[0]}-"],
                            "vector_index": True
                        },
                        "attributes": [
                            {
                                "identifier": "$.vector",
                                "attribute": "vector",
                                "type": "VECTOR"
                            }
                        ],
                        "num_docs": 0,
                        "max_doc_id": 0,
                        "num_terms": 0,
                        "num_records": 0,
                        "inverted_sz_mb": 0,
                        "vector_index_sz_mb": 0.01,
                        "total_inverted_index_blocks": 0,
                        "offset_vectors_sz_mb": 0,
                        "doc_table_size_mb": 0,
                        "key_table_size_mb": 0,
                        "records_per_doc_avg": 0,
                        "bytes_per_record_avg": 0,
                        "offsets_per_term_avg": 0,
                        "offset_bits_per_record_avg": 0,
                        "gc_stats": {},
                    }
                else:
                    return {
                        "index_name": index_name,
                        "index_options": [],
                        "index_definition": {},
                        "attributes": [],
                        "num_docs": 0,
                        "max_doc_id": 0,
                        "num_terms": 0,
                        "num_records": 0,
                        "inverted_sz_mb": 0,
                        "vector_index_sz_mb": 0,
                        "total_inverted_index_blocks": 0,
                        "offset_vectors_sz_mb": 0,
                        "doc_table_size_mb": 0,
                        "key_table_size_mb": 0,
                        "records_per_doc_avg": 0,
                        "bytes_per_record_avg": 0,
                        "offsets_per_term_avg": 0,
                        "offset_bits_per_record_avg": 0,
                        "gc_stats": {},
                    }

            if command == "ft.create":
                # Mock successful index creation
                index_name = args[0]
                
                # Check if this is a vector index creation
                is_vector_index = False
                vector_dimension = 0
                
                # Parse arguments to detect vector index creation
                for i in range(len(args)):
                    if isinstance(args[i], str):
                        if args[i].upper() == "SCHEMA" and i+1 < len(args):
                            # Look for vector field definition
                            for j in range(i+1, len(args), 2):
                                if j+1 < len(args) and "VECTOR" in str(args[j+1]).upper():
                                    is_vector_index = True
                                    # Try to extract dimension from vector args
                                    for k in range(j+1, min(j+10, len(args))):
                                        if isinstance(args[k], str) and "DIM" in args[k].upper() and k+1 < len(args):
                                            try:
                                                vector_dimension = int(args[k+1])
                                            except (ValueError, TypeError):
                                                pass
                                    break
                                            
                logger.info(f"Creating index: {index_name}" + 
                           (f" with vector dimension: {vector_dimension}" if is_vector_index else ""))
                
                # Store index info for future reference
                index_key = f"__index__{index_name}"
                self.store[index_key] = {
                    "name": index_name,
                    "is_vector": is_vector_index,
                    "vector_dimension": vector_dimension,
                    "created_at": time.time()
                }
                return "OK"

            if command == "ft.search":
                # Basic implementation of FT.SEARCH
                index_name = args[0]
                query = args[1]

                # Process optional parameters
                limit_start = 0
                limit_num = 10
                sort_by = None
                sort_direction = "ASC"
                prefix = None
                preflen = 0
                return_fields = []
                has_vector_query = False
                vector_query = None
                vector_k = 5
                
                # Parse the arguments for vector query
                i = 2
                while i < len(args):
                    if args[i].lower() == "limit":
                        limit_start = int(args[i + 1])
                        limit_num = int(args[i + 2])
                        i += 3
                    elif args[i].lower() == "sortby":
                        sort_by = args[i + 1]
                        if i + 2 < len(args) and args[i + 2].lower() in ("asc", "desc"):
                            sort_direction = args[i + 2].upper()
                            i += 3
                        else:
                            i += 2
                    elif args[i].lower() == "filter":
                        # Skip FILTER params
                        i += 3
                    elif args[i].lower() == "preflen":
                        preflen = int(args[i + 1])
                        i += 2
                    elif args[i].lower() == "prefix":
                        prefix_count = int(args[i + 1])
                        prefix = args[i + 2]
                        i += 2 + prefix_count
                    elif args[i].lower() == "params":
                        param_count = int(args[i + 1])
                        # Skip over the params
                        i += 2 + param_count
                    elif args[i].lower() == "dialect":
                        # Skip dialect
                        i += 2
                    elif args[i].lower() == "return":
                        return_count = int(args[i + 1])
                        return_fields = args[i+2:i+2+return_count]
                        i += 2 + return_count
                    elif args[i].lower() == "vector":
                        # Vector search parameters
                        has_vector_query = True
                        field_name = args[i + 1]
                        query_vector = args[i + 2] 
                        if i + 3 < len(args) and args[i + 3].lower() == "k":
                            vector_k = int(args[i + 4])
                            i += 5
                        else:
                            i += 3
                    else:
                        # Skip any other arguments we don't recognize
                        i += 1

                # Find matching keys based on prefix
                matching_keys = []
                
                # If this is a vector search
                if "_vector_idx" in index_name and has_vector_query:
                    # Simulate a vector search by returning some random keys
                    agent_prefix = index_name.split('_')[0]
                    metadata_filter = {}  # Initialize empty metadata filter
                    
                    # Extract metadata filter from args if present
                    for i in range(len(args)):
                        if args[i].lower() == "filter":
                            try:
                                metadata_filter = json.loads(args[i + 2])
                            except:
                                pass
                            break
                            
                    for key in self.store.keys():
                        if isinstance(key, str) and key.startswith(f"{agent_prefix}-"):
                            # Get the memory data to check metadata
                            memory_data = self.store.get(key)
                            if memory_data and isinstance(memory_data, dict):
                                # Check if memory has the required metadata
                                metadata = memory_data.get('metadata', {})
                                content = memory_data.get('content', {})
                                content_metadata = content.get('metadata', {}) if isinstance(content, dict) else {}
                                
                                # Check if memory matches metadata filter
                                matches_filter = True
                                for filter_key, filter_value in metadata_filter.items():
                                    # Check in top-level metadata
                                    if filter_key in metadata and metadata[filter_key] == filter_value:
                                        continue
                                    # Check in memory_type
                                    if filter_key == 'type' and 'memory_type' in metadata and metadata['memory_type'] == filter_value:
                                        continue
                                    # Check in content.metadata
                                    if filter_key in content_metadata and content_metadata[filter_key] == filter_value:
                                        continue
                                    # No match found
                                    matches_filter = False
                                    break
                                
                                if matches_filter:
                                    matching_keys.append(key)
                                    if len(matching_keys) >= vector_k:
                                        break
                    
                    # Simulate scores
                    scores = [random.uniform(0.5, 1.0) for _ in matching_keys]
                    # Sort by score in descending order
                    matching_keys = [k for _, k in sorted(zip(scores, matching_keys), reverse=True)]
                elif prefix:
                    for key in self.store.keys():
                        if isinstance(key, bytes):
                            key_str = key.decode("utf-8")
                        else:
                            key_str = key

                        if key_str.startswith(prefix):
                            matching_keys.append(key_str)
                elif query != "*":
                    # Simple text matching
                    for key in self.store.keys():
                        key_str = key
                        if isinstance(key, bytes):
                            key_str = key.decode("utf-8")
                            
                        # Get the value and check if query exists in it
                        value = self.store.get(key)
                        if isinstance(value, dict):
                            try:
                                value_str = json.dumps(value)
                                if query.replace('*', '') in value_str:
                                    matching_keys.append(key_str)
                            except:
                                pass
                        elif isinstance(value, str):
                            if query.replace('*', '') in value:
                                matching_keys.append(key_str)
                else:
                    # * means get all
                    for key in self.store.keys():
                        if isinstance(key, bytes):
                            key_str = key.decode("utf-8")
                        else:
                            key_str = key
                        matching_keys.append(key_str)

                if not matching_keys:
                    # Return empty result
                    return [0]

                # Simulate results
                results = [len(matching_keys)]  # Total count

                # Apply limit
                matching_keys = matching_keys[limit_start : limit_start + limit_num]

                for key in matching_keys:
                    # Add document ID
                    results.append(key)

                    # Get document data - ensure we store a proper dictionary
                    doc_data = self.store.get(key, {})

                    # Ensure doc_data is a dictionary
                    if isinstance(doc_data, str):
                        try:
                            doc_data = json.loads(doc_data)
                        except (json.JSONDecodeError, TypeError):
                            # Create a simple dictionary if string can't be parsed
                            doc_data = {"content": doc_data}
                    elif not isinstance(doc_data, dict):
                        doc_data = {"content": str(doc_data)}

                    # Ensure step_number is an integer if present
                    if "step_number" in doc_data and isinstance(
                        doc_data["step_number"], str
                    ):
                        try:
                            doc_data["step_number"] = int(doc_data["step_number"])
                        except (ValueError, TypeError):
                            pass

                    if "timestamp" in doc_data and isinstance(
                        doc_data["timestamp"], str
                    ):
                        try:
                            doc_data["timestamp"] = int(float(doc_data["timestamp"]))
                        except (ValueError, TypeError):
                            pass

                    # Format as expected by redis_im.py
                    doc_json = json.dumps(doc_data)

                    # Add document fields
                    if has_vector_query:
                        # For vector search, include score
                        score = random.uniform(0.5, 1.0)
                        results.append(
                            [
                                ["$", doc_json],
                                ["__embedding_score", str(score)],  # Vector score
                            ]
                        )
                    else:
                        # For regular search
                        results.append(
                            [
                                ["$", doc_json],
                            ]
                        )

                return results

            if command in self.commands:
                return self._circuit_breaker_wrapper(
                    lambda: self.commands[command](*args, **kwargs)
                )
            else:
                # For commands we don't explicitly support, return appropriate results
                # instead of warnings to allow the demo to continue without warnings
                if command.startswith("ft."):
                    logger.info(f"Redis command '{command}' called with args: {args}")
                    if command == "ft.info":
                        return self._handle_ft_info(args[0])
                    return "OK"  # For RediSearch commands return OK
                
                # For any other command, log a debug message but don't warn
                logger.debug(f"Command '{command}' not implemented in MockRedis, returning empty result")
                return None

    def _handle_ft_info(self, index_name):
        """Handle the FT.INFO command with proper index information."""
        # Handle vector index information
        if "_vector_idx" in index_name:
            prefix = index_name.split("_")[0]
            return {
                "index_name": index_name,
                "index_options": ["VEC.HNSW"],
                "index_definition": {
                    "key_type": "HASH",
                    "prefixes": [f"{prefix}-"],
                    "vector_index": True
                },
                "attributes": [
                    {
                        "identifier": "$.vector",
                        "attribute": "vector",
                        "type": "VECTOR"
                    }
                ],
                "num_docs": 0,
                "max_doc_id": 0,
                "num_terms": 0,
                "num_records": 0,
                "inverted_sz_mb": 0,
                "vector_index_sz_mb": 0.01,
                "total_inverted_index_blocks": 0,
                "offset_vectors_sz_mb": 0,
                "doc_table_size_mb": 0,
                "key_table_size_mb": 0,
                "records_per_doc_avg": 0,
                "bytes_per_record_avg": 0,
                "offsets_per_term_avg": 0,
                "offset_bits_per_record_avg": 0,
                "gc_stats": {},
            }
        else:
            # Regular index info
            return {
                "index_name": index_name,
                "index_options": [],
                "index_definition": {},
                "attributes": [],
                "num_docs": 0,
                "max_doc_id": 0,
                "num_terms": 0,
                "num_records": 0,
                "inverted_sz_mb": 0,
                "vector_index_sz_mb": 0,
                "total_inverted_index_blocks": 0,
                "offset_vectors_sz_mb": 0,
                "doc_table_size_mb": 0,
                "key_table_size_mb": 0,
                "records_per_doc_avg": 0,
                "bytes_per_record_avg": 0,
                "offsets_per_term_avg": 0,
                "offset_bits_per_record_avg": 0,
                "gc_stats": {},
            }

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
            raise CircuitBreakerError(
                "Circuit is open, commands are not being executed"
            )

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
            return None  # Redis SET returns None in Python client
        elif "return redis.call('hget'" in script:
            name = keys[0]
            field = args[0]
            return self.hget(name, field)
        elif "return redis.call('hset'" in script:
            name = keys[0]
            field = args[0]
            value = args[1]
            self.hset(name, field, value)
            return None  # Return None to match redis-py behavior
        elif "redis.call('del'" in script:
            for key in keys:
                self.delete(key)
            return None  # Return None to match redis-py behavior
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
        if not hasattr(self, "_script_cache"):
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
        if not hasattr(self, "_script_cache"):
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
        if not hasattr(self, "_script_cache"):
            self._script_cache = {}

        return [sha1 in self._script_cache for sha1 in sha1s]

    def script_flush(self):
        """
        Flush the script cache
        """
        self._script_cache = {}
        return True

    def ping(self):
        """
        Ping the Redis server

        Returns:
            True if successful
        """
        # Just return True since this is a mock
        return True

    def info(self):
        """
        Get information and statistics about the Redis server

        Returns:
            Dictionary with server information
        """
        # Return a minimal set of mock information
        return {
            "redis_version": "mock",
            "used_memory_human": "1M",
            "used_memory_peak_human": "1M",
            "used_memory_rss_human": "1M",
            "mem_fragmentation_ratio": 1.0,
            "connected_clients": 1,
            "uptime_in_seconds": 1000,
        }

    def get_latency(self):
        """
        Get simulated latency in milliseconds

        Returns:
            A small latency value in milliseconds
        """
        return 0.5  # 0.5ms simulated latency

    def store_with_retry(
        self, agent_id, state_data, store_func, priority=None, max_retries=3
    ):
        """
        Mock implementation of store_with_retry for testing.
        This method directly calls the store_func with the provided parameters.

        Args:
            agent_id: The agent ID for the operation
            state_data: The data to be stored
            store_func: The function to call to perform the actual storage operation
            priority: The priority level (ignored in mock implementation)
            max_retries: Maximum number of retries (ignored in mock implementation)

        Returns:
            The result of the store_func call, or False if an exception occurs
        """
        try:
            # Call the store function and return its result
            result = store_func(agent_id, state_data)
            
            # If result is None or contains None values, consider it a failure
            if result is None:
                return False
            if isinstance(result, list) and None in result:
                return False
                
            return result
        except Exception as e:
            logger.error(f"Error in store_with_retry: {str(e)}")
            return False

    @classmethod
    def from_url(cls, url):
        return cls()

    def _normalize_numeric_value(self, value):
        """Normalize numeric values to integers if they look like integers.

        This helps with consistent comparison of timestamp values, which might
        come in as strings, floats, or integers.

        Args:
            value: Value to normalize

        Returns:
            Normalized value
        """
        if isinstance(value, str):
            try:
                if "." in value:
                    float_val = float(value)
                    # If it represents an integer, convert to int
                    if float_val.is_integer():
                        return int(float_val)
                    return float_val
                else:
                    return int(value)
            except (ValueError, TypeError):
                return value
        return value

    def keys(self, pattern="*"):
        """Find all keys matching the given pattern.

        Args:
            pattern: Pattern to match (using glob-style patterns)

        Returns:
            List of matching keys
        """
        with self.lock:
            all_keys = list(self.store.keys())
            if pattern == "*":
                return all_keys
            
            import re
            # Convert Redis pattern to regex pattern
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            regex = re.compile(f"^{regex_pattern}$")
            return [key for key in all_keys if regex.match(key)]
