import json
import logging
import queue
import threading
import time

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
            Number of fields that were added
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

            # Handle traditional form: hset(name, key, value)
            else:
                if key is None:
                    raise ValueError(
                        "Either 'key' and 'value' or 'mapping' must be specified"
                    )

                # Handle common JSON fields
                if key in ["content", "metadata", "embedding"] and isinstance(
                    value, str
                ):
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

                # Count only newly added fields
                result = 0 if key in self.store[name] else 1
                self.store[name][key] = value
                return result

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

            # If this is a timeline or step-based query, try to match by step_number as well
            is_timeline_query = "timeline" in name or "time" in name

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
                                # Check if memory contains step_number or timestamp
                                memory_data = self.store.get(key)

                                if isinstance(memory_data, dict):
                                    # Normalize step_number and timestamp to integers
                                    step_number = memory_data.get("step_number")
                                    if step_number is not None:
                                        # Convert step_number for reliable comparison
                                        step_number = self._normalize_numeric_value(
                                            step_number
                                        )

                                        if min_score <= float(step_number) <= max_score:
                                            filtered_items.append(
                                                (member_str, float(step_number))
                                            )
                                            break

                                    # Check timestamp
                                    timestamp = memory_data.get("timestamp")
                                    if timestamp is not None:
                                        # Convert timestamp for reliable comparison
                                        timestamp = self._normalize_numeric_value(
                                            timestamp
                                        )

                                        # Check if normalized timestamp matches query range
                                        if min_score <= float(timestamp) <= max_score:
                                            filtered_items.append(
                                                (member_str, float(timestamp))
                                            )
                                            break

                                    # Also check content.timestamp if it exists (nested timestamp)
                                    content = memory_data.get("content")
                                    if (
                                        isinstance(content, dict)
                                        and "timestamp" in content
                                    ):
                                        content_timestamp = (
                                            self._normalize_numeric_value(
                                                content["timestamp"]
                                            )
                                        )

                                        if (
                                            min_score
                                            <= float(content_timestamp)
                                            <= max_score
                                        ):
                                            filtered_items.append(
                                                (member_str, float(content_timestamp))
                                            )
                                            break
            else:
                # Standard score range filtering for non-timeline queries
                filtered_items = [
                    (member, score)
                    for member, score in self.store[name].items()
                    if min_score <= float(score) <= max_score
                ]

            sorted_items = sorted(filtered_items, key=lambda x: (x[1], x[0]))

            if num is not None:
                sorted_items = sorted_items[start : start + num]
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

                # Parse the arguments
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
                        i += 1
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
                    else:
                        # Skip any other arguments we don't recognize
                        i += 1

                # Find matching keys based on prefix
                matching_keys = []
                if prefix:
                    for key in self.store.keys():
                        if isinstance(key, bytes):
                            key_str = key.decode("utf-8")
                        else:
                            key_str = key

                        if key_str.startswith(prefix):
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
                    results.append(
                        [
                            ["$", doc_json],
                            ["__embedding_score", "0.75"],  # Example score
                        ]
                    )

                return results

            if command in self.commands:
                return self._circuit_breaker_wrapper(
                    lambda: self.commands[command](*args, **kwargs)
                )
            else:
                # For commands we don't explicitly support, return empty results
                # instead of raising exceptions to allow the demo to continue
                logger.warning(
                    f"Command '{command}' not implemented in MockRedis, returning empty result"
                )
                if command.startswith("ft."):
                    return "OK"  # For RediSearch commands return OK
                return None

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
            The result of the store_func call
        """
        # Just call the store function directly
        return store_func(agent_id, state_data)

    @classmethod
    def from_url(cls, url):
        return cls()

    # Add scan method to support the scan_iter implementation
    def scan(self, cursor=0, match=None, count=None):
        """Incrementally iterate over keys.

        In the mock implementation, returns all keys at once.

        Args:
            cursor: Cursor position (ignored in mock)
            match: Pattern to match keys
            count: Number of keys per call (ignored in mock)

        Returns:
            Tuple of (0, matching_keys)
        """
        with self.lock:
            import fnmatch

            if match:
                result = [key for key in self.store if fnmatch.fnmatch(key, match)]
            else:
                result = list(self.store.keys())

            # Always return cursor 0 to indicate completion
            return 0, result

    def _normalize_numeric_value(self, value):
        """Helper method to normalize timestamps and step numbers for consistent comparison.

        Args:
            value: The value to normalize

        Returns:
            The normalized value (integer if possible, otherwise float or original)
        """
        if value is None:
            return None

        try:
            if isinstance(value, str):
                # Try to convert string to float first
                float_value = float(value)
                # Convert to int if it represents a whole number
                if float_value.is_integer():
                    return int(float_value)
                return float_value
            elif isinstance(value, float):
                # Convert to int if it represents a whole number
                if value.is_integer():
                    return int(value)
                return value
            elif isinstance(value, (int, bool)):
                return value
        except (ValueError, TypeError):
            pass

        # Return original value if conversion failed
        return value
