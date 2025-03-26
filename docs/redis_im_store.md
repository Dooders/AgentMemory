# **RedisIMStore Documentation**

## **1. Overview**

The `RedisIMStore` is a Redis-based implementation of the Intermediate Memory (IM) storage tier in the agent memory system. It provides medium-resolution storage with TTL-based expiration for memories that have transitioned from Short-Term Memory but aren't yet ready for Long-Term Memory storage.

## **2. Key Features**

- **Level 1 Compression**: Enforces medium-resolution storage with level 1 compression
- **TTL-Based Expiration**: Automatic expiry of memories after configured duration (default 7 days)
- **Multi-Index Design**: Maintains specialized indices for different query patterns
- **Resilient Redis Operations**: Uses circuit breaker pattern and retry mechanisms
- **Access-Based Importance**: Updates importance scores based on retrieval patterns
- **Comprehensive Error Handling**: Graceful degradation under failure conditions
- **Optimized Vector Search**: Uses Redis vector search capabilities when available for efficient similarity searches
- **Attribute-Based Filtering**: Leverages Redis search for efficient attribute and step range filtering
- **Lua Scripting**: Uses Lua scripts for atomic operations with fallback mechanisms

## **3. Class Structure**

### **3.1 RedisIMStore**

```python
class RedisIMStore:
    def __init__(self, config: RedisIMConfig):
        """Initialize the Redis IM store.
        
        Args:
            config: Configuration for IM Redis storage
        """
```

### **3.2 Configuration (RedisIMConfig)**

```python
@dataclass
class RedisIMConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 1  # IM uses database 1
    password: Optional[str] = None
    
    # Memory settings
    ttl: int = 604800  # 7 days
    compression_level: int = 1  # Level 1 compression
    
    namespace: str = "agent_memory:im"
```

## **4. Key Methods**

### **4.1 Memory Storage and Retrieval**

| Method | Purpose |
|--------|---------|
| `store(agent_id, memory_entry, priority)` | Store a level 1 compressed memory entry |
| `get(agent_id, memory_id)` | Retrieve a memory entry by ID |
| `get_by_timerange(agent_id, start_time, end_time, limit)` | Retrieve memories within a time range |
| `get_by_importance(agent_id, min_importance, max_importance, limit)` | Retrieve memories by importance score |
| `get_all(agent_id, limit)` | Retrieve all memories for an agent with optional limit |
| `get_size(agent_id)` | Get the approximate size in bytes of all memories for an agent |
| `delete(agent_id, memory_id)` | Delete a memory entry |
| `count(agent_id)` | Get the number of memories for an agent |
| `clear(agent_id)` | Clear all memories for an agent |

### **4.2 Advanced Search Methods**

| Method | Purpose |
|--------|---------|
| `search_similar(agent_id, query_embedding, k, memory_type)` | Find semantically similar memories using vector search |
| `search_by_attributes(agent_id, attributes, memory_type)` | Find memories matching specific content attributes |
| `search_by_step_range(agent_id, start_step, end_step, memory_type)` | Find memories within a specific step number range |

### **4.3 Monitoring and Health**

| Method | Purpose |
|--------|---------|
| `check_health()` | Check the health of the Redis store with basic metrics |
| `get_monitoring_data()` | Get comprehensive monitoring data for integration with monitoring dashboards |

### **4.4 Internal Methods**

| Method | Purpose |
|--------|---------|
| `_store_memory_entry(agent_id, memory_entry)` | Internal storage implementation |
| `_update_access_metadata(agent_id, memory_id, memory_entry)` | Update access statistics and importance |
| `_check_vector_search_available()` | Check if Redis vector search capabilities are available |
| `_check_lua_scripting()` | Check if Redis Lua scripting is fully supported |
| `_create_vector_index()` | Create vector search index for optimized searches |
| `_hash_to_memory_entry(hash_data)` | Convert Redis hash data to memory entry dictionary |
| `_get_memory_key(agent_id, memory_id)` | Construct Redis key for a memory entry |
| `_get_agent_memories_key(agent_id)` | Construct Redis key for agent memories list |
| `_get_timeline_key(agent_id)` | Construct Redis key for agent timeline index |
| `_get_importance_key(agent_id)` | Construct Redis key for agent importance index |
| `_get_agent_prefix(agent_id)` | Construct Redis key prefix for an agent |
| `_search_similar_redis_vector(agent_id, query_embedding, limit, score_threshold, memory_type)` | Optimized vector similarity search using Redis |
| `_search_similar_python(agent_id, query_embedding, k, memory_type)` | Fallback vector similarity search using Python |
| `_cosine_similarity(a, b)` | Calculate cosine similarity between two vectors |
| `_search_by_attributes_redis(agent_id, attributes, memory_type)` | Optimized attribute search using Redis |
| `_search_by_attributes_python(agent_id, attributes, memory_type)` | Fallback attribute search using Python |
| `_matches_attributes(memory, attributes)` | Check if a memory matches specified attributes |
| `_search_by_step_range_redis(agent_id, start_step, end_step, memory_type)` | Optimized step range search using Redis |
| `_search_by_step_range_python(agent_id, start_step, end_step, memory_type)` | Fallback step range search using Python |
| `_calculate_hit_rate(info)` | Calculate Redis cache hit rate from info statistics |

## **5. Data Organization**

### **5.1 Redis Key Structure**

| Key Pattern | Purpose |
|-------------|---------|
| `{namespace}:{agent_id}:memory:{memory_id}` | Individual memory entries |
| `{namespace}:{agent_id}:memories` | Sorted set of all agent memories |
| `{namespace}:{agent_id}:timeline` | Chronological index |
| `{namespace}:{agent_id}:importance` | Importance score index |
| `{namespace}_vector_idx` | Vector search index for optimized searches |

### **5.2 Memory Entry Structure**

```json
{
    "memory_id": "unique-id",
    "agent_id": "agent-id",
    "timestamp": 1234567890,
    "content": {
        // Level 1 compressed memory contents
    },
    "metadata": {
        "compression_level": 1,
        "importance_score": 0.5,
        "retrieval_count": 0,
        "creation_time": 1234567890,
        "last_access_time": 1234567890
    },
    "embedding": [0.1, 0.2, ..., 0.9],  // Vector embedding for similarity search
    "memory_type": "observation",  // Optional type classification
    "step_number": 42  // Optional step number for step-based retrieval
}
```

## **6. Error Handling**

### **6.1 Redis Operation Errors**

- Uses `ResilientRedisClient` with circuit breaker pattern
- Retries failed operations based on priority
- Handles specific exceptions:
  - `RedisUnavailableError`: When Redis is not reachable
  - `RedisTimeoutError`: When operations time out
  - `redis.RedisError`: For Redis-specific errors
- Graceful degradation when Redis is unavailable

### **6.2 Data Validation**

- Enforces level 1 compression requirement
- Validates memory entry structure
- Ensures proper TTL settings on all keys
- Handles JSON encoding/decoding errors

### **6.3 Search Fallback Mechanisms**

- Detects Redis search capabilities at initialization
- Falls back to Python-based implementations when Redis search is unavailable
- Detects Lua scripting support and uses pipeline-based approaches when not available
- Gracefully handles search errors and provides fallback paths
- Comprehensive exception handling with appropriate logging

## **7. Performance Considerations**

### **7.1 Memory Usage**

- Maintains medium-resolution data with level 1 compression
- Automatic TTL-based cleanup
- Uses hash fields for efficient storage of memory entries

### **7.2 Search Optimization**

- Uses Redis vector search for optimized similarity searches when available
- Supports FLAT vector index with COSINE distance metric
- Handles both the JSON storage format and hash storage format
- Offloads attribute and step range filtering to Redis when possible
- Maintains fallback implementations for compatibility with basic Redis installations

### **7.3 Access Patterns**

- Optimized for time-based queries
- Efficient importance-based retrieval
- Access frequency affects importance scores
- Dynamically updates access metadata (retrieval count, last access time)
- Adjusts importance scores based on retrieval frequency

### **7.4 Optimization Strategies**

- Redis pipelining for batch operations
- Lua scripts for atomic operations when supported
- TTL on indices to prevent orphaned data
- Automatic importance score adjustments
- Vector indexing for efficient similarity searches
- Redis search for attribute filtering
- Fallback implementations for environments without Redis Stack

## **8. Integration Points**

### **8.1 Memory Transition**

- Accepts memories transitioned from STM
- Verifies compression level before storage
- Updates metadata during transitions

### **8.2 Importance Scoring**

- Base importance from memory creation
- Dynamic updates based on access patterns
- Retrieval count influences importance score adjustments
- Importance scores used for memory retrieval prioritization

### **8.3 Vector Search Requirements**

- Requires Redis Stack or RediSearch module for optimized vector searches
- Automatically detects search capabilities at initialization
- Creates vector index if search capabilities are available
- Vector index uses 1536-dimensional FLOAT32 vectors with COSINE distance metric
- Falls back to Python-based cosine similarity calculation when Redis search is unavailable

### **8.4 Lua Scripting**

- Uses Lua scripts for atomic operations when supported
- Automatically detects Lua scripting support at initialization
- Falls back to pipeline-based approaches when Lua scripting is not fully supported
- Ensures data consistency with both approaches

## **9. Best Practices**

1. **Memory Entry Validation**
   - Always verify compression level
   - Ensure required fields are present
   - Validate importance scores (0.0-1.0)

2. **Error Handling**
   - Handle Redis connection issues gracefully
   - Log errors with appropriate context
   - Use retry mechanisms for critical operations
   - Implement circuit breaker pattern for resilience

3. **Performance Optimization**
   - Use Redis Stack or RediSearch module for optimized searches
   - Enable Lua scripting for atomic operations
   - Monitor memory usage
   - Implement batch operations where possible
   - Regular maintenance of indices

4. **Vector Search**
   - Include vector embeddings in memory entries for similarity search
   - Consider using HNSW index type for large vector datasets
   - Ensure embeddings are normalized for effective cosine similarity

5. **Health Monitoring**
   - Regularly check Redis health
   - Monitor cache hit rates
   - Track memory usage and growth
   - Set up alerts for circuit breaker trips

## **10. See Also**

- [Memory Tiers](memory_tiers.md)
- [Redis STM Store](redis_stm_store.md)
- [SQLite LTM Store](sqlite_ltm_store.md)
- [Agent Memory System](agent_memory_system.md)
- [Redis Search Documentation](https://redis.io/docs/stack/search/)
- [Redis Lua Scripting](https://redis.io/docs/manual/programmability/eval-intro/) 