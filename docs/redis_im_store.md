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
    memory_limit: int = 10000  # Max entries per agent
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
| `_create_vector_index()` | Create vector search index for optimized searches |
| `_search_similar_redis_vector(agent_id, query_embedding, k, memory_type)` | Optimized vector similarity search using Redis |
| `_search_similar_python(agent_id, query_embedding, k, memory_type)` | Fallback vector similarity search using Python |
| `_search_by_attributes_redis(agent_id, attributes, memory_type)` | Optimized attribute search using Redis |
| `_search_by_attributes_python(agent_id, attributes, memory_type)` | Fallback attribute search using Python |
| `_search_by_step_range_redis(agent_id, start_step, end_step, memory_type)` | Optimized step range search using Redis |
| `_search_by_step_range_python(agent_id, start_step, end_step, memory_type)` | Fallback step range search using Python |

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
    "contents": {
        // Level 1 compressed memory contents
    },
    "metadata": {
        "compression_level": 1,
        "importance_score": 0.5,
        "retrieval_count": 0,
        "creation_time": 1234567890,
        "last_access_time": 1234567890
    },
    "embedding": [0.1, 0.2, ..., 0.9]  // Vector embedding for similarity search
}
```

## **6. Error Handling**

### **6.1 Redis Operation Errors**

- Uses `ResilientRedisClient` with circuit breaker pattern
- Retries failed operations based on priority
- Graceful degradation when Redis is unavailable

### **6.2 Data Validation**

- Enforces level 1 compression requirement
- Validates memory entry structure
- Ensures proper TTL settings on all keys

### **6.3 Search Fallback Mechanisms**

- Detects Redis search capabilities at initialization
- Falls back to Python-based implementations when Redis search is unavailable
- Gracefully handles search errors and provides fallback paths

## **7. Performance Considerations**

### **7.1 Memory Usage**

- Maintains medium-resolution data with level 1 compression
- Automatic TTL-based cleanup
- Configurable memory limits per agent

### **7.2 Search Optimization**

- Uses Redis vector search for optimized similarity searches when available
- Offloads attribute and step range filtering to Redis when possible
- Maintains fallback implementations for compatibility with basic Redis installations

### **7.3 Access Patterns**

- Optimized for time-based queries
- Efficient importance-based retrieval
- Access frequency affects importance scores

### **7.4 Optimization Strategies**

- Redis pipelining for batch operations
- TTL on indices to prevent orphaned data
- Automatic importance score adjustments
- Vector indexing for efficient similarity searches
- Redis search for attribute filtering

## **8. Integration Points**

### **8.1 Memory Transition**

- Accepts memories transitioned from STM
- Verifies compression level before storage
- Updates metadata during transitions

### **8.2 Importance Scoring**

- Base importance from memory creation
- Dynamic updates based on access patterns
- Influences memory retention decisions

### **8.3 Vector Search Requirements**

- Requires Redis Stack or RediSearch module for optimized vector searches
- Automatically detects search capabilities at initialization
- Creates vector index if search capabilities are available

## **9. Best Practices**

1. **Memory Entry Validation**
   - Always verify compression level
   - Ensure required fields are present
   - Validate importance scores (0.0-1.0)

2. **Error Handling**
   - Handle Redis connection issues gracefully
   - Log errors with appropriate context
   - Use retry mechanisms for critical operations

3. **Performance Optimization**
   - Use Redis Stack or RediSearch module for optimized searches
   - Monitor memory usage
   - Implement batch operations where possible
   - Regular maintenance of indices

4. **Vector Search**
   - Include vector embeddings in memory entries for similarity search
   - Consider using HNSW index type for large vector datasets

## **10. See Also**

- [Memory Tiers](memory_tiers.md)
- [Redis STM Store](redis_stm_store.md)
- [SQLite LTM Store](sqlite_ltm_store.md)
- [Agent Memory System](agent_memory_system.md)
- [Redis Search Documentation](https://redis.io/docs/stack/search/) 