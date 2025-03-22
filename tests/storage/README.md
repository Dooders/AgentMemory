# Storage Tests

This directory contains tests for the storage components of the Agent Memory system.

## Test Files

- **test_redis_stm.py**: Unit tests for the Redis Short-Term Memory (STM) storage
- **test_redis_stm_integration.py**: Integration tests for the Redis STM storage that require a real Redis instance
- **test_redis_stm_edge_cases.py**: Tests for handling edge cases and unusual inputs in Redis STM storage
- **test_redis_im.py**: Unit tests for the Redis Intermediate Memory (IM) storage
- **test_redis_im_integration.py**: Integration tests for the Redis IM storage

## Testing Approach

The test suite uses a layered approach to test different aspects of the storage subsystem:

1. **Unit Tests**: Mock Redis to test the storage logic in isolation
2. **Edge Cases**: Test robustness with unusual inputs and error conditions
3. **Integration Tests**: Test with real Redis instances to verify end-to-end behavior

### Unit Tests

The unit tests use mock Redis clients to verify the core functionality without requiring
an actual Redis instance. These tests focus on:

- Proper handling of memory storage and retrieval
- Correct indexing and searching
- Error handling and resilience
- Validation of storage patterns

### Edge Case Tests

Edge case tests verify how the storage systems handle unusual inputs and corner cases:

- Invalid or unusual inputs
- Missing or corrupted data
- Extreme values
- Unicode and special characters
- Very large content

### Integration Tests

Integration tests require actual Redis instances and verify that the storage systems
work correctly with real Redis deployments. These tests focus on:

- End-to-end functionality
- TTL handling
- Redis command interactions
- Performance characteristics

## Running the Tests

### Unit and Edge Case Tests

To run the unit and edge case tests:

```bash
pytest tests/storage/test_redis_stm.py tests/storage/test_redis_stm_edge_cases.py -v
```

### Integration Tests

To run the integration tests, you need a Redis instance. By default, the tests will
try to connect to Redis on localhost:6379 with DB 15 (to avoid affecting other data).

You can customize the Redis connection using environment variables:

```bash
REDIS_HOST=custom-host REDIS_PORT=6380 REDIS_PASSWORD=secret REDIS_TEST_DB=14 \
pytest tests/storage/test_redis_stm_integration.py -v
```

To skip integration tests:

```bash
pytest tests/storage -k "not integration" -v
```

## Test Coverage

The tests aim to cover:

- All public methods of the storage classes
- Error handling and recovery mechanisms
- Configuration options
- Performance characteristics
- Resilience under unusual conditions

To run tests with coverage:

```bash
pytest tests/storage --cov=agent_memory.storage
```

## Redis Intermediate Memory (IM) Tests

The `test_redis_im.py` file contains comprehensive tests for the Redis-based Intermediate Memory storage implementation. These tests verify the functionality of memory storage, retrieval, querying, and management operations.

### Requirements

To run these tests, you'll need:

- Python 3.7+
- pytest
- pytest-mock
- redis (for the actual implementation, tests use mocks)

### Running the Tests

From the project root directory, run:

```bash
# Run all storage tests
pytest tests/storage/

# Run only the Redis IM tests
pytest tests/storage/test_redis_im.py

# Run a specific test
pytest tests/storage/test_redis_im.py::test_store_success

# Run with verbose output
pytest tests/storage/test_redis_im.py -v
```

### Test Coverage

The Redis IM tests cover:

1. **Initialization** - Tests that the Redis client is properly initialized with the correct parameters.

2. **Storage Operations**
   - Storing valid memories with proper compression level
   - Handling invalid memories (missing ID, wrong compression level)
   - Error handling during storage operations

3. **Retrieval Operations**
   - Getting memories by ID
   - Handling non-existent memories
   - Error handling during retrieval

4. **Query Operations**
   - Querying memories by time range
   - Querying memories by importance score
   - Error handling during queries

5. **Management Operations**
   - Counting memories
   - Deleting individual memories
   - Clearing all memories for an agent
   - Error handling during management operations

6. **Metadata Updates**
   - Updating access metadata
   - Importance score adjustment based on retrieval frequency
   - Error handling during metadata updates

7. **Error Handling**
   - Redis timeout errors
   - Redis unavailability errors
   - Different priority levels for storage operations

### Mock Strategy

The tests use pytest-mock to create a mock Redis client that simulates the behavior of a real Redis instance without requiring an actual Redis server. This approach allows for:

- Fast, deterministic tests
- Testing error conditions that would be difficult to reproduce with a real Redis server
- No external dependencies during testing
- Verifying the exact interactions with the Redis client

## Integration Tests

In addition to unit tests, we also provide integration tests in `test_redis_im_integration.py` that test the RedisIMStore against a real Redis instance.

### Running Integration Tests

Integration tests are marked with the `integration` marker and require a Redis server to be running.

```bash
# Run only integration tests
pytest tests/storage/test_redis_im_integration.py -v

# Skip integration tests
pytest tests/storage -k "not integration" -v
```

### Integration Test Coverage

The integration tests verify that the RedisIMStore correctly interacts with a real Redis instance:

1. **Storage and Retrieval** - Verifies that memories are correctly stored in Redis and can be retrieved.

2. **Time-Range Queries** - Tests that querying memories within a time range works correctly.

3. **Importance Queries** - Tests that querying memories by importance score works correctly.

4. **Deletion** - Verifies that memory entries are correctly removed from all indices.

5. **Counting and Clearing** - Tests that counting and clearing all memories works as expected.

### Redis Configuration for Tests

The integration tests use DB 15 (by default) and a randomly generated namespace to avoid conflicts with other Redis data. After each test, all keys created during the test are removed to clean up.

### Note on Redis Availability

Before running integration tests, ensure that you have a Redis server running on localhost:6379 (or update the configuration in the test file to match your Redis server). If Redis is not available, the tests will be skipped automatically. 