# Utils Module Tests

This directory contains unit tests for the utility modules in the agent_memory/utils package:

- `test_serialization.py`: Tests for serialization.py module
- `test_redis_utils.py`: Tests for redis_utils.py module
- `test_error_handling.py`: Tests for error_handling.py module

## Setup & Running Tests

To run all utility tests:

```bash
python -m pytest tests/utils
```

To run tests for a specific module:

```bash
python -m pytest tests/utils/test_serialization.py
```

## Test Coverage

The tests cover:

1. **Serialization Module**
   - Memory serialization and deserialization (JSON and pickle formats)
   - Vector serialization and deserialization
   - Custom JSON encoder/decoder for special types (datetime, set, bytes)
   - Roundtrip tests for various data types

2. **Redis Utils Module**
   - Redis connection management
   - Redis batch processing
   - Memory entry and vector serialization for Redis
   - Redis key operations (exists, scan, index creation/deletion)

3. **Error Handling Module**
   - Error class hierarchy
   - Circuit breaker pattern implementation
   - Retry policies with exponential backoff
   - Recovery queue for failed operations

## Notes on Test Failures

Some tests may fail due to differences between the test assumptions and the actual implementation. The tests were created based on reading the code without complete knowledge of the implementation details.

Common issues include:
- Attribute naming differences (e.g., `_running` vs. `running`)
- Method signatures and return values
- Connection string formats in Redis connection management

These tests should be adjusted to match the actual implementation. The passing tests can serve as a good starting point. 