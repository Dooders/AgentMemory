# Agent Memory Tests

This directory contains unit and integration tests for the Agent Memory system.

## Attribution Retrieval Tests

### `test_attribute_retrieval.py`

Primary unit test file for the `AttributeRetrieval` class. It tests the basic functionality of all methods in the class with mocked storage backends.

Key features:
- Tests for all retrieval methods (by type, importance, metadata, content, etc.)
- Tests for helper methods like `_get_value_at_path` and `_matches_metadata_filters`
- Basic coverage with standard inputs and outputs

### `test_attribute_retrieval_edge_cases.py`

Tests edge cases and error handling for the `AttributeRetrieval` class:
- Empty result sets
- Malformed memories and data
- Invalid inputs (regex patterns, paths, etc.)
- Error propagation
- None values and special inputs

### `test_attribute_retrieval_combined.py`

Tests for combined usage patterns and workflows:
- Chaining multiple retrieval operations
- Cross-tier searches
- Performance with larger datasets
- Complex queries combining multiple conditions

### `test_attribute_retrieval_integration.py`

Integration tests showing how `AttributeRetrieval` interacts with storage layers:
- Mocked Redis and SQLite backends
- Proper method calls and integration points
- Exception handling and fault tolerance
- Cross-store memory access patterns

## Running Tests

Run tests with pytest:

```bash
# Run all tests
pytest

# Run only attribute retrieval tests
pytest tests/test_attribute_retrieval*.py

# Run with verbosity
pytest -v tests/test_attribute_retrieval.py

# Run with coverage
pytest --cov=agent_memory.retrieval.attribute tests/test_attribute_retrieval*.py
``` 