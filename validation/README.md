# Memory System Validation Framework

This module provides a comprehensive framework for validating memory system components, particularly search strategies. It includes both functional validation and performance testing capabilities.

## Directory Structure

```
validation/
├── framework/                 # Core validation framework
│   └── validation_framework.py  # Base classes and utilities
├── search/                   # Search strategy validations
│   ├── attribute/           # Attribute search validation
│   └── example/             # Example validation implementation
├── memory_samples/          # Sample memory data for testing
├── logs/                    # Test execution logs
├── demo_utils.py           # Common utilities for demos and tests
└── *.py                    # Various validation scripts
```

## Framework Components

### 1. Base Validation Classes

The framework provides two main base classes:

#### ValidationTest (ABC)
- Base class for all validation tests
- Handles test setup, execution, and result validation
- Provides common utilities for logging and result storage

#### PerformanceTest (ValidationTest)
- Extends ValidationTest for performance testing
- Adds capabilities for:
  - Scaling tests
  - Concurrent load testing
  - Performance metrics collection
  - Result visualization

### 2. Key Features

- **Standardized Testing**: Consistent structure for all validation tests
- **Result Validation**: Built-in validation against expected results
- **Performance Metrics**: CPU, memory, and timing measurements
- **Visualization**: Automatic generation of performance plots
- **Logging**: Comprehensive logging of test execution
- **Results Storage**: JSON-based result storage with timestamps

## Usage Examples

### 1. Creating a Validation Test

```python
from validation.framework.validation_framework import ValidationTest

class YourStrategyValidationTest(ValidationTest):
    def setup_memory_system(self, memory_count: int = 1000):
        # Setup your memory system
        self.memory_system = create_memory_system(...)
        self.search_strategy = YourSearchStrategy(...)
    
    def run_test(self, test_name, query, expected_checksums=None, **search_params):
        # Run your test
        results = self.search_strategy.search(query, **search_params)
        return self.validate_results(results, expected_checksums)
```

### 2. Running Functional Tests

```python
# Create validation test instance
validation_test = create_validation_test("your_strategy", YourStrategy)

# Run basic tests
validation_test.run_test(
    "Basic Search",
    "test query",
    expected_memory_ids=["memory-1", "memory-2"],
    content_fields=["content.content"]
)

# Run metadata tests
validation_test.run_test(
    "Metadata Search",
    {"metadata": {"type": "test"}},
    expected_memory_ids=["memory-3"]
)

# Save results
validation_test.save_results()
```

### 3. Running Performance Tests

```python
# Create performance test instance
performance_test = create_validation_test("your_strategy", YourStrategy, is_performance=True)

# Run scaling tests
scaling_results = performance_test.run_scaling_test(
    "test query",
    memory_sizes=[1000, 5000, 10000]
)

# Run concurrent tests
concurrent_results = performance_test.run_concurrent_test(
    "test query",
    concurrency_levels=[1, 5, 10]
)

# Generate performance plots
performance_test.generate_plots(scaling_results, concurrent_results)
```

## Test Categories

### 1. Functional Tests
- Basic content search
- Case sensitivity
- Metadata filtering
- Match all conditions
- Tier-specific searching
- Regex support
- Edge cases

### 2. Performance Tests
- Memory scaling
- Concurrent load
- Response time
- Resource usage
- Scoring method comparison

## Best Practices

1. **Test Organization**
   - Group related tests together
   - Use descriptive test names
   - Include both success and failure cases

2. **Result Validation**
   - Always specify expected results
   - Validate both presence and absence of results
   - Check result ordering when relevant

3. **Performance Testing**
   - Test with realistic data sizes
   - Include warm-up runs
   - Measure both average and peak performance

4. **Documentation**
   - Document test assumptions
   - Explain expected behavior
   - Note any known limitations

## Example Validation Structure

```python
def validate_your_strategy():
    # Create test instance
    validation_test = create_validation_test("your_strategy", YourStrategy)
    
    # Basic functionality tests
    validation_test.run_test(
        "Basic Search",
        "test query",
        expected_memory_ids=["memory-1"]
    )
    
    # Advanced functionality tests
    validation_test.run_test(
        "Complex Search",
        {"content": "test", "metadata": {"type": "test"}},
        match_all=True,
        expected_memory_ids=["memory-2"]
    )
    
    # Edge case tests
    validation_test.run_test(
        "Empty Query",
        "",
        expected_memory_ids=[]
    )
    
    # Save and display results
    validation_test.save_results()
    display_summary(validation_test.results)
```

## Performance Metrics

The framework tracks several key performance metrics:

1. **Time Metrics**
   - Total execution time
   - Per-operation timing
   - Concurrent operation timing

2. **Resource Usage**
   - CPU utilization
   - Memory consumption
   - I/O operations

3. **Result Quality**
   - Result count
   - Result relevance
   - Result ordering

## Visualization

The framework automatically generates performance visualizations:

1. **Scaling Performance**
   - Memory size vs. execution time
   - Memory size vs. resource usage

2. **Concurrency Performance**
   - Concurrent users vs. execution time
   - Concurrent users vs. resource usage

## Logging

Comprehensive logging is provided through:
- Test execution logs
- Performance metrics
- Error tracking
- Result validation

Logs are stored in the `logs/` directory with timestamps for easy reference.

## Contributing

When adding new validation tests:
1. Follow the existing structure
2. Document test assumptions
3. Include both functional and performance tests
4. Add appropriate logging
5. Save and visualize results 