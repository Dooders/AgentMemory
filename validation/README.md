# Memory System Validation Framework

This module provides a comprehensive framework for validating memory system components, particularly search strategies. It includes both functional validation and performance testing capabilities.

## Directory Structure

```
validation/
├── framework/                   # Core validation framework
│   ├── validation_framework.py  # Original base classes
│   ├── test_runner.py           # Test runner implementation
│   ├── test_suite.py            # Test suite base classes 
│   ├── cli.py                   # Command-line interface
│   └── README.md                # Framework documentation
├── search/                      # Search strategy validations
│   ├── attribute/               # Attribute search validation
│   ├── importance/              # Importance search validation
│   └── ...
├── memory_samples/              # Sample memory data for testing
├── logs/                        # Test execution logs
├── run_validations.py           # Main entry point script
├── demo_utils.py                # Common utilities for demos and tests
└── README.md                    # This file
```

## Framework Components

### 1. Base Validation Classes

The framework provides several main classes:

#### ValidationTestRunner
- Handles the execution of individual tests
- Manages test setup, execution, and result validation
- Provides utilities for tracking and reporting test results

#### TestSuite (ABC)
- Abstract base class for organizing related tests
- Standardizes the test suite structure
- Groups tests into basic, advanced, and edge case categories

#### PerformanceTestSuite
- Extends TestSuite for performance testing
- Adds capabilities for scaling and concurrent load tests

#### Original Base Classes
- ValidationTest - Core validation functionality
- PerformanceTest - Performance testing capabilities

### 2. Key Features

- **Standardized Testing**: Consistent structure for all validation tests
- **Result Validation**: Built-in validation against expected results
- **Performance Metrics**: CPU, memory, and timing measurements
- **Visualization**: Automatic generation of performance plots
- **Logging**: Comprehensive logging of test execution
- **Results Storage**: JSON-based result storage with timestamps
- **Command-line Interface**: Run validations from the command line

## Usage Examples

### 1. Creating a Test Suite

```python
from validation.framework.test_suite import TestSuite
from memory.search.strategies.your_strategy import YourStrategy

class YourStrategyTestSuite(TestSuite):
    def __init__(self, logger=None):
        # Constants
        STRATEGY_NAME = "your_strategy"
        AGENT_ID = "test-agent-your-strategy"
        MEMORY_SAMPLE = os.path.join("validation", "memory_samples", "your_strategy_memory.json")
        
        # Memory ID to checksum mapping
        MEMORY_CHECKSUMS = {
            "memory-1": "checksum1",
            "memory-2": "checksum2",
            # More mappings...
        }
        
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=YourStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger
        )
    
    def run_basic_tests(self):
        # Basic functionality tests
        self.runner.run_test(
            "Basic Search",
            "test query",
            expected_memory_ids=["memory-1", "memory-2"],
            memory_checksum_map=self.memory_checksum_map
        )
    
    def run_advanced_tests(self):
        # Advanced functionality tests
        self.runner.run_test(
            "Complex Search",
            {"content": "test", "metadata": {"type": "test"}},
            match_all=True,
            expected_memory_ids=["memory-2"],
            memory_checksum_map=self.memory_checksum_map
        )
    
    def run_edge_case_tests(self):
        # Edge case tests
        self.runner.run_test(
            "Empty Query",
            "",
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map
        )
```

### 2. Running a Test Suite Programmatically

```python
# Create and run the test suite
test_suite = YourStrategyTestSuite()

# Run all tests
test_suite.run_all_tests()

# Or run specific test categories
test_suite.run_basic_tests()
test_suite.run_advanced_tests()
test_suite.run_edge_case_tests()
test_suite.runner.display_summary()
```

### 3. Running Tests from Command Line

The framework includes a command-line interface for running tests:

```bash
# Run all tests for a specific strategy
python validation/run_validations.py your_strategy

# Run only basic tests
python validation/run_validations.py your_strategy --test-type basic

# Run with performance tests
python validation/run_validations.py your_strategy --perf

# Run all available test suites
python validation/run_validations.py all
```

## Test Categories

### 1. Basic Tests
- Core functionality tests
- Simple search queries
- Basic filtering options

### 2. Advanced Tests
- Complex queries
- Multiple conditions
- Advanced filtering
- Special search options

### 3. Edge Case Tests
- Empty queries
- Invalid inputs
- Boundary conditions
- Error handling

### 4. Performance Tests
- Memory scaling
- Concurrent load
- Response time
- Resource usage

## Best Practices

### Creating Test Suites

1. **Test Organization**
   - Group related tests in appropriate test methods
   - Use clear, descriptive test names
   - Include both success and failure cases

2. **Result Validation**
   - Always specify expected results
   - Use the memory_checksum_map for consistent validation
   - Check both presence and absence of results

3. **Performance Testing**
   - Extend PerformanceTestSuite for performance tests
   - Use realistic data sizes
   - Test with varying memory sizes and concurrency levels

4. **Documentation**
   - Document test assumptions
   - Explain expected behavior
   - Note any known limitations

### Running Tests

1. **Use the CLI for quick testing**
   - The command-line interface provides easy access to all test suites
   - Filter by test type to focus on specific areas
   - Use the --verbose flag for detailed logging

2. **Use programmatic interface for integration**
   - Create custom test runners for integration with other systems
   - Automate test execution in CI/CD pipelines
   - Implement custom result handling

## Contributing

When adding new validation tests:

1. **Follow the framework structure**
   - Create a new test suite class extending TestSuite
   - Implement the required test methods
   - Use the standard validation patterns

2. **Add your tests to the discovery system**
   - Name your files following the convention (validate_*.py)
   - Place them in the appropriate directory
   - Make sure they're discoverable by the CLI

3. **Include comprehensive tests**
   - Cover all functionality of your component
   - Include edge cases and error handling
   - Add performance tests if applicable

4. **Document your tests**
   - Add docstrings to your test suite and methods
   - Comment complex test scenarios
   - Update this README if adding new test categories 