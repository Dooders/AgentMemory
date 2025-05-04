# Validation Framework

The Validation Framework provides a standardized, reusable, and extensible way to create and run validation tests for various components of the AgentMemory system, particularly for search strategies.

## Purpose

This framework aims to:
- Reduce code duplication across validation implementations
- Provide a consistent interface for all validations
- Make it easy to create new validation test suites
- Enable comprehensive testing and result reporting

## Framework Components

### 1. ValidationTestRunner (`test_runner.py`)

The test runner is responsible for:
- Setting up the memory system and test environment
- Running individual test cases
- Validating test results against expected outcomes
- Tracking test statistics and results
- Generating test summaries

### 2. TestSuite (`test_suite.py`)

The TestSuite abstract base class:
- Defines a standard structure for validation test suites
- Provides a consistent interface for running tests
- Groups tests into logical categories (basic, advanced, edge cases)
- Handles summary reporting

### 3. PerformanceTestSuite (`test_suite.py`)

Extends TestSuite with performance testing capabilities:
- Scaling tests with different memory sizes
- Concurrent load testing
- Performance metrics collection

### 4. Core ValidationTest/PerformanceTest (`validation_framework.py`)

The original validation classes that the framework builds upon:
- ValidationTest - Base class for all validation tests
- PerformanceTest - Extended capabilities for performance testing

## Usage

### Creating a New Test Suite

1. Create a new test suite class that inherits from `TestSuite` or `PerformanceTestSuite`:

```python
from validation.framework.test_suite import TestSuite

class AttributeSearchTestSuite(TestSuite):
    """Test suite for AttributeSearchStrategy."""
    
    def __init__(self, logger=None):
        # Define constants
        STRATEGY_NAME = "attribute"
        AGENT_ID = "test-agent-attribute-search"
        MEMORY_SAMPLE = os.path.join("validation", "memory_samples", "attribute_validation_memory.json")
        
        # Memory ID to checksum mapping
        MEMORY_CHECKSUMS = {
            "meeting-123456-1": "0eb0f81d07276f08e05351a604d3c994564fedee3a93329e318186da517a3c56",
            "meeting-123456-3": "f6ab36930459e74a52fdf21fb96a84241ccae3f6987365a21f9a17d84c5dae1e",
            # Additional mappings...
        }
        
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=AttributeSearchStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger
        )
```

2. Implement the required test methods:

```python
def run_basic_tests(self) -> None:
    """Run basic functionality tests."""
    # Test 1: Basic content search
    self.runner.run_test(
        "Basic Content Search",
        "meeting",
        expected_memory_ids=["meeting-123456-1", "meeting-123456-3"],
        content_fields=["content.content"],
        **self.default_params
    )
    
    # Test 2: Case sensitive search
    self.runner.run_test(
        "Case Sensitive Search",
        "Meeting",
        expected_memory_ids=["meeting-123456-1"],
        case_sensitive=True,
        **self.default_params
    )

def run_advanced_tests(self) -> None:
    """Run advanced functionality tests."""
    # Test complex queries, filters, etc.
    
def run_edge_case_tests(self) -> None:
    """Run edge case tests."""
    # Test empty queries, invalid inputs, etc.
    
    # Test exception handling
    self.runner.run_exception_test(
        "Invalid Threshold",
        ValueError,
        self.runner.strategy.search,
        -0.5,  # Invalid value
        self.agent_id
    )
```

### Running a Test Suite

```python
def main():
    # Create and run the test suite
    test_suite = AttributeSearchTestSuite()
    test_suite.run_all_tests()
    
if __name__ == "__main__":
    main()
```

## Benefits

Using this framework provides several benefits:

1. **Consistency**: All validations follow the same structure and reporting format
2. **Reusability**: Common validation logic is abstracted and reused
3. **Maintainability**: Changes to validation behavior can be made in one place
4. **Extensibility**: New validation types can be added by extending the base classes
5. **Clarity**: Test organization is clear and consistent

## Directory Structure

```
validation/
├── framework/
│   ├── README.md                # This file
│   ├── validation_framework.py  # Original base classes
│   ├── test_runner.py           # Test runner implementation
│   └── test_suite.py            # Test suite base classes
├── search/                      # Strategy-specific validations
│   ├── attribute/
│   ├── importance/
│   └── ...
└── ...
``` 