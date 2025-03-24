# Agent Memory System Benchmarking Module

This module provides a comprehensive framework for running, storing, analyzing, and visualizing benchmark results for the Agent Memory System.

## Features

- **Benchmark Execution**: Run individual benchmarks, categories of benchmarks, or all available benchmarks
- **Results Storage**: Store benchmark results in a structured format for future analysis
- **Results Analysis**: Compare results across different runs and configurations
- **Visualization**: Generate HTML reports and plots for benchmark results
- **CLI Interface**: Easy-to-use command-line interface for running benchmarks and analyzing results

## Benchmark Categories

The benchmarking module includes the following categories of benchmarks:

1. **Storage Performance**: Evaluate raw storage capabilities (write throughput, read latency, memory efficiency)
2. **Compression Effectiveness**: Measure neural compression system capabilities
3. **Memory Transition**: Test effectiveness of memory tier transitions
4. **Retrieval Performance**: Evaluate memory retrieval capabilities
5. **Scalability**: Measure system performance under increasing load
6. **Integration Tests**: Evaluate integration with existing agent systems

## Usage

### Running Benchmarks

You can run benchmarks using the provided CLI script:

```bash
# Run all benchmarks
python scripts/benchmark.py run

# Run a specific category of benchmarks
python scripts/benchmark.py run --category storage

# Run a specific benchmark
python scripts/benchmark.py run --category storage --benchmark write_throughput

# Run with custom configuration
python scripts/benchmark.py run --config custom_config.json

# Run in parallel
python scripts/benchmark.py run --parallel --workers 8
```

### Listing Available Benchmarks

```bash
# List all available benchmarks
python scripts/benchmark.py list

# List benchmarks in a specific category
python scripts/benchmark.py list --category retrieval
```

### Working with Results

```bash
# List available results
python scripts/benchmark.py results list

# Generate a report from results
python scripts/benchmark.py results report --category storage

# Compare results with a baseline
python scripts/benchmark.py compare --current-dir results/run1 --baseline-dir results/baseline
```

## Extending the Framework

### Adding New Benchmarks

1. Create a new benchmark function in the appropriate module under `agent_memory/benchmarking/benchmarks/`
2. Ensure the function name starts with `benchmark_`
3. Implement the benchmark, accepting configuration parameters and returning a dictionary of results

Example:

```python
def benchmark_my_new_test(param1: int = 100, param2: float = 0.5, **kwargs) -> Dict[str, Any]:
    """My new benchmark description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    # Benchmark implementation
    return results
```

### Custom Configuration

You can create a custom configuration file by extending the default configuration:

```python
from agent_memory.benchmarking.config import BenchmarkConfig

# Create default config
config = BenchmarkConfig()

# Modify configuration
config.storage.batch_sizes = [10, 50, 100]
config.output_dir = "my_benchmark_results"

# Save modified configuration
config.save("my_config.json")
```

## Result Format

Benchmark results are stored as JSON files with the following structure:

```json
{
  "benchmark": "benchmark_name",
  "category": "category_name",
  "timestamp": "20250323_145623",
  "results": {
    // Benchmark-specific results
  },
  "metadata": {
    "execution_time": 15.2,
    "params": {
      // Parameters used for this run
    },
    "config": {
      // Configuration for this run
    }
  }
}
```

## Requirements

The benchmarking module requires the following packages:

- matplotlib
- pandas
- numpy

These are already included in the main project requirements. 