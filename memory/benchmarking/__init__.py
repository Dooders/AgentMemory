"""
Benchmarking module for the AgentMemory system.

This module provides a comprehensive framework to run, store, analyze, and visualize
benchmark results for various aspects of the AgentMemory system. It enables systematic
performance evaluation and comparison of different memory configurations and algorithms.

Key components:

1. BenchmarkRunner: Executes benchmark tests according to specified configurations,
   collecting performance metrics and handling test scheduling.

2. BenchmarkResults: Stores, processes, and provides analysis tools for benchmark
   outcomes, supporting comparison across different test runs.

3. BenchmarkConfig: Defines benchmark parameters, test scenarios, and system
   configurations to ensure reproducible and controlled testing.

Usage example:
```python
from memory.benchmarking import BenchmarkRunner, BenchmarkConfig, BenchmarkResults

# Configure benchmark parameters
config = BenchmarkConfig(
    name="retrieval_performance",
    iterations=100,
    memory_size=10000
)

# Run benchmarks
runner = BenchmarkRunner(config)
results = runner.run()

# Analyze and visualize results
results.summary()
results.plot_metrics(["latency", "accuracy"])

# Compare with previous benchmark
previous_results = BenchmarkResults.load("previous_benchmark.json")
comparison = results.compare_with(previous_results)
```
"""

from memory.benchmarking.runner import BenchmarkRunner
from memory.benchmarking.results import BenchmarkResults
from memory.benchmarking.config import BenchmarkConfig

__all__ = ["BenchmarkRunner", "BenchmarkResults", "BenchmarkConfig"] 