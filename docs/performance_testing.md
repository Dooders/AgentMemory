# AttributeSearchStrategy Performance Testing

This document outlines the approach for performance testing the AttributeSearchStrategy component of the AgentMemory system.

## Overview

Performance testing is critical to understand how the AttributeSearchStrategy behaves under different load conditions, helping identify potential bottlenecks, optimization opportunities, and operational limits.

The testing framework (`demos/search/performance_test_attribute.py`) provides a comprehensive set of tests to evaluate the strategy's performance across several dimensions.

## Test Scenarios

### 1. Scalability Tests

Evaluates how search performance scales with increasing memory counts:
- Memory sizes: 100, 500, 1,000, 5,000, 10,000
- Measures search execution time as memory count increases
- Generates scaling plots to visualize performance trends

### 2. Concurrent Load Tests

Tests behavior under simultaneous search requests:
- Concurrency levels: 1, 5, 10, 20, 50 threads
- Measures total execution time and average per-request time
- Identifies potential thread contention issues

### 3. Query Type Performance

Compares performance across different query types:
- Simple string queries vs. dictionary queries
- Match_all=True vs. match_all=False
- With and without metadata filters
- Case-sensitive vs. case-insensitive searches

### 4. Regex Performance Impact

Evaluates the performance cost of regex searches:
- Compares regex vs. standard string searches
- Measures pattern caching effectiveness
- Tests with varying regex complexity

### 5. Scoring Method Comparison

Benchmarks all available scoring algorithms:
- length_ratio: Score based on ratio of query length to field length
- term_frequency: Score based on term frequency
- BM25: Score based on BM25 ranking algorithm
- binary: Simple binary scoring (1.0 for match, 0.0 for no match)

### 6. Memory Tier Performance

Compares search performance across memory tiers:
- STM (Short-Term Memory)
- IM (Intermediate Memory)
- LTM (Long-Term Memory)
- All tiers combined

### 7. Resource Usage Monitoring

Tracks system resource consumption during searches:
- CPU usage during search operations
- Memory consumption patterns
- Potential memory leaks during extended runs

## Metrics Tracked

### Time-based Metrics
- **Search latency**: Total execution time for a search operation
- **Throughput**: Searches per second under concurrent load
- **Scaling factor**: How execution time increases relative to memory size

### Resource-based Metrics
- **Peak memory usage**: Maximum memory consumed during search
- **CPU utilization**: Processor usage during search operations
- **Memory growth**: Patterns of memory allocation/deallocation

### Result Quality Metrics
- **Result count consistency**: Whether similar queries return similar numbers of results
- **Scoring distribution**: How scores are distributed across result sets

## Running the Tests

To run the performance tests:

```bash
cd AgentMemory
pip install -r requirements.txt  # Ensure dependencies including psutil, pandas, matplotlib
python -m demos.search.performance_test_attribute
```

The tests will generate:
1. CSV file with detailed results (`results/attribute_search_performance.csv`)
2. Performance plots for visualization:
   - `scaling_performance.png`: Memory size scaling performance
   - `concurrent_performance.png`: Concurrent load performance
   - `scoring_method_performance.png`: Scoring method comparison
   - `regex_comparison.png`: Regex vs standard query performance

## Interpreting Results

### Scaling Characteristics

The search performance should ideally scale sub-linearly or linearly with the number of memories. A steep increase in search time as memory count grows indicates potential scaling issues.

### Concurrency Performance

Ideally, average request time should remain stable under concurrent load. If average time increases dramatically with concurrency, this suggests contention issues.

### Scoring Method Efficiency

Different scoring methods offer tradeoffs between result quality and performance. This test helps identify which method provides the best balance for your specific use case.

### Memory Usage Patterns

Watch for unexpected memory growth that could indicate memory leaks, especially during long-running concurrent tests.

## Optimization Recommendations

Based on test results, consider these potential optimization strategies:

1. **Pattern Caching**: If regex searches are slow, ensure pattern caching is enabled.
2. **Tier Optimization**: If certain tiers show better performance, consider memory tier allocation strategies.
3. **Concurrency Tuning**: Adjust thread pool size based on concurrent performance results.
4. **Scoring Method Selection**: Choose the scoring method that balances performance and result quality.
5. **Memory Management**: Optimize memory allocation patterns based on usage metrics.

## Extension Points

The performance testing framework can be extended to test:

1. Custom search strategies
2. Alternative memory storage backends
3. Integration with caching layers
4. Performance impact of embeddings and vector search
5. Custom scoring algorithms

## Dependencies

The performance testing framework requires:
- psutil: For resource monitoring
- pandas: For data analysis and export
- matplotlib: For visualization
- memory_profiler: For detailed memory analysis
- concurrent.futures: For concurrency testing 