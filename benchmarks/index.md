# Agent Memory System Benchmarks

This directory contains comprehensive benchmarks for evaluating the performance, scalability, and effectiveness of the Agent Memory System.

## Benchmark Categories

### [Storage Performance](storage_performance.md)
Evaluate the raw storage capabilities across memory tiers:
- Write throughput
- Read latency
- Memory efficiency

### [Compression Effectiveness](compression_effectiveness.md)
Measure the neural compression system's capabilities:
- Embedding quality
- Compression ratio
- Training efficiency

### [Memory Transition](memory_transition.md)
Test the effectiveness of memory tier transitions:
- Transition accuracy
- Transition overhead
- Importance scoring

### [Retrieval Performance](retrieval_performance.md)
Evaluate memory retrieval capabilities:
- Vector search latency
- Search accuracy
- Cross-tier retrieval

### [Scalability](scalability.md)
Measure system performance under increasing load:
- Agent count scaling
- Memory size scaling
- Resource utilization

### [Integration Tests](integration_tests.md)
Evaluate integration with existing agent systems:
- API overhead
- Hook implementation
- Real-world agent integration

## Running the Benchmarks

To run all benchmarks:

```bash
python -m benchmarks.run_all
```

To run a specific benchmark category:

```bash
python -m benchmarks.run --category storage_performance
```

To run an individual benchmark:

```bash
python -m benchmarks.run --benchmark write_throughput
```

## Benchmark Configuration

Benchmark parameters can be configured in `benchmarks/config.py`. The default configuration is designed to provide a comprehensive evaluation while keeping run times reasonable.

## Visualizing Results

Benchmark results are stored in JSON format in the `results/` directory. To generate visualization reports:

```bash
python -m benchmarks.visualize
```

This will create HTML reports in the `reports/` directory with charts and tables showing benchmark results.

## Comparison with Baseline

To compare benchmark results with a baseline:

```bash
python -m benchmarks.compare --baseline results/baseline.json --current results/current.json
```

## Automated Regression Testing

The CI pipeline automatically runs these benchmarks and fails if performance regresses beyond configurable thresholds. See `ci/benchmark_thresholds.json` for current thresholds. 