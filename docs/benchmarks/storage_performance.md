# Storage Performance Benchmarks

These benchmarks evaluate the raw storage capabilities of the Agent Memory System across its three tiers: Short-Term Memory (STM), Intermediate Memory (IM), and Long-Term Memory (LTM).

## Write Throughput

### Benchmark Description
Measure the rate at which the system can store new memories in each tier.

### Methodology
1. Generate synthetic agent states and actions of varying complexity
2. Perform batch writes of increasing size (10, 100, 1000, 10000 items)
3. Measure throughput in operations per second for each tier

### Code Example
```python
from memory import AgentMemorySystem, MemoryConfig
import time
import random

def generate_test_state(complexity=1):
    """Generate synthetic agent state with variable complexity"""
    state = {"position": [random.random() for _ in range(complexity * 3)],
             "inventory": [random.choice(["item1", "item2", "item3"]) 
                          for _ in range(complexity * 5)],
             "attributes": {f"attr_{i}": random.random() 
                           for i in range(complexity * 10)}}
    return state

def benchmark_write_throughput(batch_sizes=[10, 100, 1000, 10000], complexity=1):
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    results = {}
    for batch_size in batch_sizes:
        # Generate test data
        test_states = [generate_test_state(complexity) for _ in range(batch_size)]
        
        # Benchmark STM write
        start_time = time.time()
        for i, state in enumerate(test_states):
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_data=state,
                step_number=i
            )
        end_time = time.time()
        
        stm_throughput = batch_size / (end_time - start_time)
        results[batch_size] = {
            "stm_throughput": stm_throughput,
            # Similar measurements for IM and LTM
        }
    
    return results
```

### Expected Metrics
- STM Write Throughput: Operations per second
- IM Write Throughput: Operations per second
- LTM Write Throughput: Operations per second

## Read Latency

### Benchmark Description
Compare retrieval times across the three memory tiers.

### Methodology
1. Pre-populate each memory tier with test data
2. Perform single-item retrievals with random IDs
3. Perform batch retrievals of 10, 50, and 100 items
4. Measure average retrieval time in milliseconds

### Code Example
```python
def benchmark_read_latency(num_samples=1000):
    # Pre-populate memory tiers with test data
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    # ... population code ...
    
    # Single item retrieval
    stm_latencies = []
    for i in range(num_samples):
        state_id = random.choice(stm_state_ids)
        start_time = time.time()
        state = memory_system.retrieve_state(state_id, tier="stm")
        end_time = time.time()
        stm_latencies.append((end_time - start_time) * 1000)  # ms
    
    # Similarly for IM and LTM...
    
    # Batch retrieval tests
    # ...
    
    return {
        "stm": {
            "single_item_latency_ms": sum(stm_latencies) / len(stm_latencies),
            # Batch results
        },
        "im": { /* ... */ },
        "ltm": { /* ... */ }
    }
```

### Expected Metrics
- Average Single-Item Retrieval Time (ms) per tier
- Average Batch Retrieval Time (ms) per tier and batch size

## Memory Efficiency

### Benchmark Description
Evaluate storage efficiency across memory tiers for identical data.

### Methodology
1. Generate standard test datasets of varying complexity
2. Store identical datasets in each memory tier
3. Measure storage footprint in bytes
4. Calculate storage efficiency ratio between tiers

### Code Example
```python
def benchmark_memory_efficiency(data_complexity_levels=[1, 5, 10]):
    results = {}
    
    for complexity in data_complexity_levels:
        test_data = generate_test_dataset(complexity)
        
        # Measure STM storage size
        memory_system = AgentMemorySystem.get_instance(MemoryConfig())
        for item in test_data:
            memory_system.store_agent_state("test_agent", item, 0)
        
        stm_size = measure_redis_memory_usage("stm_prefix")
        im_size = measure_redis_memory_usage("im_prefix")
        ltm_size = measure_sqlite_storage_size()
        
        results[complexity] = {
            "stm_size_bytes": stm_size,
            "im_size_bytes": im_size,
            "ltm_size_bytes": ltm_size,
            "im_to_stm_ratio": im_size / stm_size,
            "ltm_to_stm_ratio": ltm_size / stm_size,
        }
    
    return results
```

### Expected Metrics
- Storage Size (bytes) per tier
- Compression Ratio between tiers
- Storage Efficiency (memories per MB) per tier 