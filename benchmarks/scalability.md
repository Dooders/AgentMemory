# Scalability Benchmarks

These benchmarks evaluate how well the Agent Memory System scales with increasing load across different dimensions.

## Agent Count Scaling

### Benchmark Description
Measure how the system performance scales with an increasing number of concurrent agents.

### Methodology
1. Create memory systems with different numbers of active agents
2. Perform standardized operations with all agents concurrently
3. Measure throughput, latency, and resource usage for each agent count
4. Identify bottlenecks and scaling limitations

### Code Example
```python
from agent_memory import AgentMemorySystem, MemoryConfig
import time
import threading
import psutil
import matplotlib.pyplot as plt
import numpy as np

def benchmark_agent_scaling(agent_counts=[1, 10, 50, 100, 500]):
    results = {}
    
    for agent_count in agent_counts:
        # Create a fresh memory system
        memory_system = AgentMemorySystem.get_instance(MemoryConfig())
        
        # Create threads to simulate concurrent agent activity
        threads = []
        results_lock = threading.Lock()
        thread_results = []
        
        def agent_activity(agent_id):
            # Each agent performs standard operations:
            # 1. Store 10 states
            # 2. Retrieve 5 states
            # 3. Perform 3 similarity searches
            
            start_time = time.time()
            
            # Store states
            states = []
            for i in range(10):
                state = generate_test_state()
                states.append(state)
                memory_system.store_agent_state(
                    agent_id=f"agent_{agent_id}",
                    state_id=f"state_{agent_id}_{i}",
                    state_data=state
                )
            
            # Retrieve states
            for i in range(5):
                memory_system.retrieve_state(
                    state_id=f"state_{agent_id}_{i}",
                    agent_id=f"agent_{agent_id}"
                )
            
            # Similarity searches
            for i in range(3):
                memory_system.retrieve_similar_states(
                    agent_id=f"agent_{agent_id}",
                    query_state=states[i],
                    k=5
                )
            
            end_time = time.time()
            
            # Record thread results
            with results_lock:
                thread_results.append({
                    "agent_id": agent_id,
                    "duration_seconds": end_time - start_time
                })
        
        # Resource monitoring thread
        resource_samples = []
        stop_monitoring = threading.Event()
        
        def monitor_resources():
            while not stop_monitoring.is_set():
                resource_samples.append({
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_used": psutil.virtual_memory().used,
                    "disk_io": psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
                })
                time.sleep(0.2)  # Sample every 200ms
        
        # Start resource monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Start timing
        start_time = time.time()
        
        # Launch agent threads
        for i in range(agent_count):
            t = threading.Thread(target=agent_activity, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        end_time = time.time()
        
        # Stop resource monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Calculate metrics
        total_duration = end_time - start_time
        agent_durations = [r["duration_seconds"] for r in thread_results]
        avg_duration = sum(agent_durations) / len(agent_durations)
        max_duration = max(agent_durations)
        min_duration = min(agent_durations)
        
        throughput = agent_count / total_duration  # agents processed per second
        
        # Resource metrics
        avg_cpu = sum(s["cpu_percent"] for s in resource_samples) / len(resource_samples)
        max_cpu = max(s["cpu_percent"] for s in resource_samples)
        avg_memory = sum(s["memory_used"] for s in resource_samples) / len(resource_samples)
        max_memory = max(s["memory_used"] for s in resource_samples)
        
        results[agent_count] = {
            "total_duration_seconds": total_duration,
            "avg_agent_duration_seconds": avg_duration,
            "min_agent_duration_seconds": min_duration,
            "max_agent_duration_seconds": max_duration,
            "throughput_agents_per_second": throughput,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_memory_bytes": avg_memory,
            "max_memory_bytes": max_memory,
            "std_deviation_duration": np.std(agent_durations),
            "95th_percentile_duration": np.percentile(agent_durations, 95)
        }
    
    return results
```

### Expected Metrics
- Total execution time vs. agent count
- Average operation latency per agent
- System throughput (operations/second)
- Resource utilization (CPU, memory)
- Scaling efficiency (ideal vs. actual)

## Memory Size Scaling

### Benchmark Description
Evaluate how system performance changes as the volume of stored memories increases.

### Methodology
1. Incrementally populate the memory system with varying data volumes
2. Measure access times, retrieval performance, and resource usage
3. Test standard operations at each data volume checkpoint
4. Analyze performance degradation with scale

### Code Example
```python
def benchmark_memory_size_scaling(data_volumes=[1000, 10000, 100000, 1000000]):
    results = {}
    
    # Create a fresh memory system
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    # Test operations to perform at each volume checkpoint
    def run_test_operations():
        # Prepare test data (different from what's stored)
        test_states = [generate_test_state() for _ in range(10)]
        query_state = generate_test_state()
        
        # Test 1: Store new memory - measure latency
        store_latencies = []
        for i, state in enumerate(test_states):
            start_time = time.time()
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_id=f"test_state_{i}",
                state_data=state
            )
            store_latencies.append((time.time() - start_time) * 1000)  # ms
        
        # Test 2: Retrieve memory by ID - measure latency
        retrieve_latencies = []
        for i in range(10):
            start_time = time.time()
            memory_system.retrieve_state(
                agent_id="benchmark_agent",
                state_id=f"test_state_{i}"
            )
            retrieve_latencies.append((time.time() - start_time) * 1000)  # ms
        
        # Test 3: Vector similarity search - measure latency
        search_latencies = []
        for i in range(5):
            start_time = time.time()
            memory_system.retrieve_similar_states(
                agent_id="benchmark_agent",
                query_state=query_state,
                k=10
            )
            search_latencies.append((time.time() - start_time) * 1000)  # ms
        
        # Test 4: Cross-tier search
        cross_tier_start = time.time()
        memory_system.retrieve_similar_states(
            agent_id="benchmark_agent",
            query_state=query_state,
            k=10,
            tier="all"
        )
        cross_tier_latency = (time.time() - cross_tier_start) * 1000  # ms
        
        # Measure memory usage
        memory_usage = psutil.Process().memory_info().rss
        
        return {
            "store_latency_ms": {
                "avg": sum(store_latencies) / len(store_latencies),
                "min": min(store_latencies),
                "max": max(store_latencies)
            },
            "retrieve_latency_ms": {
                "avg": sum(retrieve_latencies) / len(retrieve_latencies),
                "min": min(retrieve_latencies),
                "max": max(retrieve_latencies)
            },
            "search_latency_ms": {
                "avg": sum(search_latencies) / len(search_latencies),
                "min": min(search_latencies),
                "max": max(search_latencies)
            },
            "cross_tier_latency_ms": cross_tier_latency,
            "memory_usage_bytes": memory_usage
        }
    
    # Initial empty system baseline
    results["baseline"] = run_test_operations()
    
    # Incrementally add data and measure at each volume
    current_count = 0
    
    for target_volume in data_volumes:
        # Add data to reach target volume
        items_to_add = target_volume - current_count
        print(f"Adding {items_to_add} items to reach {target_volume}...")
        
        for i in range(items_to_add):
            state = generate_test_state()
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_id=f"volume_state_{current_count + i}",
                state_data=state
            )
        
        current_count = target_volume
        
        # Force memory to spread across tiers
        if target_volume >= 10000:
            memory_system.force_transition_to_im(percentage=30)
        if target_volume >= 100000:
            memory_system.force_transition_to_ltm(percentage=20)
        
        # Run tests with this volume
        print(f"Running tests at volume {target_volume}...")
        results[target_volume] = run_test_operations()
    
    return results
```

### Expected Metrics
- Operation latency vs. data volume
- Memory usage vs. data volume
- Retrieval accuracy at different scales
- Search performance degradation curve

## Resource Utilization

### Benchmark Description
Measure how efficiently the Agent Memory System uses system resources under different loads.

### Methodology
1. Run memory operations under controlled load scenarios
2. Monitor CPU, memory, disk, and network utilization
3. Identify resource bottlenecks and usage patterns
4. Measure resource efficiency for different operations

### Code Example
```python
def benchmark_resource_utilization(duration_seconds=300, operation_rate=10):
    """
    Run a sustained load test while monitoring resource utilization.
    
    Args:
        duration_seconds: How long to run the benchmark
        operation_rate: Target operations per second
    """
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    # Prepare test data
    test_states = [generate_test_state() for _ in range(1000)]
    
    # Define operations
    operations = [
        # Store operation
        lambda i: memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"resource_state_{i % 1000}",
            state_data=test_states[i % 1000]
        ),
        # Retrieve operation
        lambda i: memory_system.retrieve_state(
            agent_id="benchmark_agent",
            state_id=f"resource_state_{i % 1000}"
        ),
        # Search operation
        lambda i: memory_system.retrieve_similar_states(
            agent_id="benchmark_agent",
            query_state=test_states[i % 1000],
            k=10
        )
    ]
    
    # Resource monitoring thread
    resource_samples = []
    stop_monitoring = threading.Event()
    
    def monitor_resources():
        while not stop_monitoring.is_set():
            try:
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_used": psutil.Process().memory_info().rss,
                    "system_memory": psutil.virtual_memory().used,
                    "disk_io": psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
                }
                resource_samples.append(sample)
            except Exception as e:
                print(f"Resource monitoring error: {e}")
            time.sleep(0.5)  # Sample every 500ms
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()
    
    # Run operations at the target rate
    start_time = time.time()
    operation_count = 0
    operation_times = []
    
    try:
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Determine if we should perform an operation to maintain the target rate
            target_ops = int(elapsed * operation_rate)
            
            while operation_count < target_ops:
                # Choose operation randomly (equal distribution)
                op = random.choice(operations)
                
                # Perform operation and measure time
                op_start = time.time()
                op(operation_count)
                op_duration = time.time() - op_start
                
                operation_times.append(op_duration)
                operation_count += 1
            
            # Sleep a small amount to avoid tight loop
            time.sleep(0.01)
    finally:
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
    
    # Calculate metrics
    end_time = time.time()
    actual_duration = end_time - start_time
    actual_rate = operation_count / actual_duration
    
    # Resource utilization metrics
    avg_cpu = sum(s["cpu_percent"] for s in resource_samples) / len(resource_samples)
    max_cpu = max(s["cpu_percent"] for s in resource_samples)
    avg_memory = sum(s["memory_used"] for s in resource_samples) / len(resource_samples)
    max_memory = max(s["memory_used"] for s in resource_samples)
    
    # Resource efficiency metrics
    ops_per_cpu_percent = actual_rate / avg_cpu if avg_cpu > 0 else 0
    ops_per_mb = (actual_rate * 1024 * 1024) / avg_memory if avg_memory > 0 else 0
    
    # Time series data for plotting
    time_series = {
        "timestamps": [s["timestamp"] - start_time for s in resource_samples],
        "cpu_percent": [s["cpu_percent"] for s in resource_samples],
        "memory_mb": [s["memory_used"] / (1024 * 1024) for s in resource_samples]
    }
    
    return {
        "duration_seconds": actual_duration,
        "operation_count": operation_count,
        "actual_rate_ops_per_second": actual_rate,
        "avg_operation_time_ms": sum(operation_times) / len(operation_times) * 1000,
        "p95_operation_time_ms": np.percentile(operation_times, 95) * 1000,
        "resource_utilization": {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_memory_bytes": avg_memory,
            "max_memory_bytes": max_memory
        },
        "resource_efficiency": {
            "operations_per_cpu_percent": ops_per_cpu_percent,
            "operations_per_mb": ops_per_mb
        },
        "time_series": time_series
    }
```

### Expected Metrics
- CPU utilization (%)
- Memory usage (bytes) 
- Operations per second per CPU %
- Operations per MB of memory
- Resource utilization timeline
- Memory leak detection 