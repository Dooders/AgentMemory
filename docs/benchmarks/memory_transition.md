# Memory Transition Benchmarks

These benchmarks evaluate how effectively the Agent Memory System transfers memories between tiers and maintains important information during transitions.

## Transition Accuracy

### Benchmark Description
Measure how accurately the system preserves information when transferring memories between tiers.

### Methodology
1. Create test memories with varying importance scores
2. Allow natural transition between STM → IM → LTM tiers
3. Compare original memory content with transitioned memories
4. Measure information loss and preservation rates

### Code Example
```python
from memory import AgentMemorySystem, MemoryConfig
import time
import numpy as np
from sklearn.metrics import mean_squared_error

def benchmark_transition_accuracy(num_memories=1000, importance_levels=5):
    memory_system = AgentMemorySystem.get_instance(MemoryConfig(
        # Configure short transition times for testing
        stm_config={"ttl": 60},  # 1 minute
        im_config={"ttl": 120}   # 2 minutes
    ))
    
    # Create memories with varying importance
    test_memories = []
    memory_ids = []
    for i in range(num_memories):
        # Assign varying importance (1-5)
        importance = (i % importance_levels) + 1
        
        memory = generate_test_memory(complexity=importance)
        memory_id = f"test_memory_{i}"
        memory_ids.append(memory_id)
        test_memories.append(memory)
        
        # Store in STM with importance score
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=memory_id,
            state_data=memory,
            importance=importance
        )
    
    # Wait for transitions to STM → IM (wait 90 seconds)
    print("Waiting for STM → IM transition...")
    time.sleep(90)
    
    # Check IM memories and compare to originals
    im_results = {
        "total_transitioned": 0,
        "correctly_transitioned": 0,
        "mean_similarity": [],
        "importance_retention": {}
    }
    
    for i, memory_id in enumerate(memory_ids):
        original = test_memories[i]
        importance = (i % importance_levels) + 1
        
        # Check if memory exists in IM
        im_memory = memory_system.retrieve_state(memory_id, tier="im")
        if im_memory:
            im_results["total_transitioned"] += 1
            
            # Compare with original
            similarity = calculate_semantic_similarity(original, im_memory)
            im_results["mean_similarity"].append(similarity)
            
            # Check if information correctly preserved based on threshold
            if similarity > 0.8:  # 80% similarity threshold
                im_results["correctly_transitioned"] += 1
            
            # Track by importance level
            if importance not in im_results["importance_retention"]:
                im_results["importance_retention"][importance] = {
                    "total": 0, "preserved": 0
                }
            im_results["importance_retention"][importance]["total"] += 1
            if similarity > 0.8:
                im_results["importance_retention"][importance]["preserved"] += 1
    
    # Wait for transitions to IM → LTM (wait 60 more seconds)
    print("Waiting for IM → LTM transition...")
    time.sleep(60)
    
    # Check LTM memories (similar to IM check)
    ltm_results = {
        # Similar structure to im_results
    }
    
    # Calculate final metrics
    if im_results["total_transitioned"] > 0:
        im_results["preservation_rate"] = im_results["correctly_transitioned"] / im_results["total_transitioned"]
        im_results["avg_similarity"] = sum(im_results["mean_similarity"]) / len(im_results["mean_similarity"])
        
        # Calculate preservation by importance
        for imp, data in im_results["importance_retention"].items():
            if data["total"] > 0:
                data["preservation_rate"] = data["preserved"] / data["total"]
    
    # Similar calculations for ltm_results
    
    return {
        "im_transition": im_results,
        "ltm_transition": ltm_results
    }
```

### Expected Metrics
- Preservation rate by memory tier
- Information loss during transition
- Correlation between importance score and preservation
- Average semantic similarity between original and transitioned memories

## Transition Overhead

### Benchmark Description
Evaluate the computational and time costs associated with memory tier transitions.

### Methodology
1. Measure system resource usage during memory transitions
2. Time the duration of batch transitions between tiers
3. Evaluate impact on system performance during transitions

### Code Example
```python
import time
import psutil
import threading

def measure_transition_overhead(batch_sizes=[100, 1000, 10000]):
    results = {}
    
    for batch_size in batch_sizes:
        memory_system = AgentMemorySystem.get_instance(MemoryConfig(
            # Configure immediate transitions for testing
            stm_config={"ttl": 5},  # 5 seconds 
            im_config={"ttl": 10}   # 10 seconds
        ))
        
        # Insert test memories
        for i in range(batch_size):
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_id=f"test_memory_{i}",
                state_data=generate_test_memory()
            )
        
        # Set up monitoring thread to record resource usage
        usage_records = []
        stop_monitoring = threading.Event()
        
        def monitor_resources():
            while not stop_monitoring.is_set():
                usage_records.append({
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_used": psutil.virtual_memory().used,
                    "time": time.time()
                })
                time.sleep(0.1)  # Record every 100ms
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Start time measurement
        start_time = time.time()
        
        # Wait for transitions to complete
        print(f"Waiting for transitions to complete for batch size {batch_size}...")
        time.sleep(15)  # Wait for both transitions to complete
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Calculate metrics
        max_cpu = max(record["cpu_percent"] for record in usage_records)
        max_memory = max(record["memory_used"] for record in usage_records)
        avg_cpu = sum(record["cpu_percent"] for record in usage_records) / len(usage_records)
        
        results[batch_size] = {
            "transition_duration_seconds": time.time() - start_time,
            "max_cpu_percent": max_cpu,
            "avg_cpu_percent": avg_cpu,
            "max_memory_bytes": max_memory,
            "resource_usage_timeline": usage_records
        }
    
    return results
```

### Expected Metrics
- Transition duration (seconds)
- CPU usage during transitions (%)
- Memory usage during transitions (bytes)
- Resource usage timeline for visualization

## Importance Scoring

### Benchmark Description
Evaluate how effectively the system prioritizes important memories during transitions.

### Methodology
1. Create memories with predefined importance scores
2. Force memory overflows to trigger selective retention
3. Measure retention rates based on importance scores
4. Analyze how effectively high-importance memories are preserved

### Code Example
```python
def benchmark_importance_scoring():
    # Configure memory system with small capacity limits
    memory_system = AgentMemorySystem.get_instance(MemoryConfig(
        stm_config={"memory_limit": 100},  # Only 100 items
        im_config={"memory_limit": 50},    # Only 50 items
        ltm_config={"memory_limit": 20}    # Only 20 items
    ))
    
    # Generate 200 memories with importance scores 1-5
    # (exceeding STM capacity to force selective retention)
    memories = []
    for i in range(200):
        # Distribute importance (more low importance, fewer high importance)
        if i < 100:    # 50% are importance 1
            importance = 1
        elif i < 150:  # 25% are importance 2
            importance = 2
        elif i < 180:  # 15% are importance 3
            importance = 3
        elif i < 195:  # 7.5% are importance 4
            importance = 4
        else:          # 2.5% are importance 5
            importance = 5
        
        memory = generate_test_memory()
        memory_id = f"test_memory_{i}"
        
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=memory_id,
            state_data=memory,
            importance=importance
        )
        
        memories.append((memory_id, importance))
    
    # Force immediate memory consolidation
    memory_system.force_consolidation()
    
    # Check which memories were retained
    results = {
        "stm": {1: {"retained": 0, "total": 0},
                2: {"retained": 0, "total": 0},
                3: {"retained": 0, "total": 0},
                4: {"retained": 0, "total": 0},
                5: {"retained": 0, "total": 0}},
        "im": {1: {"retained": 0, "total": 0},
               2: {"retained": 0, "total": 0},
               3: {"retained": 0, "total": 0},
               4: {"retained": 0, "total": 0},
               5: {"retained": 0, "total": 0}},
        "ltm": {1: {"retained": 0, "total": 0},
                2: {"retained": 0, "total": 0},
                3: {"retained": 0, "total": 0},
                4: {"retained": 0, "total": 0},
                5: {"retained": 0, "total": 0}}
    }
    
    # Check retention for each memory and importance level
    for memory_id, importance in memories:
        # Track totals by importance
        results["stm"][importance]["total"] += 1
        results["im"][importance]["total"] += 1
        results["ltm"][importance]["total"] += 1
        
        # Check if retained in STM
        if memory_system.memory_exists(memory_id, tier="stm"):
            results["stm"][importance]["retained"] += 1
        
        # Check if retained in IM
        if memory_system.memory_exists(memory_id, tier="im"):
            results["im"][importance]["retained"] += 1
        
        # Check if retained in LTM
        if memory_system.memory_exists(memory_id, tier="ltm"):
            results["ltm"][importance]["retained"] += 1
    
    # Calculate retention rates by importance
    for tier in ["stm", "im", "ltm"]:
        for importance in range(1, 6):
            if results[tier][importance]["total"] > 0:
                results[tier][importance]["retention_rate"] = (
                    results[tier][importance]["retained"] / 
                    results[tier][importance]["total"]
                )
    
    return results
```

### Expected Metrics
- Retention rate by importance level
- Correlation between importance and retention probability
- Tier-specific importance thresholds
- Overall prioritization effectiveness 