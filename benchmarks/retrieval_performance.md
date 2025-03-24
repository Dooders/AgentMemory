# Retrieval Performance Benchmarks

These benchmarks evaluate the memory retrieval capabilities of the Agent Memory System, focusing on the accuracy, speed, and efficiency of finding relevant memories.

## Vector Search Latency

### Benchmark Description
Measure the time required to perform similarity-based retrieval across different memory tiers.

### Methodology
1. Populate each memory tier with varying volumes of test data
2. Perform vector searches with random query states
3. Measure search latency across different data volumes and tiers
4. Analyze performance scalability with increasing data size

### Code Example
```python
from agent_memory import AgentMemorySystem, MemoryConfig
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def benchmark_vector_search_latency(data_volumes=[100, 1000, 10000, 100000]):
    results = {}
    
    for volume in data_volumes:
        # Create a fresh memory system for each test
        memory_system = AgentMemorySystem.get_instance(MemoryConfig())
        
        # Populate with test data
        print(f"Populating memory system with {volume} items...")
        test_states = []
        for i in range(volume):
            state = generate_test_state()
            test_states.append(state)
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_id=f"test_state_{i}",
                state_data=state
            )
        
        # Generate query states (randomly select from test data)
        query_indices = random.sample(range(volume), min(100, volume))
        query_states = [test_states[i] for i in query_indices]
        
        # Measure STM search latency
        stm_latencies = []
        for query_state in query_states:
            start_time = time.time()
            _ = memory_system.retrieve_similar_states(
                agent_id="benchmark_agent",
                query_state=query_state,
                k=10,
                tier="stm"
            )
            search_time = time.time() - start_time
            stm_latencies.append(search_time * 1000)  # Convert to ms
        
        # Transition some data to IM and LTM for those tests
        # (In a real benchmark, you'd properly populate all tiers)
        memory_system.force_transition_to_im(percentage=30)
        memory_system.force_transition_to_ltm(percentage=10)
        
        # Measure IM search latency
        im_latencies = []
        for query_state in query_states:
            start_time = time.time()
            _ = memory_system.retrieve_similar_states(
                agent_id="benchmark_agent",
                query_state=query_state,
                k=10,
                tier="im"
            )
            search_time = time.time() - start_time
            im_latencies.append(search_time * 1000)  # Convert to ms
        
        # Measure LTM search latency
        ltm_latencies = []
        for query_state in query_states:
            start_time = time.time()
            _ = memory_system.retrieve_similar_states(
                agent_id="benchmark_agent",
                query_state=query_state,
                k=10,
                tier="ltm"
            )
            search_time = time.time() - start_time
            ltm_latencies.append(search_time * 1000)  # Convert to ms
        
        # Measure cross-tier search latency
        cross_tier_latencies = []
        for query_state in query_states:
            start_time = time.time()
            _ = memory_system.retrieve_similar_states(
                agent_id="benchmark_agent",
                query_state=query_state,
                k=10,
                tier="all"  # Search across all tiers
            )
            search_time = time.time() - start_time
            cross_tier_latencies.append(search_time * 1000)  # Convert to ms
        
        # Calculate metrics
        results[volume] = {
            "stm": {
                "avg_latency_ms": np.mean(stm_latencies),
                "min_latency_ms": np.min(stm_latencies),
                "max_latency_ms": np.max(stm_latencies),
                "p95_latency_ms": np.percentile(stm_latencies, 95)
            },
            "im": {
                "avg_latency_ms": np.mean(im_latencies),
                "min_latency_ms": np.min(im_latencies),
                "max_latency_ms": np.max(im_latencies),
                "p95_latency_ms": np.percentile(im_latencies, 95)
            },
            "ltm": {
                "avg_latency_ms": np.mean(ltm_latencies),
                "min_latency_ms": np.min(ltm_latencies),
                "max_latency_ms": np.max(ltm_latencies),
                "p95_latency_ms": np.percentile(ltm_latencies, 95)
            },
            "cross_tier": {
                "avg_latency_ms": np.mean(cross_tier_latencies),
                "min_latency_ms": np.min(cross_tier_latencies),
                "max_latency_ms": np.max(cross_tier_latencies),
                "p95_latency_ms": np.percentile(cross_tier_latencies, 95)
            }
        }
    
    return results
```

### Expected Metrics
- Average search latency (ms) by tier
- P95 latency (ms) by tier
- Latency scaling with data volume
- Cross-tier vs. single-tier latency comparison

## Search Accuracy

### Benchmark Description
Evaluate how accurately the system retrieves semantically similar memories.

### Methodology
1. Create test datasets with known semantic relationships
2. Perform vector similarity searches with specific query states
3. Compare retrieved results against ground truth similar items
4. Calculate precision, recall, and F1 scores for retrieval

### Code Example
```python
from sklearn.metrics import precision_recall_fscore_support

def benchmark_search_accuracy():
    # Create a memory system
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    # Generate test data with known clusters
    # Each cluster represents semantically similar items
    test_clusters = generate_clustered_test_data(
        num_clusters=10,
        items_per_cluster=20,
        noise_level=0.2
    )
    
    # Store all test data
    all_items = []
    cluster_map = {}  # Maps item_id to cluster_id
    
    for cluster_id, cluster_items in enumerate(test_clusters):
        for i, item in enumerate(cluster_items):
            item_id = f"cluster_{cluster_id}_item_{i}"
            memory_system.store_agent_state(
                agent_id="benchmark_agent",
                state_id=item_id,
                state_data=item
            )
            all_items.append((item_id, item))
            cluster_map[item_id] = cluster_id
    
    # Perform retrieval tests
    results = []
    
    # Test with one query from each cluster
    for cluster_id in range(len(test_clusters)):
        # Use first item in cluster as query
        query_item = test_clusters[cluster_id][0]
        query_id = f"cluster_{cluster_id}_item_0"
        
        # Get ground truth - all items in same cluster (except query itself)
        ground_truth = [
            f"cluster_{cluster_id}_item_{i}" 
            for i in range(1, len(test_clusters[cluster_id]))
        ]
        
        # Perform similarity search
        k = len(test_clusters[cluster_id]) * 2  # Get more than in cluster to test precision
        similar_items = memory_system.retrieve_similar_states(
            agent_id="benchmark_agent",
            query_state=query_item,
            k=k,
            tier="stm"
        )
        
        # Extract IDs from results
        retrieved_ids = [item["state_id"] for item in similar_items]
        
        # Calculate metrics (excluding the query item itself)
        retrieved_set = set(retrieved_ids) - {query_id}
        ground_truth_set = set(ground_truth)
        
        true_positives = len(retrieved_set & ground_truth_set)
        false_positives = len(retrieved_set - ground_truth_set)
        false_negatives = len(ground_truth_set - retrieved_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "cluster_id": cluster_id,
            "query_id": query_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved_count": len(retrieved_set),
            "ground_truth_count": len(ground_truth_set),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        })
    
    # Calculate aggregate metrics
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1_score"] for r in results) / len(results)
    
    return {
        "per_cluster_results": results,
        "aggregate": {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1
        }
    }
```

### Expected Metrics
- Precision: Fraction of retrieved items that are relevant
- Recall: Fraction of relevant items that are retrieved
- F1 Score: Harmonic mean of precision and recall
- Per-cluster and aggregate metrics

## Cross-Tier Retrieval

### Benchmark Description
Evaluate the performance of memory retrieval that spans multiple memory tiers.

### Methodology
1. Populate all memory tiers with interrelated test data
2. Perform cross-tier searches with varying complexity
3. Measure latency and resource usage during cross-tier operations
4. Compare single-tier vs. multi-tier retrieval accuracy

### Code Example
```python
def benchmark_cross_tier_retrieval():
    # Create memory system and populate all tiers
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    # Generate test data with distinct but related items in each tier
    # Some memories in different tiers should be semantically related
    stm_states, im_states, ltm_states, related_groups = generate_cross_tier_test_data(
        stm_count=1000,
        im_count=500,
        ltm_count=200,
        related_groups=50  # Sets of related memories across tiers
    )
    
    # Store data in appropriate tiers
    for i, state in enumerate(stm_states):
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"stm_state_{i}",
            state_data=state,
            tier="stm"
        )
    
    for i, state in enumerate(im_states):
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"im_state_{i}",
            state_data=state,
            tier="im"
        )
    
    for i, state in enumerate(ltm_states):
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"ltm_state_{i}",
            state_data=state,
            tier="ltm"
        )
    
    # Test cross-tier retrieval
    results = []
    
    for group_id, related_items in enumerate(related_groups):
        # Each group contains related items across tiers
        stm_item, im_item, ltm_item = related_items
        
        # Query using the STM item
        start_time = time.time()
        cross_tier_results = memory_system.retrieve_similar_states(
            agent_id="benchmark_agent",
            query_state=stm_item,
            k=20,
            tier="all"  # Search across all tiers
        )
        retrieval_time = time.time() - start_time
        
        # Identify which related items were found
        retrieved_ids = [item["state_id"] for item in cross_tier_results]
        
        # Check if the related IM and LTM items were found
        im_item_id = f"im_state_{im_states.index(im_item)}"
        ltm_item_id = f"ltm_state_{ltm_states.index(ltm_item)}"
        
        im_found = im_item_id in retrieved_ids
        ltm_found = ltm_item_id in retrieved_ids
        
        # Count items found from each tier
        stm_count = sum(1 for id in retrieved_ids if id.startswith("stm_state_"))
        im_count = sum(1 for id in retrieved_ids if id.startswith("im_state_"))
        ltm_count = sum(1 for id in retrieved_ids if id.startswith("ltm_state_"))
        
        results.append({
            "group_id": group_id,
            "retrieval_time_ms": retrieval_time * 1000,
            "related_im_found": im_found,
            "related_ltm_found": ltm_found,
            "stm_items_found": stm_count,
            "im_items_found": im_count,
            "ltm_items_found": ltm_count
        })
    
    # Calculate aggregate metrics
    avg_retrieval_time = sum(r["retrieval_time_ms"] for r in results) / len(results)
    im_found_rate = sum(1 for r in results if r["related_im_found"]) / len(results)
    ltm_found_rate = sum(1 for r in results if r["related_ltm_found"]) / len(results)
    
    return {
        "per_query_results": results,
        "aggregate": {
            "avg_retrieval_time_ms": avg_retrieval_time,
            "im_related_found_rate": im_found_rate,
            "ltm_related_found_rate": ltm_found_rate,
            "avg_stm_items": sum(r["stm_items_found"] for r in results) / len(results),
            "avg_im_items": sum(r["im_items_found"] for r in results) / len(results),
            "avg_ltm_items": sum(r["ltm_items_found"] for r in results) / len(results)
        }
    }
```

### Expected Metrics
- Cross-tier retrieval latency (ms)
- Related item discovery rate across tiers
- Distribution of retrieved items by tier
- Tier-specific retrieval accuracy 