"""
Demo 5: Performance Optimization and Error Handling

This demo showcases the performance-oriented features and error handling capabilities including:

1. Performance optimization techniques:
   - Batch operations for efficient memory storage
   - Redis-based caching for high-throughput operations
   - Optimized memory maintenance procedures
   - Performance benchmarking for different memory operations

2. Error handling mechanisms:
   - Graceful recovery from invalid inputs
   - Exception handling for boundary conditions
   - Fallback strategies for failed operations
   - Robust handling of extreme values and edge cases

3. Performance monitoring capabilities:
   - Time measurement for critical operations
   - Storage efficiency metrics
   - Retrieval latency assessment
   - Memory maintenance overhead analysis

The demo runs through several distinct phases:
- Phase 1: Performance benchmarking with batch operations
- Phase 2: Memory maintenance and optimization
- Phase 3: Error case testing and recovery verification

This demonstrates how a memory system can be optimized for performance while
maintaining robustness against errors, enabling stable operation in
production environments with high throughput requirements.
"""

import random
import time
from typing import Any, Dict, List

# Import common utilities for demos
from demo_utils import create_memory_system, generate_random_state, print_memory_details


def measure_time(func, *args, **kwargs) -> tuple:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def batch_store_states(
    memory_system, agent_id: str, states: List[Dict[str, Any]]
) -> tuple:
    """Store multiple states and measure time."""
    start_time = time.time()

    success_count = 0
    for state in states:
        success = memory_system.store_agent_state(
            agent_id, state, state.get("step", 0), random.uniform(0.4, 0.9)
        )
        if success:
            success_count += 1

    end_time = time.time()
    return success_count, end_time - start_time


def run_performance_test(agent_id: str, memory_system) -> Dict[str, float]:
    """Run a performance test with many memory operations."""
    print("\n--- Running Performance Test ---")

    # Generate a batch of random states
    num_states = 100
    states = [
        generate_random_state(agent_id, step) for step in range(1, num_states + 1)
    ]

    # Measure batch storage performance
    print(f"Storing {num_states} states in batch...")
    success_count, storage_time = batch_store_states(memory_system, agent_id, states)

    print(f"Successfully stored {success_count}/{num_states} states")
    print(f"Total storage time: {storage_time:.3f} seconds")
    print(f"Average time per state: {storage_time/num_states*1000:.3f} ms")

    # Force memory maintenance and measure time
    print("\nRunning memory maintenance...")
    _, maintenance_time = measure_time(memory_system.force_memory_maintenance, agent_id)
    print(f"Memory maintenance completed in {maintenance_time:.3f} seconds")

    # Measure retrieval performance
    print("\nMeasuring retrieval performance...")

    # Attribute-based retrieval
    query = {"status": "exploring"}
    _, attr_time = measure_time(memory_system.retrieve_by_attributes, agent_id, query)
    print(f"Attribute-based retrieval: {attr_time*1000:.3f} ms")

    # Time-range retrieval
    _, time_range_time = measure_time(
        memory_system.retrieve_by_time_range, agent_id, 10, 20
    )
    print(f"Time-range retrieval: {time_range_time*1000:.3f} ms")

    # Similarity-based retrieval (more complex)
    sample_state = states[len(states) // 2]  # Use a middle state as query
    _, similarity_time = measure_time(
        memory_system.retrieve_similar_states, agent_id, sample_state, 5
    )
    print(f"Similarity-based retrieval: {similarity_time*1000:.3f} ms")

    # Return performance results
    return {
        "storage_time_ms": storage_time / num_states * 1000,
        "maintenance_time_ms": maintenance_time * 1000,
        "attr_retrieval_ms": attr_time * 1000,
        "time_range_retrieval_ms": time_range_time * 1000,
        "similarity_retrieval_ms": similarity_time * 1000,
    }


def test_error_handling(memory_system) -> Dict[str, bool]:
    """Test error handling and recovery mechanisms."""
    print("\n--- Testing Error Handling and Recovery ---")

    agent_id = "error_test_agent"

    # Test 1: Try to store invalid data
    print("\nTest 1: Storing invalid data")
    try:
        # Intentionally pass invalid data (non-dict)
        result = memory_system.store_agent_state(agent_id, "not_a_dict", 1, 0.5)
        print(f"  Result: {'Success' if result else 'Failed'}")
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {e}")

    # Test 2: Try to retrieve with invalid parameters
    print("\nTest 2: Retrieving with invalid parameters")
    try:
        # Intentionally pass invalid query, but wrap it in try-except
        try:
            result = memory_system.retrieve_by_attributes(agent_id, "invalid_query")
            print(f"  Result: {len(result)} memories returned")
        except TypeError as type_err:
            print(f"  Caught TypeError: {type_err}")
            # Create a valid but empty query to continue
            result = memory_system.retrieve_by_attributes(agent_id, {})
            print(f"  Recovered with empty query: {len(result)} memories returned")
    except Exception as e:
        print(f"  Unexpected exception: {type(e).__name__}: {e}")

    # Test 3: Try to use an extremely large value
    print("\nTest 3: Using extreme values")
    try:
        large_state = {
            "position": {
                "x": float("inf"),
                "y": float("inf"),
            }
        }
        result = memory_system.store_agent_state(agent_id, large_state, 9999, 0.5)
        print(f"  Result: {'Success' if result else 'Failed'}")
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {e}")

    # Return results of error handling tests
    return {
        "invalid_data_handled": True,  # Modify based on results
        "invalid_params_handled": True,
        "extreme_values_handled": True,
    }


def run_demo():
    """Run the performance and error handling demo."""
    # Initialize with performance-oriented config
    memory_system = create_memory_system(
        stm_limit=1000,  # Larger STM capacity
        stm_ttl=86400,  # 24 hour TTL
        im_limit=10000,  # Larger IM capacity
        im_compression_level=1,
        ltm_batch_size=100,  # Larger batch size for better performance
        logging_level="WARNING",  # Reduce logging for performance
        cleanup_interval=500,  # Less frequent cleanup for better performance
        description="performance and error handling demo",
    )

    # First run a performance test
    agent_id = "performance_agent"
    performance_results = run_performance_test(agent_id, memory_system)

    # Then test error handling and recovery
    error_handling_results = test_error_handling(memory_system)

    # Print system statistics
    print_memory_details(memory_system, agent_id, "Memory System Statistics")

    # Print performance summary
    print("\nPerformance Test Summary:")
    for metric, value in performance_results.items():
        print(f"  {metric}: {value:.3f} ms")

    print("\nPerformance and error handling demo completed!")


if __name__ == "__main__":
    run_demo()
