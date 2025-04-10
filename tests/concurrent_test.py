"""
Concurrent access test for AgentMemory system.
Tests performance characteristics under concurrent load.
"""

import argparse
import csv
import time
import threading
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from memory import AgentMemorySystem, MemoryConfig
from memory.config import RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig

def create_test_memory_system() -> AgentMemorySystem:
    """Initialize memory system with test configuration."""
    config = MemoryConfig(
        stm_config=RedisSTMConfig(
            host="localhost",
            port=6379,
            ttl=3600,
            memory_limit=100000
        ),
        im_config=RedisIMConfig(
            ttl=7200,
            memory_limit=200000
        ),
        ltm_config=SQLiteLTMConfig(
            db_path="test_ltm.db"
        ),
        cleanup_interval=100,
        enable_compression=True
    )
    return AgentMemorySystem.get_instance(config)

def generate_test_memory(agent_id: str, step: int) -> Dict:
    """Generate a test memory with realistic-looking data."""
    return {
        "position": {
            "x": step * 1.5,
            "y": step * 0.75,
            "location": f"sector_{step % 10}"
        },
        "stats": {
            "health": 100 - (step % 20),
            "energy": 80 - (step % 15)
        },
        "inventory": {
            "items": [
                {"name": "item_1", "count": step % 5},
                {"name": "item_2", "count": step % 3}
            ]
        },
        "status": "active",
        "step": step
    }

def measure_operation_time(operation, *args) -> Tuple[any, float]:
    """Measure time taken for an operation."""
    start_time = time.time()
    result = operation(*args)
    end_time = time.time()
    return result, (end_time - start_time) * 1000  # Convert to milliseconds

def worker_task(worker_id: int, memory_system: AgentMemorySystem, 
                agent_id: str, operations: int, write_ratio: float,
                results: Dict) -> None:
    """Worker task for concurrent testing."""
    local_latencies = {"write": [], "read": [], "mixed": []}
    success_count = 0
    error_count = 0
    
    for op in range(operations):
        try:
            # Determine if this is a write or read operation
            is_write = op % 100 < write_ratio * 100  # Convert ratio to percentage
            
            if is_write:
                # Store operation
                memory = generate_test_memory(agent_id, op)
                priority = 0.1 + (op % 10) / 10.0
                
                _, latency = measure_operation_time(
                    memory_system.store_agent_state,
                    agent_id, memory, op, priority
                )
                local_latencies["write"].append(latency)
            else:
                # Retrieve operation - randomly choose between different retrieval methods
                retrieval_type = op % 3
                
                if retrieval_type == 0:
                    # Retrieve by time range
                    _, latency = measure_operation_time(
                        memory_system.retrieve_by_time_range,
                        agent_id, max(0, op - 100), op
                    )
                elif retrieval_type == 1:
                    # Retrieve most recent
                    _, latency = measure_operation_time(
                        memory_system.retrieve_most_recent,
                        agent_id, min(10, op)
                    )
                else:
                    # Retrieve by attributes
                    _, latency = measure_operation_time(
                        memory_system.retrieve_by_attributes,
                        agent_id, {"status": "active"}, 10
                    )
                
                local_latencies["read"].append(latency)
            
            local_latencies["mixed"].append(latency)
            success_count += 1
            
            # Occasionally force memory maintenance
            if op % 100 == 0:
                memory_system.force_memory_maintenance(agent_id)
            
        except Exception as e:
            error_count += 1
            # Uncomment for debugging
            # print(f"Worker {worker_id} error: {str(e)}")
    
    # Update global results with thread-local data
    with results["lock"]:
        results["write_latencies"].extend(local_latencies["write"])
        results["read_latencies"].extend(local_latencies["read"])
        results["mixed_latencies"].extend(local_latencies["mixed"])
        results["success_count"] += success_count
        results["error_count"] += error_count

def calculate_percentile(latencies: List[float], percentile: float) -> float:
    """Calculate a percentile value from a list of latencies."""
    if not latencies:
        return 0
    sorted_latencies = sorted(latencies)
    index = int(len(sorted_latencies) * percentile / 100)
    return sorted_latencies[index]

def run_concurrent_test(num_threads: int, operations_per_thread: int, 
                        write_ratio: float = 0.5) -> Dict:
    """
    Run concurrent access test with specified parameters.
    
    Args:
        num_threads: Number of concurrent threads
        operations_per_thread: Number of operations per thread
        write_ratio: Ratio of write operations (0.0-1.0)
        
    Returns:
        Dictionary with test results
    """
    memory_system = create_test_memory_system()
    
    # Shared results dictionary with thread synchronization
    results = {
        "write_latencies": [],
        "read_latencies": [],
        "mixed_latencies": [],
        "success_count": 0,
        "error_count": 0,
        "lock": threading.Lock()
    }
    
    print(f"Starting concurrent test with {num_threads} threads, {operations_per_thread} operations each")
    print(f"Write ratio: {write_ratio:.1f} ({int(write_ratio * 100)}% writes, {100 - int(write_ratio * 100)}% reads)")
    
    # Create and start worker threads
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            agent_id = f"concurrent_agent_{thread_id}"
            future = executor.submit(
                worker_task,
                thread_id, memory_system, agent_id, 
                operations_per_thread, write_ratio, results
            )
            futures.append(future)
    
    # Wait for all threads to complete
    for future in futures:
        future.result()
    
    end_time = time.time()
    total_time = end_time - start_time
    total_operations = results["success_count"] + results["error_count"]
    
    # Calculate metrics
    throughput = results["success_count"] / total_time if total_time > 0 else 0
    success_rate = results["success_count"] / total_operations if total_operations > 0 else 0
    
    metrics = {
        # Overall latency
        "avg_latency": sum(results["mixed_latencies"]) / len(results["mixed_latencies"]) if results["mixed_latencies"] else 0,
        "p50_latency": calculate_percentile(results["mixed_latencies"], 50),
        "p95_latency": calculate_percentile(results["mixed_latencies"], 95),
        "p99_latency": calculate_percentile(results["mixed_latencies"], 99),
        
        # Write operation latency
        "write_avg_latency": sum(results["write_latencies"]) / len(results["write_latencies"]) if results["write_latencies"] else 0,
        "write_p95_latency": calculate_percentile(results["write_latencies"], 95),
        
        # Read operation latency
        "read_avg_latency": sum(results["read_latencies"]) / len(results["read_latencies"]) if results["read_latencies"] else 0,
        "read_p95_latency": calculate_percentile(results["read_latencies"], 95),
        
        # Overall stats
        "throughput": throughput,  # Operations per second
        "success_rate": success_rate,
        "total_time": total_time,
        "total_operations": total_operations,
        "success_count": results["success_count"],
        "error_count": results["error_count"]
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run concurrent access tests for AgentMemory")
    parser.add_argument("--threads", type=int, required=True, help="Number of concurrent threads")
    parser.add_argument("--operations", type=int, required=True, help="Operations per thread")
    parser.add_argument("--write-ratio", type=float, default=0.5, help="Ratio of write operations (0.0-1.0)")
    parser.add_argument("--output", type=str, default="concurrent_results.csv", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        results = run_concurrent_test(args.threads, args.operations, args.write_ratio)
        
        # Print summary results
        print("\nTest Results:")
        print(f"Total operations: {results['total_operations']} ({results['success_count']} successful, {results['error_count']} errors)")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Throughput: {results['throughput']:.2f} ops/sec")
        print(f"Success rate: {results['success_rate'] * 100:.2f}%")
        print(f"Average latency: {results['avg_latency']:.2f} ms")
        print(f"P95 latency: {results['p95_latency']:.2f} ms")
        print(f"P99 latency: {results['p99_latency']:.2f} ms")
        print(f"Write avg latency: {results['write_avg_latency']:.2f} ms")
        print(f"Read avg latency: {results['read_avg_latency']:.2f} ms")
        
        # Write results to CSV
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Check if file is empty, add header if needed
            f.seek(0, 2)  # Go to end of file
            if f.tell() == 0:  # File is empty
                writer.writerow([
                    "Threads", "OpsPerThread", "WriteRatio", 
                    "TotalOps", "SuccessRate", "Throughput", 
                    "AvgLatency", "P95Latency", "P99Latency",
                    "WriteAvgLatency", "ReadAvgLatency"
                ])
            
            writer.writerow([
                args.threads,
                args.operations,
                args.write_ratio,
                results["total_operations"],
                results["success_rate"],
                results["throughput"],
                results["avg_latency"],
                results["p95_latency"],
                results["p99_latency"],
                results["write_avg_latency"],
                results["read_avg_latency"]
            ])
        
        print(f"Test completed. Results written to {args.output}")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 