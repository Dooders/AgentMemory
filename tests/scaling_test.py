"""
Scaling test for AgentMemory system.
Tests performance characteristics under increasing load.
"""

import argparse
import csv
import time
from typing import Dict, List, Tuple

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

def run_scaling_test(num_agents: int, memories_per_agent: int) -> Dict:
    """Run scaling test with specified parameters."""
    memory_system = create_test_memory_system()
    metrics = {
        "stm_latency": [],
        "im_latency": [],
        "ltm_latency": [],
        "total_memories": 0
    }
    
    print(f"Starting test with {num_agents} agents, {memories_per_agent} memories each")
    
    # Store memories for each agent
    for agent_idx in range(num_agents):
        agent_id = f"test_agent_{agent_idx}"
        
        for step in range(memories_per_agent):
            memory = generate_test_memory(agent_id, step)
            
            # Store with varying priorities to trigger different tiers
            priority = 0.1 + (step % 10) / 10.0
            
            # Measure storage time
            _, latency = measure_operation_time(
                memory_system.store_agent_state,
                agent_id,
                memory,
                step,
                priority
            )
            
            # Track latency based on priority (which determines tier)
            if priority > 0.8:  # High priority -> STM
                metrics["stm_latency"].append(latency)
            elif priority > 0.4:  # Medium priority -> IM
                metrics["im_latency"].append(latency)
            else:  # Low priority -> LTM
                metrics["ltm_latency"].append(latency)
            
            metrics["total_memories"] += 1
            
            # Force maintenance periodically
            if step % 100 == 0:
                memory_system.force_memory_maintenance(agent_id)
        
        if agent_idx % 10 == 0:
            print(f"Completed agent {agent_idx}/{num_agents}")
    
    # Calculate average latencies
    return {
        "stm_avg_latency": sum(metrics["stm_latency"]) / len(metrics["stm_latency"]) if metrics["stm_latency"] else 0,
        "im_avg_latency": sum(metrics["im_latency"]) / len(metrics["im_latency"]) if metrics["im_latency"] else 0,
        "ltm_avg_latency": sum(metrics["ltm_latency"]) / len(metrics["ltm_latency"]) if metrics["ltm_latency"] else 0,
        "total_memories": metrics["total_memories"]
    }

def main():
    parser = argparse.ArgumentParser(description="Run scaling tests for AgentMemory")
    parser.add_argument("--agents", type=int, required=True, help="Number of agents")
    parser.add_argument("--memories", type=int, required=True, help="Memories per agent")
    parser.add_argument("--output", type=str, default="scaling_results.csv", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        results = run_scaling_test(args.agents, args.memories)
        
        # Write results to CSV
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                args.agents,
                args.memories,
                results["stm_avg_latency"],
                results["im_avg_latency"],
                results["ltm_avg_latency"],
                results["total_memories"]
            ])
        
        print(f"Test completed. Results written to {args.output}")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 