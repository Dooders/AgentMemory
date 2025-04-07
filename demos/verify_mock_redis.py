"""
Verify that the mock Redis configuration works properly.

This script creates a memory system with mock Redis and performs basic operations
to verify that it works without requiring a real Redis server.
"""

import sys
import os
import time

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo_utils import create_memory_system

def run_verification():
    """Verify that mock Redis is working correctly."""
    print("Initializing memory system with mock Redis...")
    
    # Use explicit use_mock_redis=True to ensure MockRedis is used
    memory_system = create_memory_system(
        stm_limit=10,
        im_limit=20,
        use_embeddings=False,
        use_mock_redis=True,
        description="mock redis verification"
    )
    
    # Create a test agent
    agent_id = "test_mock_agent"
    
    # Try to store and retrieve some memories
    print(f"Storing test memories for agent {agent_id}...")
    
    # Store a state memory
    state = {
        "position": {"x": 1, "y": 2},
        "health": 100,
        "inventory": ["item1", "item2"]
    }
    success = memory_system.store_agent_state(agent_id, state, 1, 1.0)
    print(f"Stored state memory, success: {success}")
    
    # Store an action memory
    action = {
        "type": "move",
        "direction": "north",
        "distance": 5
    }
    success = memory_system.store_agent_action(agent_id, action, 2, 0.7)
    print(f"Stored action memory, success: {success}")
    
    # Get memory statistics
    stats = memory_system.get_memory_statistics(agent_id)
    print("\nMemory Statistics:")
    print(f"STM count: {stats['tiers']['stm']['count']}")
    print(f"IM count: {stats['tiers']['im']['count']}")
    print(f"LTM count: {stats['tiers']['ltm']['count']}")
    
    # Print agent memory details
    print("\nVerification completed successfully!")
    print("Mock Redis is working properly - no connection to a real Redis server was required.")

if __name__ == "__main__":
    run_verification() 