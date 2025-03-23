"""
Demo 1: Basic Memory Operations

This demo shows how to initialize the AgentMemorySystem and perform basic 
memory storage operations across the three memory tiers.
"""

import time
from typing import Dict, Any

# Import common utilities for demos
from demo_utils import (
    setup_logging,
    log_print,
    create_memory_system,
    print_memory_details
)

def run_demo():
    """Run the basic memory operations demo."""
    # Setup logging
    logger = setup_logging("basic_memory_operations")
    log_print(logger, "Starting Basic Memory Operations Demo")
    
    # Initialize the memory system with custom settings
    memory_system = create_memory_system(
        stm_limit=500,              # Smaller limit to trigger transitions sooner
        stm_ttl=3600,               # Shorter TTL (1 hour)
        im_limit=1000,              # Smaller limit for demo purposes
        im_compression_level=1,
        ltm_compression_level=2,
        cleanup_interval=5,         # Check more frequently for demo
        description="basic memory operations demo"
    )
    
    # Create a test agent
    agent_id = "test_agent_1"
    
    # Store a series of state memories over simulated time steps
    for step in range(1, 21):
        # Create a sample state
        state = {
            "position": {"x": step, "y": step * 0.5},
            "health": 100 - step,
            "inventory": ["sword", "potion"] if step % 5 == 0 else ["sword"],
            "status": "exploring" if step % 3 == 0 else "idle",
        }
        
        # Store the state with different priorities
        priority = 1.0 if step % 5 == 0 else 0.5  # Higher priority for milestone steps
        success = memory_system.store_agent_state(agent_id, state, step, priority)
        
        log_print(logger, f"Step {step}: Stored state with priority {priority}, success: {success}")
        
        # Store an action every few steps
        if step % 3 == 0:
            action = {
                "type": "move",
                "direction": "north" if step % 2 == 0 else "east",
                "distance": step * 0.1,
            }
            success = memory_system.store_agent_action(agent_id, action, step, 0.7)
            log_print(logger, f"Step {step}: Stored action, success: {success}")
        
        # Store an interaction occasionally
        if step % 7 == 0:
            interaction = {
                "type": "conversation",
                "entity": f"npc_{step}",
                "content": f"Hello agent, this is step {step}!",
                "sentiment": "positive" if step % 2 == 0 else "neutral",
            }
            success = memory_system.store_agent_interaction(agent_id, interaction, step, 0.9)
            log_print(logger, f"Step {step}: Stored interaction, success: {success}")
        
        # Pause between steps to simulate time passing
        time.sleep(0.2)
        
        # Force cleanup check more frequently for demo purposes
        if step % 5 == 0:
            log_print(logger, f"\nStep {step}: Triggering memory maintenance check...")
            memory_system.force_memory_maintenance(agent_id)
    
    # After storing memories, print statistics
    stats = memory_system.get_memory_statistics(agent_id)
    log_print(logger, "\nMemory Statistics:")
    for key, value in stats.items():
        log_print(logger, f"  {key}: {value}")
    
    # Force memory maintenance to ensure tier transitions
    log_print(logger, "\nForcing final memory maintenance...")
    success = memory_system.force_memory_maintenance(agent_id)
    log_print(logger, f"Memory maintenance completed, success: {success}")
    
    # Print updated statistics after maintenance
    print_memory_details(memory_system, agent_id, "Updated Memory Statistics")
    
    log_print(logger, "\nBasic memory operations demo completed!")
    log_print(logger, f"Log file saved at: logs/{logger.handlers[0].baseFilename.split('/')[-1]}")

if __name__ == "__main__":
    run_demo() 