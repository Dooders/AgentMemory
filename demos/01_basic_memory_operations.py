"""
Demo 1: Basic Memory Operations

This demo shows how to initialize the AgentMemorySystem and perform basic
memory storage operations across the three memory tiers.
"""

import time

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    log_print,
    print_memory_details,
    setup_logging,
)


def run_demo():
    """Run the basic memory operations demo."""
    # Demo name to use for logging
    demo_name = "basic_memory_operations"

    # Setup logging
    logger = setup_logging(demo_name)
    log_print(logger, "Starting Basic Memory Operations Demo")

    # Initialize the memory system with custom settings
    memory_system = create_memory_system(
        stm_limit=10,  # Much smaller limit to trigger transitions sooner
        stm_ttl=5,  # Much shorter TTL (5 seconds)
        im_limit=20,  # Smaller limit for demo purposes
        im_compression_level=1,
        ltm_compression_level=2,
        cleanup_interval=3,  # Check more frequently for demo
        description="basic memory operations demo",
    )

    # Create a test agent
    agent_id = "test_agent_1"

    # Store a series of state memories over simulated time steps
    for step in range(1, 31):  # Increased from 20 to 30 steps
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

        log_print(
            logger,
            f"Step {step}: Stored state with priority {priority}, success: {success}",
        )

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
            success = memory_system.store_agent_interaction(
                agent_id, interaction, step, 0.9
            )
            log_print(logger, f"Step {step}: Stored interaction, success: {success}")

        # Pause between steps to simulate time passing
        time.sleep(0.5)  # Increased from 0.2 to 0.5 seconds

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

    # Add some fresh STM memories to ensure we have examples to display
    log_print(logger, "\nAdding fresh STM memories for demonstration...")
    
    # Fresh state memory
    fresh_state = {
        "position": {"x": 100, "y": 100},
        "health": 75,
        "inventory": ["sword", "shield", "health_potion"],
        "status": "ready_for_display",
    }
    memory_system.store_agent_state(agent_id, fresh_state, 100, 0.8)
    
    # Fresh action memory
    fresh_action = {
        "type": "demonstration",
        "action": "display_memory_example",
        "parameters": {"detail_level": "high"}
    }
    memory_system.store_agent_action(agent_id, fresh_action, 101, 0.7)
    
    # Fresh interaction memory
    fresh_interaction = {
        "type": "system_message",
        "content": "This is an example STM interaction memory",
        "metadata": {"purpose": "display_in_demo"}
    }
    memory_system.store_agent_interaction(agent_id, fresh_interaction, 102, 0.9)

    # Display example content from each memory tier
    log_print(logger, "\n== EXAMPLE MEMORY CONTENT FROM EACH TIER ==")
    memory_agent = memory_system.get_memory_agent(agent_id)
    
    # STM example content
    log_print(logger, "\n----- SHORT-TERM MEMORY (STM) EXAMPLES -----")
    stm_memories = memory_agent.stm_store.get_all(agent_id)
    
    if stm_memories:
        for i, memory in enumerate(stm_memories[:3]):  # Show up to 3 examples
            log_print(logger, f"STM Memory #{i+1}:")
            log_print(logger, f"  ID: {memory.get('id', 'N/A')}")
            log_print(logger, f"  Type: {memory.get('type', 'N/A')}")
            log_print(logger, f"  Timestamp: {memory.get('timestamp', 'N/A')}")
            log_print(logger, f"  Priority: {memory.get('priority', 'N/A')}")
            # Pretty print the content for better readability
            log_print(logger, f"  Content: {memory.get('content', 'N/A')}")
    else:
        log_print(logger, "  No memories in STM - this can happen if all memories have already transitioned to IM or LTM.")
        log_print(logger, "  Try reducing the maintenance frequency or increasing stm_ttl if you want to see more STM examples.")
    
    # IM example content
    log_print(logger, "\n----- INTERMEDIATE MEMORY (IM) EXAMPLES -----")
    im_memories = memory_agent.im_store.get_all(agent_id)
    
    if im_memories:
        for i, memory in enumerate(im_memories[:3]):  # Show up to 3 examples
            log_print(logger, f"IM Memory #{i+1}:")
            log_print(logger, f"  ID: {memory.get('id', 'N/A')}")
            log_print(logger, f"  Type: {memory.get('type', 'N/A')}")
            log_print(logger, f"  Timestamp: {memory.get('timestamp', 'N/A')}")
            log_print(logger, f"  Priority: {memory.get('priority', 'N/A')}")
            log_print(logger, f"  Content: {memory.get('content', 'N/A')}")
    else:
        log_print(logger, "  No memories in IM")
    
    # LTM example content
    log_print(logger, "\n----- LONG-TERM MEMORY (LTM) EXAMPLES -----")
    ltm_memories = memory_agent.ltm_store.get_all(agent_id)
    
    if ltm_memories:
        for i, memory in enumerate(ltm_memories[:3]):  # Show up to 3 examples
            log_print(logger, f"LTM Memory #{i+1}:")
            log_print(logger, f"  ID: {memory.get('id', 'N/A')}")
            log_print(logger, f"  Type: {memory.get('type', 'N/A')}")
            log_print(logger, f"  Timestamp: {memory.get('timestamp', 'N/A')}")
            log_print(logger, f"  Priority: {memory.get('priority', 'N/A')}")
            log_print(logger, f"  Content: {memory.get('content', 'N/A')}")
    else:
        log_print(logger, "  No memories in LTM")

    # Validate memory operations
    log_print(logger, "\nValidating memory operations...")
    
    # 1. Get memory statistics to check counts in each tier
    stats = memory_system.get_memory_statistics(agent_id)
    stm_count = stats["tiers"]["stm"]["count"]
    im_count = stats["tiers"]["im"]["count"]
    ltm_count = stats["tiers"]["ltm"]["count"]
    
    log_print(logger, f"STM count: {stm_count}")
    log_print(logger, f"IM count: {im_count}")
    log_print(logger, f"LTM count: {ltm_count}")
    
    # Validate STM is within limits
    assert stm_count <= 10, f"STM exceeds limit: {stm_count} > 10"
    log_print(logger, "[PASS] STM is within configured limits")
    
    # 2. Check tier transitions - ensure memories are transitioning to IM and LTM
    total_memories = stm_count + im_count + ltm_count
    expected_min = 40  # At least this many memories should be stored across all tiers
    assert total_memories >= expected_min, f"Total memories {total_memories} is less than expected minimum {expected_min}"
    log_print(logger, f"[PASS] Total memories ({total_memories}) meets expected minimum")
    
    # 3. Check high-priority memories in different tiers
    # We expect most high-priority memories to be preserved in higher tiers (IM/LTM)
    high_priority_steps = [5, 10, 15, 20, 25, 30]
    memory_agent = memory_system.get_memory_agent(agent_id)
    
    # Query for high-priority memories across all tiers
    milestone_memories = []
    
    # Try using retrieval from the memory system API instead of direct store access
    log_print(logger, "Checking for milestone memories:")
    for step in high_priority_steps:
        # Use the higher-level API method that knows how to search across all tiers
        memories = memory_system.retrieve_by_time_range(agent_id, step, step)
        for memory in memories:
            # Try to determine which tier it was from
            tier = "Unknown"
            memory_id = memory.get("id", "")
            if memory_id:
                # Check which tier contains this memory
                if memory_agent.stm_store.exists(memory_id):
                    tier = "STM"
                elif memory_agent.im_store.exists(memory_id):
                    tier = "IM"
                elif memory_agent.ltm_store.exists(memory_id):
                    tier = "LTM"
            
            log_print(logger, f"Found milestone memory for step {step} in tier: {tier}")
            milestone_memories.append(memory)
    
    # We expect most high-priority memories to be preserved
    expected_milestone_min = 4  # At least this many milestone memories should remain
    assert len(milestone_memories) >= expected_milestone_min, f"Only found {len(milestone_memories)} milestone memories, expected at least {expected_milestone_min}"
    log_print(logger, f"[PASS] Found {len(milestone_memories)} milestone memories as expected")
    
    log_print(logger, "\nValidation completed successfully!")

    log_print(logger, "\nBasic memory operations demo completed!")
    log_print(logger, f"Log file saved at: logs/{demo_name}.log")


if __name__ == "__main__":
    run_demo()
