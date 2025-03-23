"""
Demo 2: Memory Retrieval Capabilities

This demo showcases the different memory retrieval methods including:
- Similarity-based retrieval
- Attribute-based retrieval  
- Temporal retrieval
- Cross-tier retrieval
"""

import random
from typing import Dict, Any, List

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    pretty_print_memories
)

def run_demo():
    """Run the memory retrieval demo."""
    # Initialize with custom config for demo purposes
    memory_system = create_memory_system(
        stm_limit=500,          # Smaller limit to trigger transitions sooner
        stm_ttl=3600,           # Shorter TTL (1 hour)
        im_limit=1000,          # Smaller limit for demo purposes
        im_compression_level=1,
        cleanup_interval=10,    # Check for cleanup more frequently for demo
        description="retrieval demo"
    )
    
    # Use test agent
    agent_id = "retrieval_agent"
    
    # Populate with sample memories across multiple steps
    locations = ["forest", "mountain", "village", "dungeon", "castle"]
    actions = ["attack", "defend", "heal", "move", "interact", "observe"]
    items = ["sword", "shield", "potion", "map", "gem", "key"]
    
    print("Populating memory system with sample data...")
    
    # Generate 30 steps of varied memories
    for step in range(1, 31):
        # Create varied states
        state = {
            "position": {
                "x": random.uniform(-100, 100),
                "y": random.uniform(-100, 100),
                "location": random.choice(locations),
            },
            "health": random.randint(50, 100),
            "energy": random.randint(30, 100),
            "inventory": random.sample(items, random.randint(1, 3)),
            "level": step // 5 + 1,
        }
        
        # Store states with varying priorities
        priority = random.uniform(0.3, 1.0)
        memory_system.store_agent_state(agent_id, state, step, priority)
        
        # Store some actions (about 60% of steps)
        if random.random() < 0.6:
            action = {
                "type": random.choice(actions),
                "target": f"entity_{random.randint(1, 10)}",
                "success": random.choice([True, False, True]),  # Bias toward success
                "location": state["position"]["location"],
            }
            memory_system.store_agent_action(agent_id, action, step, random.uniform(0.4, 0.9))
        
        # Store some interactions (about 30% of steps)
        if random.random() < 0.3:
            interaction = {
                "type": "encounter",
                "entity": f"npc_{random.randint(1, 5)}",
                "mood": random.choice(["friendly", "hostile", "neutral"]),
                "outcome": random.choice(["trade", "information", "quest", "combat"]),
                "location": state["position"]["location"],
            }
            memory_system.store_agent_interaction(agent_id, interaction, step, random.uniform(0.5, 1.0))
        
        # Trigger maintenance every 10 steps to ensure memories are distributed
        if step % 10 == 0:
            memory_system.force_memory_maintenance(agent_id)
    
    # Force final memory maintenance to ensure memories are distributed across tiers
    print("Forcing memory maintenance to distribute memories across tiers...")
    memory_system.force_memory_maintenance(agent_id)
    
    # Print memory statistics
    stats = memory_system.get_memory_statistics(agent_id)
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demo 1: Similarity-based Retrieval
    print("\n--- Demo: Similarity-based Retrieval ---")
    query_state = {
        "position": {
            "location": "forest",
        },
        "health": 75,
        "inventory": ["sword", "potion"],
    }
    print("Query state:", query_state)
    
    similar_memories = memory_system.retrieve_similar_states(agent_id, query_state, k=3)
    pretty_print_memories(similar_memories, "Most Similar States")
    
    # Demo 2: Attribute-based Retrieval
    print("\n--- Demo: Attribute-based Retrieval ---")
    attr_query = {
        "position.location": "village",
    }
    print("Attribute query:", attr_query)
    
    attribute_memories = memory_system.retrieve_by_attributes(agent_id, attr_query)
    pretty_print_memories(attribute_memories[:3], "Memories with location='village' (showing first 3)")
    
    # Demo 3: Temporal Retrieval
    print("\n--- Demo: Temporal Retrieval ---")
    start_step = 10
    end_step = 15
    print(f"Time range: steps {start_step} to {end_step}")
    
    temporal_memories = memory_system.retrieve_by_time_range(agent_id, start_step, end_step)
    pretty_print_memories(temporal_memories, f"Memories from steps {start_step}-{end_step}")
    
    # Demo 4: Specific Memory Type Retrieval
    print("\n--- Demo: Specific Memory Type Retrieval ---")
    action_memories = memory_system.retrieve_by_attributes(agent_id, {"type": "attack"}, memory_type="action")
    pretty_print_memories(action_memories, "Attack Action Memories")
    
    # Demo 5: Content-based Search
    print("\n--- Demo: Content-based Search ---")
    content_query = "potion"  # Search for memories involving potions
    print(f"Content query: '{content_query}'")
    
    content_memories = memory_system.search_by_content(agent_id, content_query, k=3)
    pretty_print_memories(content_memories, f"Memories matching '{content_query}'")
    
    print("\nMemory retrieval demo completed!")

if __name__ == "__main__":
    run_demo() 