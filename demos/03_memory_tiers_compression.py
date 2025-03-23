"""
Demo 3: Memory Tiers and Compression

This demo showcases the hierarchical memory architecture including:
- Memory transitions between STM, IM, and LTM tiers
- Memory compression during tier transitions
- Information retention across compression
"""

import time
import json
from typing import Dict, Any, List

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    print_memory_details
)

def create_detailed_state(step: int) -> Dict[str, Any]:
    """Create a detailed state with rich information."""
    return {
        "agent_id": "compression_agent",
        "step": step,
        "position": {
            "x": step * 1.5,
            "y": step * 0.75,
            "z": step * 0.3,
            "orientation": step % 360,
            "location_name": f"sector_{step // 5}",
            "terrain_type": "forest" if step % 3 == 0 else "plains" if step % 3 == 1 else "mountain",
        },
        "stats": {
            "health": 100 - (step % 10),
            "energy": 80 - (step % 15),
            "stamina": 90 - (step % 20),
            "morale": 75 + (step % 10),
            "hunger": step % 30,
            "thirst": step % 25,
        },
        "equipment": {
            "head": "helmet" if step % 8 < 4 else "cap",
            "chest": "armor" if step % 10 < 7 else "tunic",
            "legs": "greaves" if step % 6 < 3 else "pants",
            "feet": "boots" if step % 4 < 2 else "sandals",
            "main_hand": "sword" if step % 5 < 2 else "axe" if step % 5 < 4 else "staff",
            "off_hand": "shield" if step % 7 < 4 else "torch" if step % 7 < 6 else "empty",
        },
        "inventory": {
            "items": [
                {"name": "health_potion", "count": step % 5 + 1},
                {"name": "energy_potion", "count": step % 3 + 1},
                {"name": "gold_coins", "count": step * 10},
                {"name": "arrows", "count": step * 5},
                {"name": f"quest_item_{step % 10}", "count": 1},
            ],
            "capacity": {
                "used": (step % 10) + 10,
                "total": 50,
            }
        },
        "status_effects": [
            {"name": "rested", "duration": step % 10 + 5} if step % 6 == 0 else None,
            {"name": "fortified", "duration": step % 8 + 3} if step % 7 == 0 else None,
            {"name": "poisoned", "duration": step % 5 + 2} if step % 11 == 0 else None,
        ],
        "objectives": {
            "current": f"objective_{step // 3}",
            "progress": step % 100,
            "subgoals": [
                {"id": f"subgoal_{step % 5}_1", "completed": step % 2 == 0},
                {"id": f"subgoal_{step % 5}_2", "completed": step % 3 == 0},
            ]
        },
        "relationships": {
            f"npc_{step % 10}": {"affinity": step % 100, "trust": step % 50 + 50},
            f"npc_{(step + 5) % 10}": {"affinity": step % 80 + 20, "trust": step % 40 + 30},
        }
    }

def run_demo():
    """Run the memory tiers and compression demo."""
    # Initialize with custom config for demo purposes
    memory_system = create_memory_system(
        stm_limit=8,                 # Very small limit to trigger transitions quickly
        stm_ttl=3600,                # Shorter TTL (1 hour)
        im_limit=16,                 # Small limit for quick LTM transitions
        im_compression_level=1,      # Level 1 compression for IM
        ltm_compression_level=2,     # Level 2 compression for LTM
        ltm_batch_size=5,            # Smaller batch size for demo
        cleanup_interval=5,          # Check for cleanup frequently for demo
        description="memory tier demo"
    )
    
    print("Memory system configuration:")
    print(f"  STM memory limit: 8 entries")
    print(f"  IM memory limit: 16 entries")
    print(f"  IM compression level: 1")
    print(f"  LTM compression level: 2")
    
    agent_id = "compression_agent"
    
    # PHASE 1: Store detailed memories in STM
    print("\n--- PHASE 1: Storing memories in STM ---")
    for step in range(1, 11):
        state = create_detailed_state(step)
        success = memory_system.store_agent_state(agent_id, state, step, 0.8)
        print(f"Step {step}: Stored detailed state ({len(json.dumps(state))} bytes)")
        
        # Force maintenance every few steps to ensure transitions happen
        if step % 3 == 0:
            memory_system.force_memory_maintenance(agent_id)
    
    # Print memory details after STM filling
    print_memory_details(memory_system, agent_id, "After STM population")
    
    # PHASE 2: Force transition to move some memories to IM with compression
    print("\n--- PHASE 2: Transitioning to Intermediate Memory ---")
    # Store a few more states to trigger automatic transitions
    for step in range(11, 16):
        state = create_detailed_state(step)
        success = memory_system.store_agent_state(agent_id, state, step, 0.7)
        print(f"Step {step}: Stored detailed state")
        
        # Force maintenance after each step for demo purposes
        memory_system.force_memory_maintenance(agent_id)
    
    # Print memory details after IM transition
    print_memory_details(memory_system, agent_id, "After IM transition")
    
    # PHASE 3: Move more memories to push some to LTM
    print("\n--- PHASE 3: Transitioning to Long-Term Memory ---")
    # Store more states to push oldest memories to LTM
    for step in range(16, 26):
        state = create_detailed_state(step)
        success = memory_system.store_agent_state(agent_id, state, step, 0.6)
        print(f"Step {step}: Stored detailed state")
        
        # Force maintenance after each step for demo purposes
        if step % 2 == 0:
            memory_system.force_memory_maintenance(agent_id)
    
    # Force final maintenance to ensure transitions happen
    memory_system.force_memory_maintenance(agent_id)
    print("Forced memory maintenance - some memories should move to LTM with higher compression")
    
    # Print memory details after LTM transition
    print_memory_details(memory_system, agent_id, "After LTM transition")
    
    # PHASE 4: Demonstrate information retention despite compression
    print("\n--- PHASE 4: Testing Information Retention ---")
    
    # Retrieve a memory from each tier
    recent_memory = memory_system.retrieve_by_time_range(agent_id, 24, 25)
    im_memory = memory_system.retrieve_by_time_range(agent_id, 14, 15)
    old_memory = memory_system.retrieve_by_time_range(agent_id, 4, 5)
    
    print("\nRecent Memory from STM (Step 24-25):")
    if recent_memory:
        memory = recent_memory[0]
        print(f"  Position: {json.dumps(memory.get('position', {}))}")
        print(f"  Stats: {json.dumps(memory.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
    
    print("\nIntermediate Memory from IM (Step 14-15):")
    if im_memory:
        memory = im_memory[0]
        print(f"  Position: {json.dumps(memory.get('position', {}))}")
        print(f"  Stats: {json.dumps(memory.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
    
    print("\nOldest Memory from LTM (Step 4-5):")
    if old_memory:
        memory = old_memory[0]
        print(f"  Position: {json.dumps(memory.get('position', {}))}")
        print(f"  Stats: {json.dumps(memory.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
    
    # Compare compression ratios and information retention
    if recent_memory and old_memory:
        recent_size = len(json.dumps(recent_memory[0]))
        old_size = len(json.dumps(old_memory[0]))
        ratio = recent_size / max(1, old_size)
        print(f"\nCompression comparison:")
        print(f"  Recent memory size: {recent_size} bytes")
        print(f"  Old memory size: {old_size} bytes")
        print(f"  Compression ratio: {ratio:.2f}x")
    
    print("\nMemory tiers and compression demo completed!")

if __name__ == "__main__":
    run_demo() 