"""
Demo 3: Memory Tiers and Compression

This demo showcases the hierarchical memory architecture of the AgentMemorySystem focusing on:

1. Memory transitions between three memory tiers:
   - Short-Term Memory (STM): Recent, readily accessible, detailed memories
   - Intermediate Memory (IM): Important memories with level 1 compression
   - Long-Term Memory (LTM): Critical long-term memories with level 2 compression

2. Memory compression mechanisms:
   - Progressive information reduction during tier transitions
   - Priority-based retention of critical details
   - Size reduction while preserving essential information

3. Information retention across compression levels:
   - Comparison of memory content before and after compression
   - Compression ratios between memory tiers
   - Preservation of critical data despite size reduction

The demo runs through four distinct phases:
- Phase 1: Populating STM with detailed agent states
- Phase 2: Forcing transitions to IM with level 1 compression
- Phase 3: Triggering transitions to LTM with level 2 compression
- Phase 4: Testing information retention by comparing memories across tiers

This demonstrates how an agent can maintain a large memory store efficiently by
compressing older, less immediately relevant memories while preserving their
essential information for future reference.
"""

import json
import time
from typing import Any, Dict, List

# Import common utilities for demos
from demo_utils import create_memory_system, print_memory_details


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
            "terrain_type": (
                "forest" if step % 3 == 0 else "plains" if step % 3 == 1 else "mountain"
            ),
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
            "main_hand": (
                "sword" if step % 5 < 2 else "axe" if step % 5 < 4 else "staff"
            ),
            "off_hand": (
                "shield" if step % 7 < 4 else "torch" if step % 7 < 6 else "empty"
            ),
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
            },
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
            ],
        },
        "relationships": {
            f"npc_{step % 10}": {"affinity": step % 100, "trust": step % 50 + 50},
            f"npc_{(step + 5) % 10}": {
                "affinity": step % 80 + 20,
                "trust": step % 40 + 30,
            },
        },
    }


def run_demo():
    """Run the memory tiers and compression demo."""
    # Initialize with custom config for demo purposes
    memory_system = create_memory_system(
        stm_limit=8,  # Very small limit to trigger transitions quickly
        stm_ttl=3600,  # Shorter TTL (1 hour)
        im_limit=16,  # Small limit for quick LTM transitions
        im_compression_level=1,  # Level 1 compression for IM
        ltm_compression_level=2,  # Level 2 compression for LTM
        ltm_batch_size=5,  # Smaller batch size for demo
        cleanup_interval=5,  # Check for cleanup frequently for demo
        description="memory tier demo",
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
    print(
        "Forced memory maintenance - some memories should move to LTM with higher compression"
    )

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
        # Fix: Access position and stats from within the content field
        content = memory.get("content", {})
        print(f"  Position: {json.dumps(content.get('position', {}))}")
        print(f"  Stats: {json.dumps(content.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
        print(f"  Content size: {len(json.dumps(content))} bytes")

    print("\nIntermediate Memory from IM (Step 14-15):")
    if im_memory:
        memory = im_memory[0]
        # Fix: Access position and stats from within the content field
        content = memory.get("content", {})
        print(f"  Position: {json.dumps(content.get('position', {}))}")
        print(f"  Stats: {json.dumps(content.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
        print(f"  Content size: {len(json.dumps(content))} bytes")

    print("\nOldest Memory from LTM (Step 4-5):")
    if old_memory:
        memory = old_memory[0]
        # Fix: Access position and stats from within the content field
        content = memory.get("content", {})
        print(f"  Position: {json.dumps(content.get('position', {}))}")
        print(f"  Stats: {json.dumps(content.get('stats', {}))}")
        print(f"  Size: {len(json.dumps(memory))} bytes")
        print(f"  Content size: {len(json.dumps(content))} bytes")

    # Compare compression ratios and information retention
    if recent_memory and old_memory:
        recent_memory_obj = recent_memory[0]
        old_memory_obj = old_memory[0]

        # Compare total memory sizes
        recent_size = len(json.dumps(recent_memory_obj))
        old_size = len(json.dumps(old_memory_obj))
        total_ratio = recent_size / max(1, old_size)

        # Also compare just the content sizes to see compression effect
        recent_content_size = len(json.dumps(recent_memory_obj.get("content", {})))
        old_content_size = len(json.dumps(old_memory_obj.get("content", {})))
        content_ratio = recent_content_size / max(1, old_content_size)

        print(f"\nCompression comparison:")
        print(f"  Recent memory size: {recent_size} bytes")
        print(f"  Old memory size: {old_size} bytes")
        print(f"  Total size ratio: {total_ratio:.2f}x")
        print(f"  Recent content size: {recent_content_size} bytes")
        print(f"  Old content size: {old_content_size} bytes")
        print(f"  Content compression ratio: {content_ratio:.2f}x")

    # PHASE 5: Visualization of ideal compression behavior
    print("\n--- PHASE 5: Visualization of Theoretical Compression ---")
    print(
        "This section demonstrates how compression would ideally reduce memory details across tiers."
    )

    # Get a sample memory for demonstration
    if recent_memory:
        sample_memory = recent_memory[0].get("content", {})

        # Define compression levels
        print("\nSTM (Short-Term Memory) - Full Detail:")
        print(json.dumps(sample_memory, indent=2))

        # Level 1 compression (IM) - Reduce some details
        def simulate_im_compression(memory):
            """Simulate level 1 compression for Intermediate Memory."""
            compressed = memory.copy()

            # Simplify position - keep only essential location data
            if "position" in compressed:
                position = compressed["position"].copy()
                # Keep only core position values and location
                compressed["position"] = {
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "location_name": position.get("location_name"),
                    "terrain_type": position.get("terrain_type"),
                }

            # Simplify stats - keep only primary stats
            if "stats" in compressed:
                stats = compressed["stats"].copy()
                # Keep only the most important stats
                compressed["stats"] = {
                    "health": stats.get("health"),
                    "energy": stats.get("energy"),
                }

            # Simplify equipment - keep only summary
            if "equipment" in compressed:
                equipment = compressed["equipment"].copy()
                compressed["equipment"] = {
                    "main_hand": equipment.get("main_hand"),
                    "armor_equipped": (
                        True if equipment.get("chest") == "armor" else False
                    ),
                }

            # Simplify inventory - summarize items
            if "inventory" in compressed:
                inventory = compressed["inventory"].copy()
                # Create a simplified count of items
                item_count = len(inventory.get("items", []))
                compressed["inventory"] = {
                    "item_count": item_count,
                    "key_items": [
                        item.get("name")
                        for item in inventory.get("items", [])
                        if "quest" in item.get("name", "")
                    ],
                }

            # Remove less important sections
            compressed.pop("status_effects", None)

            # Simplify objectives
            if "objectives" in compressed:
                objectives = compressed["objectives"].copy()
                compressed["objectives"] = {
                    "current": objectives.get("current"),
                    "progress": objectives.get("progress"),
                }

            # Remove relationships data
            compressed.pop("relationships", None)

            return compressed

        # Level 2 compression (LTM) - Keep only critical information
        def simulate_ltm_compression(memory):
            """Simulate level 2 compression for Long-Term Memory."""
            compressed = memory.copy()

            # Keep only minimal position info
            if "position" in compressed:
                position = compressed["position"].copy()
                compressed["position"] = {
                    "location_name": position.get("location_name")
                }

            # Keep only health
            if "stats" in compressed:
                stats = compressed["stats"].copy()
                compressed["stats"] = {"health": stats.get("health")}

            # Remove equipment details
            compressed.pop("equipment", None)

            # Summarize inventory to just quest items
            if "inventory" in compressed:
                inventory = compressed["inventory"].copy()
                quest_items = [
                    item.get("name")
                    for item in inventory.get("items", [])
                    if "quest" in item.get("name", "")
                ]
                if quest_items:
                    compressed["inventory"] = {"quest_items": quest_items}
                else:
                    compressed.pop("inventory", None)

            # Remove status effects
            compressed.pop("status_effects", None)

            # Simplify objectives to just the current objective
            if "objectives" in compressed:
                objectives = compressed["objectives"].copy()
                compressed["objectives"] = {"current": objectives.get("current")}

            # Remove relationships
            compressed.pop("relationships", None)

            return compressed

        # Display simulated compression results
        im_compressed = simulate_im_compression(sample_memory)
        ltm_compressed = simulate_ltm_compression(sample_memory)

        print("\nIM (Intermediate Memory) - Level 1 Compression:")
        print(json.dumps(im_compressed, indent=2))

        print("\nLTM (Long-Term Memory) - Level 2 Compression:")
        print(json.dumps(ltm_compressed, indent=2))

        # Compare sizes
        original_size = len(json.dumps(sample_memory))
        im_size = len(json.dumps(im_compressed))
        ltm_size = len(json.dumps(ltm_compressed))

        im_ratio = im_size / original_size
        ltm_ratio = ltm_size / original_size

        print("\nTheoretical Compression Summary:")
        print(f"  Original size: {original_size} bytes")
        print(
            f"  IM compressed: {im_size} bytes ({im_ratio:.2f}x, {(1-im_ratio)*100:.1f}% reduction)"
        )
        print(
            f"  LTM compressed: {ltm_size} bytes ({ltm_ratio:.2f}x, {(1-ltm_ratio)*100:.1f}% reduction)"
        )

    print("\nMemory tiers and compression demo completed!")


if __name__ == "__main__":
    run_demo()
