"""
Demo 4: Memory Hooks and Integration Capabilities

This demo showcases the event-driven memory system features including:
- Memory event hooks for automatic memory formation
- Integration with external systems via event triggers
- Custom memory filtering and processing
"""

import os
import sys
import time
import random
from typing import Dict, Any, List, Callable
from pprint import pprint

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_memory.core import AgentMemorySystem
from agent_memory.config import MemoryConfig, RedisSTMConfig, RedisIMConfig

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    print_memory_details
)

# Define some custom memory hook functions
def significant_event_hook(event_data: Dict[str, Any], agent_id: str, memory_system: Any) -> bool:
    """Hook that processes significant events and decides whether to store them.
    
    Args:
        event_data: Data about the event
        agent_id: ID of the agent experiencing the event
        memory_system: Reference to the memory system
        
    Returns:
        True if processing was successful
    """
    print(f"Significant event hook triggered for agent {agent_id}")
    print(f"  Event type: {event_data.get('type', 'unknown')}")
    
    # Only store events that meet certain criteria (e.g., high importance)
    if event_data.get('importance', 0) >= 0.7:
        # Create a formatted memory entry
        memory_data = {
            "event_type": event_data.get('type', 'unknown'),
            "description": event_data.get('description', ''),
            "location": event_data.get('location', {}),
            "entities": event_data.get('entities', []),
            "importance": event_data.get('importance', 0),
            "emotional_impact": event_data.get('emotional_impact', {}),
        }
        
        # Store with high priority
        step = event_data.get('step', 0)
        memory_system.store_agent_action(agent_id, memory_data, step, 0.9)
        print(f"  Stored high-importance event (importance: {event_data.get('importance', 0):.2f})")
        return True
    else:
        print(f"  Event ignored (importance: {event_data.get('importance', 0):.2f} below threshold)")
        return False

def state_change_hook(event_data: Dict[str, Any], agent_id: str, memory_system: Any) -> bool:
    """Hook that detects and records significant state changes.
    
    Args:
        event_data: Data about the state change
        agent_id: ID of the agent experiencing the change
        memory_system: Reference to the memory system
        
    Returns:
        True if processing was successful
    """
    print(f"State change hook triggered for agent {agent_id}")
    
    # Extract current and previous state
    current_state = event_data.get('current_state', {})
    previous_state = event_data.get('previous_state', {})
    
    # Identify what changed
    changes = {}
    significance = 0.0
    
    # Check for health changes
    if 'health' in current_state and 'health' in previous_state:
        health_change = current_state['health'] - previous_state['health']
        if abs(health_change) >= 10:  # Significant health change
            changes['health_change'] = health_change
            # Health drops are more significant than gains
            significance += abs(health_change) / 100 * (1.5 if health_change < 0 else 1.0)
    
    # Check for location changes
    if 'position' in current_state and 'position' in previous_state:
        curr_loc = current_state['position'].get('location_name', '')
        prev_loc = previous_state['position'].get('location_name', '')
        if curr_loc != prev_loc and curr_loc and prev_loc:
            changes['location_change'] = {'from': prev_loc, 'to': curr_loc}
            significance += 0.3  # Location changes are moderately significant
    
    # Check inventory changes
    if 'inventory' in current_state and 'inventory' in previous_state:
        # This is simplified - a real implementation would do a deeper comparison
        curr_items = set(str(item) for item in current_state['inventory'].get('items', []))
        prev_items = set(str(item) for item in previous_state['inventory'].get('items', []))
        
        gained_items = curr_items - prev_items
        lost_items = prev_items - curr_items
        
        if gained_items or lost_items:
            changes['inventory_changes'] = {
                'gained': list(gained_items),
                'lost': list(lost_items)
            }
            significance += 0.2  # Inventory changes are somewhat significant
    
    # Only store if changes are significant enough
    if significance >= 0.25:
        memory_data = {
            "type": "state_change",
            "changes": changes,
            "significance": significance,
            "step": event_data.get('step', 0),
        }
        
        memory_system.store_agent_state(agent_id, memory_data, event_data.get('step', 0), significance)
        print(f"  Stored significant state change (significance: {significance:.2f})")
        return True
    else:
        print(f"  State change ignored (significance: {significance:.2f} below threshold)")
        return False

def interaction_hook(event_data: Dict[str, Any], agent_id: str, memory_system: Any) -> bool:
    """Hook for processing social interactions between agents.
    
    Args:
        event_data: Data about the interaction
        agent_id: ID of the agent experiencing the interaction
        memory_system: Reference to the memory system
        
    Returns:
        True if processing was successful
    """
    print(f"Interaction hook triggered for agent {agent_id}")
    
    interaction_type = event_data.get('interaction_type', 'unknown')
    target_agent = event_data.get('target_agent', 'unknown')
    outcome = event_data.get('outcome', 'neutral')
    
    # Calculate social impact
    social_impact = {
        "positive": 0.8,
        "neutral": 0.5,
        "negative": 0.9,  # Negative interactions often more memorable
    }.get(outcome, 0.5)
    
    # More important if it's a new entity
    is_new_entity = event_data.get('is_new_entity', False)
    if is_new_entity:
        social_impact += 0.2
    
    # Format and store the interaction
    interaction_data = {
        "type": "social_interaction",
        "interaction_type": interaction_type,
        "target_agent": target_agent,
        "outcome": outcome,
        "details": event_data.get('details', {}),
        "social_impact": social_impact,
    }
    
    memory_system.store_agent_interaction(
        agent_id, 
        interaction_data, 
        event_data.get('step', 0), 
        social_impact
    )
    
    print(f"  Stored social interaction with impact {social_impact:.2f}")
    return True

def run_demo():
    """Run the memory hooks and integration demo."""
    # Initialize with custom config
    memory_system = create_memory_system(
        stm_limit=100,              # Smaller limit for demo purposes
        stm_ttl=3600,               # 1 hour TTL
        im_limit=200,               # Smaller limit for demo purposes
        im_compression_level=1,
        cleanup_interval=10,        # More frequent cleanup
        enable_hooks=True,          # Ensure hooks are enabled
        description="hooks and integration demo"
    )
    
    # Set up agent
    agent_id = "hook_agent"
    
    # Register memory hooks
    memory_system.register_memory_hook(agent_id, "significant_event", significant_event_hook)
    memory_system.register_memory_hook(agent_id, "state_change", state_change_hook)
    memory_system.register_memory_hook(agent_id, "interaction", interaction_hook)
    
    print(f"Registered memory hooks for agent {agent_id}")
    
    # Simulate a sequence of agent experiences that will trigger hooks
    print("\n--- Simulating Agent Experiences ---")
    
    # Generate initial state
    current_state = {
        "health": 100,
        "energy": 80,
        "position": {
            "x": 100,
            "y": 200,
            "location_name": "village_square"
        },
        "inventory": {
            "items": [
                {"name": "sword", "count": 1},
                {"name": "gold_coins", "count": 50}
            ]
        }
    }
    
    # Run through several steps with different types of events
    for step in range(1, 11):
        print(f"\nStep {step}:")
        
        # Store the current state directly (without hooks)
        memory_system.store_agent_state(agent_id, current_state, step, 0.5)
        
        # Clone previous state before modifications
        previous_state = {k: v.copy() if isinstance(v, dict) else v for k, v in current_state.items()}
        
        # SIMULATION: Modify the state based on random events
        
        # 1. Sometimes trigger a significant event
        if random.random() < 0.4:  # 40% chance
            event_types = ["discovery", "combat", "achievement", "obstacle"]
            event_importance = random.uniform(0.5, 1.0)
            
            event_data = {
                "type": random.choice(event_types),
                "description": f"A {event_types} event occurred in step {step}",
                "location": current_state["position"],
                "entities": [f"entity_{random.randint(1, 5)}" for _ in range(random.randint(1, 3))],
                "importance": event_importance,
                "emotional_impact": {
                    "surprise": random.uniform(0, 1),
                    "fear": random.uniform(0, 1) if "combat" in event_types else 0,
                    "joy": random.uniform(0, 1) if "achievement" in event_types else 0,
                },
                "step": step
            }
            
            # Trigger the significant event hook
            memory_system.trigger_memory_event(agent_id, "significant_event", event_data)
        
        # 2. Sometimes change location
        if random.random() < 0.3:  # 30% chance
            locations = ["village_square", "forest_path", "mountain_cave", "riverside", "castle"]
            new_location = random.choice([l for l in locations if l != current_state["position"]["location_name"]])
            
            # Update position
            current_state["position"]["location_name"] = new_location
            current_state["position"]["x"] += random.randint(-50, 50)
            current_state["position"]["y"] += random.randint(-50, 50)
        
        # 3. Sometimes change health
        if random.random() < 0.4:  # 40% chance
            health_change = random.randint(-20, 10)  # More likely to lose health
            current_state["health"] = max(0, min(100, current_state["health"] + health_change))
        
        # 4. Sometimes change inventory
        if random.random() < 0.3:  # 30% chance
            possible_items = [
                {"name": "health_potion", "count": random.randint(1, 3)},
                {"name": "map_fragment", "count": 1},
                {"name": "gold_coins", "count": random.randint(5, 25)},
                {"name": "gemstone", "count": 1}
            ]
            
            # Random add or remove
            if random.random() < 0.7 or not current_state["inventory"]["items"]:  # 70% add, or forced add if empty
                # Add a new item
                current_state["inventory"]["items"].append(random.choice(possible_items))
            else:
                # Remove an item if we have any
                if current_state["inventory"]["items"]:
                    current_state["inventory"]["items"].pop(random.randrange(len(current_state["inventory"]["items"])))
        
        # 5. Trigger state change hook if state changed
        if current_state != previous_state:
            state_change_data = {
                "current_state": current_state,
                "previous_state": previous_state,
                "step": step
            }
            memory_system.trigger_memory_event(agent_id, "state_change", state_change_data)
        
        # 6. Sometimes trigger social interactions
        if random.random() < 0.4:  # 40% chance
            interaction_types = ["conversation", "trade", "quest", "conflict"]
            outcomes = ["positive", "neutral", "negative"]
            
            interaction_data = {
                "interaction_type": random.choice(interaction_types),
                "target_agent": f"npc_{random.randint(1, 5)}",
                "outcome": random.choice(outcomes),
                "is_new_entity": random.random() < 0.2,  # 20% chance it's a new entity
                "details": {
                    "duration": random.randint(1, 10),
                    "intensity": random.uniform(0.1, 0.9)
                },
                "step": step
            }
            
            memory_system.trigger_memory_event(agent_id, "interaction", interaction_data)
        
        # Pause to make the demo more readable
        time.sleep(0.5)
    
    # After simulation, display summary
    print("\n--- Memory System Results ---")
    
    # Print memory statistics
    print_memory_details(memory_system, agent_id, "Final Memory State")
    
    # Retrieve memories formed through hooks
    print("\n--- Retrieving Memories Formed Through Hooks ---")
    
    # Get significant event memories
    event_memories = memory_system.retrieve_by_attributes(
        agent_id, {"event_type": {"$exists": True}}
    )
    print(f"\nSignificant Event Memories: {len(event_memories)} found")
    for i, memory in enumerate(event_memories[:3]):  # Show first 3
        print(f"  Event {i+1}: {memory.get('event_type', 'unknown')} - " + 
              f"Importance: {memory.get('importance', 0):.2f}")
    
    # Get state change memories
    state_change_memories = memory_system.retrieve_by_attributes(
        agent_id, {"type": "state_change"}
    )
    print(f"\nState Change Memories: {len(state_change_memories)} found")
    for i, memory in enumerate(state_change_memories[:3]):  # Show first 3
        print(f"  Change {i+1}: significance {memory.get('significance', 0):.2f}")
        for change_type, change_data in memory.get('changes', {}).items():
            print(f"    {change_type}: {change_data}")
    
    # Get interaction memories
    interaction_memories = memory_system.retrieve_by_attributes(
        agent_id, {"type": "social_interaction"}
    )
    print(f"\nSocial Interaction Memories: {len(interaction_memories)} found")
    for i, memory in enumerate(interaction_memories[:3]):  # Show first 3
        print(f"  Interaction {i+1}: {memory.get('interaction_type', 'unknown')} with " + 
              f"{memory.get('target_agent', 'unknown')} - " +
              f"Outcome: {memory.get('outcome', 'unknown')}")
    
    print("\nMemory hooks and integration demo completed!")

if __name__ == "__main__":
    run_demo() 