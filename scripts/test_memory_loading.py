"""
Test script to verify if memory IDs and checksums are preserved when loading memories from JSON.
"""

import json
import os
import sys
from pprint import pprint

# Add the project root to the Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation.demo_utils import create_memory_system


def compare_memories(original_memory, loaded_memory):
    """Compare original and loaded memories to check for differences"""
    print(f"Comparing memory: {original_memory.get('memory_id')}")

    # Check memory_id
    if original_memory.get("memory_id") != loaded_memory.get("memory_id"):
        print(
            f"❌ memory_id mismatch: original={original_memory.get('memory_id')}, loaded={loaded_memory.get('memory_id')}"
        )
    else:
        print(f"✓ memory_id preserved: {original_memory.get('memory_id')}")

    # Check checksum
    original_checksum = original_memory.get("metadata", {}).get("checksum")
    loaded_checksum = loaded_memory.get("metadata", {}).get("checksum")

    if original_checksum != loaded_checksum:
        print(
            f"❌ checksum mismatch: original={original_checksum}, loaded={loaded_checksum}"
        )
    else:
        print(f"✓ checksum preserved: {original_checksum}")

    # Check content
    if original_memory.get("content") != loaded_memory.get("content"):
        print("❌ content mismatch")
        print("Original:", original_memory.get("content"))
        print("Loaded:", loaded_memory.get("content"))
    else:
        print("✓ content preserved")

    # Check metadata
    for key in original_memory.get("metadata", {}):
        if key != "checksum" and key in loaded_memory.get("metadata", {}):
            # Skip operational tracking fields that are expected to change
            if key in ["last_access_time", "retrieval_count"]:
                print(
                    f"ℹ️ {key} changed as expected: original={original_memory['metadata'][key]}, loaded={loaded_memory['metadata'][key]}"
                )
            elif original_memory["metadata"][key] != loaded_memory["metadata"][key]:
                print(
                    f"❌ metadata mismatch for {key}: original={original_memory['metadata'][key]}, loaded={loaded_memory['metadata'][key]}"
                )
            else:
                print(
                    f"✓ metadata preserved for {key}: {original_memory['metadata'][key]}"
                )

    print()


def main():
    """Main test function"""
    # Sample memory file path and agent ID
    memory_file = "simple"  # Use the predefined sample
    agent_id = "demo_agent"

    print(f"Testing memory loading from: {memory_file}")
    print(f"Using agent: {agent_id}")

    # First, parse the original JSON file directly
    json_file_path = os.path.join(
        os.path.dirname(__file__), "demos", "memory_samples", "simple_agent_memory.json"
    )
    with open(json_file_path, "r") as f:
        original_data = json.load(f)

    # Extract original memories from the JSON
    original_memories = original_data["agents"][agent_id]["memories"]
    print(f"Original memories from JSON: {len(original_memories)}")
    for memory in original_memories:
        print(f"Memory ID: {memory.get('memory_id')}")
        if "checksum" in memory.get("metadata", {}):
            print(f"Checksum: {memory.get('metadata', {}).get('checksum')}")

    # Load the memory system using create_memory_system
    print("\nLoading memory system using create_memory_system...")
    memory_system = create_memory_system(
        use_mock_redis=True, memory_file=memory_file, clear_db=False
    )

    # Get the loaded memories from the memory system
    memory_agent = memory_system.get_memory_agent(agent_id)
    loaded_memories = memory_agent.stm_store.get_all(agent_id)

    print(f"Loaded memories from system: {len(loaded_memories)}")

    # Compare original JSON memories with loaded memories
    print("\nComparing original JSON memories with loaded memories:")
    print("=" * 50)

    # Create maps for easier lookup
    original_map = {mem["memory_id"]: mem for mem in original_memories}
    loaded_map = {mem["memory_id"]: mem for mem in loaded_memories}

    # Check if all original memory IDs are preserved
    for memory_id, original_memory in original_map.items():
        if memory_id in loaded_map:
            compare_memories(original_memory, loaded_map[memory_id])
        else:
            print(f"❌ Memory {memory_id} not found in loaded system")

    # Check if there are any new memory IDs
    for memory_id in loaded_map:
        if memory_id not in original_map:
            print(
                f"❌ New memory {memory_id} found in loaded system that wasn't in original JSON"
            )


if __name__ == "__main__":
    main()
