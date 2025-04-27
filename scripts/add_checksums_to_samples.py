"""
Add checksums to sample memory files.

This script processes all memory JSON files in the demos/memory_samples directory,
adds checksums to each memory entry's metadata, and saves the updated files.
"""

import json
import os
from pathlib import Path
import sys

# Add project root to path so we can import the memory module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory.utils.checksums import add_checksum_to_memory

def process_memory_file(file_path):
    """Process a memory file and add checksums to all memory entries."""
    print(f"Processing {file_path}...")
    
    # Read the memory file
    with open(file_path, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)
    
    # Track if we made any changes
    changes_made = False
    
    # Process single agent memory format
    if "agents" in memory_data:
        for agent_id, agent_data in memory_data["agents"].items():
            if "memories" in agent_data:
                for i, memory_entry in enumerate(agent_data["memories"]):
                    # Add checksum if not already present
                    if "metadata" not in memory_entry or "checksum" not in memory_entry["metadata"]:
                        # Identify the content field (could be content or contents)
                        if "content" in memory_entry:
                            agent_data["memories"][i] = add_checksum_to_memory(memory_entry)
                            changes_made = True
                        elif "contents" in memory_entry:
                            agent_data["memories"][i] = add_checksum_to_memory(memory_entry)
                            changes_made = True
    
    # Process array of memories format (for other sample files)
    elif isinstance(memory_data, list):
        for i, memory_entry in enumerate(memory_data):
            # Add checksum if not already present
            if "metadata" not in memory_entry or "checksum" not in memory_entry["metadata"]:
                # Identify the content field (could be content or contents)
                if "content" in memory_entry:
                    memory_data[i] = add_checksum_to_memory(memory_entry)
                    changes_made = True
                elif "contents" in memory_entry:
                    memory_data[i] = add_checksum_to_memory(memory_entry)
                    changes_made = True
    
    # Save the updated file with checksums
    if changes_made:
        # Create backup of original file
        backup_path = str(file_path) + '.backup'
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # Write updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2)
        print(f"Updated {file_path} with checksums")
    else:
        print(f"No changes made to {file_path}")

def main():
    # Get the directory containing the sample memory files
    samples_dir = Path('demos/memory_samples')
    
    # Check if directory exists
    if not samples_dir.exists():
        print(f"Error: Directory {samples_dir} does not exist")
        return
    
    # Process all JSON files in the directory
    json_files = list(samples_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {samples_dir}")
        return
    
    print(f"Found {len(json_files)} memory files to process")
    
    for file_path in json_files:
        if '.backup' not in str(file_path):  # Skip backup files
            process_memory_file(file_path)
    
    print("Processing complete")

if __name__ == "__main__":
    main() 