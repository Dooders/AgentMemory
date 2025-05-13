"""Unit tests for saving memory systems to JSON files.

This test suite covers the functionality of saving memory system data to JSON files,
validating their structure, and ensuring the proper serialization of memory agents and their memories.

The tests focus on creating memory systems, adding content, and then saving them to JSON files.
Each test focuses on a different scenario for saving memory system data.
"""

import json
import logging
import os
from pathlib import Path
import tempfile
import traceback

import pytest

from memory.config import MemoryConfig
from memory.core import AgentMemorySystem
from memory.schema import validate_memory_system_json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@pytest.fixture(autouse=True)
def cleanup_memory_system():
    """Reset the AgentMemorySystem singleton before and after each test."""
    AgentMemorySystem._instance = None
    yield
    AgentMemorySystem._instance = None


def test_save_empty_memory_system(cleanup_memory_system):
    """Test saving an empty memory system with default configuration."""
    # Create an empty memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem(config)

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        filepath = temp_file.name
    
    logger.info(f"Using temporary file: {filepath}")

    try:
        # Save the memory system to the temporary file
        success = memory_system.save_to_json(filepath)
        logger.info(f"save_to_json result: {success}")
        assert success, "Failed to save empty memory system"

        # Verify the file exists
        file_exists = os.path.exists(filepath)
        logger.info(f"File exists: {file_exists}")
        assert file_exists, "Output file was not created"

        # Verify the JSON structure is valid
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded JSON data with keys: {data.keys()}")

        assert "config" in data, "Saved JSON doesn't contain config section"
        assert "agents" in data, "Saved JSON doesn't contain agents section"
        assert isinstance(data["agents"], dict), "Agents section is not a dictionary"
        
        # Validate schema
        schema_valid = validate_memory_system_json(data)
        logger.info(f"Schema validation result: {schema_valid}")
        assert schema_valid, "Saved JSON fails schema validation"

    except Exception as e:
        logger.error(f"Exception in test: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_save_memory_system_with_agent(cleanup_memory_system):
    """Test saving a memory system with one agent and some memories."""
    # Create a memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem(config)

    # Create an agent and add some memories
    agent_id = "test_agent"
    memory_agent = memory_system.get_memory_agent(agent_id)
    logger.info(f"Created memory agent with ID: {agent_id}")

    # Add state memory
    state_result = memory_system.store_agent_state(
        agent_id, 
        {"name": "Test Agent", "status": "active"}, 
        step_number=1,
        priority=0.8
    )
    logger.info(f"store_agent_state result: {state_result}")

    # Add interaction memory
    interaction_result = memory_system.store_agent_interaction(
        agent_id,
        {"agent_id": agent_id, "other_agent": "user", "content": "Hello world"},
        step_number=2,
        priority=0.9
    )
    logger.info(f"store_agent_interaction result: {interaction_result}")

    # Check if memories were stored
    try:
        stm_memories = memory_agent.stm_store.get_all(agent_id)
        logger.info(f"STM memories count: {len(stm_memories)}")
        if stm_memories:
            logger.info(f"First STM memory keys: {stm_memories[0].keys()}")
    except Exception as e:
        logger.error(f"Error retrieving STM memories: {e}")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        filepath = temp_file.name
    
    logger.info(f"Using temporary file: {filepath}")

    try:
        # Save the memory system to the temporary file
        logger.info("Calling save_to_json...")
        success = memory_system.save_to_json(filepath)
        logger.info(f"save_to_json result: {success}")
        assert success, "Failed to save memory system with agent"

        # Verify the file exists
        file_exists = os.path.exists(filepath)
        logger.info(f"File exists after save: {file_exists}")
        assert file_exists, "Output file was not created"

        # Verify the JSON structure is valid
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded JSON data with keys: {data.keys()}")
            if "agents" in data:
                logger.info(f"Agents in data: {list(data['agents'].keys())}")

        assert "config" in data, "Saved JSON doesn't contain config section"
        assert "agents" in data, "Saved JSON doesn't contain agents section"
        assert agent_id in data["agents"], f"Agent {agent_id} not found in saved data"
        
        # Verify memories were saved
        agent_data = data["agents"][agent_id]
        assert "memories" in agent_data, "Agent data doesn't contain memories"
        assert len(agent_data["memories"]) >= 2, "Not all memories were saved"
        
        # Verify schema validation
        schema_valid = validate_memory_system_json(data)
        logger.info(f"Schema validation result: {schema_valid}")
        assert schema_valid, "Saved JSON fails schema validation"

    except Exception as e:
        logger.error(f"Exception in test: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_save_memory_system_with_multiple_agents(cleanup_memory_system):
    """Test saving a memory system with multiple agents."""
    # Create a memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem(config)

    # Create multiple agents and add memories
    agent_ids = ["agent1", "agent2", "agent3"]
    
    for i, agent_id in enumerate(agent_ids):
        # Add state memory for each agent
        memory_system.store_agent_state(
            agent_id, 
            {"name": f"Agent {i+1}", "status": "active", "index": i}, 
            step_number=i+1,
            priority=0.7 + (i * 0.1)
        )

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        filepath = temp_file.name

    try:
        # Save the memory system to the temporary file
        success = memory_system.save_to_json(filepath)
        assert success, "Failed to save memory system with multiple agents"

        # Verify the file exists
        assert os.path.exists(filepath), "Output file was not created"

        # Verify the JSON structure is valid
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify all agents were saved
        for agent_id in agent_ids:
            assert agent_id in data["agents"], f"Agent {agent_id} not found in saved data"
            
            # Verify memories for each agent
            agent_data = data["agents"][agent_id]
            assert "memories" in agent_data, f"Agent {agent_id} data doesn't contain memories"
            assert len(agent_data["memories"]) >= 1, f"No memories saved for agent {agent_id}"
        
        # Verify schema validation
        assert validate_memory_system_json(data), "Saved JSON fails schema validation"

    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_save_and_load_roundtrip(cleanup_memory_system):
    """Test round-trip save and load of a memory system."""
    # Create an initial memory system
    config = MemoryConfig()
    original_system = AgentMemorySystem(config)

    # Create an agent and add some memories
    agent_id = "test_agent"
    memory_agent = original_system.get_memory_agent(agent_id)

    # Add state memory with some content to check after reload
    test_content = {"name": "Test Agent", "status": "active", "data": {"key1": "value1", "key2": 42}}
    
    # Explicitly use state type
    state_result = original_system.store_agent_state(
        agent_id, 
        test_content, 
        step_number=1,
        priority=0.8
    )
    logger.info(f"store_agent_state result: {state_result}")
    
    # Verify the memory was stored with the correct type
    try:
        stm_memories = memory_agent.stm_store.get_all(agent_id)
        logger.info(f"Before save: STM memories count: {len(stm_memories)}")
        for i, mem in enumerate(stm_memories):
            logger.info(f"Memory {i} type: {mem.get('type')}")
            logger.info(f"Memory {i} metadata.memory_type: {mem.get('metadata', {}).get('memory_type')}")
    except Exception as e:
        logger.error(f"Error retrieving STM memories: {e}")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        filepath = temp_file.name
    
    logger.info(f"Using temporary file: {filepath}")

    try:
        # Save the memory system to the temporary file
        success = original_system.save_to_json(filepath)
        assert success, "Failed to save memory system"
        
        # Examine the saved file
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Log the memory types in the saved file
        if "agents" in data and agent_id in data["agents"] and "memories" in data["agents"][agent_id]:
            memories = data["agents"][agent_id]["memories"]
            logger.info(f"Saved file has {len(memories)} memories")
            for i, mem in enumerate(memories):
                logger.info(f"Saved memory {i} type: {mem.get('type')}")
                logger.info(f"Saved memory {i} metadata.memory_type: {mem.get('metadata', {}).get('memory_type')}")

        # Reset the singleton to enable loading a new instance
        AgentMemorySystem._instance = None

        # Load the memory system back from the file
        loaded_system = AgentMemorySystem.load_from_json(filepath, use_mock_redis=True)
        print(f"Load result: {loaded_system}")
        if loaded_system is None:
            print("Failed to load memory system, checking what went wrong...")
            try:
                AgentMemorySystem._instance = None
                AgentMemorySystem.load_from_json(filepath, use_mock_redis=True)
            except Exception as e:
                print(f"Error during load: {e}")
        assert loaded_system is not None, "Failed to load memory system"

        # Verify the agent exists in the loaded system
        assert agent_id in loaded_system.agents, f"Agent {agent_id} not found in loaded system"
        
        # Get all memories for the agent
        loaded_agent = loaded_system.get_memory_agent(agent_id)
        stm_memories = loaded_agent.stm_store.get_all(agent_id)
        im_memories = loaded_agent.im_store.get_all(agent_id)
        ltm_memories = loaded_agent.ltm_store.get_all(agent_id)
        
        all_memories = stm_memories + im_memories + ltm_memories
        logger.info(f"After load: All memories count: {len(all_memories)}")
        logger.info(f"STM: {len(stm_memories)}, IM: {len(im_memories)}, LTM: {len(ltm_memories)}")
        
        # Log memory types after loading
        for i, mem in enumerate(all_memories):
            logger.info(f"Loaded memory {i} type: {mem.get('type')}")
            logger.info(f"Loaded memory {i} step_number: {mem.get('step_number')}")
            logger.info(f"Loaded memory {i} metadata.memory_type: {mem.get('metadata', {}).get('memory_type')}")
        
        assert len(all_memories) > 0, "No memories were loaded"

        # Find a memory with the test content
        matching_memory = None
        for memory in all_memories:
            memory_type = memory.get("type", "generic")
            metadata_type = memory.get("metadata", {}).get("memory_type", "generic")
            step = memory.get("step_number")
            
            logger.info(f"Checking memory: type={memory_type}, metadata_type={metadata_type}, step={step}")
            
            # Check both type and metadata.memory_type since they might be different
            if (memory_type == "state" or metadata_type == "state") and step == 1:
                matching_memory = memory
                logger.info("Found matching memory!")
                break

        assert matching_memory is not None, "Could not find the test memory"
        
        # Verify the content was preserved
        content = matching_memory.get("content", {})
        assert "name" in content, "Memory content missing 'name' field"
        assert content["name"] == test_content["name"], "Memory content 'name' doesn't match"
        assert "data" in content, "Memory content missing 'data' field"
        assert content["data"].get("key1") == test_content["data"]["key1"], "Nested content doesn't match"
        assert content["data"].get("key2") == test_content["data"]["key2"], "Nested content doesn't match"

    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_save_with_invalid_path(cleanup_memory_system):
    """Test saving to an invalid file path."""
    # Create a memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem(config)

    # Try to save to an invalid path with characters or structures invalid for the current OS
    if os.name == "nt":  # Windows
        invalid_path = os.path.join(tempfile.gettempdir(), "invalid*path?", "file|name<>.json")
    else:  # POSIX (Linux/macOS)
        invalid_path = os.path.join(tempfile.gettempdir(), "invalid_path", "\0file.json")  # Null byte is invalid
    
    logger.info(f"Testing invalid path: {invalid_path}")
    
    # The method should handle this gracefully
    success = memory_system.save_to_json(invalid_path)
    logger.info(f"save_to_json with invalid path result: {success}")
    assert not success, "save_to_json should return False for invalid paths"


def test_save_with_complex_memory_content(cleanup_memory_system):
    """Test saving a memory system with complex memory content."""
    # Create a memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem(config)

    # Create an agent 
    agent_id = "test_agent"
    memory_agent = memory_system.get_memory_agent(agent_id)

    # Add state memory with complex nested content
    complex_content = {
        "name": "Complex Test",
        "nested": {
            "level1": {
                "level2": {
                    "level3": "deep value",
                    "numbers": [1, 2, 3, 4, 5],
                    "mixed": [{"a": 1}, {"b": 2}, None, False, True]
                }
            }
        },
        "arrays": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        "special_values": {
            "null": None,
            "boolean": True,
            "empty_string": "",
            "zero": 0
        }
    }
    
    memory_system.store_agent_state(
        agent_id, 
        complex_content, 
        step_number=1,
        priority=0.8
    )

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        filepath = temp_file.name

    try:
        # Save the memory system to the temporary file
        success = memory_system.save_to_json(filepath)
        assert success, "Failed to save memory system with complex content"

        # Verify the file exists and can be parsed
        assert os.path.exists(filepath), "Output file was not created"
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify schema validation still passes with complex content
        assert validate_memory_system_json(data), "Saved JSON fails schema validation"
        
        # Optional: Load back and verify complex content was preserved
        AgentMemorySystem._instance = None
        loaded_system = AgentMemorySystem.load_from_json(filepath, use_mock_redis=True)
        print(f"Load result: {loaded_system}")
        if loaded_system is None:
            print("Failed to load memory system, checking what went wrong...")
            try:
                AgentMemorySystem._instance = None
                AgentMemorySystem.load_from_json(filepath, use_mock_redis=True)
            except Exception as e:
                print(f"Error during load: {e}")
        assert loaded_system is not None, "Failed to load memory system"

    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 