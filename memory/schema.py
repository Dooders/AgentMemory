"""Schema definitions for the Tiered Adaptive Semantic Memory (TASM) system.

This module contains JSON schema definitions for validating memory system data structures,
especially for serialization and deserialization operations.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# JSON Schema for memory system serialization/deserialization
MEMORY_SYSTEM_SCHEMA = {
    "type": "object",
    "required": ["config", "agents"],
    "properties": {
        "config": {
            "type": "object",
            "description": "Memory system configuration",
            "properties": {
                "logging_level": {"type": "string"},
                "cleanup_interval": {"type": "integer"},
                "enable_memory_hooks": {"type": "boolean"},
                # Other config properties would be listed here
            }
        },
        "agents": {
            "type": "object",
            "description": "Map of agent IDs to agent memory data",
            "additionalProperties": {
                "type": "object",
                "required": ["agent_id", "memories"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "memories": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/memory_entry"}
                    }
                }
            }
        }
    },
    "definitions": {
        "memory_entry": {
            "type": "object",
            "required": ["memory_id", "agent_id", "content", "metadata", "type"],
            "properties": {
                "memory_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "step_number": {"type": "integer"},
                "timestamp": {"type": "integer"},
                "content": {"type": "object"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "creation_time": {"type": "integer"},
                        "last_access_time": {"type": "integer"},
                        "compression_level": {"type": "integer"},
                        "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "retrieval_count": {"type": "integer"},
                        "memory_type": {"type": "string", "enum": ["state", "interaction", "action", "generic"]},
                        "current_tier": {"type": "string", "enum": ["stm", "im", "ltm"]}
                    }
                },
                "type": {"type": "string", "enum": ["state", "interaction", "action", "generic"]},
                "embeddings": {
                    "type": "object",
                    "properties": {
                        "full_vector": {"type": "array", "items": {"type": "number"}},
                        "compressed_vector": {"type": "array", "items": {"type": "number"}},
                        "abstract_vector": {"type": "array", "items": {"type": "number"}}
                    }
                }
            }
        }
    }
}

def validate_memory_system_json(data: Dict[str, Any]) -> bool:
    """Validate memory system JSON data against the schema.
    
    Args:
        data: The JSON data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # This is a placeholder for actual JSON Schema validation
        # In a real implementation, you would use a library like jsonschema
        
        # Basic structure validation
        if not isinstance(data, dict):
            logger.error("Memory system data must be a dictionary")
            return False
            
        if "config" not in data or "agents" not in data:
            logger.error("Memory system data must contain 'config' and 'agents' keys")
            return False
            
        # Config validation
        if not isinstance(data["config"], dict):
            logger.error("Config must be a dictionary")
            return False
            
        # Agents validation
        if not isinstance(data["agents"], dict):
            logger.error("Agents must be a dictionary")
            return False
            
        # Validate each agent
        for agent_id, agent_data in data["agents"].items():
            if not isinstance(agent_data, dict):
                logger.error(f"Agent data for {agent_id} must be a dictionary")
                return False
                
            if "agent_id" not in agent_data or "memories" not in agent_data:
                logger.error(f"Agent data for {agent_id} must contain 'agent_id' and 'memories' keys")
                return False
                
            if not isinstance(agent_data["memories"], list):
                logger.error(f"Memories for agent {agent_id} must be a list")
                return False
                
            # Validate each memory
            for i, memory in enumerate(agent_data["memories"]):
                if not isinstance(memory, dict):
                    logger.error(f"Memory {i} for agent {agent_id} must be a dictionary")
                    return False
                    
                required_keys = ["memory_id", "agent_id", "content", "metadata", "type"]
                for key in required_keys:
                    if key not in memory:
                        logger.error(f"Memory {i} for agent {agent_id} is missing required key '{key}'")
                        return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating memory system JSON: {e}")
        return False

def get_schema_as_json() -> str:
    """Get the memory system schema as a JSON string.
    
    Returns:
        JSON string representation of the schema
    """
    return json.dumps(MEMORY_SYSTEM_SCHEMA, indent=2)

def save_schema_to_file(filepath: str) -> bool:
    """Save the memory system schema to a JSON file.
    
    Args:
        filepath: Path to save the schema
        
    Returns:
        True if successful
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(MEMORY_SYSTEM_SCHEMA, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving schema to file: {e}")
        return False

def load_and_validate(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a memory system JSON file and validate it against the schema.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Validated data if valid, None otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if validate_memory_system_json(data):
            return data
        else:
            logger.error(f"Invalid memory system JSON in {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading memory system JSON from {filepath}: {e}")
        return None 