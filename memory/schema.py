"""Schema definitions for the Tiered Adaptive Semantic Memory (TASM) system.

This module contains JSON schema definitions for validating memory system data structures,
especially for serialization and deserialization operations.
"""

import json
import logging
from typing import Any, Dict, Optional

import jsonschema

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
            },
            "additionalProperties": True,  # Allow additional config properties
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
                        "items": {"$ref": "#/definitions/memory_entry"},
                    },
                },
                "additionalProperties": True,  # Allow additional agent properties
            },
        },
    },
    "additionalProperties": True,  # Allow additional top-level properties
    "definitions": {
        "memory_entry": {
            "type": "object",
            "required": ["memory_id", "agent_id", "content", "metadata", "type"],
            "properties": {
                "memory_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "step_number": {"type": ["integer", "null"]},
                "timestamp": {"type": ["integer", "number", "null"]},
                "content": {"type": "object", "additionalProperties": True},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "creation_time": {"type": ["integer", "number", "null"]},
                        "last_access_time": {"type": ["integer", "number", "null"]},
                        "compression_level": {"type": ["integer", "null"]},
                        "importance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "retrieval_count": {"type": ["integer", "null"]},
                        "memory_type": {
                            "type": "string",
                            "enum": ["state", "interaction", "action", "generic"],
                        },
                        "current_tier": {
                            "type": "string",
                            "enum": ["stm", "im", "ltm"],
                        },
                    },
                    "additionalProperties": True,  # Allow additional metadata properties
                },
                "type": {
                    "type": "string",
                    "enum": ["state", "interaction", "action", "generic"],
                },
                "embeddings": {
                    "type": "object",
                    "properties": {
                        "full_vector": {"type": ["array", "null"], "items": {"type": "number"}},
                        "compressed_vector": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                        },
                        "abstract_vector": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                        },
                    },
                    "additionalProperties": True,  # Allow additional embedding properties
                },
            },
            "additionalProperties": True,  # Allow additional memory entry properties
        }
    },
}


def validate_memory_system_json(data: Dict[str, Any]) -> bool:
    """Validate memory system JSON data against the schema.

    Args:
        data: The JSON data to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate against schema
        jsonschema.validate(instance=data, schema=MEMORY_SYSTEM_SCHEMA)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Schema validation error: {e.message}")
        return False
    except Exception as e:
        logger.error(f"Error validating memory system JSON: {e}")
        print(f"UNEXPECTED ERROR: {e}")
        import traceback
        print(traceback.format_exc())
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
        with open(filepath, "w", encoding="utf-8") as f:
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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if validate_memory_system_json(data):
            return data
        else:
            logger.error(f"Invalid memory system JSON in {filepath}")
            return None

    except Exception as e:
        logger.error(f"Error loading memory system JSON from {filepath}: {e}")
        return None
