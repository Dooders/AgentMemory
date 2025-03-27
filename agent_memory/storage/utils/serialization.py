"""Serialization utilities for memory storage.

This module provides common serialization and deserialization functions
used across different storage backends to handle memory entries.
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypeVar, Union, Type, cast

from agent_memory.storage.models import (
    BaseMemoryEntry,
    MemoryEntry,
    MemoryMetadata,
    MemoryEmbeddings,
)

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=Dict[str, Any])


def serialize_memory_entry(memory_entry: Dict[str, Any]) -> str:
    """Serialize a memory entry to JSON string.
    
    Args:
        memory_entry: Memory entry to serialize
        
    Returns:
        JSON string representation of the memory entry
        
    Raises:
        ValueError: If serialization fails
    """
    try:
        # Handle numpy arrays or other non-serializable types
        sanitized_entry = _sanitize_for_json(memory_entry)
        return json.dumps(sanitized_entry)
    except Exception as e:
        logger.error(f"Failed to serialize memory entry: {e}")
        raise ValueError(f"Failed to serialize memory entry: {e}")


def deserialize_memory_entry(data: str) -> Dict[str, Any]:
    """Deserialize a JSON string to a memory entry.
    
    Args:
        data: JSON string to deserialize
        
    Returns:
        Deserialized memory entry
        
    Raises:
        ValueError: If deserialization fails
    """
    try:
        return json.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize memory entry: {e}")
        raise ValueError(f"Failed to deserialize memory entry: {e}")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization.
    
    Handles numpy arrays, sets, and other non-serializable types.
    
    Args:
        obj: Object to sanitize
        
    Returns:
        JSON-serializable version of the object
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def hash_to_memory_entry(hash_data: Dict[str, str]) -> Dict[str, Any]:
    """Convert a Redis hash to a memory entry.
    
    Args:
        hash_data: Redis hash data
        
    Returns:
        Memory entry dictionary
    """
    memory_entry: Dict[str, Any] = {}
    
    # Basic properties
    if "memory_id" in hash_data:
        memory_entry["memory_id"] = hash_data["memory_id"]
    if "agent_id" in hash_data:
        memory_entry["agent_id"] = hash_data["agent_id"]
    if "timestamp" in hash_data:
        memory_entry["timestamp"] = float(hash_data["timestamp"])
    if "memory_type" in hash_data:
        memory_entry["memory_type"] = hash_data["memory_type"]
    if "step_number" in hash_data:
        memory_entry["step_number"] = int(hash_data["step_number"])
        
    # Content (complex object stored as JSON)
    if "content" in hash_data and hash_data["content"]:
        try:
            memory_entry["content"] = json.loads(hash_data["content"])
        except Exception as e:
            logger.warning(f"Failed to deserialize content: {e}")
            memory_entry["content"] = hash_data["content"]
            
    # Metadata (complex object stored as JSON)
    if "metadata" in hash_data and hash_data["metadata"]:
        try:
            memory_entry["metadata"] = json.loads(hash_data["metadata"])
        except Exception as e:
            logger.warning(f"Failed to deserialize metadata: {e}")
            memory_entry["metadata"] = {}
            
    # Embeddings (stored as JSON)
    if "embeddings" in hash_data and hash_data["embeddings"]:
        try:
            memory_entry["embeddings"] = json.loads(hash_data["embeddings"])
        except Exception as e:
            logger.warning(f"Failed to deserialize embeddings: {e}")
            memory_entry["embeddings"] = {}
            
    return memory_entry


def memory_entry_to_hash(memory_entry: Dict[str, Any]) -> Dict[str, str]:
    """Convert a memory entry to a Redis hash.
    
    Args:
        memory_entry: Memory entry dictionary
        
    Returns:
        Redis hash dictionary with string values
    """
    hash_data: Dict[str, str] = {}
    
    # Basic properties
    if "memory_id" in memory_entry:
        hash_data["memory_id"] = str(memory_entry["memory_id"])
    if "agent_id" in memory_entry:
        hash_data["agent_id"] = str(memory_entry["agent_id"])
    if "timestamp" in memory_entry:
        hash_data["timestamp"] = str(memory_entry["timestamp"])
    if "memory_type" in memory_entry:
        hash_data["memory_type"] = str(memory_entry["memory_type"])
    if "step_number" in memory_entry:
        hash_data["step_number"] = str(memory_entry["step_number"])
        
    # Complex objects (serialize to JSON)
    if "content" in memory_entry:
        hash_data["content"] = json.dumps(_sanitize_for_json(memory_entry["content"]))
        
    if "metadata" in memory_entry:
        hash_data["metadata"] = json.dumps(_sanitize_for_json(memory_entry["metadata"]))
        
    if "embeddings" in memory_entry:
        hash_data["embeddings"] = json.dumps(_sanitize_for_json(memory_entry["embeddings"]))
        
    return hash_data 