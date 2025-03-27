"""Key management utilities for storage backends.

This module provides consistent key construction patterns for different
storage backends to minimize duplication and ensure consistency.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def construct_memory_key(prefix: str, agent_id: str, memory_id: str) -> str:
    """Construct a key for a memory entry.
    
    Creates a consistent key format for memory entries across different storage backends.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        memory_id: Memory identifier
        
    Returns:
        Formatted key string
    """
    return f"{prefix}:{agent_id}:memory:{memory_id}"


def construct_agent_memories_key(prefix: str, agent_id: str) -> str:
    """Construct a key for an agent's memories collection.
    
    Creates a consistent key format for agent memory collections.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        
    Returns:
        Formatted key string
    """
    return f"{prefix}:{agent_id}:memories"


def construct_timeline_key(prefix: str, agent_id: str) -> str:
    """Construct a key for an agent's memory timeline index.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        
    Returns:
        Formatted key string
    """
    return f"{prefix}:{agent_id}:timeline"


def construct_importance_key(prefix: str, agent_id: str) -> str:
    """Construct a key for an agent's memory importance index.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        
    Returns:
        Formatted key string
    """
    return f"{prefix}:{agent_id}:importance"


def construct_vector_key(prefix: str, agent_id: str, memory_id: str) -> str:
    """Construct a key for a memory's vector embedding.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        memory_id: Memory identifier
        
    Returns:
        Formatted key string
    """
    return f"{prefix}:{agent_id}:vector:{memory_id}"


def construct_agent_prefix(prefix: str, agent_id: str) -> str:
    """Construct a key prefix for all an agent's data.
    
    Args:
        prefix: Namespace or prefix for the key
        agent_id: Agent identifier
        
    Returns:
        Formatted key prefix string
    """
    return f"{prefix}:{agent_id}"


def parse_memory_key(key: str, prefix: str) -> tuple[Optional[str], Optional[str]]:
    """Extract agent_id and memory_id from a memory key.
    
    Args:
        key: The full memory key to parse
        prefix: The namespace or prefix used in the key
        
    Returns:
        Tuple of (agent_id, memory_id) or (None, None) if parsing fails
    """
    try:
        parts = key.split(":")
        if len(parts) >= 4 and parts[0] == prefix and parts[2] == "memory":
            return parts[1], parts[3]
    except Exception as e:
        logger.warning(f"Failed to parse memory key '{key}': {e}")
    
    return None, None 