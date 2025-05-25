"""
Identity utilities for the memory system.

This module provides helper functions for generating and managing unique identifiers
for memories and related entities in the system. It ensures consistent ID generation
across different types of memories and provides a standardized way to create
unique, traceable identifiers.

Usage Examples:
    Basic memory ID generation:
        >>> from memory.utils.identity import generate_memory_id
        >>> memory_type = 'action'
        >>> agent_id = ABC123
        >>> step_number = 5
        >>> memory_id = generate_memory_id(memory_type, agent_id, step_number)
        >>> print(memory_id)
        'action_ABC123_5'

    Different memory types:
        >>> memory_type = 'interaction'
        >>> step_number = 10
        >>> memory_id = generate_memory_id(memory_type, agent_id, step_number)
        >>> print(memory_id)
        'interaction_ABC123_10'

    State memory:
        >>> memory_type = 'state'
        >>> step_number = 15
        >>> memory_id = generate_memory_id(memory_type, agent_id, step_number)
        >>> print(memory_id)
        'state_ABC123_15'

    Already formatted ID:
        >>> memory_type = 'action'
        >>> agent_id = 'action_ABC123_5'  # Already in correct format
        >>> step_number = 5
        >>> memory_id = generate_memory_id(memory_type, agent_id, step_number)
        >>> print(memory_id)
        'action_ABC123_5'
"""

import re
from typing import Union


def generate_memory_id(
    memory_type: str, agent_id: Union[int, str], step_number: int
) -> str:
    """
    Generate a unique memory ID based on memory type, agent ID, and step number.
    If the agent_id is already in the correct format, it will be returned as is.

    Args:
        memory_type: Type of memory (e.g., 'action', 'interaction', 'state')
        agent_id: ID from the agent's entity (e.g., action_id, interaction_id, state_id)
        step_number: Step number when the memory was created

    Returns:
        A unique memory ID string in the format: memory_type_agent_id_step_number
        If agent_id is already in this format, it will be returned unchanged.

    Examples:
        >>> generate_memory_id('action', 123, 5)
        'action_123_5'
        >>> generate_memory_id('action', 'action_123_5', 5)
        'action_123_5'
    """
    # Check if agent_id is already in the correct format
    if isinstance(agent_id, str):
        pattern = f"^{memory_type}_[^_]+_{step_number}$"
        if re.match(pattern, agent_id):
            return agent_id

    # Convert agent_id to string if it's not already
    agent_id_str = str(agent_id)

    # Build and return the memory ID
    return f"{memory_type}_{agent_id_str}_{step_number}"
