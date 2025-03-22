"""Memory module for agent memory systems.

This package provides memory systems for agents to store and retrieve
their experiences, enabling learning and adaptation.
"""

from agent_memory.config import MemoryConfig
from agent_memory.core import AgentMemorySystem
from agent_memory.memory_agent import MemoryAgent

__all__ = [
    "AgentMemorySystem",  # New unified system
    "MemoryAgent",  # New memory agent
    "MemoryConfig",  # New configuration
]

"""Agent memory system package."""
