"""API for the Agent Memory System.

This package provides the public API for integrating the Agent Memory System
with agents, including memory hooks and data models.
"""

from memory.api.hooks import BaseAgent, install_memory_hooks, with_memory
from memory.api.models import ActionData, ActionResult, AgentState

# Make type definitions available
from memory.api.types import (
    MemoryChangeRecord,
    MemoryEmbeddings,
    MemoryEntry,
    MemoryImportanceScore,
    MemoryMetadata,
    MemoryStatistics,
    MemoryStore,
    MemoryTier,
    MemoryTypeFilter,
)

__all__ = [
    # Hooks API
    "BaseAgent",
    "install_memory_hooks",
    "with_memory",
    # Models API
    "AgentState",
    "ActionData",
    "ActionResult",
    # Types API
    "MemoryEntry",
    "MemoryMetadata",
    "MemoryEmbeddings",
    "MemoryChangeRecord",
    "MemoryStatistics",
    "MemoryTier",
    "MemoryTypeFilter",
    "MemoryImportanceScore",
    "MemoryStore",
]
