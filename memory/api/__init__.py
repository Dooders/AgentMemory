"""Agent Memory API Module

This module provides the public API for integrating the Agent Memory System
with agents, enabling efficient storage, retrieval, and utilization of memories
to support context-aware agent reasoning and behavior.

Key components:

1. AgentMemoryAPI: The main interface for storing, retrieving, and managing
   memories across different memory tiers (STM, IM, LTM).

2. Hooks: Utilities for installing memory hooks into agent lifecycles, enabling
   automatic memory capture and retrieval during agent operation.

3. Models: Data models for structured representation of agent states, actions,
   and results to ensure consistent memory storage and retrieval.

4. Types: Type definitions for memory entries, metadata, embeddings, and other
   core concepts used throughout the memory system.

5. Exceptions: Specialized exception classes for different categories of memory
   system errors to facilitate precise error handling.

Usage example:
```python
from memory.api import AgentMemoryAPI, AgentState

# Initialize the memory API
memory_api = AgentMemoryAPI()

# Store an agent state in memory
agent_state = AgentState(
    agent_id="agent-001",
    timestamp=1649879872,
    content={"observation": "User asked about weather", "thought": "I should check the forecast"}
)
memory_id = memory_api.store(agent_state)

# Retrieve similar memories
query = "weather forecast"
similar_memories = memory_api.search_by_content(query, limit=3)

# Use memories to inform agent's response
for memory in similar_memories:
    print(f"Related memory: {memory.content}")
```

For automatic memory integration with agents:
```python
from memory.api import with_memory, BaseAgent

@with_memory
class MyAgent(BaseAgent):
    def process_input(self, user_input):
        # Memories automatically captured and made available
        relevant_memories = self.memory.retrieve_relevant(user_input)
        return self.generate_response(user_input, context=relevant_memories)
```
"""

# Agent hooks and decorators
from memory.api.hooks import BaseAgent, install_memory_hooks, with_memory

# Main API class
from memory.api.memory_api import (  # Exception classes; Utility functions
    AgentMemoryAPI,
    MemoryAPIException,
    MemoryConfigException,
    MemoryMaintenanceException,
    MemoryRetrievalException,
    MemoryStoreException,
    cacheable,
    log_with_context,
)

# Data models
from memory.api.models import ActionData, ActionResult, AgentState

# Type definitions
from memory.api.types import (
    ConfigUpdate,
    MemoryChangeRecord,
    MemoryEmbeddings,
    MemoryEntry,
    MemoryImportanceScore,
    MemoryMetadata,
    MemoryStatistics,
    MemoryStore,
    MemoryTier,
    MemoryTypeFilter,
    QueryResult,
    SimilaritySearchResult,
)

__all__ = [
    # Main API
    "AgentMemoryAPI",
    # Exception classes
    "MemoryAPIException",
    "MemoryConfigException",
    "MemoryMaintenanceException",
    "MemoryRetrievalException",
    "MemoryStoreException",
    # Utility functions
    "cacheable",
    "log_with_context",
    # Hooks API
    "BaseAgent",
    "install_memory_hooks",
    "with_memory",
    # Models API
    "AgentState",
    "ActionData",
    "ActionResult",
    # Types API
    "ConfigUpdate",
    "MemoryChangeRecord",
    "MemoryEmbeddings",
    "MemoryEntry",
    "MemoryImportanceScore",
    "MemoryMetadata",
    "MemoryStatistics",
    "MemoryStore",
    "MemoryTier",
    "MemoryTypeFilter",
    "QueryResult",
    "SimilaritySearchResult",
]
