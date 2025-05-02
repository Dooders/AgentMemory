# Agent Memory API

The `memory/api` module provides a comprehensive interface for integrating the Agent Memory System with AI agents, enabling efficient storage, retrieval, and utilization of memories to support context-aware agent reasoning and behavior.

## Overview

The Agent Memory System implements a tiered memory architecture inspired by human cognitive systems:

- **Short-Term Memory (STM)**: Recent, high-fidelity memories with detailed information
- **Intermediate Memory (IM)**: Medium-term memories with moderate compression
- **Long-Term Memory (LTM)**: Persistent, compressed memories retaining core information

This architecture allows agents to efficiently manage memories with different levels of detail and importance across varying time horizons.

## Module Components

### Main API Class

[`memory_api.py`](memory_api.py) - Provides the primary interface class `AgentMemoryAPI` for interacting with the memory system:
- Store agent states, actions, and interactions
- Retrieve memories by various criteria (ID, time range, attributes)
- Perform semantic search across memory tiers
- Manage memory lifecycle and maintenance

### Agent Integration

[`hooks.py`](hooks.py) - Offers decorators and utility functions for automatic memory integration:
- `install_memory_hooks`: Class decorator to add memory capabilities to agent classes
- `with_memory`: Instance decorator for adding memory to existing agent instances
- `BaseAgent`: Minimal interface with standard lifecycle methods for memory-aware agents

### Data Models

[`models.py`](models.py) - Defines structured representations of agent data:
- `AgentState`: Standardized representation of an agent's state
- `ActionData`: Record of an agent action with associated states and metrics
- `ActionResult`: Lightweight result of an action execution

### Type Definitions

[`types.py`](types.py) - Establishes core type definitions for the memory system:
- Memory entry structures (metadata, embeddings, content)
- Memory tiers and filtering types
- Statistics and query result types
- Protocol definitions for memory stores

## Getting Started

### Basic Usage

```python
from memory.api import AgentMemoryAPI

# Initialize the memory API
memory_api = AgentMemoryAPI()

# Store an agent state
state_data = {
    "agent_id": "agent-001",
    "step_number": 42,
    "content": {
        "observation": "User asked about weather", 
        "thought": "I should check the forecast"
    }
}
memory_id = memory_api.store_agent_state("agent-001", state_data, step_number=42)

# Retrieve similar memories
query = "weather forecast"
similar_memories = memory_api.search_by_content("agent-001", query, k=3)

# Use memories to inform agent's response
for memory in similar_memories:
    print(f"Related memory: {memory['contents']}")
```

### Automatic Memory Integration

```python
from memory.api import install_memory_hooks, BaseAgent

@install_memory_hooks
class MyAgent(BaseAgent):
    def __init__(self, config=None, agent_id=None):
        super().__init__(config, agent_id)
        # Agent-specific initialization
        
    def act(self, observation):
        # Memory hooks automatically capture state before this method
        self.step_number += 1
        action_result = self._process(observation)
        # Memory hooks automatically capture state after this method
        return action_result
        
    def get_state(self):
        # Return current agent state
        state = super().get_state()
        state.extra_data["custom_field"] = self.some_internal_state
        return state

# Create an agent with memory enabled
agent = MyAgent(agent_id="agent-001")

# Use the agent normally - memories are created automatically
result = agent.act({"user_input": "What's the weather today?"})
```

## Advanced Features

### Memory Maintenance

```python
# Run memory maintenance to consolidate and optimize memories
memory_api.force_memory_maintenance("agent-001")

# Get memory statistics
stats = memory_api.get_memory_statistics("agent-001")
print(f"Total memories: {stats['total_memories']}")
print(f"STM: {stats['stm_count']}, IM: {stats['im_count']}, LTM: {stats['ltm_count']}")
```

### Memory Search

```python
# Search by content similarity
similar_memories = memory_api.search_by_content(
    agent_id="agent-001",
    content_query="user asked about calendar appointments",
    k=5
)

# Retrieve memories by time range
recent_memories = memory_api.retrieve_by_time_range(
    agent_id="agent-001",
    start_step=100,
    end_step=120
)

# Retrieve memories by attributes
filtered_memories = memory_api.retrieve_by_attributes(
    agent_id="agent-001",
    attributes={"action_type": "calendar_query"}
)
```

### Memory Configuration

```python
from memory.api import AgentMemoryAPI
from memory.config import MemoryConfig

# Custom configuration
config = MemoryConfig(
    stm_config={"memory_limit": 1000},
    im_config={"memory_limit": 10000},
    ltm_config={"memory_limit": 100000}
)

# Initialize API with custom configuration
memory_api = AgentMemoryAPI(config)

# Update configuration
memory_api.configure_memory_system({
    "stm_config": {"memory_limit": 2000}
})
```

## Error Handling

```python
from memory.api import AgentMemoryAPI
from memory.api.memory_api import MemoryStoreException, MemoryRetrievalException

memory_api = AgentMemoryAPI()

try:
    memory = memory_api.retrieve_state_by_id("agent-001", "non_existent_id")
except MemoryRetrievalException as e:
    print(f"Memory retrieval error: {e}")

try:
    memories = memory_api.search_by_content("agent-001", "query", k=-1)
except MemoryConfigException as e:
    print(f"Configuration error: {e}")
``` 