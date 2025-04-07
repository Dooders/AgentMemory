# Agent Memory System Documentation

## Overview

The Agent Memory System is a hierarchical memory management system designed for intelligent agents. It provides a sophisticated way to store, retrieve, and manage agent experiences and states across different memory tiers with varying levels of detail and persistence.

## System Architecture

```mermaid
graph TB
    subgraph "Main Components"
        AMS[Agent Memory System]
        MA[Memory Agent]
        RS[Redis Storage]
        SQ[SQLite Storage]
    end
    
    AMS --> |Creates and manages| MA
    MA --> |Uses| RS
    MA --> |Uses| SQ
    
    subgraph "Memory Tiers"
        STM[Short-Term Memory]
        IM[Intermediate Memory]
        LTM[Long-Term Memory]
    end
    
    RS --> STM
    RS --> IM
    SQ --> LTM
    
    subgraph "Supporting Components"
        CE[Compression Engine]
        AE[Autoencoder Embedding Engine]
        TE[Text Embedding Engine]
        EH[Event Hooks]
    end
    
    MA --> |Uses| CE
    MA --> |Uses| AE
    MA --> |Uses| TE
    MA --> |Manages| EH
```

The system is composed of an AgentMemorySystem that serves as the singleton entry point, managing MemoryAgent instances for individual agents. Each MemoryAgent manages three memory tiers: Short-Term Memory (Redis-based), Intermediate Memory (Redis-based with TTL), and Long-Term Memory (SQLite-based), with distinct compression levels and persistence characteristics.

## Core Components

### 1. Agent Memory System (`AgentMemorySystem` class)
The central manager and entry point for the memory system, providing agent-specific memory access.

```python
from memory.core import AgentMemorySystem
from memory.config import MemoryConfig

# Get the singleton instance with default configuration
memory_system = AgentMemorySystem.get_instance()

# Or initialize with custom configuration
config = MemoryConfig(
    cleanup_interval=100,
    stm_config=RedisSTMConfig(memory_limit=1000),
    im_config=RedisIMConfig(ttl=604800)  # 7 days
)
memory_system = AgentMemorySystem.get_instance(config)
```

### 2. Memory Agent (`MemoryAgent` class)
Manages memory operations for a specific agent across all memory tiers.

```python
# The AgentMemorySystem creates and manages MemoryAgent instances
memory_agent = memory_system.get_memory_agent(agent_id="agent-123")
```

### 3. Memory Tiers
- **Short-Term Memory (STM)**: Redis-based, full resolution storage with 24-hour TTL
- **Intermediate Memory (IM)**: Redis-based with 7-day TTL and level 1 compression
- **Long-Term Memory (LTM)**: SQLite-based with level 2 compression

### 4. Supporting Components
- **Compression Engine**: Handles data compression for different memory tiers
- **Autoencoder Embedding Engine**: Neural network for embedding generation
- **Text Embedding Engine**: For text-based embeddings (optional)
- **Event Hook System**: For custom event handling

## Key Features

### 1. Memory Storage
```python
# Store a new state
memory_system.store_agent_state(
    agent_id="agent-123",
    state_data={"position": [x, y], "resources": 42},
    step_number=1234,
    priority=0.8
)

# Store an interaction
memory_system.store_agent_interaction(
    agent_id="agent-123",
    interaction_data={"type": "collision", "with": "agent-456"},
    step_number=1234,
    priority=0.9
)

# Store an action
memory_system.store_agent_action(
    agent_id="agent-123",
    action_data={"action": "move", "direction": "north"},
    step_number=1234,
    priority=0.7
)
```

### 2. Memory Retrieval
```python
# Retrieve similar states
similar_states = memory_system.retrieve_similar_states(
    agent_id="agent-123",
    query_state=current_state,
    k=5,
    threshold=0.6,
    context_weights={"position": 0.8, "resources": 0.2}
)

# Retrieve by time range
historical_states = memory_system.retrieve_by_time_range(
    agent_id="agent-123",
    start_step=1000,
    end_step=2000,
    memory_type="state"
)

# Retrieve by attributes
matching_states = memory_system.retrieve_by_attributes(
    agent_id="agent-123",
    attributes={"resource_level": "high"},
    memory_type="state"
)

# Advanced: hybrid retrieval
hybrid_results = memory_system.hybrid_retrieve(
    agent_id="agent-123",
    query_state=current_state,
    k=5,
    vector_weight=0.7,
    attribute_weight=0.3
)
```

### 3. Event Hooks
```python
# Register a custom hook
memory_system.register_memory_hook(
    agent_id="agent-123",
    event_type="critical_resource_change",
    hook_function=custom_hook_function,
    priority=7
)

# Trigger an event
memory_system.trigger_memory_event(
    agent_id="agent-123",
    event_type="critical_resource_change",
    event_data={"resource_delta": -50}
)
```

## Memory Entry Structure

```mermaid
classDiagram
    class MemoryEntry {
        +string memory_id
        +string agent_id
        +int step_number
        +timestamp timestamp
        +Dict contents
        +Dict metadata
        +Dict embeddings
    }
    
    class Contents {
        +Dict state_data
        +Dict interaction_data
        +Dict action_data
        +string memory_type
    }
    
    class Metadata {
        +timestamp creation_time
        +timestamp last_access_time
        +int compression_level
        +float importance_score
        +int retrieval_count
        +string memory_type
    }
    
    class Embeddings {
        +array full_vector
        +array compressed_vector
        +array abstract_vector
    }
    
    MemoryEntry *-- Contents
    MemoryEntry *-- Metadata
    MemoryEntry *-- Embeddings
```

Each memory entry follows a standardized structure containing identifiers, contents, metadata, and embeddings for efficient storage and retrieval across all memory tiers. The exact format is:

```json
{
  "memory_id": "agent-123-1234-1679233344",
  "agent_id": "agent-123",
  "step_number": 1234,
  "timestamp": 1679233344,
  
  "contents": {
    "position": [x, y],
    "resources": 42,
    "health": 0.85
  },
  
  "metadata": {
    "creation_time": 1679233344,
    "last_access_time": 1679233400,
    "compression_level": 0,
    "importance_score": 0.75,
    "retrieval_count": 3,
    "memory_type": "state"
  },
  
  "embeddings": {
    "full_vector": [...],
    "compressed_vector": [...],
    "abstract_vector": [...]
  }
}
```

## Performance Monitoring

The system provides comprehensive statistics:
```python
stats = memory_system.get_memory_statistics(agent_id="agent-123")
```

Statistics include:
- Memory counts per tier
- Average importance scores
- Compression ratios
- Access patterns
- Memory type distribution

## Memory Maintenance

The system includes automatic memory maintenance:
```python
# Force memory maintenance for a specific agent
memory_system.force_memory_maintenance(agent_id="agent-123")

# Force maintenance for all agents
memory_system.force_memory_maintenance()
```

## Configuration

The system is highly configurable through dataclasses:
```python
from memory.config import MemoryConfig, RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig, AutoencoderConfig

config = MemoryConfig(
    stm_config=RedisSTMConfig(
        host="localhost",
        port=6379,
        db=0,
        ttl=86400,  # 24 hours
        memory_limit=1000
    ),
    im_config=RedisIMConfig(
        ttl=604800,  # 7 days
        memory_limit=10000,
        compression_level=1
    ),
    ltm_config=SQLiteLTMConfig(
        db_path="memory.db",
        compression_level=2
    ),
    autoencoder_config=AutoencoderConfig(
        input_dim=64,
        stm_dim=384,
        im_dim=128,
        ltm_dim=32,
        model_path="model.pt",
        use_neural_embeddings=True
    ),
    cleanup_interval=100,
    memory_priority_decay=0.95,
    enable_memory_hooks=True,
    logging_level="INFO"
)
```

## Using the Agent Memory System

```python
from memory.core import AgentMemorySystem

# Get the singleton instance
memory_system = AgentMemorySystem.get_instance()

# Store an agent state
memory_system.store_agent_state(
    agent_id="agent_001",
    state_data={"position": [10, 20], "health": 100, "inventory": {"gold": 50}},
    step_number=1500
)

# Retrieve similar states
similar_states = memory_system.retrieve_similar_states(
    agent_id="agent_001",
    query_state={"position": [12, 18]},
    k=5
)

# Search by content
content_results = memory_system.search_by_content(
    agent_id="agent_001",
    content_query="gold mining",
    k=5
)

# Get statistics
stats = memory_system.get_memory_statistics(agent_id="agent_001") 