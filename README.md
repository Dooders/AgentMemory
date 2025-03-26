# AgentMemory

AgentMemory is a hierarchical memory system for intelligent agents that provides efficient storage, retrieval, and compression of agent states and experiences across multiple memory tiers.

## Overview

The Agent Memory System implements a biologically-inspired memory architecture with three distinct memory tiers:

1. **Short-Term Memory (STM)**: High-resolution, rapid-access storage using Redis
2. **Intermediate Memory (IM)**: Compressed mid-term storage with Redis + TTL
3. **Long-Term Memory (LTM)**: Highly compressed long-term storage using SQLite

## Features

- **Hierarchical Storage**: Automatic memory transition between STM, IM, and LTM tiers
- **Neural Compression**: Autoencoder-based embedding generation for efficient storage (in-development)
- **Flexible Integration**: Easy integration with existing agent systems via API or hooks
- **Priority-Based Memory**: Importance scoring for intelligent memory retention
- **Vector Search**: Similarity-based memory retrieval using embeddings
- **Configurable**: Extensive configuration options for all components
- **Direct Imports**: Import all key classes directly from the root package

## Directory Structure

```
agent_memory/
├── __init__.py
├── core.py              # Core memory system implementation
├── config.py            # Configuration classes
├── memory_agent.py      # Memory agent implementation
├── embeddings/          # Neural embedding components
│   ├── autoencoder.py  # Autoencoder for compression
│   ├── vector_store.py # Vector storage utilities
│   └── compression.py  # Compression algorithms
├── storage/            # Storage backend implementations
│   ├── redis_stm.py   # Redis STM storage
│   ├── redis_im.py    # Redis IM storage
│   └── sqlite_ltm.py  # SQLite LTM storage
├── retrieval/          # Memory retrieval components
│   ├── similarity.py  # Similarity search
│   ├── temporal.py    # Time-based retrieval
│   └── attribute.py   # Attribute-based retrieval
├── api/               # API interfaces
│   ├── memory_api.py # Main API interface
│   └── hooks.py      # Agent integration hooks
└── utils/            # Utility functions
    ├── serialization.py
    └── redis_utils.py
```

## Requirements

- Python 3.8+
- PyTorch
- Redis
- SQLite
- NumPy
- SQLAlchemy

## Installation

1. Install Redis server:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
from agent_memory import AgentMemorySystem, MemoryConfig

# Initialize memory system
memory_system = AgentMemorySystem.get_instance(MemoryConfig())

# Store agent state
memory_system.store_agent_state(
    agent_id="agent1",
    state_data={"position": [0, 0], "health": 100},
    step_number=1
)

# Store agent action
memory_system.store_agent_action(
    agent_id="agent1",
    action_data={"action": "move", "direction": "north"},
    step_number=1
)
```

### Using Memory Hooks

```python
from agent_memory import MemoryConfig
from agent_memory.api.hooks import install_memory_hooks
from agent_memory.api.hooks import BaseAgent

@install_memory_hooks
class MyAgent(BaseAgent):
    def act(self, observation):
        # Memory hooks will automatically store states and actions
        return super().act(observation)
```

### Custom Configuration

```python
from agent_memory import MemoryConfig, RedisSTMConfig

config = MemoryConfig(
    stm_config=RedisSTMConfig(
        host="localhost",
        port=6379,
        ttl=86400,  # 24 hours
        memory_limit=1000
    )
)

memory_system = AgentMemorySystem.get_instance(config)
```

## Memory Tiers

### Short-Term Memory (STM)
- High-resolution storage of recent experiences
- Fast access and retrieval
- Limited capacity with automatic cleanup
- Full feature vectors with minimal compression

### Intermediate Memory (IM)
- Medium-term storage with moderate compression
- TTL-based expiration
- Balanced between resolution and storage efficiency
- Compressed feature vectors

### Long-Term Memory (LTM)
- Long-term persistent storage
- Highly compressed representations
- Efficient storage of essential information
- Abstract feature vectors

## Advanced Features

### Neural Compression (in-development)
The system uses an autoencoder architecture to generate compressed embeddings:
- STM: 384-dimensional embeddings
- IM: 128-dimensional embeddings
- LTM: 32-dimensional embeddings

### Memory Transitions
Memories automatically transition between tiers based on:
- Age
- Importance score
- Access frequency
- Storage capacity limits

### Vector Similarity Search
Find similar memories using embedding-based similarity search:
```python
from agent_memory import AgentMemorySystem

similar_states = AgentMemorySystem.get_instance().retrieve_similar_states(
    agent_id="agent1",
    query_state=current_state,
    k=10
)
```

## System Advantages

The Agent Memory System offers several distinct advantages for intelligent agent implementations:

### Performance-Oriented Architecture
- Optimized database backends (Redis for fast access, SQLite for persistence)
- Automatic data compression that balances resolution and storage requirements
- Efficient retrieval mechanisms that scale with growing memory size

### Biologically-Inspired Design
- Memory organization that mimics human memory consolidation processes
- Intelligent forgetting mechanisms to prevent information overload
- Preservation of critical information through importance scoring

### Technical Implementation
- Unified API across all memory tiers for simplified development
- Parameter-based transition logic with configurable thresholds
- Dimensional reduction through neural techniques preserves semantic meaning

### Practical Benefits
- Reduced memory footprint through progressive compression
- Improved recall performance for both recent and distant experiences
- Seamless integration with existing agent frameworks

### Development Advantages
- Clean separation of memory concerns from agent logic
- Extensive configuration options without code changes
- Standardized components that can be extended or replaced

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
