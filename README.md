# TASM (Tiered Adaptive Semantic Memory)
![Status](https://img.shields.io/badge/status-In%20Development%20–%20Experimental%20%26%20Aspirational-blue)

[TASM](docs/memory_system.md) is a tiered adaptive semantic memory system for intelligent agents that provides efficient storage, retrieval, and compression of agent states and experiences across multiple memory tiers.

For detailed theoretical background, see the [Tiered Adaptive Semantic Memory white paper](https://github.com/Dooders/.github/blob/main/whitepapers/tasm/tiered_adaptive_semantic_memory.md).

## Overview

The TASM system implements a biologically-inspired memory architecture with three distinct memory tiers:

1. **Short-Term Memory (STM)**: High-resolution, rapid-access storage using Redis
2. **Intermediate Memory (IM)**: Compressed mid-term storage with Redis + TTL
3. **Long-Term Memory (LTM)**: Highly compressed long-term storage using SQLite

## Features

- **Tiered Storage**: Automatic memory transition between STM, IM, and LTM tiers
- **Adaptive Management**: Intelligent memory maintenance based on importance and usage
- **Semantic Representation**: Neural embeddings for meaningful content storage and retrieval
- **Flexible Integration**: Easy integration with existing agent systems via API or hooks
- **Priority-Based Memory**: Importance scoring for intelligent memory retention
- **Vector Search**: Similarity-based memory retrieval using embeddings
- **Configurable**: Extensive configuration options for all components
- **Direct Imports**: Import all key classes directly from the root package
- **Hybrid Retrieval**: Combine vector and attribute-based search for optimal recall
- **Asynchronous Support**: Async interfaces for high-performance applications

## Directory Structure

```
memory/
├── __init__.py
├── core.py                     # Core memory system implementation
├── config.py                   # Configuration classes
├── memory.py                   # Agent memory implementation
├── config/                     # Configuration components
├── embeddings/                 # Neural embedding components
│   ├── __init__.py
│   ├── autoencoder.py          # Autoencoder for compression
│   ├── vector_store.py         # Vector storage utilities
│   ├── compression.py          # Compression algorithms
│   ├── text_embeddings.py      # Text embedding utilities
│   ├── utils.py                # Embedding utility functions
│   └── vector_compression.py   # Vector compression methods
├── storage/                    # Storage backend implementations
│   ├── __init__.py
│   ├── redis_stm.py            # Redis STM storage
│   ├── redis_im.py             # Redis IM storage
│   ├── sqlite_ltm.py           # SQLite LTM storage
│   ├── redis_client.py         # Redis client implementation
│   ├── async_redis_client.py   # Async Redis client
│   ├── redis_factory.py        # Redis connection factory
│   └── mockredis/              # Mock Redis implementation for testing
├── retrieval/                  # Memory retrieval components
│   ├── similarity.py           # Similarity search
│   ├── temporal.py             # Time-based retrieval
│   └── attribute.py            # Attribute-based retrieval
├── search/                     # Search components
├── api/                        # API interfaces
│   ├── memory_api.py           # Main API interface
│   └── hooks.py                # Agent integration hooks
├── utils/                      # Utility functions
│   ├── serialization.py
│   └── redis_utils.py
└── benchmarking/               # Performance benchmarking tools
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/csmangum/AgentMemory.git
   cd AgentMemory
   ```

2. Install Redis server:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
from memory import AgentMemorySystem, MemoryConfig

# Initialize TASM system
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
from memory import MemoryConfig
from memory.api.hooks import install_memory_hooks
from memory.api.hooks import BaseAgent

@install_memory_hooks
class MyAgent(BaseAgent):
    def act(self, observation):
        # Memory hooks will automatically store states and actions
        return super().act(observation)
```

### Custom Configuration

```python
from memory import MemoryConfig, RedisSTMConfig

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

### Using MockRedis for Development and Testing

For development or testing without a real Redis server, you can use the built-in MockRedis implementation:

```python
from memory import MemoryConfig, RedisSTMConfig, RedisIMConfig

# Configure STM with MockRedis
stm_config = RedisSTMConfig(
    use_mock=True,  # Use MockRedis instead of real Redis
    namespace="agent-stm",
    ttl=3600  # 1 hour
)

# Configure IM with MockRedis
im_config = RedisIMConfig(
    use_mock=True,  # Use MockRedis instead of real Redis
    namespace="agent-im",
    ttl=86400  # 24 hours
)

# Create memory system with MockRedis for both STM and IM
config = MemoryConfig(
    stm_config=stm_config,
    im_config=im_config
)

memory_system = AgentMemorySystem.get_instance(config)
```

This allows you to develop and test your agent memory system without setting up a Redis server.

### Redis Connection Options

| Option | Description | Default |
|--------|-------------|---------|
| host | Redis server hostname | "localhost" |
| port | Redis server port | 6379 |
| db | Redis database number | 0 (STM), 1 (IM) |
| password | Redis password | None |
| use_mock | Use MockRedis instead of real Redis | False |
| ttl | Time-to-live for memories in seconds | 86400 (STM), 604800 (IM) |

## Documentation

Explore the comprehensive documentation to understand the system components and APIs:

- [Memory API](docs/memory_api.md) - Complete API reference for working with the TASM system
- [Agent Memory](docs/agent_memory.md) - Core MemoryAgent implementation and usage
- [Memory System](docs/memory_system.md) - Overview of the tiered memory architecture
- [Memory Configuration](docs/memory_config.md) - Configuration options for all components
- [Memory Tiers](docs/memory_tiers.md) - Details on the three memory tiers and their characteristics
- [Embeddings](docs/embeddings.md) - Neural embedding generation for memory compression
- [Benchmarking](docs/benchmarking.md) - Performance benchmarks and optimization

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

### Vector Similarity Search (in-development)
Find similar memories using embedding-based similarity search:
```python
from memory import AgentMemorySystem

similar_states = AgentMemorySystem.get_instance().retrieve_similar_states(
    agent_id="agent1",
    query_state=current_state,
    k=10
)
```

### Hybrid Retrieval
Combine vector-based and attribute-based search for optimal recall:
```python
from memory import AgentMemorySystem

memories = AgentMemorySystem.get_instance().hybrid_retrieve(
    agent_id="agent1",
    query_state=current_state,
    k=10,
    vector_weight=0.7,
    attribute_weight=0.3
)
```
---
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
