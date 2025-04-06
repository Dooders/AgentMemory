# Agent Memory - Embeddings Module

## Overview

The Agent Memory Embeddings module provides tools for creating, managing, and retrieving vector embeddings that represent agent memories in a semantic vector space. These embeddings enable efficient semantic search and retrieval of memories based on their meaning rather than exact keyword matching.

This module is a core component of the AgentMemory system, providing the semantic encoding that allows agents to store and retrieve memories based on conceptual similarity.

## Key Components

### 1. TextEmbeddingEngine (WIP)

Uses sentence-transformers models to convert text and structured data into vector embeddings that capture semantic meaning.

- Simple interface for encoding any data into vector embeddings
- Support for memory tier-specific encoding (STM, IM, LTM)
- Optional context weighting to emphasize important attributes

### 2. AutoEncoder

Provides dimensionality reduction through neural network-based compression of embedding vectors, reducing storage requirements while preserving semantic similarity relationships.

- Learns a compressed representation of high-dimensional embeddings
- Can be trained on agent-specific data for optimal performance
- Reduces storage footprint for long-term memory

### 3. Compression

Contains utilities for compressing memory entries as they move through the memory hierarchy.

- Reduces detail while preserving essential information
- Multiple compression levels for different memory tiers
- Binary compression for long-term storage efficiency

### 4. VectorStore

Implements vector storage and similarity search capabilities for efficient memory retrieval.

- Multiple backend options (in-memory and Redis)
- Efficient similarity search based on cosine similarity
- Support for metadata filtering during retrieval

### 5. Utils

Common utility functions used across the embedding modules, including cosine similarity calculations, dictionary flattening, and text conversion.

## Usage Examples

### Basic Usage with TextEmbeddingEngine

```python
from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.vector_store import VectorStore

# Initialize embedding engine
embedding_engine = TextEmbeddingEngine(model_name="all-MiniLM-L6-v2")  # Smaller model

# Create vector embeddings for a memory
memory_data = {"content": "The user asked about machine learning applications."}
vector = embedding_engine.encode(memory_data)

# Store in vector database
vector_store = VectorStore()
vector_store.store_memory_vectors({
    "id": "memory_1",
    "stm_vector": vector,
    "metadata": memory_data
})

# Retrieve similar memories
query = "What AI technologies are relevant?"
query_vector = embedding_engine.encode(query)
similar_memories = vector_store.find_similar_memories(
    query_vector, 
    tier="stm",  # Search in short-term memory tier
    limit=5      # Return top 5 results
)

# Process results
for memory in similar_memories:
    print(f"Memory ID: {memory['id']}, Score: {memory['score']}")
    print(f"Content: {memory['metadata'].get('content', 'No content')}")
```

### Using Memory Compression

```python
from memory.embeddings.compression import CompressionEngine
from memory.config import AutoencoderConfig

# Initialize compression engine
config = AutoencoderConfig()
compression_engine = CompressionEngine(config)

# Original memory entry
memory = {
    "id": "memory_1",
    "content": {
        "observation": "The user asked about the capital of France. I responded with Paris.",
        "detailed_state": "The conversation was about European geography...",
        "raw_observation": "Very detailed and verbose observation data...",
    },
    "metadata": {
        "timestamp": 1649852400,
        "importance": 0.75,
        "category": "user_interaction"
    }
}

# Compress for intermediate memory (level 1)
im_memory = compression_engine.compress(memory, level=1)

# Compress for long-term memory (level 2)
ltm_memory = compression_engine.compress(memory, level=2)

# Later, decompress when needed
original_memory = compression_engine.decompress(ltm_memory)
```

### Working with VectorStore and Multiple Memory Tiers

```python
from memory.embeddings import TextEmbeddingEngine, VectorStore
import redis

# Create Redis client for production use
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize vector store with Redis backend
vector_store = VectorStore(
    redis_client=redis_client,
    stm_dimension=384,  # Dimension for short-term memory vectors
    im_dimension=128,   # Dimension for intermediate memory vectors
    ltm_dimension=32,   # Dimension for long-term memory vectors
    namespace="agent_1" # Namespace to separate different agents
)

# Create embedding engine
embedding_engine = TextEmbeddingEngine()

# Process a memory
memory_content = {"content": "The temperature in San Francisco is 68°F"}
memory_id = "weather_sf_1"

# Create vectors for different memory tiers
# In a real implementation, you would use dimensionality reduction
# via the AutoEncoder for IM and LTM tiers
stm_vector = embedding_engine.encode_stm(memory_content)
im_vector = embedding_engine.encode_im(memory_content)  # Would normally compress
ltm_vector = embedding_engine.encode_ltm(memory_content)  # Would normally compress further

# Store vectors for all tiers
vector_store.store_memory_vectors({
    "id": memory_id,
    "stm_vector": stm_vector,
    "im_vector": im_vector,
    "ltm_vector": ltm_vector,
    "metadata": memory_content
})

# Query using different tiers
query = "What's the weather like in California?"
query_vector = embedding_engine.encode(query)

# Search in different memory tiers
stm_results = vector_store.find_similar_memories(query_vector, tier="stm", limit=3)
im_results = vector_store.find_similar_memories(query_vector, tier="im", limit=3)
ltm_results = vector_store.find_similar_memories(query_vector, tier="ltm", limit=3)
```

## Installation Requirements

The embeddings module requires the following dependencies:

```
sentence-transformers
numpy
redis (optional, for Redis backend)
torch (required for AutoEncoder)
```

Install with:

```bash
pip install sentence-transformers numpy redis torch
```

## Best Practices

1. **Choose the right model**: For `TextEmbeddingEngine`, the default "all-mpnet-base-v2" model provides good quality but is larger (420MB). For smaller footprint, try "all-MiniLM-L6-v2" (80MB).

2. **Optimize memory hierarchy**: Use compression appropriately for different memory tiers:
   - STM: Full fidelity, no compression
   - IM: Moderate compression, reduced precision
   - LTM: High compression, essential information only

3. **Context-aware encoding**: Use the `context_weights` parameter in `encode()` to emphasize important attributes when generating embeddings.

4. **Redis for production**: Use the Redis backend for production deployments to enable persistence and scaling.

5. **Custom training**: For specialized domains, consider training the AutoEncoder on domain-specific data for improved performance. 