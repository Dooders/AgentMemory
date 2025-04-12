"""Agent Memory Embeddings Module

This module provides tools for creating, managing, and retrieving vector embeddings
that represent agent memories in a semantic vector space. These embeddings enable
efficient semantic search and retrieval of memories based on their meaning.

Key components:

1. TextEmbeddingEngine: Uses sentence-transformers models to convert text and
   structured data into vector embeddings that capture semantic meaning.

2. CustomEmbeddingEngine: A lightweight alternative to TextEmbeddingEngine that
   eliminates the sentence-transformers dependency. It uses a custom LSTM-based
   model trained through knowledge distillation.

3. AutoEncoder: Provides dimensionality reduction through neural network-based
   compression of embedding vectors, reducing storage requirements while
   preserving semantic similarity relationships.

4. Compression: Contains utilities for compressing high-dimensional vectors into
   lower-dimensional representations for efficient storage and retrieval.

5. VectorStore: Implements vector storage and similarity search capabilities,
   including in-memory and Redis-backed vector indices for efficient retrieval.

6. Utils: Common utility functions used across the embedding modules, including
   cosine similarity calculations, dictionary flattening, and text conversion.

This module works with the agent memory system to encode memories into vector
representations, store them efficiently, and retrieve them based on semantic
similarity to support context-aware agent reasoning.

Usage example:
```python
from memory.embeddings.custom_embeddings import CustomEmbeddingEngine
from memory.embeddings.vector_store import VectorStore

# Initialize embedding engine
embedding_engine = CustomEmbeddingEngine()

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
similar_memories = vector_store.find_similar_memories(vector, tier="stm", limit=5)
```
"""

from memory.embeddings.autoencoder import (
    AgentStateDataset,
    AutoencoderEmbeddingEngine,
    NumericExtractor,
    StateAutoencoder,
)
# from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.custom_embeddings import CustomEmbeddingEngine
from memory.embeddings.utils import (
    cosine_similarity,
    filter_dict_keys,
    flatten_dict,
    object_to_text,
)
from memory.embeddings.vector_store import (
    InMemoryVectorIndex,
    VectorIndex,
    VectorStore,
)
from memory.embeddings.vector_compression import (
    quantize_vector,
    dequantize_vector,
    compress_vector_rp,
    decompress_vector_rp,
    CompressionConfig,
)

__all__ = [
    # "TextEmbeddingEngine",
    "CustomEmbeddingEngine",
    "AutoencoderEmbeddingEngine",
    "NumericExtractor",
    "StateAutoencoder",
    "AgentStateDataset",
    "VectorStore",
    "VectorIndex",
    "InMemoryVectorIndex",
    "cosine_similarity",
    "flatten_dict",
    "object_to_text",
    "filter_dict_keys",
    "quantize_vector",
    "dequantize_vector",
    "compress_vector_rp",
    "decompress_vector_rp",
    "CompressionConfig",
]
