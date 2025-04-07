"""Memory module for agent memory systems.

This package provides memory systems for agents to store and retrieve
their experiences, enabling learning and adaptation.
"""

# Core components
from memory.config import (
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.core import AgentMemorySystem

# Embedding components
from memory.embeddings.autoencoder import (
    AutoencoderEmbeddingEngine,
    StateAutoencoder,
)
# from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.compression import CompressionEngine
from memory.agent_memory import MemoryAgent

# Retrieval components
from memory.retrieval.attribute import AttributeRetrieval
from memory.retrieval.similarity import SimilarityRetrieval
from memory.retrieval.temporal import TemporalRetrieval
from memory.storage.redis_im import RedisIMStore

# Storage components
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore

__all__ = [
    # Core classes
    "AgentMemorySystem",
    "MemoryAgent",
    "MemoryConfig",
    # Config classes
    "RedisSTMConfig",
    "RedisIMConfig",
    "SQLiteLTMConfig",
    # Storage classes
    "RedisSTMStore",
    "RedisIMStore",
    "SQLiteLTMStore",
    # Embedding classes
    "AutoencoderEmbeddingEngine",
    "StateAutoencoder",
    "TextEmbeddingEngine",
    "CompressionEngine",
    # Retrieval classes
    "AttributeRetrieval",
    "SimilarityRetrieval",
    "TemporalRetrieval",
]

"""Agent memory system package."""
