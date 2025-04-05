"""Memory module for agent memory systems.

This package provides memory systems for agents to store and retrieve
their experiences, enabling learning and adaptation.
"""

# Core components
from agent_memory.config import (
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from agent_memory.core import AgentMemorySystem

# Embedding components
from agent_memory.embeddings.autoencoder import (
    AutoencoderEmbeddingEngine,
    StateAutoencoder,
)
# from agent_memory.embeddings.text_embeddings import TextEmbeddingEngine
from agent_memory.embeddings.compression import CompressionEngine
from agent_memory.memory_agent import MemoryAgent

# Retrieval components
from agent_memory.retrieval.attribute import AttributeRetrieval
from agent_memory.retrieval.similarity import SimilarityRetrieval
from agent_memory.retrieval.temporal import TemporalRetrieval
from agent_memory.storage.redis_im import RedisIMStore

# Storage components
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore

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
