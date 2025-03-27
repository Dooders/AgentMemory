"""Memory models for agent memory system.

This module defines standardized TypedDict models for memory entries
across all storage tiers to ensure consistency and type safety.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


class MemoryMetadata(TypedDict, total=False):
    """Metadata information for a memory entry.
    
    This model is used across all memory tiers (STM, IM, LTM)
    with tier-specific considerations.
    
    Attributes:
        compression_level: Compression level (0=none, 1=light, 2=heavy)
        importance_score: Importance score (0.0-1.0)
        retrieval_count: Number of times this memory has been retrieved
        creation_time: Unix timestamp of creation
        last_access_time: Unix timestamp of last access
    """

    compression_level: int
    importance_score: float
    retrieval_count: int
    creation_time: float
    last_access_time: float


class MemoryEmbeddings(TypedDict, total=False):
    """Vector embeddings for a memory entry.
    
    Different tiers may store different resolution embeddings.
    
    Attributes:
        full_vector: Full resolution embedding (STM only)
        compressed_vector: Compressed embedding (IM, LTM)
    """

    full_vector: List[float]       # Full resolution embedding (STM)
    compressed_vector: List[float] # Compressed embedding (IM, LTM)


class BaseMemoryEntry(TypedDict, total=False):
    """Base structure for a memory entry across all memory tiers.
    
    This serves as the common base for all memory entry types.
    
    Attributes:
        memory_id: Unique identifier for the memory
        agent_id: ID of the agent that owns this memory
        timestamp: Unix timestamp when the memory was created
        content: The actual content of the memory (any structured data)
        metadata: Metadata about the memory
        memory_type: Optional type classification
        step_number: Optional step number for step-based retrieval
    """

    memory_id: str
    agent_id: str
    timestamp: float
    content: Any  # Can be any structured data
    metadata: MemoryMetadata
    memory_type: Optional[str]
    step_number: Optional[int]


class STMMemoryEntry(BaseMemoryEntry):
    """Structure of a memory entry in the short-term memory store.
    
    STM entries contain full resolution embeddings and no compression.
    
    Attributes:
        embeddings: Vector embeddings at full resolution
    """

    embeddings: MemoryEmbeddings  # Full resolution embeddings


class IMMemoryEntry(BaseMemoryEntry):
    """Structure of a memory entry in the intermediate memory store.
    
    IM entries contain compressed embeddings with Level 1 compression.
    
    Attributes:
        embeddings: Vector embeddings with light compression
    """

    embeddings: MemoryEmbeddings  # Light compressed embeddings


class LTMMemoryEntry(BaseMemoryEntry):
    """Structure of a memory entry in the long-term memory store.
    
    LTM entries contain highly compressed embeddings with Level 2 compression.
    
    Attributes:
        embeddings: Vector embeddings with heavy compression
    """

    embeddings: MemoryEmbeddings  # Heavily compressed embeddings


# Type aliases for cleaner type hints
MemoryEntry = Union[STMMemoryEntry, IMMemoryEntry, LTMMemoryEntry] 