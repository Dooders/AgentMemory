"""Agent Memory Types Module

This module defines core type definitions for the agent memory system, establishing
a consistent type system for memory storage, retrieval, and manipulation. These types
serve as the foundation for type checking, data validation, and API contracts throughout
the memory system.

Key components:

1. Memory Entry Types: Structured definitions for memory entries, including metadata,
   embeddings, and content structures that represent different kinds of agent memories.

2. Memory Tiers: Type definitions for the hierarchical memory system's tiers
   (STM, IM, LTM) that organize memories by recency, importance, and abstraction level.

3. Statistics Types: Types for representing memory usage statistics, enabling
   monitoring and optimization of memory resources.

4. Protocol Definitions: Runtime-checkable protocols that define interfaces for
   memory stores and other pluggable components, allowing for different implementations
   while maintaining type safety.

5. Search Result Types: Specialized types for representing search results with
   additional metadata like similarity scores.

These type definitions enable static type checking throughout the codebase, provide
clear documentation for API consumers, and establish contracts between different
parts of the memory system.

Usage example:
```python
from memory.api.types import MemoryEntry, MemoryMetadata, MemoryTier, MemoryStore
from typing import List, Optional

# Define a function that works with typed memory entries
def filter_important_memories(
    memories: List[MemoryEntry],
    importance_threshold: float = 0.7
) -> List[MemoryEntry]:
    "Filter memories by importance score."
    return [
        memory for memory in memories
        if memory["metadata"].get("importance_score", 0) >= importance_threshold
    ]

# Create a typed memory entry
memory_entry: MemoryEntry = {
    "memory_id": "mem_12345",
    "agent_id": "agent_001",
    "step_number": 42,
    "timestamp": 1649879872.123,
    "contents": {"observation": "User asked about weather"},
    "metadata": {
        "importance_score": 0.8,
        "memory_type": "interaction",
        "creation_time": 1649879872.123,
        "retrieval_count": 0
    },
    "embeddings": {
        "full_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "compressed_vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    }
}

# Specify memory tier for operations
tier: MemoryTier = "stm"  # or "im" or "ltm"
```
"""

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

# Memory tier types
MemoryTier = Literal["stm", "im", "ltm"]
MemoryTypeFilter = Literal["state", "action", "interaction", None]
MemoryImportanceScore = float  # Constrained to 0.0-1.0


# Memory entry structures
class MemoryMetadata(TypedDict, total=False):
    """Metadata for a memory entry.

    This structure contains metadata about a memory entry, including temporal information,
    importance scoring, and retrieval statistics. It supports the memory lifecycle by
    providing context for memory maintenance operations like consolidation and forgetting.

    Attributes:
        creation_time: Unix timestamp when the memory was first created
        last_access_time: Unix timestamp when the memory was last accessed/retrieved
        compression_level: Level of compression applied to this memory (0=none, 1=partial, 2=high)
        importance_score: Subjective importance of this memory (0.0-1.0)
        retrieval_count: Number of times this memory has been retrieved
        memory_type: Classification of the memory content (state/action/interaction)
    """

    creation_time: float
    last_access_time: float
    compression_level: int
    importance_score: MemoryImportanceScore
    retrieval_count: int
    memory_type: Literal["state", "action", "interaction"]


class MemoryEmbeddings(TypedDict, total=False):
    """Embeddings for a memory entry.

    This structure contains vector embeddings at different resolutions for the memory entry,
    enabling semantic search across different memory tiers. Each tier uses a different
    resolution embedding to balance search quality with storage and computation requirements.

    Attributes:
        full_vector: High-dimensional vector for STM with full semantic detail
        compressed_vector: Medium-dimensional vector for IM with reduced detail
        abstract_vector: Low-dimensional vector for LTM with core semantic essence
    """

    full_vector: List[float]  # STM embedding
    compressed_vector: List[float]  # IM embedding
    abstract_vector: List[float]  # LTM embedding


class MemoryEntry(TypedDict):
    """A memory entry in the agent memory system.

    This is the core data structure representing a discrete memory in the system.
    It includes identification, content, metadata, and vector embeddings that enable
    storage, retrieval, and semantic search operations across memory tiers.

    Memory entries move through the memory hierarchy from STM to IM to LTM, with
    increasing levels of compression and abstraction applied at each stage.

    Attributes:
        memory_id: Unique identifier for this memory entry
        agent_id: Identifier for the agent that owns this memory
        step_number: Sequential step number in the agent's lifecycle
        timestamp: Unix timestamp when the memory was recorded
        contents: Arbitrary structured data representing the memory content
        metadata: Additional information about the memory for management
        embeddings: Vector representations for semantic search and retrieval
    """

    memory_id: str
    agent_id: str
    step_number: int
    timestamp: float
    contents: Dict[str, Any]
    metadata: MemoryMetadata
    embeddings: Optional[MemoryEmbeddings]


class MemoryChangeRecord(TypedDict):
    """Record of a change to a memory attribute.

    This structure captures the history of changes to specific memory attributes,
    enabling temporal reasoning about how agent attributes evolve over time.
    Change records support debugging, causal analysis, and explanation generation.

    Attributes:
        memory_id: Identifier for the memory where the change occurred
        step_number: Step in the agent's lifecycle when the change happened
        timestamp: Unix timestamp when the change was recorded
        previous_value: Value of the attribute before the change (None if new)
        new_value: Value of the attribute after the change
    """

    memory_id: str
    step_number: int
    timestamp: float
    previous_value: Optional[Any]
    new_value: Any


class MemoryTypeDistribution(TypedDict, total=False):
    """Distribution of memory types in the agent's memory.

    This structure captures the count of different memory types, providing insight
    into the composition of an agent's memory and supporting memory management decisions.

    Attributes:
        state: Number of state memories (agent's internal state)
        action: Number of action memories (agent's actions and their outcomes)
        interaction: Number of interaction memories (exchanges with environment/users)
    """

    state: int
    action: int
    interaction: int


class MemoryStatistics(TypedDict):
    """Statistics about memory usage for an agent.

    This structure provides a comprehensive view of an agent's memory usage across
    different tiers, supporting monitoring, diagnostics, and optimization of the
    memory system. It helps identify potential memory bottlenecks and imbalances.

    Attributes:
        total_memories: Total number of memories across all tiers
        stm_count: Number of memories in short-term memory
        im_count: Number of memories in intermediate memory
        ltm_count: Number of memories in long-term memory
        memory_type_distribution: Breakdown of memories by type
        last_maintenance_time: When memory maintenance was last performed
        insert_count_since_maintenance: New memories added since last maintenance
    """

    total_memories: int
    stm_count: int
    im_count: int
    ltm_count: int
    memory_type_distribution: MemoryTypeDistribution
    last_maintenance_time: Optional[float]
    insert_count_since_maintenance: int


class SimilaritySearchResult(MemoryEntry):
    """Memory entry with similarity score.

    This structure extends the base MemoryEntry with a similarity score field,
    representing how closely the entry matches a query vector or text. It is used
    to return and rank search results from semantic memory queries.

    The similarity score is typically computed using cosine similarity between
    the query embedding and memory embedding vectors, with scores closer to 1.0
    indicating higher similarity.

    Attributes:
        _similarity_score: Numeric similarity score (typically 0.0-1.0)
        (Plus all attributes from MemoryEntry)
    """

    _similarity_score: float


# Type for configuration updates
class ConfigUpdate(Dict[str, Any]):
    """Type for configuration updates.

    This dictionary type represents updates to the memory system configuration.
    It supports both flat updates to top-level configuration parameters and
    nested updates to subsystem configurations using dot notation.

    Configuration updates are applied atomically to ensure consistency across
    the memory system, with validation performed before changes take effect.

    Examples:
        {"cleanup_interval": 200}  # Simple flat update
        {"stm_config.memory_limit": 10000}  # Nested path update
        {"im_config.ttl": 86400, "ltm_config.db_path": "/data/memory.db"}  # Multiple updates
    """

    pass


# Generic query result
class QueryResult(TypedDict):
    """Generic result from a memory query.

    This structure represents a simplified memory entry returned from query operations,
    containing the essential fields for displaying and utilizing query results.
    It provides a consistent interface for different types of memory queries.

    Attributes:
        memory_id: Unique identifier for this memory entry
        agent_id: Identifier for the agent that owns this memory
        step_number: Sequential step number in the agent's lifecycle
        timestamp: Unix timestamp when the memory was recorded
        contents: Arbitrary structured data representing the memory content
        metadata: Additional information about the memory for management
    """

    memory_id: str
    agent_id: str
    step_number: int
    timestamp: float
    contents: Dict[str, Any]
    metadata: MemoryMetadata


# Protocol for memory stores
@runtime_checkable
class MemoryStore(Protocol):
    """Protocol defining the interface for memory stores.

    This protocol establishes a contract that all memory store implementations must follow,
    enabling the memory system to interact with different storage backends interchangeably.
    Memory stores are responsible for persisting, retrieving, and querying memory entries.

    Implementations might include in-memory stores for STM, Redis-backed stores for IM,
    and SQLite or file-based stores for LTM, all adhering to this common interface.
    """

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID.

        Args:
            memory_id: Unique identifier for the memory to retrieve

        Returns:
            The memory entry if found, or None if not present
        """
        ...

    def get_recent(
        self, count: int, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get recent memories.

        Args:
            count: Maximum number of memories to retrieve
            memory_type: Optional filter for specific memory types

        Returns:
            List of most recent memory entries, ordered by recency
        """
        ...

    def get_by_step_range(
        self, start_step: int, end_step: int, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get memories in a step range.

        Args:
            start_step: Beginning of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries within the specified step range
        """
        ...

    def get_by_attributes(
        self, attributes: Dict[str, Any], memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get memories matching attributes.

        Args:
            attributes: Dictionary of attribute-value pairs to match
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries matching the specified attributes
        """
        ...

    def search_by_vector(
        self, vector: List[float], k: int = 5, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Search memories by vector similarity.

        Args:
            vector: Embedding vector to search with
            k: Maximum number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity to query vector
        """
        ...

    def search_by_content(
        self, content_query: Dict[str, Any], k: int = 5
    ) -> List[MemoryEntry]:
        """Search memories by content.

        Args:
            content_query: Dictionary of content fields to search for
            k: Maximum number of results to return

        Returns:
            List of memory entries matching the content query
        """
        ...

    def contains(self, memory_id: str) -> bool:
        """Check if a memory exists.

        Args:
            memory_id: Unique identifier for the memory to check

        Returns:
            True if the memory exists in this store, False otherwise
        """
        ...

    def update(self, memory: MemoryEntry) -> bool:
        """Update a memory.

        Args:
            memory: Updated memory entry with the same memory_id

        Returns:
            True if update was successful, False otherwise
        """
        ...

    def count(self) -> int:
        """Count memories.

        Returns:
            Total number of memories in this store
        """
        ...

    def count_by_type(self) -> Dict[str, int]:
        """Count memories by type.

        Returns:
            Dictionary mapping memory types to counts
        """
        ...

    def clear(self) -> bool:
        """Clear all memories.

        Returns:
            True if clearing was successful
        """
        ...
