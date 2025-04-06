"""Type definitions for the agent memory system."""

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
    """Metadata for a memory entry."""

    creation_time: float
    last_access_time: float
    compression_level: int
    importance_score: MemoryImportanceScore
    retrieval_count: int
    memory_type: Literal["state", "action", "interaction"]


class MemoryEmbeddings(TypedDict, total=False):
    """Embeddings for a memory entry."""

    full_vector: List[float]  # STM embedding
    compressed_vector: List[float]  # IM embedding
    abstract_vector: List[float]  # LTM embedding


class MemoryEntry(TypedDict):
    """A memory entry in the agent memory system."""

    memory_id: str
    agent_id: str
    step_number: int
    timestamp: float
    contents: Dict[str, Any]
    metadata: MemoryMetadata
    embeddings: Optional[MemoryEmbeddings]


class MemoryChangeRecord(TypedDict):
    """Record of a change to a memory attribute."""

    memory_id: str
    step_number: int
    timestamp: float
    previous_value: Optional[Any]
    new_value: Any


class MemoryStatistics(TypedDict):
    """Statistics about memory usage for an agent."""

    total_memories: int
    stm_count: int
    im_count: int
    ltm_count: int
    memory_type_distribution: Dict[str, int]
    last_maintenance_time: Optional[float]
    insert_count_since_maintenance: int


# Protocol for memory stores
@runtime_checkable
class MemoryStore(Protocol):
    """Protocol defining the interface for memory stores."""

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        ...

    def get_recent(
        self, count: int, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get recent memories."""
        ...

    def get_by_step_range(
        self, start_step: int, end_step: int, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get memories in a step range."""
        ...

    def get_by_attributes(
        self, attributes: Dict[str, Any], memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get memories matching attributes."""
        ...

    def search_by_vector(
        self, vector: List[float], k: int = 5, memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Search memories by vector similarity."""
        ...

    def search_by_content(
        self, content_query: Dict[str, Any], k: int = 5
    ) -> List[MemoryEntry]:
        """Search memories by content."""
        ...

    def contains(self, memory_id: str) -> bool:
        """Check if a memory exists."""
        ...

    def update(self, memory: MemoryEntry) -> bool:
        """Update a memory."""
        ...

    def count(self) -> int:
        """Count memories."""
        ...

    def count_by_type(self) -> Dict[str, int]:
        """Count memories by type."""
        ...

    def clear(self) -> bool:
        """Clear all memories."""
        ...


class MemoryTypeDistribution(TypedDict, total=False):
    state: int
    action: int
    interaction: int


class MemoryStatistics(TypedDict):
    """Statistics about memory usage for an agent."""

    total_memories: int
    stm_count: int
    im_count: int
    ltm_count: int
    memory_type_distribution: MemoryTypeDistribution
    last_maintenance_time: Optional[float]
    insert_count_since_maintenance: int


class SimilaritySearchResult(MemoryEntry):
    """Memory entry with similarity score."""

    _similarity_score: float


# Type for configuration updates
class ConfigUpdate(Dict[str, Any]):
    """Type for configuration updates.

    Can include flat updates like {"cleanup_interval": 200}
    or nested updates like {"stm_config.memory_limit": 10000}
    """

    pass


# Generic query result
class QueryResult(TypedDict):
    """Generic result from a memory query."""

    memory_id: str
    agent_id: str
    step_number: int
    timestamp: float
    contents: Dict[str, Any]
    metadata: MemoryMetadata
