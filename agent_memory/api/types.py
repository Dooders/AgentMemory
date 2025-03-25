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
MemoryTypeFilter = Literal["state", "interaction", "action"]
MemoryImportanceScore = float  # 0.0 to 1.0


# Memory entry structures
class MemoryMetadata(TypedDict):
    """Metadata for a memory entry."""

    creation_time: int
    last_access_time: int
    compression_level: int
    importance_score: float
    retrieval_count: int
    memory_type: str


class MemoryEmbeddings(TypedDict, total=False):
    """Embeddings for a memory entry."""

    full_vector: List[float]  # STM embedding
    compressed_vector: Optional[List[float]]  # IM embedding
    abstract_vector: Optional[List[float]]  # LTM embedding


class MemoryEntry(TypedDict):
    """A memory entry in the agent memory system."""

    memory_id: str
    agent_id: str
    step_number: int
    timestamp: int
    contents: Dict[str, Any]
    metadata: MemoryMetadata
    embeddings: Optional[MemoryEmbeddings]
    _similarity_score: Optional[float]


class MemoryChangeRecord(TypedDict):
    """Record of a change in a memory attribute."""

    memory_id: str
    step_number: int
    timestamp: int
    previous_value: Optional[Any]
    new_value: Any


class MemoryStatistics(TypedDict):
    """Statistics about an agent's memory usage."""

    total_memories: int
    stm_count: int
    im_count: int
    ltm_count: int
    memory_type_distribution: Dict[str, int]
    last_maintenance_time: Optional[int]
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
