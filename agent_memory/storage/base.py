"""Base abstract memory store for agent memory system.

This module defines the abstract base class for all memory store implementations,
establishing a common interface and shared functionality.
"""

import abc
import logging
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union

from agent_memory.utils.error_handling import Priority, MemoryError

logger = logging.getLogger(__name__)

# Type variable for memory entry
M = TypeVar('M', bound=Dict[str, Any])


class BaseMemoryStore(Generic[M], abc.ABC):
    """Abstract base class for all memory store implementations.

    This class defines the common interface that all memory stores must implement,
    regardless of the underlying storage technology (Redis, SQLite, etc.)
    or memory tier (STM, IM, LTM).

    Attributes:
        store_type: The type of memory store (STM, IM, LTM)
        config: Configuration for the memory store
    """

    def __init__(self, store_type: str):
        """Initialize the base memory store.

        Args:
            store_type: The type of memory store (STM, IM, LTM)
        """
        self.store_type = store_type
        logger.info(f"Initializing {store_type} memory store")

    @abc.abstractmethod
    def store(self, memory_entry: M, priority: Priority = Priority.NORMAL) -> bool:
        """Store a memory entry.

        Args:
            memory_entry: Memory entry to store
            priority: Priority level for this operation

        Returns:
            True if the operation succeeded, False otherwise
        """
        pass

    @abc.abstractmethod
    def get(self, memory_id: str) -> Optional[M]:
        """Retrieve a memory entry by ID.

        Args:
            memory_id: ID of the memory to retrieve

        Returns:
            Memory entry or None if not found
        """
        pass

    @abc.abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if the memory was deleted, False otherwise
        """
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Count memories.

        Returns:
            Number of memories in the store
        """
        pass

    @abc.abstractmethod
    def clear(self) -> bool:
        """Clear all memories.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def get_size(self) -> int:
        """Get the size of the memory store in bytes.

        Returns:
            Memory usage in bytes
        """
        pass

    @abc.abstractmethod
    def get_by_timerange(
        self, start_time: float, end_time: float, limit: int = 100
    ) -> List[M]:
        """Get memories in a time range.

        Args:
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)
            limit: Maximum number of entries to return

        Returns:
            List of memory entries in the time range
        """
        pass

    @abc.abstractmethod
    def get_by_importance(
        self, min_importance: float = 0.0, max_importance: float = 1.0, limit: int = 100
    ) -> List[M]:
        """Get memories by importance score.

        Args:
            min_importance: Minimum importance score
            max_importance: Maximum importance score
            limit: Maximum number of entries to return

        Returns:
            List of memory entries in the importance range
        """
        pass

    @abc.abstractmethod
    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[M]:
        """Search memories by vector similarity.

        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_type: Optional filter by memory type

        Returns:
            List of memory entries ordered by similarity
        """
        pass

    @abc.abstractmethod
    def search_by_attributes(
        self,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None
    ) -> List[M]:
        """Search memories by attribute matching.

        Args:
            attributes: Attributes to match
            memory_type: Optional filter by memory type

        Returns:
            List of memory entries matching the attributes
        """
        pass

    @abc.abstractmethod
    def search_by_step_range(
        self,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None
    ) -> List[M]:
        """Get memories in a step range.

        Args:
            start_step: Start step number
            end_step: End step number
            memory_type: Optional filter by memory type

        Returns:
            List of memory entries in the step range
        """
        pass

    @abc.abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the memory store.

        Returns:
            Dictionary with health status information
        """
        pass

    def get_all(self, limit: int = 1000) -> List[M]:
        """Get all memories up to limit.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of memory entries
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_all"
        )

    def _update_access_metadata(self, memory_id: str, memory_entry: M) -> None:
        """Update access metadata for a memory entry.

        This is a common operation for all memory stores. 
        It updates the retrieval count and last access time.

        Args:
            memory_id: ID of the memory
            memory_entry: Memory entry to update
        """
        if "metadata" not in memory_entry:
            memory_entry["metadata"] = {}

        # Update retrieval count and last access time
        metadata = memory_entry["metadata"]
        retrieval_count = metadata.get("retrieval_count", 0) + 1
        metadata["retrieval_count"] = retrieval_count
        metadata["last_access_time"] = float(time.time())

    @staticmethod
    def _matches_attributes(memory: M, attributes: Dict[str, Any]) -> bool:
        """Check if a memory entry matches the given attributes.

        This is a common implementation for attribute matching across all stores.

        Args:
            memory: Memory entry to check
            attributes: Attributes to match

        Returns:
            True if the memory matches all attributes, False otherwise
        """
        for key, value in attributes.items():
            # Handle nested attributes with dot notation (e.g., "metadata.importance_score")
            if "." in key:
                parts = key.split(".")
                curr = memory
                for part in parts[:-1]:
                    if part not in curr:
                        return False
                    curr = curr[part]
                last_part = parts[-1]
                if last_part not in curr or curr[last_part] != value:
                    return False
            # Handle direct attributes
            elif key not in memory or memory[key] != value:
                return False
        return True 