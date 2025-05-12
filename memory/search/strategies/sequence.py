from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .base import SearchStrategy
from memory.core import AgentMemorySystem


class NarrativeSequenceStrategy(SearchStrategy):
    """
    Strategy for retrieving a sequence of memories surrounding a reference memory
    to form a contextual narrative.
    """

    def __init__(self, memory_system: AgentMemorySystem):
        """
        Initialize the narrative sequence strategy.

        Args:
            memory_system: The memory system containing the stores
        """
        self.stm_store = memory_system.stm_store
        self.im_store = memory_system.im_store
        self.ltm_store = memory_system.ltm_store

    def name(self) -> str:
        """Return the name of the strategy."""
        return "sequence"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Retrieves a sequence of memories before and after a reference memory to form a narrative"

    def search(
        self,
        query: Union[str, Dict[str, Any]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for a narrative sequence of memories surrounding a reference memory.

        Args:
            query: Either a string reference_id or a dict containing query parameters
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters

        Returns:
            List of memory entries forming a narrative sequence
        """
        # Handle string query (reference_id as a string)
        if isinstance(query, str):
            reference_id = query
            sequence_size = kwargs.get("sequence_size", 5)
            before_count = None
            after_count = None
            time_window_minutes = None
            timestamp_field = "timestamp"
        else:
            # Extract parameters from query dict
            reference_id = query.get("reference_id")
            sequence_size = query.get("sequence_size")
            before_count = query.get("before_count")
            after_count = query.get("after_count")
            time_window_minutes = query.get("time_window_minutes")
            timestamp_field = query.get("timestamp_field", "timestamp")

        # Validate required parameters
        if not reference_id:
            raise ValueError("reference_id is required for sequence search")

        # Validate parameter combinations
        if sequence_size is not None and (
            before_count is not None or after_count is not None
        ):
            raise ValueError(
                "Cannot specify both sequence_size and before_count/after_count"
            )

        if sequence_size is not None and sequence_size < 1:
            raise ValueError("sequence_size must be a positive integer")

        if time_window_minutes is not None and time_window_minutes < 0:
            raise ValueError("time_window_minutes must be a non-negative number")

        # Default values if not specified
        if (
            sequence_size is None
            and before_count is None
            and after_count is None
            and time_window_minutes is None
        ):
            sequence_size = 5

        if sequence_size is not None:
            # Equal distribution around reference if possible
            total_context = sequence_size - 1
            before_count = total_context // 2
            after_count = total_context - before_count

        # Get the appropriate store based on tier
        store = self._get_store_for_tier(tier)

        # Try to get the reference memory
        reference_memory = store.get(agent_id, reference_id)
        if not reference_memory:
            raise ValueError(f"Reference memory with ID {reference_id} not found")

        # Get all memories for filtering
        all_memories = store.get_all(agent_id)

        # Get reference memory type from content metadata
        reference_type = None
        if isinstance(reference_memory.get("content"), dict):
            reference_type = reference_memory.get("content", {}).get("metadata", {}).get("type")
        else:
            # If content is a string, check metadata directly
            reference_type = reference_memory.get("metadata", {}).get("type")

        # Filter memories by type if reference type exists
        if reference_type:
            all_memories = [
                memory
                for memory in all_memories
                if (isinstance(memory.get("content"), dict) and 
                    memory.get("content", {}).get("metadata", {}).get("type") == reference_type) or
                   (not isinstance(memory.get("content"), dict) and 
                    memory.get("metadata", {}).get("type") == reference_type)
            ]

        # Sort memories by timestamp
        sorted_memories = self._sort_memories_by_timestamp(
            all_memories, timestamp_field
        )

        # Find reference memory index
        reference_index = None
        for i, memory in enumerate(sorted_memories):
            memory_id = self._extract_memory_id(memory)
            if memory_id == reference_id:
                reference_index = i
                break

        if reference_index is None:
            raise ValueError(
                f"Reference memory with ID {reference_id} not found in sorted list"
            )

        # Determine range of memories to include
        if time_window_minutes is not None:
            # Use time window instead of counts
            reference_timestamp = self._get_timestamp_value(
                reference_memory, timestamp_field
            )
            if reference_timestamp is None:
                raise ValueError(
                    f"Could not find timestamp at path {timestamp_field} in reference memory"
                )

            # Convert timestamp to datetime
            if isinstance(reference_timestamp, str):
                try:
                    reference_dt = datetime.fromisoformat(reference_timestamp)
                except ValueError:
                    reference_dt = datetime.fromtimestamp(float(reference_timestamp))
            else:
                reference_dt = datetime.fromtimestamp(float(reference_timestamp))
            
            # Convert window to seconds for Unix timestamp comparison
            window_seconds = time_window_minutes * 60
            window_start_ts = reference_dt.timestamp() - window_seconds
            window_end_ts = reference_dt.timestamp() + window_seconds

            results = []
            for memory in sorted_memories:
                memory_timestamp = self._get_timestamp_value(memory, timestamp_field)
                if memory_timestamp is None:
                    continue

                # Convert memory timestamp to float for comparison
                if isinstance(memory_timestamp, str):
                    try:
                        memory_dt = datetime.fromisoformat(memory_timestamp)
                        memory_ts = memory_dt.timestamp()
                    except ValueError:
                        memory_ts = float(memory_timestamp)
                else:
                    memory_ts = float(memory_timestamp)
                
                # Compare timestamps
                if window_start_ts <= memory_ts <= window_end_ts:
                    # Mark the reference memory
                    memory_id = memory.get("memory_id") or memory.get("id")
                    if memory_id == reference_id:
                        memory = dict(
                            memory
                        )  # Create a copy to avoid modifying the original
                        memory.setdefault("metadata", {})["is_reference_memory"] = True
                    results.append(memory)
        else:
            # Use before/after counts
            start_index = max(0, reference_index - before_count)
            end_index = min(len(sorted_memories) - 1, reference_index + after_count)

            results = sorted_memories[start_index : end_index + 1]

            # Mark the reference memory
            for i, memory in enumerate(results):
                memory_id = memory.get("memory_id") or memory.get("id")
                if memory_id == reference_id:
                    results[i] = dict(
                        memory
                    )  # Create a copy to avoid modifying the original
                    results[i].setdefault("metadata", {})["is_reference_memory"] = True
                    break

        # Apply metadata filters
        if metadata_filter:
            results = [
                memory
                for memory in results
                if self._matches_metadata_filter(memory, metadata_filter)
            ]

        return results[:limit]

    def _get_store_for_tier(self, tier):
        """Get the appropriate memory store based on the specified tier."""
        if tier == "stm":
            return self.stm_store
        elif tier == "im":
            return self.im_store
        elif tier == "ltm":
            return self.ltm_store
        else:
            # Default to STM if not specified
            return self.stm_store

    def _sort_memories_by_timestamp(self, memories, timestamp_field):
        """
        Sort memories by timestamp field.

        Args:
            memories: List of memories to sort
            timestamp_field: Field path to use for timestamp (e.g., "metadata.timestamp")

        Returns:
            List of memories sorted by timestamp
        """

        def get_timestamp(memory):
            timestamp = self._get_timestamp_value(memory, timestamp_field)
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp).timestamp()
            elif isinstance(timestamp, (int, float)):
                return float(timestamp)
            else:
                return 0.0  # Default timestamp for invalid values

        return sorted(memories, key=get_timestamp)

    def _get_timestamp_value(self, memory, timestamp_field):
        """
        Extract timestamp value from a memory.

        Args:
            memory: Memory object
            timestamp_field: Field path to use for timestamp (e.g., "metadata.timestamp")

        Returns:
            Timestamp value as int/float or None if not found
        """
        if "." in timestamp_field:
            parts = timestamp_field.split(".")
            value = memory
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        else:
            return memory.get(timestamp_field)

    def _matches_metadata_filter(self, memory, metadata_filter):
        """
        Check if a memory matches the specified metadata filter.

        Args:
            memory: Memory to check
            metadata_filter: Dictionary of metadata key-value pairs to match

        Returns:
            True if memory matches all filter criteria, False otherwise
        """
        if not metadata_filter:
            return True

        memory_metadata = memory.get("metadata", {})
        for key, value in metadata_filter.items():
            if memory_metadata.get(key) != value:
                return False

        return True

    def _extract_memory_id(self, memory: Dict[str, Any]) -> str:
        """
        Extract memory ID from a memory object.

        Args:
            memory: Memory object to extract ID from

        Returns:
            Memory ID as string
        """
        return memory.get("memory_id") or memory.get("id")
