from typing import Dict, List, Any, Optional, Union
from .base import SearchStrategy


class ImportanceStrategy(SearchStrategy):
    """
    Strategy for retrieving memories based on their importance score,
    allowing agents to focus on significant information.
    """

    def __init__(self, stm_store, im_store, ltm_store):
        """
        Initialize the importance strategy.

        Args:
            stm_store: Short-term memory store
            im_store: Intermediate memory store
            ltm_store: Long-term memory store
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store

    def name(self) -> str:
        """Return the name of the strategy."""
        return "importance"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Retrieves memories that meet or exceed a specified importance threshold"

    def search(
        self,
        query: Union[float, Dict[str, Any]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for memories meeting a specified importance threshold.

        Args:
            query: Either importance threshold as a float or dict with search parameters
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters:
                - sort_order: "asc" or "desc" for importance sorting (default: "desc")

        Returns:
            List of memory entries meeting the importance threshold
        """
        # Extract and validate parameters
        min_importance = 0
        max_importance = float("inf")
        top_n = None

        if isinstance(query, (int, float)):
            min_importance = float(query)
            if min_importance < 0:
                raise ValueError("Importance value cannot be negative")
        else:
            # Handle min_importance
            if "min_importance" in query:
                try:
                    min_importance = float(query["min_importance"])
                    if min_importance < 0:
                        raise ValueError("min_importance cannot be negative")
                except (ValueError, TypeError):
                    raise ValueError("min_importance must be a number")

            # Handle max_importance
            if "max_importance" in query:
                try:
                    max_importance = float(query["max_importance"])
                    if max_importance < 0:
                        raise ValueError("max_importance cannot be negative")
                except (ValueError, TypeError):
                    raise ValueError("max_importance must be a number")

            # Handle top_n
            if "top_n" in query:
                try:
                    top_n = int(query["top_n"])
                    if top_n <= 0:
                        raise ValueError("top_n must be greater than 0")
                except (ValueError, TypeError):
                    raise ValueError("top_n must be a positive integer")

        # Validate min/max relationship
        if min_importance > max_importance:
            raise ValueError("min_importance cannot be greater than max_importance")

        sort_order = kwargs.get("sort_order", "desc")

        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)

        # Retrieve memories
        results = []
        for store in stores:
            # Get all memories for the agent
            memories = store.list(agent_id)

            # Filter memories by importance
            for memory in memories:
                metadata = memory.get("metadata", {})
                # Skip memories without importance metadata
                if "importance" not in metadata:
                    continue

                importance = metadata["importance"]
                if min_importance <= importance <= max_importance:
                    results.append(memory)

        # Apply metadata filtering
        if metadata_filter:
            results = [
                memory
                for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]

        # Sort by importance
        reverse_sort = sort_order.lower() == "desc"
        results.sort(
            key=lambda x: x.get("metadata", {}).get("importance", 0),
            reverse=reverse_sort,
        )

        # If top_n is specified, return only the top N results
        if top_n is not None:
            return results[:top_n]

        # Otherwise, return up to the specified limit
        return results[:limit]

    def _get_stores_for_tier(self, tier):
        """Get the appropriate memory stores based on the specified tier."""
        if tier == "stm":
            return [self.stm_store]
        elif tier == "im":
            return [self.im_store]
        elif tier == "ltm":
            return [self.ltm_store]
        else:
            return [self.stm_store, self.im_store, self.ltm_store]

    def _matches_metadata_filters(self, memory, metadata_filter):
        """Check if a memory matches the specified metadata filters."""
        if not metadata_filter:
            return True

        memory_metadata = memory.get("metadata", {})
        for key, value in metadata_filter.items():
            if memory_metadata.get(key) != value:
                return False

        return True
