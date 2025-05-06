from typing import Any, Dict, List, Optional, Union
import logging

from .base import SearchStrategy

logger = logging.getLogger(__name__)


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

        # Mapping for string importance values to numeric values
        self.importance_mapping = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
        }

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
        elif isinstance(query, dict):
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
            memories = store.get_all(agent_id)

            # Filter memories by importance
            for memory in memories:
                metadata = memory.get("metadata", {})

                # Get importance - check both field names for compatibility
                importance_value = metadata.get("importance_score")
                if importance_value is None:
                    importance_value = metadata.get("importance")

                # Skip memories without importance metadata
                if importance_value is None:
                    continue

                # Parse importance value to numeric
                try:
                    if isinstance(importance_value, str):
                        if importance_value.lower() in self.importance_mapping:
                            importance = self.importance_mapping[importance_value.lower()]
                        else:
                            importance = float(importance_value)
                    else:
                        importance = float(importance_value)
                except (ValueError, TypeError):
                    continue

                # Round importance to 2 decimal places to avoid floating point precision issues
                importance = round(importance, 2)
                min_importance = round(min_importance, 2)
                max_importance = round(max_importance, 2)

                # Debug logging for importance comparison
                logger.debug(f"Memory {memory.get('memory_id')}: importance={importance}, min={min_importance}, max={max_importance}")
                
                # Check if memory meets importance threshold
                if min_importance <= importance <= max_importance:
                    logger.debug(f"Memory {memory.get('memory_id')} passed importance check")
                    # Store the parsed importance in the memory metadata for sorting
                    memory["metadata"]["_parsed_importance"] = importance
                    results.append(memory)
                else:
                    logger.debug(f"Memory {memory.get('memory_id')} failed importance check: {importance} not in [{min_importance}, {max_importance}]")

        # Apply metadata filtering
        if metadata_filter:
            filtered_results = []
            for memory in results:
                if self._matches_metadata_filters(memory, metadata_filter):
                    filtered_results.append(memory)
            results = filtered_results

        # Sort by importance
        reverse_sort = sort_order.lower() == "desc"

        def get_importance(memory):
            metadata = memory.get("metadata", {})
            return metadata.get("_parsed_importance", 0)

        # Sort the memories by importance
        results.sort(
            key=get_importance,
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

        def get_nested_value(obj, path):
            """Get a value from a nested dictionary using dot notation."""
            parts = path.split('.')
            current = obj
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
                if current is None:
                    return None
            return current

        for key, filter_value in metadata_filter.items():
            # Handle nested paths
            memory_value = get_nested_value(memory, key)
            if memory_value is None:
                return False

            # Special case for 'tags' which is often a list
            if key.endswith('.tags'):
                if isinstance(memory_value, list) and filter_value in memory_value:
                    continue
                elif memory_value == filter_value:
                    continue
                return False

            # Special case for 'type'
            elif key.endswith('.type'):
                if memory_value == filter_value:
                    continue
                return False

            # For other list metadata values, check membership
            elif isinstance(memory_value, list):
                if filter_value in memory_value:
                    continue
                return False

            # For dict metadata values, check keys and values
            elif isinstance(memory_value, dict):
                if filter_value in memory_value or filter_value in memory_value.values():
                    continue
                return False

            # For direct equality comparison
            elif memory_value == filter_value:
                continue
            else:
                return False

        # All filter criteria matched
        return True
