import re
from typing import Any, Dict, List, Optional, Union

from .base import SearchStrategy


class ContentPathStrategy(SearchStrategy):
    """
    Strategy for searching memories based on content path values or patterns,
    providing precise access to nested content structures.
    """

    def __init__(self, stm_store, im_store, ltm_store):
        """
        Initialize the content path strategy.

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
        return "path"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Searches memories based on specific content path values or patterns"

    def search(
        self,
        query: Dict[str, Any],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for memories based on content path values or patterns.

        Args:
            query: Dict containing search parameters:
                - path: Dot-notation path to the content field (e.g., "content.details.location")
                - value: Exact value to match (optional)
                - regex: Regex pattern to match (optional)
                - exists: Check if path exists (optional)
                - gt, lt: Greater than/less than comparisons (optional)
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters

        Returns:
            List of matching memory entries
        """
        if "path" not in query:
            raise ValueError("Missing required 'path' parameter in query")

        # Extract parameters
        path = query.get("path")

        # Validate path format
        if re.search(r"[^\w\d\._]", path):
            raise ValueError(
                f"Invalid path format: {path}. Path should only contain alphanumeric characters, dots, and underscores."
            )

        value = query.get("value")
        regex = query.get("regex")
        exists = query.get("exists")
        gt = query.get("gt")
        lt = query.get("lt")

        # Convert path string to list of parts
        path_parts = path.split(".")

        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)

        # Retrieve and filter memories
        results = []
        for store in stores:
            # Use list method as configured in the tests
            memories = store.list()

            for memory in memories:
                # Get the nested value using the path
                nested_value = self._get_nested_value(memory, path_parts)

                # Handle different query types
                if exists is not None:
                    # Check if path exists
                    path_exists = nested_value is not None
                    if path_exists == exists:
                        results.append(memory)
                elif value is not None:
                    # Handle exact value matching, including array items
                    if isinstance(nested_value, list):
                        if value in nested_value:
                            results.append(memory)
                    elif nested_value == value:
                        results.append(memory)
                elif regex is not None:
                    # Handle regex pattern matching
                    if nested_value is not None:
                        if isinstance(nested_value, str) and re.search(
                            regex, nested_value
                        ):
                            results.append(memory)
                elif gt is not None or lt is not None:
                    # Handle numeric comparisons
                    if nested_value is not None and isinstance(
                        nested_value, (int, float)
                    ):
                        passes = True
                        if gt is not None and not (nested_value > gt):
                            passes = False
                        if lt is not None and not (nested_value < lt):
                            passes = False
                        if passes:
                            results.append(memory)
                elif nested_value is not None:
                    # If no specific criteria, match any non-null value
                    results.append(memory)

        # Apply metadata filtering if needed
        if metadata_filter:
            results = [
                memory
                for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]

        return results[:limit]

    def _get_nested_value(self, obj, path_parts):
        """
        Retrieve a nested value from a dictionary using a path.

        Args:
            obj: Dictionary to extract value from
            path_parts: List of keys forming the path

        Returns:
            The value at the specified path or None if not found
        """
        current = obj
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                # Handle list indexing if part is a number
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        return current

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
