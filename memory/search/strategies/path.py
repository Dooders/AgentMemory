from typing import Dict, List, Any, Optional, Union
import re
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
        return "content_path"
    
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
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for memories based on content path values or patterns.
        
        Args:
            query: Dict containing search parameters:
                - path: Dot-notation path to the content field (e.g., "location.name")
                - value: Exact value to match (optional)
                - pattern: Pattern to match (optional)
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters:
                - use_regex: Whether to use regex for pattern matching (default: False)
                - case_sensitive: Whether to use case-sensitive matching (default: False)
                
        Returns:
            List of matching memory entries
        """
        # Extract parameters
        path = query.get("path")
        value = query.get("value")
        pattern = query.get("pattern")
        
        if not path:
            return []
            
        use_regex = kwargs.get("use_regex", False)
        case_sensitive = kwargs.get("case_sensitive", False)
        
        # Convert path string to list of parts
        path_parts = path.split(".")
        
        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)
        
        # Retrieve and filter memories
        results = []
        for store in stores:
            memories = store.get_all(agent_id)
            for memory in memories:
                # Get the content field value using the path
                content_value = self._get_nested_value(memory.get("contents", {}), path_parts)
                
                if content_value is not None:
                    # Check for matches based on provided criteria
                    if pattern is not None:
                        # Convert content value to string for pattern matching
                        content_str = str(content_value)
                        if not case_sensitive:
                            content_str = content_str.lower()
                            pattern = pattern.lower()
                            
                        if use_regex:
                            # Use regex pattern matching
                            if re.search(pattern, content_str):
                                results.append(memory)
                        else:
                            # Use simple substring matching
                            if pattern in content_str:
                                results.append(memory)
                    elif value is not None:
                        # Use exact value matching
                        if content_value == value:
                            results.append(memory)
                    else:
                        # If neither pattern nor value specified, match any non-null value
                        results.append(memory)
        
        # Apply metadata filtering
        if metadata_filter:
            results = [
                memory for memory in results
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