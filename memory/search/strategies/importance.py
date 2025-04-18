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
        # Extract parameters
        if isinstance(query, (int, float)):
            min_importance = float(query)
        else:
            min_importance = float(query.get("min_importance", 0.7))
            
        sort_order = kwargs.get("sort_order", "desc")
        
        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)
        
        # Retrieve memories meeting the importance threshold
        results = []
        for store in stores:
            # Try to use optimized importance query if available
            if hasattr(store, "get_by_min_importance"):
                important_memories = store.get_by_min_importance(
                    agent_id, 
                    min_importance
                )
                results.extend(important_memories)
            else:
                # Fallback to filtering all memories
                memories = store.get_all(agent_id)
                for memory in memories:
                    importance = memory.get("metadata", {}).get("importance", 0)
                    if importance >= min_importance:
                        results.append(memory)
        
        # Apply metadata filtering
        if metadata_filter:
            results = [
                memory for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]
        
        # Sort by importance
        reverse_sort = sort_order.lower() == "desc"
        results.sort(
            key=lambda x: x.get("metadata", {}).get("importance", 0), 
            reverse=reverse_sort
        )
        
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