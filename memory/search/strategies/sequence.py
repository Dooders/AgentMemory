from typing import Dict, List, Any, Optional, Union
from .base import SearchStrategy


class NarrativeSequenceStrategy(SearchStrategy):
    """
    Strategy for retrieving a sequence of memories surrounding a reference memory
    to form a contextual narrative.
    """
    
    def __init__(self, stm_store, im_store, ltm_store):
        """
        Initialize the narrative sequence strategy.
        
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
        return "narrative_sequence"
    
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
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for a narrative sequence of memories surrounding a reference memory.
        
        Args:
            query: Either the memory_id as a string or a dict containing memory_id
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters:
                - context_before: Number of memories to include before the reference (default: 3)
                - context_after: Number of memories to include after the reference (default: 3)
                
        Returns:
            List of memory entries forming a narrative sequence
        """
        # Extract parameters
        memory_id = query if isinstance(query, str) else query.get("memory_id")
        context_before = kwargs.get("context_before", 3)
        context_after = kwargs.get("context_after", 3)
        
        if not memory_id:
            return []
        
        # Get the stores to search based on tier
        stores = self._get_stores_for_tier(tier)
        
        # Search logic for narrative sequence
        results = []
        for store in stores:
            # Find the reference memory
            reference_memory = store.get(memory_id, agent_id)
            if reference_memory:
                # Get memories before the reference
                before_memories = store.get_range_by_step(
                    agent_id,
                    reference_memory["step_number"] - context_before,
                    reference_memory["step_number"] - 1
                )
                
                # Get memories after the reference
                after_memories = store.get_range_by_step(
                    agent_id,
                    reference_memory["step_number"] + 1,
                    reference_memory["step_number"] + context_after
                )
                
                # Combine into sequence
                sequence = before_memories + [reference_memory] + after_memories
                
                # Apply any metadata filters
                if metadata_filter:
                    sequence = [
                        memory for memory in sequence
                        if self._matches_metadata_filters(memory, metadata_filter)
                    ]
                
                # Add to results and break since we found the reference memory
                results.extend(sequence)
                break
                
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