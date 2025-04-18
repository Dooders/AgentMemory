from typing import Dict, List, Any, Optional, Union
from .base import SearchStrategy


class ExampleMatchingStrategy(SearchStrategy):
    """
    Strategy for finding memories that match a provided example pattern
    using semantic similarity.
    """
    
    def __init__(self, vector_store, embedding_engine, stm_store, im_store, ltm_store):
        """
        Initialize the example matching strategy.
        
        Args:
            vector_store: Vector store for similarity calculations
            embedding_engine: Embedding engine for encoding examples
            stm_store: Short-term memory store
            im_store: Intermediate memory store
            ltm_store: Long-term memory store
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def name(self) -> str:
        """Return the name of the strategy."""
        return "example_matching"
    
    def description(self) -> str:
        """Return the description of the strategy."""
        return "Finds memories that match a provided example pattern using semantic similarity"
    
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
        Search for memories that match the provided example pattern.
        
        Args:
            query: Example pattern to match
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters:
                - min_score: Minimum similarity score threshold (default: 0.6)
                - embedding_type: Type of embedding to use (default: "full_vector")
                
        Returns:
            List of matching memory entries
        """
        # Extract parameters
        example = query
        min_score = kwargs.get("min_score", 0.6)
        embedding_type = kwargs.get("embedding_type", "full_vector")
        
        # Encode the example
        example_vector = self.embedding_engine.encode(example)
        if not example_vector:
            return []
        
        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)
        
        # Retrieve and process memories
        all_memories = []
        for store in stores:
            memories = store.get_all(agent_id)
            all_memories.extend(memories)
            
        # Calculate similarity and filter by score
        results = []
        for memory in all_memories:
            memory_vector = memory.get("embeddings", {}).get(embedding_type)
            if memory_vector:
                similarity = self.vector_store.calculate_similarity(example_vector, memory_vector)
                if similarity >= min_score:
                    memory["metadata"]["similarity_score"] = similarity
                    results.append(memory)
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["metadata"]["similarity_score"], reverse=True)
        
        # Apply metadata filtering
        if metadata_filter:
            results = [
                memory for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]
        
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