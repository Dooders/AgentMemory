"""Similarity-based search strategy for the agent memory search model."""

import logging
from typing import Any, Dict, List, Optional, Union

from agent_memory.embeddings.autoencoder import AutoencoderEmbeddingEngine
from agent_memory.embeddings.vector_store import VectorStore
from agent_memory.search.strategies.base import SearchStrategy
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)


class SimilaritySearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on semantic similarity.
    
    This strategy uses vector embeddings to find memories that are semantically
    similar to the provided query.
    
    Attributes:
        vector_store: Vector store for similarity queries
        embedding_engine: Engine for generating embeddings
        stm_store: Short-Term Memory store
        im_store: Intermediate Memory store
        ltm_store: Long-Term Memory store
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: AutoencoderEmbeddingEngine,
        stm_store: RedisSTMStore,
        im_store: RedisIMStore,
        ltm_store: SQLiteLTMStore,
    ):
        """Initialize the similarity search strategy.
        
        Args:
            vector_store: Vector store for similarity queries
            embedding_engine: Engine for generating embeddings
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def name(self) -> str:
        """Return the name of the search strategy.
        
        Returns:
            String name of the strategy
        """
        return "similarity"
    
    def description(self) -> str:
        """Return a description of the search strategy.
        
        Returns:
            String description of the strategy
        """
        return "Searches for memories based on semantic similarity using vector embeddings"
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        min_score: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories based on semantic similarity.
        
        Args:
            query: Search query, can be a text string, state dictionary, or embedding vector
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            min_score: Minimum similarity score threshold (0.0-1.0)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Initialize results
        results = []
        
        # Process all tiers or only the specified one
        tiers_to_search = ["stm", "im", "ltm"] if tier is None else [tier]
        
        for current_tier in tiers_to_search:
            # Skip if tier is not supported
            if current_tier not in ["stm", "im", "ltm"]:
                logger.warning("Unsupported memory tier: %s", current_tier)
                continue
            
            # Generate query vector from input
            query_vector = self._generate_query_vector(query, current_tier)
            
            # Skip if vector generation failed
            if query_vector is None:
                logger.warning("Failed to generate query vector for tier: %s", current_tier)
                continue
            
            # Find similar vectors
            similar_vectors = self.vector_store.find_similar_memories(
                query_vector,
                tier=current_tier,
                limit=limit * 2,  # Get extra results to allow for score filtering
                metadata_filter=metadata_filter or {},
            )
            
            # Filter by score
            filtered_vectors = [v for v in similar_vectors if v["score"] >= min_score]
            
            # Limit results
            filtered_vectors = filtered_vectors[:limit]
            
            # Fetch full memory entries
            for result in filtered_vectors:
                memory_id = result["id"]
                memory = None
                
                # Look up in the appropriate store
                if current_tier == "stm":
                    memory = self.stm_store.get(memory_id)
                elif current_tier == "im":
                    memory = self.im_store.get(memory_id)
                else:  # ltm
                    memory = self.ltm_store.get(memory_id)
                
                if memory:
                    # Attach similarity score and tier information
                    if "metadata" not in memory:
                        memory["metadata"] = {}
                    memory["metadata"]["similarity_score"] = result["score"]
                    memory["metadata"]["memory_tier"] = current_tier
                    results.append(memory)
        
        # Sort by similarity score (descending)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("similarity_score", 0.0),
            reverse=True
        )
        
        # Limit final results
        return results[:limit]
    
    def _generate_query_vector(
        self, query: Union[str, Dict[str, Any], List[float]], tier: str
    ) -> Optional[List[float]]:
        """Generate a query vector from various input types.
        
        Args:
            query: Input query (string, dictionary, or vector)
            tier: Memory tier to generate vector for
            
        Returns:
            Vector embedding or None if generation failed
        """
        # If query is already a vector, use it directly
        if isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            return query
        
        # If query is a string, convert to a simple state dictionary
        if isinstance(query, str):
            query = {"content": query}
        
        # If query is a dictionary, encode it based on tier
        if isinstance(query, dict):
            if tier == "stm":
                return self.embedding_engine.encode_stm(query)
            elif tier == "im":
                return self.embedding_engine.encode_im(query)
            elif tier == "ltm":
                return self.embedding_engine.encode_ltm(query)
        
        # If we get here, we couldn't generate a vector
        logger.warning("Could not generate vector for query type: %s", type(query))
        return None 