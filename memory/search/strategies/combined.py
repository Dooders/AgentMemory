"""Combined search strategy that integrates results from multiple strategies."""

import logging
from typing import Any, Dict, List, Optional, Union

from memory.search.strategies.base import SearchStrategy

logger = logging.getLogger(__name__)


class CombinedSearchStrategy(SearchStrategy):
    """Search strategy that combines results from multiple strategies.
    
    This strategy enables more sophisticated searches by combining and 
    reranking results from multiple underlying search strategies.
    
    Attributes:
        strategies: Dictionary of strategy name to strategy instance
        weights: Dictionary of strategy name to weight for result ranking
    """
    
    def __init__(self, strategies: Dict[str, SearchStrategy], weights: Optional[Dict[str, float]] = None):
        """Initialize the combined search strategy.
        
        Args:
            strategies: Dictionary of strategy name to strategy instance
            weights: Optional dictionary of strategy name to weight for result ranking
        """
        self.strategies = strategies
        
        # If weights not provided, use equal weights for all strategies
        if weights is None:
            self.weights = {name: 1.0 for name in strategies.keys()}
        else:
            # Ensure all strategies have weights
            self.weights = weights.copy()
            for name in strategies.keys():
                if name not in self.weights:
                    self.weights[name] = 1.0
    
    def name(self) -> str:
        """Return the name of the search strategy.
        
        Returns:
            String name of the strategy
        """
        return "combined"
    
    def description(self) -> str:
        """Return a description of the search strategy.
        
        Returns:
            String description of the strategy
        """
        strategy_names = ", ".join(self.strategies.keys())
        return f"Combines results from multiple search strategies: {strategy_names}"
    
    def set_weights(self, weights: Dict[str, float]) -> bool:
        """Set the weights for different strategies.
        
        Args:
            weights: Dictionary of strategy name to weight
            
        Returns:
            True if weights were set successfully
        """
        # Update only the weights for strategies that exist
        for name, weight in weights.items():
            if name in self.strategies:
                self.weights[name] = weight
        
        logger.debug("Updated strategy weights: %s", self.weights)
        return True
    
    def _get_individual_results(self, strategy_name, strategy, strategy_query, agent_id, limit, 
                               metadata_filter, tier, strategy_kwargs):
        """Get results from an individual strategy with error handling."""
        try:
            # Log what query is being sent to which strategy
            logger.debug(f"Sending query of type {type(strategy_query)} to strategy {strategy_name}: {strategy_query}")
            
            results = strategy.search(
                query=strategy_query,
                agent_id=agent_id,
                limit=limit * 2,  # Get more results for better combination
                metadata_filter=metadata_filter,
                tier=tier,
                **strategy_kwargs
            )
            return results
        except Exception as e:
            logger.warning(f"Error in strategy {strategy_name}: {str(e)}")
            return []
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories using multiple strategies.
        
        Args:
            query: Search query - can be string, dict with query types for each strategy, or vector
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search
            strategy_params: Optional parameters specific to each strategy
            **kwargs: Additional parameters
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Default strategy parameters
        if strategy_params is None:
            strategy_params = {}
        
        # Initialize combined results
        all_results: Dict[str, Dict[str, Any]] = {}
        
        # Handle direct strategy routing if query is a dict
        if isinstance(query, dict):
            logger.debug(f"Combined search with dictionary query: {query}")
        
        # Get results from each strategy
        for strategy_name, strategy in self.strategies.items():
            # Get strategy-specific parameters
            strategy_kwargs = strategy_params.get(strategy_name, {})
            strategy_kwargs.update(kwargs)
            
            # Extract appropriate query based on strategy type
            strategy_query = None
            
            # First priority: direct strategy mappings in query dict
            if isinstance(query, dict) and strategy_name in query:
                strategy_query = query[strategy_name]
                logger.debug(f"Using direct strategy mapping for {strategy_name}: {strategy_query}")
            
            # Second priority: infer from known field names
            elif isinstance(query, dict):
                if strategy_name == "temporal" and "timestamp" in query:
                    strategy_query = query["timestamp"]
                    logger.debug(f"Using timestamp for temporal strategy: {strategy_query}")
                elif strategy_name == "attribute" and "text" in query:
                    strategy_query = query["text"]
                    logger.debug(f"Using text for attribute strategy: {strategy_query}")
                elif strategy_name == "similarity" and "vector" in query:
                    strategy_query = query["vector"]
                    logger.debug(f"Using vector for similarity strategy: {strategy_query}")
            
            # Default: use original query if no matching pattern was found
            if strategy_query is None:
                strategy_query = query
                logger.debug(f"Using original query for {strategy_name}: {strategy_query}")
            
            # Execute the search with the appropriate query
            results = self._get_individual_results(
                strategy_name, 
                strategy,
                strategy_query,
                agent_id,
                limit,
                metadata_filter,
                tier,
                strategy_kwargs
            )
            
            # Extract strategy weight
            strategy_weight = self.weights.get(strategy_name, 1.0)
            
            # Add results to the combined pool, using memory ID as key to avoid duplicates
            for memory in results:
                memory_id = memory.get("id", "")
                
                if not memory_id:
                    # Try alternative keys if "id" isn't present
                    memory_id = memory.get("memory_id", "")
                    if not memory_id:
                        continue
                
                # Calculate score based on strategy weight and position in results
                base_score = memory.get("metadata", {}).get("similarity_score", 0.0)
                position_score = 1.0 - (results.index(memory) / len(results)) if results else 0.0
                combined_score = (base_score * 0.7 + position_score * 0.3) * strategy_weight
                
                # Store or update in combined results
                if memory_id not in all_results or combined_score > all_results[memory_id].get("metadata", {}).get("combined_score", 0.0):
                    memory_copy = memory.copy()
                    if "metadata" not in memory_copy:
                        memory_copy["metadata"] = {}
                    
                    memory_copy["metadata"]["source_strategy"] = strategy_name
                    memory_copy["metadata"]["combined_score"] = combined_score
                    all_results[memory_id] = memory_copy
        
        # Convert dictionary to list and sort by combined score
        combined_list = list(all_results.values())
        combined_list.sort(
            key=lambda x: x.get("metadata", {}).get("combined_score", 0.0),
            reverse=True
        )
        
        # Return limited results
        return combined_list[:limit] 