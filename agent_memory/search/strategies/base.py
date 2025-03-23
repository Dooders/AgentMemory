"""Base search strategy for the agent memory search model."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class SearchStrategy(ABC):
    """Abstract base class for all search strategies.
    
    Search strategies define different approaches to retrieving agent memories,
    such as by similarity, temporal attributes, content attributes, etc.
    
    Subclasses must implement the search method to define their specific
    search behavior.
    """
    
    @abstractmethod
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories according to the strategy's implementation.
        
        Args:
            query: Search query, which could be a string, dictionary, or embedding
                  depending on the specific strategy implementation
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of memory entries matching the search criteria
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the search strategy.
        
        Returns:
            String name of the strategy
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return a description of the search strategy.
        
        Returns:
            String description of the strategy
        """
        pass 