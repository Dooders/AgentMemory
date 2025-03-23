"""Search model for agent memory retrieval."""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from agent_memory.config import MemoryConfig
from agent_memory.search.strategies.base import SearchStrategy

logger = logging.getLogger(__name__)


class SearchModel:
    """Search model for retrieving agent memories using various strategies.
    
    This class provides a unified interface for searching agent memories using 
    different search strategies. It manages a registry of available strategies 
    and allows for easy switching between them.
    
    Attributes:
        config: Configuration for the memory system
        strategies: Dictionary of registered search strategies
        default_strategy: The default search strategy to use
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the search model.
        
        Args:
            config: Configuration for the memory system
        """
        self.config = config
        self.strategies: Dict[str, SearchStrategy] = {}
        self.default_strategy: Optional[str] = None
        
        logger.debug("SearchModel initialized with configuration: %s", config)
    
    def register_strategy(self, strategy: SearchStrategy, make_default: bool = False) -> bool:
        """Register a search strategy with the model.
        
        Args:
            strategy: The search strategy to register
            make_default: Whether to set this as the default strategy
            
        Returns:
            True if registration was successful
        """
        strategy_name = strategy.name()
        self.strategies[strategy_name] = strategy
        
        if make_default or self.default_strategy is None:
            self.default_strategy = strategy_name
            logger.debug("Set default search strategy to %s", strategy_name)
        
        logger.debug("Registered search strategy: %s", strategy_name)
        return True
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a search strategy from the model.
        
        Args:
            strategy_name: Name of the strategy to unregister
            
        Returns:
            True if unregistration was successful
        """
        if strategy_name not in self.strategies:
            logger.warning("Attempted to unregister unknown strategy: %s", strategy_name)
            return False
        
        # If removing the default strategy, set a new default if possible
        if strategy_name == self.default_strategy and self.strategies:
            # Pick the first available strategy as the new default
            self.default_strategy = next(iter(self.strategies.keys()))
            logger.debug("Set new default search strategy to %s", self.default_strategy)
        elif strategy_name == self.default_strategy:
            # No other strategies available
            self.default_strategy = None
        
        del self.strategies[strategy_name]
        logger.debug("Unregistered search strategy: %s", strategy_name)
        return True
    
    def set_default_strategy(self, strategy_name: str) -> bool:
        """Set the default search strategy.
        
        Args:
            strategy_name: Name of the strategy to set as default
            
        Returns:
            True if setting the default was successful
        """
        if strategy_name not in self.strategies:
            logger.warning("Attempted to set unknown strategy as default: %s", strategy_name)
            return False
        
        self.default_strategy = strategy_name
        logger.debug("Set default search strategy to %s", strategy_name)
        return True
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str, 
        strategy_name: Optional[str] = None,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories using the specified or default strategy.
        
        Args:
            query: Search query, format depends on the strategy used
            agent_id: ID of the agent whose memories to search
            strategy_name: Name of the strategy to use, or None for default
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Use default strategy if none specified
        strategy_name = strategy_name or self.default_strategy
        
        if not strategy_name or strategy_name not in self.strategies:
            logger.error("No valid search strategy available for search")
            return []
        
        strategy = self.strategies[strategy_name]
        
        logger.debug(
            "Performing search with strategy %s for agent %s", 
            strategy_name, 
            agent_id
        )
        
        return strategy.search(
            query=query,
            agent_id=agent_id,
            limit=limit,
            metadata_filter=metadata_filter,
            tier=tier,
            **kwargs
        )
    
    def get_available_strategies(self) -> Dict[str, str]:
        """Get a dictionary of available search strategies.
        
        Returns:
            Dictionary mapping strategy names to their descriptions
        """
        return {name: strategy.description() for name, strategy in self.strategies.items()} 