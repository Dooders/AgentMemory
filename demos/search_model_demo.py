#!/usr/bin/env python
"""Demonstration of the extensible search model.

This script demonstrates how to use the search model with different search
strategies, and how to create custom search strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from memory.config import MemoryConfig
from memory.core import AgentMemorySystem
from memory.search.model import SearchModel
from memory.search.strategies.base import SearchStrategy
from memory.search.strategies.similarity import SimilaritySearchStrategy
from memory.search.strategies.combined import CombinedSearchStrategy


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomSearchStrategy(SearchStrategy):
    """Example of a custom search strategy.
    
    This strategy demonstrates how to implement a custom search approach
    that could be integrated with the search model.
    """
    
    def __init__(self, memory_system: AgentMemorySystem):
        """Initialize the custom search strategy.
        
        Args:
            memory_system: The agent memory system to use
        """
        self.memory_system = memory_system
    
    def name(self) -> str:
        """Return the name of the search strategy."""
        return "custom"
    
    def description(self) -> str:
        """Return a description of the search strategy."""
        return "Custom search strategy for demonstration purposes"
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Implement a custom search approach.
        
        This example simply delegates to the memory_agent's retrieve_by_attributes
        method for simplicity. A real custom strategy would implement its own
        logic here.
        
        Args:
            query: Search query
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search
            **kwargs: Additional parameters
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Get the memory agent for this agent
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        
        # Create attribute query from input
        if isinstance(query, str):
            attributes = {"content": query}
        elif isinstance(query, dict):
            attributes = query
        else:
            # Can't handle list/vector inputs in this simple example
            return []
        
        # Apply metadata filter if provided
        if metadata_filter:
            attributes.update(metadata_filter)
        
        # Perform the search
        results = memory_agent.retrieve_by_attributes(attributes)
        
        # Limit results
        return results[:limit]


def main():
    """Run the search model demonstration."""
    # Initialize the agent memory system
    config = MemoryConfig()
    memory_system = AgentMemorySystem.get_instance(config)
    
    # Create a test agent and add some memories
    agent_id = "demo_agent"
    memory_agent = memory_system.get_memory_agent(agent_id)
    
    # Store some test memories
    for i in range(10):
        memory_system.store_agent_state(
            agent_id=agent_id,
            state_data={
                "content": f"This is test memory {i}",
                "location": f"location_{i % 3}",
                "value": i * 10
            },
            step_number=i,
            priority=0.5 + (i * 0.05)
        )
    
    logger.info("Stored 10 test memories for agent %s", agent_id)
    
    # Create the search model
    search_model = SearchModel(config)
    
    # Get a memory agent with its dependencies for similarity search
    agent = memory_system.get_memory_agent(agent_id)
    
    # Create and register the similarity search strategy
    similarity_strategy = SimilaritySearchStrategy(
        vector_store=agent.embedding_engine.vector_store,
        embedding_engine=agent.embedding_engine,
        stm_store=agent.stm_store,
        im_store=agent.im_store,
        ltm_store=agent.ltm_store
    )
    search_model.register_strategy(similarity_strategy, make_default=True)
    
    # Create and register a custom search strategy
    custom_strategy = CustomSearchStrategy(memory_system)
    search_model.register_strategy(custom_strategy)
    
    # Create a combined strategy using both
    combined_strategy = CombinedSearchStrategy(
        strategies={
            "similarity": similarity_strategy,
            "custom": custom_strategy
        },
        weights={
            "similarity": 0.7,
            "custom": 0.3
        }
    )
    search_model.register_strategy(combined_strategy)
    
    # List available strategies
    available_strategies = search_model.get_available_strategies()
    logger.info("Available search strategies:")
    for name, description in available_strategies.items():
        logger.info("  - %s: %s", name, description)
    
    # Perform a search using the default strategy
    logger.info("\nPerforming search with default strategy:")
    results = search_model.search(
        query="test memory",
        agent_id=agent_id,
        limit=5
    )
    
    for i, result in enumerate(results):
        logger.info(
            "Result %d: %s (score: %.2f)",
            i + 1,
            result.get("content", "Unknown"),
            result.get("metadata", {}).get("similarity_score", 0.0)
        )
    
    # Perform a search with a specific strategy
    logger.info("\nPerforming search with combined strategy:")
    results = search_model.search(
        query="test memory",
        agent_id=agent_id,
        strategy_name="combined",
        limit=5
    )
    
    for i, result in enumerate(results):
        logger.info(
            "Result %d: %s (score: %.2f, source: %s)",
            i + 1,
            result.get("content", "Unknown"),
            result.get("metadata", {}).get("combined_score", 0.0),
            result.get("metadata", {}).get("source_strategy", "Unknown")
        )


if __name__ == "__main__":
    main() 