"""Core classes for the agent memory system."""

import logging
from typing import Any, Dict, List, Optional, Union
import time
import uuid

from agent_memory.config import MemoryConfig
from agent_memory.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


class AgentMemorySystem:
    """Central manager for all agent memory components.

    This class serves as the main entry point for the agent memory system,
    managing memory agents for multiple agents and providing global configuration.

    Attributes:
        config: Configuration for the memory system
        agents: Dictionary of agent_id to MemoryAgent instances
    """

    _instance = None

    @classmethod
    def get_instance(cls, config: Optional[MemoryConfig] = None) -> "AgentMemorySystem":
        """Get or create the singleton instance of the AgentMemorySystem."""
        if cls._instance is None:
            cls._instance = cls(config or MemoryConfig())
        return cls._instance

    def __init__(self, config: MemoryConfig):
        """Initialize the AgentMemorySystem.

        Args:
            config: Configuration for the memory system
        """
        self.config = config
        self.agents: Dict[str, MemoryAgent] = {}

        # Configure logging
        logging.basicConfig(level=getattr(logging, config.logging_level))
        logger.info("AgentMemorySystem initialized with configuration: %s", config)

    def get_memory_agent(self, agent_id: str) -> MemoryAgent:
        """Get or create a MemoryAgent for the specified agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            MemoryAgent instance for the specified agent
        """
        if agent_id not in self.agents:
            self.agents[agent_id] = MemoryAgent(agent_id, self.config)
            logger.debug("Created new MemoryAgent for agent %s", agent_id)

        return self.agents[agent_id]

    def store_agent_state(
        self,
        agent_id: str,
        state_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store an agent's state in memory.

        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary of state attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_state(state_data, step_number, priority)

    def store_agent_interaction(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store information about an agent's interaction.

        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary of interaction attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_interaction(interaction_data, step_number, priority)

    def store_agent_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store information about an agent's action.

        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary of action attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_action(action_data, step_number, priority)

    # Add memory retrieval methods
    def retrieve_similar_states(
        self,
        agent_id: str,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar past states to the provided query state.

        Args:
            agent_id: Unique identifier for the agent
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity to query state
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.retrieve_similar_states(query_state, k, memory_type)

    def retrieve_by_time_range(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a specific time/step range.

        Args:
            agent_id: Unique identifier for the agent
            start_step: Beginning of time range
            end_step: End of time range
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries within the specified time range
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.retrieve_by_time_range(start_step, end_step, memory_type)

    def retrieve_by_attributes(
        self,
        agent_id: str,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories matching specific attribute values.

        Args:
            agent_id: Unique identifier for the agent
            attributes: Dictionary of attribute-value pairs to match
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries matching the specified attributes
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.retrieve_by_attributes(attributes, memory_type)

    # Add memory statistics method
    def get_memory_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics about an agent's memory usage.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary containing memory statistics
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.get_memory_statistics()

    # Add memory management methods
    def force_memory_maintenance(self, agent_id: Optional[str] = None) -> bool:
        """Force memory tier transitions and cleanup operations.

        Args:
            agent_id: Optional agent ID to restrict maintenance to a single agent

        Returns:
            True if maintenance was successful
        """
        if agent_id:
            memory_agent = self.get_memory_agent(agent_id)
            return memory_agent.force_maintenance()
        else:
            # Perform maintenance for all agents
            success = True
            for agent_id, memory_agent in self.agents.items():
                if not memory_agent.force_maintenance():
                    logger.error("Maintenance failed for agent %s", agent_id)
                    success = False
            return success

    # Add advanced query methods
    def search_by_embedding(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_tiers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find memories by raw embedding vector similarity.

        Args:
            agent_id: Identifier for the agent
            query_embedding: Embedding vector to search with
            k: Number of results to return
            memory_tiers: Optional list of tiers to search

        Returns:
            List of memory entries sorted by similarity
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.search_by_embedding(query_embedding, k, memory_tiers)

    def search_by_content(
        self, agent_id: str, content_query: Union[str, Dict[str, Any]], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for memories based on content text/attributes.

        Args:
            agent_id: Identifier for the agent
            content_query: String or dict to search for in memory contents
            k: Number of results to return

        Returns:
            List of memory entries matching the content query
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.search_by_content(content_query, k)

    # Add memory hook methods
    def register_memory_hook(
        self, agent_id: str, event_type: str, hook_function: callable, priority: int = 5
    ) -> bool:
        """Register a hook function for memory formation events.

        Args:
            agent_id: Identifier for the agent
            event_type: Type of event to hook into
            hook_function: Function to call when event is triggered
            priority: Priority level (1-10, 10 being highest)

        Returns:
            True if hook was registered successfully
        """
        if not self.config.enable_memory_hooks:
            logger.warning("Memory hooks are disabled in configuration")
            return False

        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.register_hook(event_type, hook_function, priority)

    def trigger_memory_event(
        self, agent_id: str, event_type: str, event_data: Dict[str, Any]
    ) -> bool:
        """Trigger memory formation event hooks.

        Args:
            agent_id: Identifier for the agent
            event_type: Type of event that occurred
            event_data: Data related to the event

        Returns:
            True if event was processed successfully
        """
        if not self.config.enable_memory_hooks:
            logger.warning("Memory hooks are disabled in configuration")
            return False

        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.trigger_event(event_type, event_data)

    def clear_all_memories(self) -> bool:
        """Clear all memory data for all agents.

        Returns:
            True if clearing was successful
        """
        success = True
        for agent_id, memory_agent in self.agents.items():
            if not memory_agent.clear_memory():
                logger.error("Failed to clear memory for agent %s", agent_id)
                success = False

        # Reset agent dictionary
        self.agents = {}
        return success

    def add_memory(self, memory_data: Dict[str, Any]) -> str:
        """Add a memory entry to the system.
        
        Args:
            memory_data: Dictionary containing memory data
            
        Returns:
            memory_id of the added memory
        """
        # Generate memory_id if not provided
        memory_id = memory_data.get("memory_id", f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}")
        memory_data["memory_id"] = memory_id
        
        # Get agent_id from memory or use a default
        agent_id = memory_data.get("agent_id", "default_agent")
        
        # Get or create memory agent
        memory_agent = self.get_memory_agent(agent_id)
        
        # Store in STM
        step_number = memory_data.get("step_number", 0)
        priority = memory_data.get("metadata", {}).get("importance_score", 1.0)
        memory_type = memory_data.get("type", "generic")
        
        # Choose appropriate method based on memory type
        if memory_type == "state":
            memory_agent.store_state(memory_data.get("content", {}), step_number, priority)
        elif memory_type == "interaction":
            memory_agent.store_interaction(memory_data.get("content", {}), step_number, priority)
        elif memory_type == "action":
            memory_agent.store_action(memory_data.get("content", {}), step_number, priority)
        else:
            # For generic types, use store_state as a fallback
            memory_agent.store_state(memory_data.get("content", {}), step_number, priority)
            
        return memory_id
        
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by its memory_id.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory entry or None if not found
        """
        # Check in all agents
        for agent_id, memory_agent in self.agents.items():
            # Try STM first
            memory = memory_agent.stm_store.get(agent_id, memory_id)
            if memory:
                return memory
                
            # Try IM next
            memory = memory_agent.im_store.get(agent_id, memory_id)
            if memory:
                return memory
                
            # Try LTM last
            memory = memory_agent.ltm_store.get(memory_id)
            if memory:
                return memory
                
        # Memory not found
        logger.warning(f"Memory with ID {memory_id} not found in any agent's stores")
        return None
