"""Core classes for the Tiered Adaptive Semantic Memory (TASM) system."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from memory.agent_memory import MemoryAgent
from memory.config import MemoryConfig

logger = logging.getLogger(__name__)

# System constants
SYSTEM_NAME = "TASM"  # Tiered Adaptive Semantic Memory


class AgentMemorySystem:
    """Central manager for the Tiered Adaptive Semantic Memory (TASM) system.

    This class serves as the main entry point for the TASM system,
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
        logger.info("TASM system initialized with configuration: %s", config)

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
        tier: str = "stm",
    ) -> bool:
        """Store an agent's state in memory.

        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary of state attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_state(state_data, step_number, priority, tier)

    def store_agent_interaction(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
        tier: str = "stm",
    ) -> bool:
        """Store information about an agent's interaction.

        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary of interaction attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_interaction(
            interaction_data, step_number, priority, tier
        )

    def store_agent_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
        tier: str = "stm",
    ) -> bool:
        """Store information about an agent's action.

        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary of action attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_action(action_data, step_number, priority, tier)

    # Add memory retrieval methods
    def retrieve_similar_states(
        self,
        agent_id: str,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
        threshold: float = 0.6,
        context_weights: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar states to the provided query.

        Args:
            agent_id: ID of the agent to retrieve memories for
            query_state: Query state to compare against
            k: Number of results to return
            memory_type: Optional memory type filter
            threshold: Minimum similarity score threshold (0.0-1.0)
            context_weights: Optional dictionary mapping keys to importance weights

        Returns:
            List of memory entries sorted by similarity
        """
        self._check_agent_exists(agent_id)
        memory_agent = self._get_agent(agent_id)
        return memory_agent.retrieve_similar_states(
            query_state, k, memory_type, threshold, context_weights
        )

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
    def get_memory_statistics(
        self, agent_id: str, simplified: bool = False
    ) -> Dict[str, Any]:
        """Get statistics about an agent's memory usage.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary containing memory statistics
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.get_memory_statistics(simplified)

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
        memory_id = memory_data.get(
            "memory_id", f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        )
        memory_data["memory_id"] = memory_id

        # Get agent_id from memory or use a default
        agent_id = memory_data.get("agent_id", "default_agent")

        # Get or create memory agent
        memory_agent = self.get_memory_agent(agent_id)

        # Store in STM
        step_number = memory_data.get("step_number", 0)
        priority = memory_data.get("metadata", {}).get("importance_score", 1.0)
        memory_type = memory_data.get("type", "generic")
        tier = memory_data.get("metadata", {}).get(
            "tier", "stm"
        )  # Get tier from metadata

        # Choose appropriate method based on memory type
        if memory_type == "state":
            memory_agent.store_state(
                memory_data.get("content", {}), step_number, priority, tier
            )
        elif memory_type == "interaction":
            memory_agent.store_interaction(
                memory_data.get("content", {}), step_number, priority, tier
            )
        elif memory_type == "action":
            memory_agent.store_action(
                memory_data.get("content", {}), step_number, priority, tier
            )
        else:
            # For generic types, use store_state as a fallback
            memory_agent.store_state(
                memory_data.get("content", {}), step_number, priority, tier
            )

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

    def hybrid_retrieve(
        self,
        agent_id: str,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
        vector_weight: float = 0.7,
        attribute_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Combine similarity and attribute-based search for more robust retrieval.

        Args:
            agent_id: ID of the agent to retrieve memories for
            query_state: State data to use for querying
            k: Number of results to return
            memory_type: Optional filter for specific memory types
            vector_weight: Weight to assign to vector similarity scores (0.0-1.0)
            attribute_weight: Weight to assign to attribute match scores (0.0-1.0)

        Returns:
            List of memory entries sorted by hybrid score
        """
        self._check_agent_exists(agent_id)
        memory_agent = self._get_agent(agent_id)
        return memory_agent.hybrid_retrieve(
            query_state, k, memory_type, vector_weight, attribute_weight
        )

    def save_to_json(self, filepath: str) -> bool:
        """Save the memory system to a JSON file.

        Args:
            filepath: Path to save the JSON file

        Returns:
            True if saving was successful
        """
        from memory.utils.serialization import save_memory_system_to_json

        return save_memory_system_to_json(self, filepath)

    @classmethod
    def load_from_json(
        cls, filepath: str, use_mock_redis: bool = False
    ) -> Optional["AgentMemorySystem"]:
        """Load a memory system from a JSON file.

        Args:
            filepath: Path to the JSON file
            use_mock_redis: Whether to use MockRedis for Redis storage

        Returns:
            AgentMemorySystem instance or None if loading failed
        """
        from memory.utils.serialization import load_memory_system_from_json

        return load_memory_system_from_json(filepath, use_mock_redis)

    def _check_agent_exists(self, agent_id: str) -> None:
        """Check if an agent exists and raise an error if not.

        Args:
            agent_id: ID of the agent to check

        Raises:
            ValueError: If the agent doesn't exist
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist")

    def _get_agent(self, agent_id: str) -> MemoryAgent:
        """Get the memory agent for an agent ID.

        This is a helper method that verifies the agent exists before returning.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            MemoryAgent instance for the specified agent

        Raises:
            ValueError: If the agent doesn't exist
        """
        self._check_agent_exists(agent_id)
        return self.agents[agent_id]
