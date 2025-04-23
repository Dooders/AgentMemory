"""Core classes for the Tiered Adaptive Semantic Memory (TASM) system."""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from memory.agent_memory import MemoryAgent
from memory.config import MemoryConfig
from memory.schema import MEMORY_SYSTEM_SCHEMA, validate_memory_system_json

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

        # Choose appropriate method based on memory type
        if memory_type == "state":
            memory_agent.store_state(
                memory_data.get("content", {}), step_number, priority
            )
        elif memory_type == "interaction":
            memory_agent.store_interaction(
                memory_data.get("content", {}), step_number, priority
            )
        elif memory_type == "action":
            memory_agent.store_action(
                memory_data.get("content", {}), step_number, priority
            )
        else:
            # For generic types, use store_state as a fallback
            memory_agent.store_state(
                memory_data.get("content", {}), step_number, priority
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
        try:
            # Create the data structure to save with proper config serialization
            config_dict = {}
            for key, value in self.config.__dict__.items():
                # Skip complex objects that aren't JSON serializable
                if (
                    isinstance(value, (str, int, float, bool, list, dict))
                    or value is None
                ):
                    config_dict[key] = value

            # Create the data structure to save
            data = {"config": config_dict, "agents": {}}

            # Save agent data
            for agent_id, agent in self.agents.items():
                # Get all memories from different tiers
                stm_memories = []
                im_memories = []
                ltm_memories = []

                try:
                    stm_memories = agent.stm_store.get_all(agent_id)
                    logger.info(
                        f"Retrieved {len(stm_memories)} STM memories for agent {agent_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get STM memories for agent {agent_id}: {e}"
                    )

                try:
                    im_memories = agent.im_store.get_all(agent_id)
                    logger.info(
                        f"Retrieved {len(im_memories)} IM memories for agent {agent_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get IM memories for agent {agent_id}: {e}"
                    )

                try:
                    ltm_memories = agent.ltm_store.get_all(agent_id)
                    logger.info(
                        f"Retrieved {len(ltm_memories)} LTM memories for agent {agent_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get LTM memories for agent {agent_id}: {e}"
                    )

                # Combine all memories
                all_memories = stm_memories + im_memories + ltm_memories
                logger.info(f"Total memories for agent {agent_id}: {len(all_memories)}")

                # Clean up non-serializable objects in memories
                clean_memories = []
                for i, memory in enumerate(all_memories):
                    # Make a copy of the memory to avoid modifying the original
                    clean_memory = {}
                    for k, v in memory.items():
                        # Skip non-serializable embeddings
                        if k == "embeddings":
                            clean_memory[k] = {}
                            for embed_key, embed_val in v.items():
                                # Convert numpy arrays to lists if needed
                                if hasattr(embed_val, "tolist"):
                                    clean_memory[k][embed_key] = embed_val.tolist()
                                elif (
                                    isinstance(
                                        embed_val, (list, dict, str, int, float, bool)
                                    )
                                    or embed_val is None
                                ):
                                    clean_memory[k][embed_key] = embed_val
                        else:
                            clean_memory[k] = v

                    # Ensure all required fields for schema validation are present
                    if "memory_id" not in clean_memory:
                        clean_memory["memory_id"] = (
                            f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                        )
                        logger.debug(f"Added memory_id to memory {i}")

                    if "agent_id" not in clean_memory:
                        clean_memory["agent_id"] = agent_id
                        logger.debug(f"Added agent_id to memory {i}")

                    if "content" not in clean_memory:
                        clean_memory["content"] = {}
                        logger.debug(f"Added empty content to memory {i}")

                    # Ensure the memory type is set correctly and consistently
                    memory_type = memory.get("type", "generic")
                    clean_memory["type"] = memory_type
                    logger.debug(f"Set memory type to {memory_type} for memory {i}")

                    # Ensure metadata is present with all required fields
                    if "metadata" not in clean_memory:
                        clean_memory["metadata"] = {}
                        logger.debug(f"Added empty metadata to memory {i}")

                    metadata = clean_memory["metadata"]
                    if "creation_time" not in metadata:
                        metadata["creation_time"] = int(time.time())

                    if "last_access_time" not in metadata:
                        metadata["last_access_time"] = int(time.time())

                    if "importance_score" not in metadata:
                        metadata["importance_score"] = memory.get("priority", 1.0)

                    if "retrieval_count" not in metadata:
                        metadata["retrieval_count"] = 0

                    if "current_tier" not in metadata:
                        metadata["current_tier"] = "stm"

                    # Ensure memory_type in metadata matches the top-level type
                    metadata["memory_type"] = memory_type

                    if "step_number" not in clean_memory:
                        clean_memory["step_number"] = 0

                    if "timestamp" not in clean_memory:
                        clean_memory["timestamp"] = int(time.time())

                    # Ensure embeddings structure is valid if present
                    if "embeddings" not in clean_memory:
                        clean_memory["embeddings"] = {}

                    # Log the memory structure after preparation
                    logger.debug(f"Memory {i} keys: {list(clean_memory.keys())}")
                    logger.debug(
                        f"Memory {i} metadata keys: {list(clean_memory.get('metadata', {}).keys())}"
                    )

                    clean_memories.append(clean_memory)

                logger.info(
                    f"Prepared {len(clean_memories)} memories for agent {agent_id}"
                )
                data["agents"][agent_id] = {
                    "agent_id": agent_id,
                    "memories": clean_memories,
                }

            # Log a summary of the data structure
            logger.info(f"Data keys: {list(data.keys())}")
            logger.info(f"Number of agents: {len(data.get('agents', {}))}")
            for a_id, a_data in data.get("agents", {}).items():
                logger.info(
                    f"Agent {a_id} has {len(a_data.get('memories', []))} memories"
                )

            # Validate against schema
            logger.info("Validating against schema...")
            if not validate_memory_system_json(data):
                logger.error("Generated JSON does not conform to schema")
                return False

            # Check if the directory exists
            dir_path = os.path.dirname(os.path.abspath(filepath))

            # Ensure directory exists
            try:
                # Only create directory if it doesn't exist
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                return False

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Memory system saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save memory system to {filepath}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

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
        try:
            # Read from file
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate against schema
            if not validate_memory_system_json(data):
                logger.error(f"JSON file {filepath} does not conform to schema")
                return None

            # Create config with Redis mock settings if requested
            config_data = data.get("config", {})

            # Extract nested configs
            from memory.config import RedisIMConfig, RedisSTMConfig, SQLiteLTMConfig

            # Create clean config data without nested configs
            clean_config_data = {}
            for key, value in config_data.items():
                if key not in ["stm_config", "im_config", "ltm_config"]:
                    clean_config_data[key] = value

            # Create config instance
            config = MemoryConfig(**clean_config_data)

            # Set up STM config
            if "stm_config" in config_data:
                stm_config_data = config_data["stm_config"]
                stm_config = RedisSTMConfig(**stm_config_data)
                if use_mock_redis:
                    stm_config.use_mock = True
                config.stm_config = stm_config

            # Set up IM config
            if "im_config" in config_data:
                im_config_data = config_data["im_config"]
                im_config = RedisIMConfig(**im_config_data)
                if use_mock_redis:
                    im_config.use_mock = True
                config.im_config = im_config

            # Set up LTM config
            if "ltm_config" in config_data:
                ltm_config_data = config_data["ltm_config"]
                ltm_config = SQLiteLTMConfig(**ltm_config_data)
                config.ltm_config = ltm_config

            # Create memory system
            memory_system = cls(config)

            # Load agents and their memories
            for agent_id, agent_data in data.get("agents", {}).items():
                # Get or create agent
                memory_agent = memory_system.get_memory_agent(agent_id)

                # Add memories
                for memory in agent_data.get("memories", []):
                    # First check the top-level type
                    memory_type = memory.get("type", "generic")

                    # Also check metadata.memory_type as a fallback
                    if memory_type == "generic" and "metadata" in memory:
                        metadata_type = memory["metadata"].get("memory_type")
                        if metadata_type in ["state", "interaction", "action"]:
                            memory_type = metadata_type
                            logger.debug(
                                f"Using memory_type from metadata: {memory_type}"
                            )

                    content = memory.get("content", {})
                    step_number = memory.get("step_number", 0)
                    priority = memory.get("metadata", {}).get("importance_score", 1.0)

                    # Log memory being loaded for debugging
                    logger.debug(
                        f"Loading memory of type {memory_type} for agent {agent_id}"
                    )
                    logger.debug(f"Memory content keys: {list(content.keys())}")
                    logger.debug(f"Memory step: {step_number}, priority: {priority}")

                    # Store according to memory type
                    if memory_type == "state":
                        memory_agent.store_state(content, step_number, priority)
                    elif memory_type == "interaction":
                        memory_agent.store_interaction(content, step_number, priority)
                    elif memory_type == "action":
                        memory_agent.store_action(content, step_number, priority)
                    else:
                        # For generic types, use store_state as fallback
                        memory_agent.store_state(content, step_number, priority)

            logger.info(f"Memory system loaded from {filepath}")
            return memory_system

        except Exception as e:
            logger.error(f"Failed to load memory system from {filepath}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

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
