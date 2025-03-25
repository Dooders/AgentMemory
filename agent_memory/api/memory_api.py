"""API interface for the agent memory system."""

import logging
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar

from agent_memory.config import MemoryConfig
from agent_memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentMemoryAPI:
    """Interface for storing and retrieving agent states in the hierarchical memory system.

    This class provides a clean, standardized API for interacting with the
    agent memory system, abstracting away the details of the underlying
    storage mechanisms.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the AgentMemoryAPI.

        Args:
            config: Configuration for the memory system
        """
        self.memory_system = AgentMemorySystem.get_instance(config)
        
    def _aggregate_results(
        self,
        memory_agent,
        query_fn: Callable,
        k: int = None,
        memory_type: Optional[str] = None,
        sort_key: Optional[Callable] = None,
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """Aggregate results across memory tiers using the provided query function.
        
        Args:
            memory_agent: The memory agent to query
            query_fn: Function that takes (store, k, memory_type) and returns results
            k: Maximum number of results to return
            memory_type: Optional filter for specific memory types
            sort_key: Optional function for sorting results
            reverse: Whether to reverse sort order (default: False)
            
        Returns:
            List of aggregated results, optionally sorted
        """
        results = []
        
        # If k is None, we're not limiting results
        remaining = float('inf') if k is None else k
        
        for store in [memory_agent.stm_store, memory_agent.im_store, memory_agent.ltm_store]:
            if remaining <= 0:
                break
                
            store_limit = None if k is None else remaining
            partial_results = query_fn(store, store_limit, memory_type)
            results.extend(partial_results)
            
            if k is not None:
                remaining = k - len(results)
        
        # Sort results if a sort key was provided
        if sort_key and results:
            results.sort(key=sort_key, reverse=reverse)
            
            # Limit to k results if specified
            if k is not None:
                results = results[:k]
                
        return results

    def store_agent_state(
        self,
        agent_id: str,
        state_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store an agent's state in short-term memory.

        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_state(
            agent_id, state_data, step_number, priority
        )

    def store_agent_interaction(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store information about an agent's interaction with environment or other agents.

        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_interaction(
            agent_id, interaction_data, step_number, priority
        )

    def store_agent_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
    ) -> bool:
        """Store information about an action taken by an agent.

        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_action(
            agent_id, action_data, step_number, priority
        )

    def retrieve_state_by_id(
        self, agent_id: str, memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.

        Args:
            agent_id: Unique identifier for the agent
            memory_id: Unique identifier for the memory

        Returns:
            Memory entry or None if not found
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        # Try retrieving from each tier in order
        memory = memory_agent.stm_store.get(memory_id)
        if not memory:
            memory = memory_agent.im_store.get(memory_id)
        if not memory:
            memory = memory_agent.ltm_store.get(memory_id)
        return memory

    def retrieve_recent_states(
        self, agent_id: str, count: int = 10, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent states for an agent.

        Args:
            agent_id: Unique identifier for the agent
            count: Maximum number of states to retrieve
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        return memory_agent.stm_store.get_recent(count, memory_type)

    def retrieve_similar_states(
        self,
        agent_id: str,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve states similar to the query state.

        Args:
            agent_id: Unique identifier for the agent
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity to query state
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        if not memory_agent.embedding_engine:
            logger.warning(
                "Vector similarity search requires embedding engine to be enabled"
            )
            return []

        # Generate embedding for query state
        query_embedding = memory_agent.embedding_engine.encode_stm(query_state)
        
        def query_function(store, limit, mem_type):
            # Determine which tier this store represents
            if store == memory_agent.stm_store:
                tier = "stm"
            elif store == memory_agent.im_store:
                tier = "im"
            else:  # ltm_store
                tier = "ltm"
                
            # Convert embedding to appropriate dimensions for this tier
            try:
                tier_embedding = memory_agent.embedding_engine.ensure_embedding_dimensions(
                    query_embedding, tier
                )
            except Exception as e:
                logger.warning(f"Error converting embedding to {tier} format: {e}")
                # If conversion fails, use original embedding which may not work properly
                tier_embedding = query_embedding
                
            return store.search_by_vector(tier_embedding, k=limit, memory_type=mem_type)
        
        return self._aggregate_results(
            memory_agent,
            query_function,
            k=k,
            memory_type=memory_type,
            sort_key=lambda x: x.get("_similarity_score", 0),
            reverse=True
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
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        def query_function(store, _, mem_type):
            return store.get_by_step_range(start_step, end_step, mem_type)
        
        return self._aggregate_results(
            memory_agent,
            query_function,
            memory_type=memory_type,
            sort_key=lambda x: x.get("step_number", 0)
        )

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
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        def query_function(store, _, mem_type):
            return store.get_by_attributes(attributes, mem_type)
        
        return self._aggregate_results(
            memory_agent,
            query_function,
            memory_type=memory_type,
            sort_key=lambda x: x.get("step_number", 0),
            reverse=True
        )

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
            memory_tiers: Optional list of tiers to search (e.g., ["stm", "im"])

        Returns:
            List of memory entries sorted by similarity
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        # Determine which tiers to search
        tiers = memory_tiers or ["stm", "im", "ltm"]
        results = []

        tier_stores = {
            "stm": memory_agent.stm_store,
            "im": memory_agent.im_store,
            "ltm": memory_agent.ltm_store
        }
        
        # Check if embedding engine is available for conversion
        has_embedding_engine = memory_agent.embedding_engine is not None
        
        for tier in tiers:
            if len(results) >= k:
                break
                
            store = tier_stores.get(tier)
            if not store:
                continue
                
            # Ensure embedding has the right dimensions for this tier
            tier_embedding = query_embedding
            
            if has_embedding_engine:
                try:
                    # Automatically convert the embedding to the target tier format
                    tier_embedding = memory_agent.embedding_engine.ensure_embedding_dimensions(
                        query_embedding, tier
                    )
                except Exception as e:
                    logger.warning(
                        f"Error converting embedding for {tier.upper()} tier: {str(e)}. "
                        f"Using original embedding with potential dimension mismatch."
                    )
            
            # Search with the properly dimensioned embedding
            tier_results = store.search_by_vector(tier_embedding, k=k-len(results))
            results.extend(tier_results)

        # Sort by similarity score
        return sorted(
            results, key=lambda x: x.get("_similarity_score", 0), reverse=True
        )

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
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        # Convert string query to dict if needed
        if isinstance(content_query, str):
            query_dict = {"text": content_query}
        else:
            query_dict = content_query

        def query_function(store, limit, _):
            return store.search_by_content(query_dict, k=limit)
        
        return self._aggregate_results(
            memory_agent,
            query_function,
            k=k
        )

    def get_memory_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics about an agent's memory usage.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary containing memory statistics
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        # Gather statistics from each memory tier
        stm_count = memory_agent.stm_store.count()
        im_count = memory_agent.im_store.count()
        ltm_count = memory_agent.ltm_store.count()

        # Get memory type counts in STM
        memory_type_counts = memory_agent.stm_store.count_by_type()

        return {
            "total_memories": stm_count + im_count + ltm_count,
            "stm_count": stm_count,
            "im_count": im_count,
            "ltm_count": ltm_count,
            "memory_type_distribution": memory_type_counts,
            "last_maintenance_time": memory_agent.last_maintenance_time,
            "insert_count_since_maintenance": memory_agent._insert_count,
        }

    def force_memory_maintenance(self, agent_id: Optional[str] = None) -> bool:
        """Force memory tier transitions and cleanup operations.

        Args:
            agent_id: Optional agent ID to restrict maintenance to a single agent

        Returns:
            True if maintenance was successful
        """
        if agent_id:
            # Maintain single agent
            memory_agent = self.memory_system.get_memory_agent(agent_id)
            return memory_agent._perform_maintenance()
        else:
            # Maintain all agents
            success = True
            for agent_id, memory_agent in self.memory_system.agents.items():
                if not memory_agent._perform_maintenance():
                    logger.error(f"Maintenance failed for agent {agent_id}")
                    success = False
            return success

    def clear_agent_memory(
        self, agent_id: str, memory_tiers: Optional[List[str]] = None
    ) -> bool:
        """Clear an agent's memory in specified tiers.

        Args:
            agent_id: Unique identifier for the agent
            memory_tiers: Optional list of tiers to clear (e.g., ["stm", "im"])
                          If None, clears all tiers

        Returns:
            True if clearing was successful
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        if not memory_tiers:
            # Clear all tiers
            return memory_agent.clear_memory()

        success = True

        # Clear specified tiers
        if "stm" in memory_tiers:
            if not memory_agent.stm_store.clear():
                logger.error(f"Failed to clear STM for agent {agent_id}")
                success = False

        if "im" in memory_tiers:
            if not memory_agent.im_store.clear():
                logger.error(f"Failed to clear IM for agent {agent_id}")
                success = False

        if "ltm" in memory_tiers:
            if not memory_agent.ltm_store.clear():
                logger.error(f"Failed to clear LTM for agent {agent_id}")
                success = False

        return success

    def set_importance_score(
        self, agent_id: str, memory_id: str, importance_score: float
    ) -> bool:
        """Update the importance score for a specific memory.

        Args:
            agent_id: Identifier for the agent
            memory_id: Unique identifier for the memory entry
            importance_score: New importance score (0.0 to 1.0)

        Returns:
            True if update was successful
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        # Find memory and update importance score
        memory = self.retrieve_state_by_id(agent_id, memory_id)
        if not memory:
            logger.error(f"Memory {memory_id} not found for agent {agent_id}")
            return False

        # Update importance score
        memory["metadata"]["importance_score"] = max(0.0, min(1.0, importance_score))

        # Determine which store contains the memory and update
        if memory_agent.stm_store.contains(memory_id):
            return memory_agent.stm_store.update(memory)
        elif memory_agent.im_store.contains(memory_id):
            return memory_agent.im_store.update(memory)
        elif memory_agent.ltm_store.contains(memory_id):
            return memory_agent.ltm_store.update(memory)

        return False

    def get_memory_snapshots(
        self, agent_id: str, steps: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get agent memory snapshots at specific steps.

        Args:
            agent_id: Identifier for the agent
            steps: List of step numbers to get snapshots for

        Returns:
            Dictionary mapping step numbers to state snapshots
        """
        result = {}

        for step in steps:
            # Retrieve state for this step
            memories = self.retrieve_by_time_range(
                agent_id, start_step=step, end_step=step, memory_type="state"
            )

            if memories:
                # Use the first state memory for this step
                result[step] = memories[0]
            else:
                # No state memory found for this step
                result[step] = None

        return result

    def configure_memory_system(self, config: Dict[str, Any]) -> bool:
        """Update configuration parameters for the memory system.

        Args:
            config: Dictionary of configuration parameters

        Returns:
            True if configuration was updated successfully
        """
        # Update configuration
        try:
            for key, value in config.items():
                if hasattr(self.memory_system.config, key):
                    setattr(self.memory_system.config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")

            # Apply configuration to existing memory agents
            for agent_id, memory_agent in self.memory_system.agents.items():
                memory_agent.config = self.memory_system.config

                # Update store configurations
                memory_agent.stm_store.config = self.memory_system.config.stm_config
                memory_agent.im_store.config = self.memory_system.config.im_config
                memory_agent.ltm_store.config = self.memory_system.config.ltm_config

                # Update embedding engine if needed
                if memory_agent.embedding_engine:
                    memory_agent.embedding_engine.configure(
                        self.memory_system.config.autoencoder_config
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_attribute_change_history(
        self,
        agent_id: str,
        attribute_name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get history of changes for a specific attribute.

        Args:
            agent_id: Unique identifier for the agent
            attribute_name: Name of the attribute to track
            start_step: Optional start step for filtering
            end_step: Optional end step for filtering

        Returns:
            List of change records for the specified attribute
        """
        # Get state memories for the specified range
        memories = self.retrieve_by_time_range(
            agent_id, start_step or 0, end_step or float("inf"), memory_type="state"
        )

        # Track changes to the attribute
        changes = []
        previous_value = None

        for memory in memories:
            if attribute_name in memory["contents"]:
                current_value = memory["contents"][attribute_name]

                # Check if value changed
                if previous_value is None or previous_value != current_value:
                    changes.append(
                        {
                            "memory_id": memory["memory_id"],
                            "step_number": memory["step_number"],
                            "timestamp": memory["timestamp"],
                            "previous_value": previous_value,
                            "new_value": current_value,
                        }
                    )
                    previous_value = current_value

        return changes
