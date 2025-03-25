"""API interface for the agent memory system."""

import heapq
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import ValidationError

from agent_memory.api.types import (
    MemoryChangeRecord,
    MemoryEntry,
    MemoryImportanceScore,
    MemoryStatistics,
    MemoryTier,
    MemoryTypeFilter,
)
from agent_memory.config import MemoryConfig
from agent_memory.config.models import MemoryConfigModel
from agent_memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Module-level caching decorator that doesn't rely on instance state
def cacheable(ttl=60):
    """Decorator to cache function results with TTL.

    Args:
        ttl: Time-to-live in seconds (default: 60)

    Returns:
        Decorated function with caching
    """
    cache = {}
    cache_ttl = {}

    def make_hashable(obj):
        """Convert unhashable types to hashable for cache keys."""
        if isinstance(obj, dict):
            return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
        elif isinstance(obj, (list, tuple)):
            return tuple(make_hashable(x) for x in obj)
        elif isinstance(obj, set):
            return frozenset(make_hashable(x) for x in obj)
        return obj

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable versions of arguments
            hashable_args = tuple(make_hashable(arg) for arg in args)
            hashable_kwargs = {k: make_hashable(v) for k, v in kwargs.items()}

            # Create cache key
            cache_key = f"{func.__name__}:{hash(hashable_args)}:{hash(frozenset(hashable_kwargs.items()))}"

            # Check cache
            if cache_key in cache:
                expiry_time = cache_ttl.get(cache_key, 0)
                if time.time() < expiry_time:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[cache_key]

            # Call function if not in cache or expired
            result = func(*args, **kwargs)

            # Store in cache
            cache[cache_key] = result
            cache_ttl[cache_key] = time.time() + ttl

            return result

        # Attach cache management methods to the function
        wrapper.cache = cache
        wrapper.cache_ttl = cache_ttl

        def clear_cache():
            """Clear the cache for this function."""
            cache.clear()
            cache_ttl.clear()

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


class MemoryAPIException(Exception):
    """Base exception class for all memory API exceptions."""

    pass


class MemoryStoreException(MemoryAPIException):
    """Exception raised for storage-related errors."""

    pass


class MemoryRetrievalException(MemoryAPIException):
    """Exception raised for retrieval-related errors."""

    pass


class MemoryMaintenanceException(MemoryAPIException):
    """Exception raised for maintenance-related errors."""

    pass


class MemoryConfigException(MemoryAPIException):
    """Exception raised for configuration-related errors."""

    pass


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
        self._default_cache_ttl = 60  # Default TTL in seconds

    def clear_all_caches(self):
        """Clear all caches for all methods in this API."""
        # Clear caches for decorated methods
        if hasattr(self.retrieve_similar_states, "clear_cache"):
            self.retrieve_similar_states.clear_cache()

        if hasattr(self.search_by_content, "clear_cache"):
            self.search_by_content.clear_cache()

    def clear_cache(self):
        """Clear the query result cache (alias for clear_all_caches for backward compatibility)."""
        self.clear_all_caches()

    def set_cache_ttl(self, ttl: int):
        """Set the default cache TTL.

        Args:
            ttl: Default time-to-live for cache entries in seconds
        """
        if ttl <= 0:
            raise ValueError("Cache TTL must be positive")
        self._default_cache_ttl = ttl

    def _merge_sorted_lists(
        self,
        lists: List[List[Dict[str, Any]]],
        key_fn: Callable[[Dict[str, Any]], Any],
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """Efficiently merge multiple sorted lists.

        Args:
            lists: List of sorted lists to merge
            key_fn: Function to extract the sort key from each element
            reverse: Whether the lists are sorted in descending order

        Returns:
            A single merged and sorted list
        """
        # Filter out empty lists
        lists = [lst for lst in lists if lst]
        if not lists:
            return []

        # If only one list, return it directly
        if len(lists) == 1:
            return lists[0]

        # For ascending order
        if not reverse:
            # Create iterators for each list
            iterators = [iter(lst) for lst in lists]
            # Get the first element from each iterator
            current_items = []
            for i, it in enumerate(iterators):
                try:
                    item = next(it)
                    # Store (key value, original item, iterator index)
                    current_items.append((key_fn(item), item, i))
                except StopIteration:
                    pass

            # Create a min heap for efficient merging
            heapq.heapify(current_items)

            # Merge the lists
            result = []
            while current_items:
                # Get the smallest item
                _, item, iterator_idx = heapq.heappop(current_items)
                result.append(item)

                # Get the next item from the same iterator
                try:
                    next_item = next(iterators[iterator_idx])
                    heapq.heappush(
                        current_items, (key_fn(next_item), next_item, iterator_idx)
                    )
                except StopIteration:
                    pass

            return result
        else:
            # For descending order, invert the key function
            def inverted_key_fn(item):
                return (
                    -key_fn(item)
                    if isinstance(key_fn(item), (int, float))
                    else key_fn(item)
                )

            # Use the same algorithm but with the inverted key function
            return self._merge_sorted_lists(lists, inverted_key_fn, reverse=False)

    def _aggregate_results(
        self,
        memory_agent,
        query_fn: Callable,
        k: int = None,
        memory_type: Optional[str] = None,
        sort_key: Optional[Callable] = None,
        reverse: bool = False,
        merge_sorted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Aggregate results across memory tiers using the provided query function.

        Args:
            memory_agent: The memory agent to query
            query_fn: Function that takes (store, k, memory_type) and returns results
            k: Maximum number of results to return
            memory_type: Optional filter for specific memory types
            sort_key: Optional function for sorting results
            reverse: Whether to reverse sort order (default: False)
            merge_sorted: Whether to use merge sort for already sorted results

        Returns:
            List of aggregated results, optionally sorted
        """
        # If k is None, we're not limiting results
        remaining = float("inf") if k is None else k

        # Collect results from each store
        all_results = []
        store_results = []

        for store in [
            memory_agent.stm_store,
            memory_agent.im_store,
            memory_agent.ltm_store,
        ]:
            if remaining <= 0:
                break

            store_limit = None if k is None else remaining
            partial_results = query_fn(store, store_limit, memory_type)

            if partial_results:
                store_results.append(partial_results)
                all_results.extend(partial_results)

                if k is not None:
                    remaining = k - len(all_results)

        # If no results, return empty list
        if not all_results:
            return []

        # If sorting is needed and results exist
        if sort_key:
            if merge_sorted and len(store_results) > 1:
                # Use efficient merge algorithm if results are already sorted
                return self._merge_sorted_lists(store_results, sort_key, reverse)
            else:
                # Fall back to regular sort
                all_results.sort(key=sort_key, reverse=reverse)

            # Limit to k results if specified
            if k is not None:
                all_results = all_results[:k]

        return all_results

    def store_agent_state(
        self,
        agent_id: str,
        state_data: Dict[str, Any],
        step_number: int,
        priority: MemoryImportanceScore = 1.0,
    ) -> bool:
        """Store an agent's state in short-term memory.

        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful

        Raises:
            MemoryStoreException: If there is an error during storage operation
        """
        try:
            # Validate input parameters
            if not agent_id:
                raise MemoryStoreException("Agent ID cannot be empty")
            if not isinstance(state_data, dict):
                raise MemoryStoreException(
                    f"State data must be a dictionary, got {type(state_data)}"
                )
            if not isinstance(step_number, int) or step_number < 0:
                raise MemoryStoreException(
                    f"Step number must be a non-negative integer, got {step_number}"
                )
            if (
                not isinstance(priority, (int, float))
                or priority < 0.0
                or priority > 1.0
            ):
                raise MemoryStoreException(
                    f"Priority must be a float between 0.0 and 1.0, got {priority}"
                )

            return self.memory_system.store_agent_state(
                agent_id, state_data, step_number, priority
            )
        except MemoryStoreException:
            # Re-raise custom exceptions as they're already properly formatted
            raise
        except Exception as e:
            logger.error(f"Failed to store agent state for agent {agent_id}: {e}")
            # Convert to custom exception with context for caller
            raise MemoryStoreException(
                f"Unexpected error storing agent state: {str(e)}"
            )

    def store_agent_interaction(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any],
        step_number: int,
        priority: MemoryImportanceScore = 1.0,
    ) -> bool:
        """Store information about an agent's interaction with environment or other agents.

        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful

        Raises:
            MemoryStoreException: If there is an error during storage operation
        """
        try:
            # Validate input parameters
            if not agent_id:
                raise MemoryStoreException("Agent ID cannot be empty")
            if not isinstance(interaction_data, dict):
                raise MemoryStoreException(
                    f"Interaction data must be a dictionary, got {type(interaction_data)}"
                )
            if not isinstance(step_number, int) or step_number < 0:
                raise MemoryStoreException(
                    f"Step number must be a non-negative integer, got {step_number}"
                )
            if (
                not isinstance(priority, (int, float))
                or priority < 0.0
                or priority > 1.0
            ):
                raise MemoryStoreException(
                    f"Priority must be a float between 0.0 and 1.0, got {priority}"
                )

            return self.memory_system.store_agent_interaction(
                agent_id, interaction_data, step_number, priority
            )
        except MemoryStoreException:
            # Re-raise custom exceptions as they're already properly formatted
            raise
        except Exception as e:
            logger.error(f"Failed to store agent interaction for agent {agent_id}: {e}")
            # Convert to custom exception with context for caller
            raise MemoryStoreException(
                f"Unexpected error storing agent interaction: {str(e)}"
            )

    def store_agent_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        step_number: int,
        priority: MemoryImportanceScore = 1.0,
    ) -> bool:
        """Store information about an action taken by an agent.

        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)

        Returns:
            True if storage was successful

        Raises:
            MemoryStoreException: If there is an error during storage operation
        """
        try:
            # Validate input parameters
            if not agent_id:
                raise MemoryStoreException("Agent ID cannot be empty")
            if not isinstance(action_data, dict):
                raise MemoryStoreException(
                    f"Action data must be a dictionary, got {type(action_data)}"
                )
            if not isinstance(step_number, int) or step_number < 0:
                raise MemoryStoreException(
                    f"Step number must be a non-negative integer, got {step_number}"
                )
            if (
                not isinstance(priority, (int, float))
                or priority < 0.0
                or priority > 1.0
            ):
                raise MemoryStoreException(
                    f"Priority must be a float between 0.0 and 1.0, got {priority}"
                )

            return self.memory_system.store_agent_action(
                agent_id, action_data, step_number, priority
            )
        except MemoryStoreException:
            # Re-raise custom exceptions as they're already properly formatted
            raise
        except Exception as e:
            logger.error(f"Failed to store agent action for agent {agent_id}: {e}")
            # Convert to custom exception with context for caller
            raise MemoryStoreException(
                f"Unexpected error storing agent action: {str(e)}"
            )

    def retrieve_state_by_id(
        self, agent_id: str, memory_id: str
    ) -> Optional[MemoryEntry]:
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
        self,
        agent_id: str,
        count: int = 10,
        memory_type: Optional[MemoryTypeFilter] = None,
    ) -> List[MemoryEntry]:
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

    @cacheable(ttl=60)
    def retrieve_similar_states(
        self,
        agent_id: str,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[MemoryTypeFilter] = None,
    ) -> List[MemoryEntry]:
        """Retrieve states similar to the query state.

        Args:
            agent_id: Unique identifier for the agent
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity to query state

        Raises:
            MemoryRetrievalException: If there is an error during retrieval
        """
        try:
            # Validate input parameters
            if not agent_id:
                raise MemoryRetrievalException("Agent ID cannot be empty")
            if not isinstance(query_state, dict):
                raise MemoryRetrievalException(
                    f"Query state must be a dictionary, got {type(query_state)}"
                )
            if not isinstance(k, int) or k <= 0:
                raise MemoryRetrievalException(f"k must be a positive integer, got {k}")

            memory_agent = self.memory_system.get_memory_agent(agent_id)

            if not memory_agent.embedding_engine:
                error_msg = (
                    "Vector similarity search requires embedding engine to be enabled"
                )
                logger.warning(error_msg)
                raise MemoryRetrievalException(error_msg)

            # Generate embedding for query state
            try:
                query_embedding = memory_agent.embedding_engine.encode_stm(query_state)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query state: {e}")
                raise MemoryRetrievalException(
                    f"Failed to encode query state: {str(e)}"
                )

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
                    tier_embedding = (
                        memory_agent.embedding_engine.ensure_embedding_dimensions(
                            query_embedding, tier
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error converting embedding to {tier} format: {e}")
                    # If conversion fails, use original embedding which may not work properly
                    tier_embedding = query_embedding

                try:
                    return store.search_by_vector(
                        tier_embedding, k=limit, memory_type=mem_type
                    )
                except Exception as e:
                    logger.error(f"Error searching the {tier} store: {e}")
                    # Return empty list on store search failure
                    return []

            return self._aggregate_results(
                memory_agent,
                query_function,
                k=k,
                memory_type=memory_type,
                sort_key=lambda x: x.get("_similarity_score", 0),
                reverse=True,
                merge_sorted=True,
            )
        except MemoryRetrievalException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving similar states for agent {agent_id}: {e}"
            )
            raise MemoryRetrievalException(
                f"Failed to retrieve similar states: {str(e)}"
            )

    def retrieve_by_time_range(
        self,
        agent_id: str,
        start_step: int,
        end_step: int,
        memory_type: Optional[MemoryTypeFilter] = None,
    ) -> List[MemoryEntry]:
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

        # Use merge sort since results from each store are already sorted by step number
        return self._aggregate_results(
            memory_agent,
            query_function,
            memory_type=memory_type,
            sort_key=lambda x: x.get("step_number", 0),
            merge_sorted=True,
        )

    def retrieve_by_attributes(
        self,
        agent_id: str,
        attributes: Dict[str, Any],
        memory_type: Optional[MemoryTypeFilter] = None,
    ) -> List[MemoryEntry]:
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
            reverse=True,
        )

    def search_by_embedding(
        self,
        agent_id: str,
        query_embedding: List[float],
        k: int = 5,
        memory_tiers: Optional[List[MemoryTier]] = None,
    ) -> List[MemoryEntry]:
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
            "ltm": memory_agent.ltm_store,
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
                    tier_embedding = (
                        memory_agent.embedding_engine.ensure_embedding_dimensions(
                            query_embedding, tier
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Error converting embedding for {tier.upper()} tier: {str(e)}. "
                        f"Using original embedding with potential dimension mismatch."
                    )

            # Search with the properly dimensioned embedding
            tier_results = store.search_by_vector(tier_embedding, k=k - len(results))
            results.extend(tier_results)

        # Sort by similarity score
        return sorted(
            results, key=lambda x: x.get("_similarity_score", 0), reverse=True
        )

    @cacheable(ttl=60)
    def search_by_content(
        self, agent_id: str, content_query: Union[str, Dict[str, Any]], k: int = 5
    ) -> List[MemoryEntry]:
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
            memory_agent, query_function, k=k, merge_sorted=True
        )

    def get_memory_statistics(self, agent_id: str) -> MemoryStatistics:
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

        Raises:
            MemoryMaintenanceException: If there is an error during maintenance
        """
        try:
            if agent_id:
                # Validate agent ID
                if not isinstance(agent_id, str):
                    raise MemoryMaintenanceException(
                        f"Agent ID must be a string, got {type(agent_id)}"
                    )

                # Maintain single agent
                try:
                    memory_agent = self.memory_system.get_memory_agent(agent_id)
                except Exception as e:
                    logger.error(
                        f"Failed to get memory agent for agent {agent_id}: {e}"
                    )
                    raise MemoryMaintenanceException(
                        f"Agent {agent_id} not found or error accessing memory agent: {str(e)}"
                    )

                try:
                    result = memory_agent._perform_maintenance()
                    if not result:
                        logger.error(f"Maintenance failed for agent {agent_id}")
                        raise MemoryMaintenanceException(
                            f"Maintenance failed for agent {agent_id}"
                        )
                    return result
                except Exception as e:
                    logger.error(f"Error during maintenance for agent {agent_id}: {e}")
                    raise MemoryMaintenanceException(
                        f"Error during maintenance for agent {agent_id}: {str(e)}"
                    )
            else:
                # Maintain all agents
                success = True
                failed_agents = []

                for current_agent_id, memory_agent in self.memory_system.agents.items():
                    try:
                        if not memory_agent._perform_maintenance():
                            logger.error(
                                f"Maintenance failed for agent {current_agent_id}"
                            )
                            success = False
                            failed_agents.append(current_agent_id)
                    except Exception as e:
                        logger.error(
                            f"Error during maintenance for agent {current_agent_id}: {e}"
                        )
                        success = False
                        failed_agents.append(current_agent_id)

                if not success:
                    raise MemoryMaintenanceException(
                        f"Maintenance failed for agents: {', '.join(failed_agents)}"
                    )

                return success
        except MemoryMaintenanceException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error during memory maintenance: {e}")
            raise MemoryMaintenanceException(
                f"Unexpected error during memory maintenance: {str(e)}"
            )

    def clear_agent_memory(
        self, agent_id: str, memory_tiers: Optional[List[MemoryTier]] = None
    ) -> bool:
        """Clear an agent's memory in specified tiers.

        Args:
            agent_id: Unique identifier for the agent
            memory_tiers: Optional list of tiers to clear (e.g., ["stm", "im"])
                          If None, clears all tiers

        Returns:
            True if clearing was successful

        Raises:
            MemoryMaintenanceException: If there is an error during memory clearing
        """
        try:
            # Validate input parameters
            if not agent_id:
                raise MemoryMaintenanceException("Agent ID cannot be empty")
            if memory_tiers is not None and not isinstance(memory_tiers, list):
                raise MemoryMaintenanceException(
                    f"Memory tiers must be a list or None, got {type(memory_tiers)}"
                )
            if memory_tiers is not None:
                valid_tiers = ["stm", "im", "ltm"]
                invalid_tiers = [t for t in memory_tiers if t not in valid_tiers]
                if invalid_tiers:
                    raise MemoryMaintenanceException(
                        f"Invalid memory tiers: {invalid_tiers}. Valid tiers are: {valid_tiers}"
                    )

            # Get memory agent - this could raise if agent_id doesn't exist
            try:
                memory_agent = self.memory_system.get_memory_agent(agent_id)
            except Exception as e:
                logger.error(f"Failed to get memory agent for agent {agent_id}: {e}")
                raise MemoryMaintenanceException(
                    f"Agent {agent_id} not found or error accessing memory agent: {str(e)}"
                )

            if not memory_tiers:
                # Clear all tiers
                try:
                    result = memory_agent.clear_memory()
                    if not result:
                        logger.error(f"Failed to clear memory for agent {agent_id}")
                    return result
                except Exception as e:
                    logger.error(f"Error clearing memory for agent {agent_id}: {e}")
                    raise MemoryMaintenanceException(
                        f"Error clearing memory for agent {agent_id}: {str(e)}"
                    )

            success = True
            failed_tiers = []

            # Clear specified tiers
            if "stm" in memory_tiers:
                try:
                    if not memory_agent.stm_store.clear():
                        logger.error(f"Failed to clear STM for agent {agent_id}")
                        success = False
                        failed_tiers.append("stm")
                except Exception as e:
                    logger.error(f"Error clearing STM for agent {agent_id}: {e}")
                    success = False
                    failed_tiers.append("stm")

            if "im" in memory_tiers:
                try:
                    if not memory_agent.im_store.clear():
                        logger.error(f"Failed to clear IM for agent {agent_id}")
                        success = False
                        failed_tiers.append("im")
                except Exception as e:
                    logger.error(f"Error clearing IM for agent {agent_id}: {e}")
                    success = False
                    failed_tiers.append("im")

            if "ltm" in memory_tiers:
                try:
                    if not memory_agent.ltm_store.clear():
                        logger.error(f"Failed to clear LTM for agent {agent_id}")
                        success = False
                        failed_tiers.append("ltm")
                except Exception as e:
                    logger.error(f"Error clearing LTM for agent {agent_id}: {e}")
                    success = False
                    failed_tiers.append("ltm")

            if not success:
                raise MemoryMaintenanceException(
                    f"Failed to clear memory tiers: {', '.join(failed_tiers)}"
                )

            return success
        except MemoryMaintenanceException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error clearing memory for agent {agent_id}: {e}")
            raise MemoryMaintenanceException(
                f"Unexpected error clearing memory: {str(e)}"
            )

    def set_importance_score(
        self, agent_id: str, memory_id: str, importance_score: MemoryImportanceScore
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
    ) -> Dict[int, Optional[MemoryEntry]]:
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

        Raises:
            MemoryConfigException: If there is an error during configuration update
        """
        try:
            if not isinstance(config, dict):
                raise MemoryConfigException(
                    f"Configuration must be a dictionary, got {type(config)}"
                )

            # Convert the current configuration to a dict for Pydantic validation
            current_config = {
                "cleanup_interval": self.memory_system.config.cleanup_interval,
                "memory_priority_decay": self.memory_system.config.memory_priority_decay,
            }

            # Add tier configurations
            if hasattr(self.memory_system.config, "stm_config"):
                stm_config = self.memory_system.config.stm_config
                current_config["stm_config"] = {
                    attr: getattr(stm_config, attr)
                    for attr in ["host", "port", "memory_limit", "ttl"]
                    if hasattr(stm_config, attr)
                }

            if hasattr(self.memory_system.config, "im_config"):
                im_config = self.memory_system.config.im_config
                current_config["im_config"] = {
                    attr: getattr(im_config, attr)
                    for attr in ["host", "port", "memory_limit", "ttl"]
                    if hasattr(im_config, attr)
                }

            if hasattr(self.memory_system.config, "ltm_config"):
                ltm_config = self.memory_system.config.ltm_config
                current_config["ltm_config"] = {
                    attr: getattr(ltm_config, attr)
                    for attr in ["db_path"]
                    if hasattr(ltm_config, attr)
                }

            if hasattr(self.memory_system.config, "autoencoder_config"):
                ac_config = self.memory_system.config.autoencoder_config
                current_config["autoencoder_config"] = {
                    attr: getattr(ac_config, attr)
                    for attr in ["stm_dim", "im_dim", "ltm_dim"]
                    if hasattr(ac_config, attr)
                }

            # Handle nested configuration updates (e.g., 'stm_config.memory_limit')
            processed_config = current_config.copy()
            flat_updates = {}
            nested_updates = {}

            for key, value in config.items():
                if "." in key:
                    # Nested configuration like 'stm_config.memory_limit'
                    main_key, sub_key = key.split(".", 1)
                    if main_key not in nested_updates:
                        nested_updates[main_key] = {}
                    nested_updates[main_key][sub_key] = value
                else:
                    # Top-level configuration
                    flat_updates[key] = value

            # Apply nested updates
            for main_key, sub_dict in nested_updates.items():
                if main_key in processed_config and isinstance(
                    processed_config[main_key], dict
                ):
                    processed_config[main_key].update(sub_dict)
                else:
                    processed_config[main_key] = sub_dict

            # Apply flat updates
            processed_config.update(flat_updates)

            # Validate with Pydantic
            try:
                validated_model = MemoryConfigModel(**processed_config)
            except ValidationError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise MemoryConfigException(f"Invalid configuration: {str(e)}")

            # Apply the validated configuration to the actual config object
            validated_model.to_config_object(self.memory_system.config)

            # Apply configuration to existing memory agents
            for agent_id, memory_agent in self.memory_system.agents.items():
                try:
                    # Update agent config
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
                except Exception as e:
                    logger.error(
                        f"Failed to update configuration for agent {agent_id}: {e}"
                    )
                    raise MemoryConfigException(
                        f"Failed to update configuration for agent {agent_id}: {str(e)}"
                    )

            return True

        except MemoryConfigException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise MemoryConfigException(
                f"Unexpected error updating configuration: {str(e)}"
            )

    def get_attribute_change_history(
        self,
        agent_id: str,
        attribute_name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[MemoryChangeRecord]:
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
