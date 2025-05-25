"""Memory Agent implementation for agent state management."""

import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from memory.config import MemoryConfig
from memory.embeddings.autoencoder import AutoencoderEmbeddingEngine
from memory.embeddings.compression import CompressionEngine
from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.vector_store import VectorStore
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore
from memory.utils.identity import generate_memory_id

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Manages an agent's memory across hierarchical storage tiers.

    This class provides a unified interface for storing and retrieving
    agent memories across different storage tiers with varying levels
    of compression and resolution.

    Attributes:
        agent_id: Unique identifier for the agent
        config: Configuration for the memory agent
        stm_store: Short-Term Memory store (Redis)
        im_store: Intermediate Memory store (Redis with TTL)
        ltm_store: Long-Term Memory store (SQLite)
        vector_store: Vector store for similarity search
        compression_engine: Engine for compressing memory entries
        embedding_engine: Optional neural embedding engine
    """

    def __init__(self, agent_id: str, config: MemoryConfig):
        """Initialize the MemoryAgent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the memory agent
        """
        self.agent_id = agent_id
        self.config = config

        # Initialize memory stores
        self.stm_store = RedisSTMStore(config.stm_config)
        self.im_store = RedisIMStore(config.im_config)
        self.ltm_store = SQLiteLTMStore(agent_id, config.ltm_config)

        # Initialize vector store
        self.vector_store = VectorStore(
            redis_client=self.stm_store.redis,
            stm_dimension=config.autoencoder_config.stm_dim,
            im_dimension=config.autoencoder_config.im_dim,
            ltm_dimension=config.autoencoder_config.ltm_dim,
            namespace=f"agent-{agent_id}",
        )

        # Initialize compression engine
        self.compression_engine = CompressionEngine(config.autoencoder_config)

        # Initialize embedding engine based on configuration
        if config.use_embedding_engine:
            # Use text embedding model
            self.embedding_engine = TextEmbeddingEngine(
                model_name=config.text_model_name
            )
            logger.info(f"Using text embeddings with model {config.text_model_name}")

        else:
            self.embedding_engine = None
            logger.warning(
                "Neural embeddings disabled - similarity search will not be available"
            )

        # Internal state
        self._insert_count = 0

        logger.debug("MemoryAgent initialized for agent %s", agent_id)

    def store_state(
        self,
        state_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
        tier: str = "stm",
    ) -> bool:
        """Store an agent state in memory.

        Args:
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            state_data, step_number, "state", priority, tier
        )

        # Store in specified memory tier
        success = self._store_in_tier(tier, memory_entry)

        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()

        return success

    def store_interaction(
        self,
        interaction_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
        tier: str = "stm",
    ) -> bool:
        """Store an agent interaction in memory.

        Args:
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            interaction_data, step_number, "interaction", priority, tier
        )

        # Store in specified memory tier
        success = self._store_in_tier(tier, memory_entry)

        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()

        return success

    def store_action(
        self,
        action_data: Dict[str, Any],
        step_number: int,
        priority: float = 1.0,
        tier: str = "stm",
    ) -> bool:
        """Store an agent action in memory.

        Args:
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            tier: Memory tier to store in ("stm", "im", or "ltm", default: "stm")

        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            action_data, step_number, "action", priority, tier
        )

        # Store in specified memory tier
        success = self._store_in_tier(tier, memory_entry)

        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()

        return success

    def _store_in_tier(self, tier: str, memory_entry: Dict[str, Any]) -> bool:
        """Store a memory entry in the specified tier.

        Args:
            tier: Memory tier to store in ("stm", "im", or "ltm")
            memory_entry: Memory entry to store

        Returns:
            True if storage was successful
        """
        if tier == "stm":
            return self.stm_store.store(self.agent_id, memory_entry)
        elif tier == "im":
            return self.im_store.store(self.agent_id, memory_entry)
        elif tier == "ltm":
            return self.ltm_store.store(memory_entry)
        else:
            logger.warning(f"Unknown memory tier '{tier}', defaulting to STM")
            return self.stm_store.store(self.agent_id, memory_entry)

    def _create_memory_entry(
        self,
        data: Dict[str, Any],
        step_number: int,
        memory_type: str,
        priority: float,
        tier: str = "stm",
    ) -> Dict[str, Any]:
        """Create a standardized memory entry.

        Args:
            data: Raw data to store
            step_number: Current simulation step
            memory_type: Type of memory ("state", "interaction", "action")
            priority: Importance of this memory (0.0-1.0)
            tier: Target memory tier for storage ("stm", "im", or "ltm")

        Returns:
            Formatted memory entry
        """
        # Generate memory ID using agent_id, step number and timestamp
        timestamp = int(time.time())
        memory_id = generate_memory_id(memory_type, self.agent_id, step_number)

        # Generate embeddings if available
        embeddings = {}
        if self.embedding_engine:
            embeddings = {
                "full_vector": self.embedding_engine.encode_stm(data),
                "compressed_vector": self.embedding_engine.encode_im(data),
                "abstract_vector": self.embedding_engine.encode_ltm(data),
            }

        # Determine compression level based on tier
        compression_level = 0
        if tier == "im":
            compression_level = 1
            if self.compression_engine:
                data = self.compression_engine.compress(data, level=1)
        elif tier == "ltm":
            compression_level = 2
            if self.compression_engine:
                data = self.compression_engine.compress(data, level=2)

        # Create standardized memory entry
        return {
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "step_number": step_number,
            "timestamp": timestamp,
            "content": data,
            "metadata": {
                "creation_time": timestamp,
                "last_access_time": timestamp,
                "compression_level": compression_level,
                "importance_score": priority,
                "retrieval_count": 0,
                "memory_type": memory_type,
                "current_tier": tier,
            },
            "embeddings": embeddings,
        }

    def _check_memory_transition(self) -> None:
        """Check if memories need to be transitioned between tiers.

        This method implements a hybrid age-importance based memory transition
        mechanism that determines when memories should move between tiers
        based on both capacity constraints and importance scores.
        """
        current_time = time.time()

        # Check if STM is at capacity
        stm_count = self.stm_store.count(self.agent_id)
        if stm_count > self.config.stm_config.memory_limit:
            # Calculate transition scores for all STM memories
            stm_memories = self.stm_store.get_all(self.agent_id)
            transition_candidates = []

            for memory in stm_memories:
                # Calculate memory importance
                importance_score = self._calculate_importance(memory)

                # Calculate age factor (normalized by TTL)
                age = (
                    current_time - memory["metadata"]["creation_time"]
                ) / self.config.stm_config.ttl
                age_factor = min(1.0, max(0.0, age))

                # Calculate transition score: higher means more likely to transition
                transition_score = age_factor * (1.0 - importance_score)

                transition_candidates.append((memory, transition_score))

            # Sort by transition score (highest first)
            transition_candidates.sort(key=lambda x: x[1], reverse=True)

            # Get overflow count memories with highest transition scores
            overflow = stm_count - self.config.stm_config.memory_limit
            to_transition = transition_candidates[:overflow]

            # Compress and move to IM
            for memory, _ in to_transition:
                # Apply level 1 compression
                compressed_entry = self.compression_engine.compress(memory, level=1)
                compressed_entry["metadata"]["compression_level"] = 1
                compressed_entry["metadata"]["last_transition_time"] = current_time
                compressed_entry["metadata"]["current_tier"] = "im"  # Update tier to IM

                # Store in IM
                self.im_store.store(self.agent_id, compressed_entry)

                # Remove from STM
                self.stm_store.delete(self.agent_id, memory["memory_id"])

            logger.debug(
                "Transitioned %d memories from STM to IM for agent %s",
                overflow,
                self.agent_id,
            )

        # Check if IM is at capacity
        im_count = self.im_store.count(self.agent_id)
        if im_count > self.config.im_config.memory_limit:
            # Calculate transition scores for all IM memories
            im_memories = self.im_store.get_all(self.agent_id)
            transition_candidates = []

            for memory in im_memories:
                # Calculate memory importance
                importance_score = self._calculate_importance(memory)

                # Calculate age factor (normalized by TTL)
                age = (
                    current_time - memory["metadata"]["creation_time"]
                ) / self.config.im_config.ttl
                age_factor = min(1.0, max(0.0, age))

                # Calculate transition score: higher means more likely to transition
                transition_score = age_factor * (1.0 - importance_score)

                transition_candidates.append((memory, transition_score))

            # Sort by transition score (highest first)
            transition_candidates.sort(key=lambda x: x[1], reverse=True)

            # Get overflow count memories with highest transition scores
            overflow = im_count - self.config.im_config.memory_limit
            to_transition = transition_candidates[:overflow]

            # Compress and move to LTM
            batch = []
            for memory, _ in to_transition:
                # Apply level 2 compression
                compressed_entry = self.compression_engine.compress(memory, level=2)
                compressed_entry["metadata"]["compression_level"] = 2
                compressed_entry["metadata"]["last_transition_time"] = current_time
                compressed_entry["metadata"][
                    "current_tier"
                ] = "ltm"  # Update tier to LTM

                # Add to batch
                batch.append(compressed_entry)

                # Remove from IM
                self.im_store.delete(self.agent_id, memory["memory_id"])

                # Process in batches
                if len(batch) >= self.config.ltm_config.batch_size:
                    self.ltm_store.store_batch(batch)
                    batch = []

            # Store any remaining entries
            if batch:
                self.ltm_store.store_batch(batch)

            logger.debug(
                "Transitioned %d memories from IM to LTM for agent %s",
                overflow,
                self.agent_id,
            )

    def _calculate_importance(self, memory: Dict[str, Any]) -> float:
        """Calculate an importance score for a memory.

        This uses several factors to determine how important a memory is:
        - Recency: More recent memories are more important
        - Reward: Higher rewards increase importance
        - Relevance: How relevant the memory is to current goals
        - Surprise: How unexpected or novel the memory is

        Args:
            memory: Memory to calculate importance for

        Returns:
            Importance score between 0 and 1
        """
        if memory is None:
            return 0.5  # Default value for None memories

        # Reward magnitude component (40%)
        reward = 0
        if (
            "content" in memory
            and memory["content"]
            and isinstance(memory["content"], dict)
        ):
            reward = memory["content"].get("reward", 0)
        reward_importance = min(1.0, abs(reward) / 10.0) * 0.4

        # Calculate recency factor (20%)
        # More recent memories get higher importance
        current_time = time.time()
        timestamp = memory.get("timestamp", current_time)
        time_diff = max(0, current_time - timestamp)
        recency_factor = math.exp(-time_diff / (24 * 3600))  # Decay over 24 hours
        recency_importance = recency_factor * 0.2

        # Retrieval frequency component (30%)
        retrieval_count = 0
        if (
            "metadata" in memory
            and memory["metadata"]
            and isinstance(memory["metadata"], dict)
        ):
            retrieval_count = memory["metadata"].get("retrieval_count", 0)
        # Cap at 5 for max importance
        retrieval_importance = min(retrieval_count / 5.0, 1.0) * 0.3

        # Surprise factor (10%)
        surprise_factor = 0.0
        if (
            "metadata" in memory
            and memory["metadata"]
            and isinstance(memory["metadata"], dict)
        ):
            surprise_factor = memory["metadata"].get("surprise_factor", 0.0)
        surprise_importance = surprise_factor * 0.1

        # Combine factors
        importance = (
            reward_importance
            + recency_importance
            + retrieval_importance
            + surprise_importance
        )

        # Cap to [0, 1] range
        importance = max(0.0, min(1.0, importance))

        return importance

    def clear_memory(self) -> bool:
        """Clear all memory data for this agent.

        Returns:
            True if clearing was successful
        """
        stm_success = self.stm_store.clear(self.agent_id)
        im_success = self.im_store.clear(self.agent_id)
        ltm_success = self.ltm_store.clear()

        return stm_success and im_success and ltm_success

    def retrieve_similar_states(
        self,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
        threshold: float = 0.6,
        context_weights: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar past states to the provided query state.

        Args:
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types
            threshold: Minimum similarity score threshold (0.0-1.0)
            context_weights: Optional dictionary mapping keys to importance weights

        Returns:
            List of memory entries sorted by similarity to query state
        """
        # If no embedding engine is available, return an empty list or fall back to attribute-based search
        if not self.embedding_engine:
            logger.warning(
                "Neural embeddings disabled - similarity search unavailable. Falling back to attribute-based search."
            )
            # Fallback: Use attribute search for the memory type if specified
            if memory_type:
                return self.retrieve_by_attributes({"memory_type": memory_type})[:k]
            # Otherwise just return an empty list
            return []

        # Generate query embedding
        query_embedding = self.embedding_engine.encode_stm(query_state, context_weights)

        # Search in each tier with appropriate embedding level
        results = []

        # Search STM (full resolution)
        stm_results = self.stm_store.search_similar(
            self.agent_id, query_embedding, k=k, memory_type=memory_type
        )
        results.extend(stm_results)

        # If we need more results, search IM
        if len(results) < k:
            remaining = k - len(results)
            im_query = self.embedding_engine.encode_im(query_state, context_weights)
            im_results = self.im_store.search_similar(
                self.agent_id, im_query, k=remaining, memory_type=memory_type
            )
            results.extend(im_results)

        # If still need more, search LTM
        if len(results) < k:
            remaining = k - len(results)
            ltm_query = self.embedding_engine.encode_ltm(query_state, context_weights)
            ltm_results = self.ltm_store.search_similar(
                ltm_query, k=remaining, memory_type=memory_type
            )
            results.extend(ltm_results)

        # Filter by similarity threshold
        filtered_results = [
            r for r in results if r.get("similarity_score", 0) >= threshold
        ]

        # Sort by similarity score
        filtered_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return filtered_results[:k]

    def retrieve_by_time_range(
        self, start_step: int, end_step: int, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a specific time/step range.

        Args:
            start_step: Beginning of time range
            end_step: End of time range
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries within the specified time range
        """
        results = []

        # Search each tier
        stm_results = self.stm_store.search_by_step_range(
            self.agent_id, start_step, end_step, memory_type
        )
        results.extend(stm_results)

        im_results = self.im_store.search_by_step_range(
            self.agent_id, start_step, end_step, memory_type
        )
        results.extend(im_results)

        ltm_results = self.ltm_store.search_by_step_range(
            start_step, end_step, memory_type
        )
        results.extend(ltm_results)

        # Sort by step number
        results.sort(key=lambda x: x["step_number"])
        return results

    def retrieve_by_attributes(
        self, attributes: Dict[str, Any], memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories matching specific attribute values.

        Args:
            attributes: Dictionary of attribute-value pairs to match
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries matching the specified attributes
        """
        results = []

        # Search each tier
        stm_results = self.stm_store.search_by_attributes(
            self.agent_id, attributes, memory_type
        )
        results.extend(stm_results)

        im_results = self.im_store.search_by_attributes(
            self.agent_id, attributes, memory_type
        )
        results.extend(im_results)

        ltm_results = self.ltm_store.search_by_attributes(
            self.agent_id, attributes, memory_type
        )
        results.extend(ltm_results)

        # Sort by recency (most recent first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def get_memory_statistics(self, simplified: bool = False) -> Dict[str, Any]:
        """Get statistics about memory usage and performance.

        Args:
            simplified: If True, returns only basic tier statistics

        Returns:
            Dictionary containing memory statistics including:
            - Counts per tier
            - Average importance scores
            - Compression ratios
            - Access patterns
            - Memory transitions
        """
        stats = {
            "timestamp": int(time.time()),
            "tiers": {
                "stm": {
                    "count": self.stm_store.count(self.agent_id),
                    "size_bytes": self.stm_store.get_size(self.agent_id),
                    "avg_importance": self._calculate_tier_importance("stm"),
                },
                "im": {
                    "count": self.im_store.count(self.agent_id),
                    "size_bytes": self.im_store.get_size(self.agent_id),
                    "avg_importance": self._calculate_tier_importance("im"),
                    "compression_ratio": self._calculate_compression_ratio("im"),
                },
                "ltm": {
                    "count": self.ltm_store.count(),
                    "size_bytes": self.ltm_store.get_size(),
                    "avg_importance": self._calculate_tier_importance("ltm"),
                    "compression_ratio": self._calculate_compression_ratio("ltm"),
                },
            },
        }

        if not simplified:
            stats.update(
                {
                    "total_memories": (
                        self.stm_store.count(self.agent_id)
                        + self.im_store.count(self.agent_id)
                        + self.ltm_store.count()
                    ),
                    "memory_types": self._get_memory_type_distribution(),
                    "access_patterns": self._get_access_patterns(),
                }
            )

        return stats

    def force_maintenance(self) -> bool:
        """Force memory tier transitions and cleanup operations.

        Returns:
            True if maintenance was successful
        """
        try:
            # Temporarily reduce the STM limit to force transition
            original_stm_limit = self.config.stm_config.memory_limit
            stm_count = self.stm_store.count(self.agent_id)

            # Only modify the limit if there are memories to transition
            if stm_count > 0:
                # Set the limit to 1 less than current count to force transition
                self.config.stm_config.memory_limit = stm_count - 1

                # Run the memory transition
                self._check_memory_transition()

                # Restore the original limit
                self.config.stm_config.memory_limit = original_stm_limit
            else:
                # Just run normal transition if no memories
                self._check_memory_transition()

            return True
        except Exception as e:
            logger.error("Failed to perform maintenance: %s", e)
            return False

    def search_by_embedding(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_tiers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find memories by raw embedding vector similarity.

        Args:
            query_embedding: Embedding vector to search with
            k: Number of results to return
            memory_tiers: Optional list of tiers to search

        Returns:
            List of memory entries sorted by similarity
        """
        results = []
        tiers = memory_tiers or ["stm", "im", "ltm"]

        for tier in tiers:
            if tier == "stm" and "stm" in tiers:
                stm_results = self.stm_store.search_by_embedding(
                    self.agent_id, query_embedding, k=k
                )
                results.extend(stm_results)

            if tier == "im" and "im" in tiers:
                # Compress query for IM search
                im_query = self.compression_engine.compress_embedding(
                    query_embedding, level=1
                )
                im_results = self.im_store.search_by_embedding(
                    self.agent_id, im_query, k=k
                )
                results.extend(im_results)

            if tier == "ltm" and "ltm" in tiers:
                # Compress query for LTM search
                ltm_query = self.compression_engine.compress_embedding(
                    query_embedding, level=2
                )
                ltm_results = self.ltm_store.search_by_embedding(ltm_query, k=k)
                results.extend(ltm_results)

        # Sort by similarity score
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return results[:k]

    def search_by_content(
        self, content_query: Union[str, Dict[str, Any]], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for memories based on content text/attributes.

        Args:
            content_query: String or dict to search for in memory contents
            k: Number of results to return

        Returns:
            List of memory entries matching the content query
        """
        results = []

        # Convert string query to dict if needed
        if isinstance(content_query, str):
            content_query = {"text": content_query}

        # Search each tier
        stm_results = self.stm_store.search_by_content(self.agent_id, content_query, k)
        results.extend(stm_results)

        if len(results) < k:
            remaining = k - len(results)
            im_results = self.im_store.search_by_content(
                self.agent_id, content_query, remaining
            )
            results.extend(im_results)

        if len(results) < k:
            remaining = k - len(results)
            ltm_results = self.ltm_store.search_by_content(content_query, remaining)
            results.extend(ltm_results)

        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:k]

    def register_hook(
        self, event_type: str, hook_function: callable, priority: int = 5
    ) -> bool:
        """Register a hook function for memory formation events.

        Args:
            event_type: Type of event to hook into
            hook_function: Function to call when event is triggered
            priority: Priority level (1-10, 10 being highest)

        Returns:
            True if hook was registered successfully
        """
        if not hasattr(self, "_event_hooks"):
            self._event_hooks = {}

        if event_type not in self._event_hooks:
            self._event_hooks[event_type] = []

        self._event_hooks[event_type].append(
            {"function": hook_function, "priority": priority}
        )

        # Sort hooks by priority (highest first)
        self._event_hooks[event_type].sort(key=lambda x: x["priority"], reverse=True)

        return True

    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Trigger memory formation event hooks.

        Args:
            event_type: Type of event that occurred
            event_data: Data related to the event

        Returns:
            True if event was processed successfully
        """
        if not hasattr(self, "_event_hooks") or event_type not in self._event_hooks:
            return False

        success = True
        event_data["timestamp"] = time.time()

        for hook in self._event_hooks[event_type]:
            try:
                result = hook["function"](event_data, self)

                # Process hook results
                if isinstance(result, dict):
                    if result.get("store_memory", False):
                        self.store_state(
                            result.get("memory_data", event_data),
                            result.get("step_number", 0),
                            result.get("priority", 1.0),
                        )
            except Exception as e:
                logger.error("Hook execution failed: %s", e)
                success = False

        return success

    def _calculate_tier_importance(self, tier: str) -> float:
        """Calculate average importance score for a memory tier."""
        store = getattr(self, f"{tier}_store")

        memories = store.get_all(self.agent_id)

        if not memories:
            return 0.0

        total_importance = sum(
            memory["metadata"]["importance_score"] for memory in memories
        )
        return total_importance / len(memories)

    def _calculate_compression_ratio(self, tier: str) -> float:
        """Calculate compression ratio for a memory tier."""
        store = getattr(self, f"{tier}_store")
        if tier in ["stm", "im"]:
            compressed_size = store.get_size(self.agent_id)
        else:
            compressed_size = store.get_size()

        if compressed_size == 0:
            return 0.0

        # Get original size from metadata
        memories = store.get_all(self.agent_id)

        if not memories:
            return 0.0

        original_size = sum(
            memory["metadata"].get("original_size", compressed_size)
            for memory in memories
        )

        return original_size / compressed_size if compressed_size > 0 else 0.0

    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types across all tiers."""
        distribution = {}

        for store_name in ["stm_store", "im_store", "ltm_store"]:
            store = getattr(self, store_name)

            memories = store.get_all(self.agent_id)

            for memory in memories:
                memory_type = memory["metadata"]["memory_type"]
                distribution[memory_type] = distribution.get(memory_type, 0) + 1

        return distribution

    def _get_access_patterns(self) -> Dict[str, Any]:
        """Get statistics about memory access patterns."""
        patterns = {
            "most_accessed": [],
            "least_accessed": [],
            "avg_accesses": 0,
            "total_accesses": 0,
        }

        all_memories = []
        for store_name in ["stm_store", "im_store", "ltm_store"]:
            store = getattr(self, store_name)
            all_memories.extend(store.get_all(self.agent_id))

        if not all_memories:
            return patterns

        # Calculate access statistics
        access_counts = [
            (memory["metadata"].get("retrieval_count", 0), memory)
            for memory in all_memories
        ]

        patterns["total_accesses"] = sum(count for count, _ in access_counts)
        patterns["avg_accesses"] = patterns["total_accesses"] / len(all_memories)

        # Get most and least accessed
        access_counts.sort(key=lambda x: x[0], reverse=True)
        patterns["most_accessed"] = [m for _, m in access_counts[:5]]
        patterns["least_accessed"] = [m for _, m in access_counts[-5:]]

        return patterns

    def calculate_reward_score(self, memory: Dict[str, Any]) -> float:
        """Calculate a reward score for a memory.

        This is used for reinforcement learning applications to give
        higher importance to memories with higher rewards.

        Args:
            memory: Memory to calculate score for

        Returns:
            Reward score between 0 and 1
        """
        # Extract reward from memory if available
        reward = memory.get("content", {}).get("reward", 0)

        # Cap and normalize reward to [0, 1]
        max_reward = self.config.autoencoder_config.max_reward_score
        normalized_reward = min(max(reward, 0), max_reward) / max_reward

        return normalized_reward

    def hybrid_retrieve(
        self,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None,
        vector_weight: float = 0.4,
        attribute_weight: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Combine similarity and attribute-based search for more robust retrieval.

        Args:
            query_state: State data to use for querying
            k: Number of results to return
            memory_type: Optional filter for specific memory types
            vector_weight: Weight to assign to vector similarity scores (0.0-1.0)
            attribute_weight: Weight to assign to attribute match scores (0.0-1.0)

        Returns:
            List of memory entries sorted by hybrid score
        """
        # Extract key attributes for exact matching
        attribute_filters = {}
        if isinstance(query_state, dict):
            # Look for location information which is often important for context
            if "position" in query_state and isinstance(query_state["position"], dict):
                if "location" in query_state["position"]:
                    attribute_filters["position.location"] = query_state["position"][
                        "location"
                    ]

            # Add other important exact-match attributes
            for key in ["health", "energy", "level"]:
                if key in query_state:
                    attribute_filters[key] = query_state[key]

            # Handle inventory items if present
            if (
                "inventory" in query_state
                and isinstance(query_state["inventory"], list)
                and query_state["inventory"]
            ):
                # Process all inventory items for attribute matching
                attribute_filters["inventory"] = query_state["inventory"]

        # Get more results than needed from both methods
        vector_results = []
        if self.embedding_engine:
            vector_results = self.retrieve_similar_states(
                query_state, k=k * 2, memory_type=memory_type, threshold=0.2
            )
        else:
            logger.warning(
                "Neural embeddings disabled - using only attribute-based search for hybrid retrieval"
            )
            # When no embedding engine is available, use only attribute-based search with higher weight
            attribute_weight = 1.0
            vector_weight = 0.0

        # Only use attribute search if we have filters
        attr_results = []
        if attribute_filters:
            attr_results = self.retrieve_by_attributes(
                attribute_filters, memory_type=memory_type
            )

            # If we don't have any attribute results, try searching with fewer attributes
            if not attr_results and len(attribute_filters) > 1:
                # Try with just location, which is often most important
                if "position.location" in attribute_filters:
                    reduced_filters = {
                        "position.location": attribute_filters["position.location"]
                    }
                    attr_results = self.retrieve_by_attributes(
                        reduced_filters, memory_type=memory_type
                    )

        # Merge and rank results
        combined = {}
        for result in vector_results:
            memory_id = result["memory_id"]
            combined[memory_id] = {
                "memory": result,
                "vector_score": result.get("similarity_score", 0),
                "attr_score": 0,
            }

        for result in attr_results:
            memory_id = result["memory_id"]
            if memory_id in combined:
                combined[memory_id]["attr_score"] = 1.0
            else:
                combined[memory_id] = {
                    "memory": result,
                    "vector_score": 0,
                    "attr_score": 1.0,
                }

        # Calculate final scores and sort
        scored_results = []
        for memory_id, data in combined.items():
            final_score = (data["vector_score"] * vector_weight) + (
                data["attr_score"] * attribute_weight
            )
            data["memory"]["hybrid_score"] = final_score
            scored_results.append(data["memory"])

        # Sort by final score
        scored_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return scored_results[:k]

    def flush_to_ltm(
        self, include_stm: bool = True, include_im: bool = True, force: bool = False
    ) -> Dict[str, int]:
        """Flush memories from STM and/or IM to LTM storage.

        Args:
            include_stm: Whether to flush STM memories
            include_im: Whether to flush IM memories
            force: If True, bypass LTM whitelist filtering for memory types

        Returns:
            Dictionary containing counts of memories processed:
            {
                'stm_stored': int,
                'stm_filtered': int,
                'im_stored': int,
                'im_filtered': int
            }

        Raises:
            Exception: If the memory flush fails after maximum retry attempts
        """
        results = {"stm_stored": 0, "stm_filtered": 0, "im_stored": 0, "im_filtered": 0}

        max_retries = 3
        retry_count = 0
        backoff_factor = 2  # Exponential backoff factor

        while retry_count <= max_retries:
            try:
                # Flush STM memories if requested
                if include_stm:
                    stm_memories = self.stm_store.get_all()
                    if stm_memories:
                        # Update the current tier for all memories to be flushed
                        for memory in stm_memories:
                            memory["metadata"]["current_tier"] = "ltm"

                        stored, filtered = self.ltm_store.flush_memories(
                            stm_memories, force=force
                        )
                        results["stm_stored"] = stored
                        results["stm_filtered"] = filtered
                        # Clear STM after successful flush
                        if stored > 0:
                            self.stm_store.clear()

                # Flush IM memories if requested
                if include_im:
                    im_memories = self.im_store.get_all()
                    if im_memories:
                        # Update the current tier for all memories to be flushed
                        for memory in im_memories:
                            memory["metadata"]["current_tier"] = "ltm"

                        stored, filtered = self.ltm_store.flush_memories(
                            im_memories, force=force
                        )
                        results["im_stored"] = stored
                        results["im_filtered"] = filtered
                        # Clear IM after successful flush
                        if stored > 0:
                            self.im_store.clear()

                logger.info(
                    f"Memory flush complete - STM: {results['stm_stored']} stored, "
                    f"{results['stm_filtered']} filtered; IM: {results['im_stored']} stored, "
                    f"{results['im_filtered']} filtered"
                )
                return results

            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = backoff_factor**retry_count
                    logger.warning(
                        f"Error during memory flush (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to flush memory after {max_retries} attempts: {e}"
                    )
                    # Either propagate the error or return partial results
                    if results["stm_stored"] > 0 or results["im_stored"] > 0:
                        logger.info(
                            "Returning partial results from successful operations"
                        )
                        return results
                    else:
                        raise Exception(f"Memory flush failed completely: {e}")
