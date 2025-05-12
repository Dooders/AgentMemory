"""Similarity-based search strategy for the agent memory search model."""

import logging
from typing import Any, Dict, List, Optional, Union

from memory.core import AgentMemorySystem
from memory.search.strategies.base import SearchStrategy

logger = logging.getLogger(__name__)


class SimilaritySearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on semantic similarity.

    This strategy uses vector embeddings to find memories that are semantically
    similar to the provided query.

    Attributes:
        vector_store: Vector store for similarity queries
        embedding_engine: Engine for generating embeddings
        memory_system: The memory system instance
        config: Memory configuration
    """

    # Default minimum similarity score threshold
    DEFAULT_MIN_SCORE = 0.6

    def __init__(self, memory_system: AgentMemorySystem):
        """Initialize the similarity search strategy.

        Args:
            memory_system: The memory system instance
        """
        self.memory_system = memory_system
        self.vector_store = self.memory_system.vector_store
        self.embedding_engine = self.memory_system.embedding_engine
        self.config = self.memory_system.config

    def name(self) -> str:
        """Return the name of the search strategy.

        Returns:
            String name of the strategy
        """
        return "similarity"

    def description(self) -> str:
        """Return a description of the search strategy.

        Returns:
            String description of the strategy
        """
        return (
            "Searches for memories based on semantic similarity using vector embeddings"
        )

    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        min_score: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories based on semantic similarity.

        Args:
            query: Search query, can be a text string, state dictionary, or embedding vector
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            min_score: Minimum similarity score threshold (0.0-1.0), defaults to DEFAULT_MIN_SCORE
            **kwargs: Additional parameters (ignored)

        Returns:
            List of memory entries matching the search criteria
        """
        # Use provided min_score or fall back to default
        min_score = min_score if min_score is not None else self.DEFAULT_MIN_SCORE

        logger.debug(
            "Starting similarity search for agent %s with query type %s, limit %d, min_score %.2f",
            agent_id,
            type(query).__name__,
            limit,
            min_score,
        )
        if metadata_filter:
            logger.debug("Using metadata filter: %s", metadata_filter)

        # Initialize results - we'll keep a dict to preserve order and handle duplicates
        results_dict = {}  # memory_id -> memory_dict

        # Process all tiers or only the specified one
        tiers_to_search = ["stm", "im", "ltm"] if tier is None else [tier]
        logger.debug("Searching memory tiers: %s", tiers_to_search)

        # Get the memory agent
        memory_agent = self.memory_system.get_memory_agent(agent_id)

        for current_tier in tiers_to_search:
            # Skip if tier is not supported
            if current_tier not in ["stm", "im", "ltm"]:
                logger.warning("Unsupported memory tier: %s", current_tier)
                continue

            logger.debug("Processing tier: %s", current_tier)

            # Generate query vector from input
            query_vector = self._generate_query_vector(query, current_tier)

            # Skip if vector generation failed
            if query_vector is None:
                logger.warning(
                    "Failed to generate query vector for tier: %s", current_tier
                )
                continue

            logger.debug(
                "Generated query vector of length %d for tier %s",
                len(query_vector),
                current_tier,
            )

            # Find similar vectors
            logger.debug(
                "Calling vector_store.find_similar_memories for tier %s", current_tier
            )
            similar_vectors = self.vector_store.find_similar_memories(
                query_vector,
                tier=current_tier,
                limit=limit * 2,  # Get extra results to allow for score filtering
                metadata_filter=metadata_filter or {},
            )

            logger.debug(
                "Vector store returned %d results for tier %s. Raw results: %s",
                len(similar_vectors),
                current_tier,
                similar_vectors,
            )

            # Filter by score
            filtered_vectors = [v for v in similar_vectors if v["score"] >= min_score]
            logger.debug(
                "After score filtering (min_score=%.2f): %d vectors remaining in tier %s. Filtered results: %s",
                min_score,
                len(filtered_vectors),
                current_tier,
                filtered_vectors,
            )

            # Limit results
            filtered_vectors = filtered_vectors[:limit]

            # Fetch full memory entries
            for result in filtered_vectors:
                memory_id = result["id"]
                score = result["score"]

                # Check if we've already found this memory with a higher score
                if (
                    memory_id in results_dict
                    and results_dict[memory_id]["metadata"]["similarity_score"] >= score
                ):
                    logger.debug(
                        "Skipping duplicate memory %s with lower score %.2f",
                        memory_id,
                        score,
                    )
                    continue

                memory = None

                # Look up in the appropriate store
                logger.debug(
                    "Fetching memory %s from %s store", memory_id, current_tier
                )
                if current_tier == "stm":
                    memory = memory_agent.stm_store.get(agent_id, memory_id)
                elif current_tier == "im":
                    memory = memory_agent.im_store.get(agent_id, memory_id)
                else:  # ltm
                    memory = memory_agent.ltm_store.get(memory_id)

                logger.debug("Memory store returned: %s", memory)

                if memory:
                    # Ensure memory is a dictionary to avoid issues with MagicMock objects
                    if not isinstance(memory, dict):
                        try:
                            # Try to convert to dictionary if not already
                            memory = dict(memory)
                            logger.debug("Converted memory to dictionary: %s", memory)
                        except (TypeError, ValueError):
                            # Skip this result if conversion fails
                            logger.warning(
                                "Skipping result with non-dictionary memory: %s",
                                memory_id,
                            )
                            continue

                    # Attach similarity score and tier information
                    if "metadata" not in memory:
                        memory["metadata"] = {}
                    memory["metadata"]["similarity_score"] = score
                    memory["metadata"]["memory_tier"] = current_tier

                    # Store in results dict
                    results_dict[memory_id] = memory
                    logger.debug(
                        "Added memory %s from tier %s with score %.2f. Full memory: %s",
                        memory_id,
                        current_tier,
                        score,
                        memory,
                    )

        # Convert to list and ensure they're all dictionaries
        results = [r for r in results_dict.values() if isinstance(r, dict)]
        logger.debug("Final results before sorting: %s", results)

        # Sort by similarity score (descending)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("similarity_score", 0.0),
            reverse=True,
        )

        # Limit final results
        final_results = results[:limit]
        logger.debug(
            "Search complete. Returning %d results with scores ranging from %.2f to %.2f. Final results: %s",
            len(final_results),
            final_results[-1]["metadata"]["similarity_score"] if final_results else 0.0,
            final_results[0]["metadata"]["similarity_score"] if final_results else 0.0,
            final_results,
        )
        return final_results

    def _generate_query_vector(
        self, query: Union[str, Dict[str, Any], List[float]], tier: str
    ) -> Optional[List[float]]:
        """Generate a query vector from various input types.

        Args:
            query: Input query (string, dictionary, or vector)
            tier: Memory tier to generate vector for

        Returns:
            Vector embedding or None if generation failed
        """
        logger.debug(
            "Generating query vector for tier %s from input type %s. Input: %s",
            tier,
            type(query).__name__,
            query,
        )

        # If query is already a vector, use it directly
        if isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            logger.debug("Using provided vector directly (length: %d)", len(query))
            return query

        # If query is a string, convert to a simple state dictionary
        if isinstance(query, str):
            logger.debug("Converting string query to state dictionary")
            query = {"content": query}

        # If query is a dictionary, encode it based on tier
        if isinstance(query, dict):
            logger.debug(
                "Encoding dictionary query for tier %s. Query dict: %s", tier, query
            )
            if tier == "stm":
                vector = self.embedding_engine.encode_stm(query)
            elif tier == "im":
                vector = self.embedding_engine.encode_im(query)
            elif tier == "ltm":
                vector = self.embedding_engine.encode_ltm(query)

            if vector is not None:
                logger.debug(
                    "Successfully generated vector of length %d: %s",
                    len(vector),
                    vector,
                )
            else:
                logger.warning("Failed to generate vector for tier %s", tier)
            return vector

        # If we get here, we couldn't generate a vector
        logger.warning("Could not generate vector for query type: %s", type(query))
        return None
