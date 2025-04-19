from typing import Any, Dict, List, Optional, Union

from .base import SearchStrategy


class ExampleMatchingStrategy(SearchStrategy):
    """
    Strategy for finding memories that match a provided example pattern
    using semantic similarity.
    """

    def __init__(self, vector_store, embedding_engine, stm_store, im_store, ltm_store):
        """
        Initialize the example matching strategy.

        Args:
            vector_store: Vector store for similarity calculations
            embedding_engine: Embedding engine for encoding examples
            stm_store: Short-term memory store
            im_store: Intermediate memory store
            ltm_store: Long-term memory store
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store

    def name(self) -> str:
        """Return the name of the strategy."""
        return "match"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Finds memories that match a provided example pattern using semantic similarity"

    def search(
        self,
        query: Dict[str, Any],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        min_score: float = 0.6,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for memories that match the provided example pattern.

        Args:
            query: Dictionary containing example pattern and optional fields
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            min_score: Minimum similarity score threshold
            **kwargs: Additional parameters

        Returns:
            List of matching memory entries
        """
        # Extract parameters
        if "example" not in query:
            raise ValueError("Query must include an 'example' memory to match against")

        example = query["example"]
        fields_mask = query.get("fields")

        if not isinstance(example, dict):
            raise ValueError("Example must be a dictionary")

        if fields_mask is not None and not isinstance(fields_mask, list):
            raise ValueError("Fields mask must be a list of field paths")

        # Apply fields mask if provided
        if fields_mask:
            example_to_encode = self._extract_fields(example, fields_mask)
        else:
            example_to_encode = example

        # Encode the example based on the tier
        vector = None
        if tier == "stm":
            vector = self.embedding_engine.encode_stm(example_to_encode)
        elif tier == "im":
            vector = self.embedding_engine.encode_im(example_to_encode)
        elif tier == "ltm":
            vector = self.embedding_engine.encode_ltm(example_to_encode)
        else:
            vector = self.embedding_engine.encode(example_to_encode)

        if not vector:
            return []

        # Find similar memories
        search_limit = limit * 2  # Get more results for post-filtering
        similar_memories = self.vector_store.find_similar_memories(
            vector, tier=tier, limit=search_limit, metadata_filter=metadata_filter
        )

        if not similar_memories:
            return []

        # Retrieve full memories and add match scores
        store = self._get_store_for_tier(tier)
        results = []

        for match in similar_memories:
            memory_id = match["id"]
            score = match["score"]

            if score < min_score:
                continue

            memory = store.get(memory_id)
            if memory:
                memory["metadata"]["match_score"] = score
                results.append(memory)

        return results[:limit]

    def _get_store_for_tier(self, tier):
        """Get the appropriate memory store based on the specified tier."""
        if tier == "stm":
            return self.stm_store
        elif tier == "im":
            return self.im_store
        elif tier == "ltm":
            return self.ltm_store
        else:
            return self.stm_store  # Default to STM

    def _extract_fields(self, memory, field_paths):
        """
        Extract specific fields from a memory based on a field mask.

        Args:
            memory: The memory object to extract fields from
            field_paths: List of dot-notation paths to fields to extract

        Returns:
            New memory object containing only the specified fields
        """
        result = {}

        for path in field_paths:
            parts = path.split(".")

            # Navigate to the correct nested dict
            current_src = memory
            current_dest = result

            for i, part in enumerate(parts):
                # Create nested dicts as needed
                if i < len(parts) - 1:
                    if part not in current_src:
                        # Skip this field path if an intermediate key is missing
                        break
                        continue

                    if part not in current_dest:
                        current_dest[part] = {}

                    current_src = current_src.get(part, {})
                    current_dest = current_dest[part]
                else:
                    # Set the leaf value
                    if part in current_src:
                        current_dest[part] = current_src[part]

        return result
