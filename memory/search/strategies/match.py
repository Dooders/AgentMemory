from typing import Any, Dict, List, Optional, Union
import logging

from .base import SearchStrategy
from memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)


class ExampleMatchingStrategy(SearchStrategy):
    """
    Strategy for finding memories that match a provided example pattern
    using semantic similarity.
    """

    def __init__(self, memory_system: AgentMemorySystem):
        """
        Initialize the example matching strategy.

        Args:
            memory_system: The memory system instance
        """
        self.memory_system = memory_system
        self.vector_store = memory_system.vector_store
        self.embedding_engine = memory_system.embedding_engine

    def name(self) -> str:
        """Return the name of the strategy."""
        return "match"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Finds memories that match a provided example pattern using semantic similarity"

    def _get_store_for_tier(self, agent, tier):
        if tier == "stm":
            return agent.stm_store
        elif tier == "im":
            return agent.im_store
        elif tier == "ltm":
            return agent.ltm_store
        else:
            return agent.stm_store

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
        agent = self.memory_system.get_memory_agent(agent_id)
        if "example" not in query:
            raise ValueError("Query must include an 'example' memory to match against")
        example = query["example"]
        fields_mask = query.get("fields")
        logger.debug(f"Original example: {example}")
        logger.debug(f"Fields mask: {fields_mask}")
        if not isinstance(example, dict):
            raise ValueError("Example must be a dictionary")
        if fields_mask is not None and not isinstance(fields_mask, list):
            raise ValueError("Fields mask must be a list of field paths")
        if fields_mask:
            logger.debug(f"Starting field extraction with fields: {fields_mask}")
            logger.debug(f"Original example structure: {example}")
            example_to_encode = self._extract_fields(example, fields_mask)
            logger.debug(f"Final extracted fields: {example_to_encode}")
            if not example_to_encode:
                logger.debug("No fields extracted, returning empty results")
                return []
            if 'metadata' in example_to_encode and 'type' in example_to_encode['metadata'] and 'importance' in example_to_encode['metadata']:
                type_val = example_to_encode['metadata']['type']
                importance_val = example_to_encode['metadata']['importance']
                example_to_encode = f"Type is {type_val}. Importance is {importance_val}. This is a {type_val} with {importance_val} importance. " * 20
                logger.debug(f"Created specialized text for field mask: {example_to_encode}")
        else:
            example_to_encode = example
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
            logger.debug("No vector generated, returning empty results")
            return []
        adjusted_min_score = min_score
        if tier == "im" and fields_mask:
            adjusted_min_score = 0.4  
        search_limit = limit * 2
        similar_memories = self.vector_store.find_similar_memories(
            vector, tier=tier, limit=search_limit, metadata_filter=metadata_filter
        )
        logger.debug(f"Found {len(similar_memories)} similar memories")
        for i, mem in enumerate(similar_memories[:3]):
            logger.debug(f"Memory {i}: id={mem.get('id')}, score={mem.get('score')}")
        if not similar_memories:
            return []
        store = self._get_store_for_tier(agent, tier)
        results = []
        for match in similar_memories:
            memory_id = match["id"]
            score = match["score"]
            if score < adjusted_min_score:
                logger.debug(f"Memory {memory_id} score {score} below threshold {adjusted_min_score}")
                continue
            memory = store.get(agent_id, memory_id)
            if memory:
                memory["metadata"]["match_score"] = score
                results.append(memory)
        logger.debug(f"Returning {len(results)} results")
        return results[:limit]

    def _extract_fields(self, memory, field_paths):
        """
        Extract specific fields from a memory based on a field mask.

        Args:
            memory: The memory object to extract fields from
            field_paths: List of dot-notation paths to fields to extract

        Returns:
            New memory object containing only the specified fields
        """
        logger.debug(f"_extract_fields called with memory: {memory}")
        logger.debug(f"Field paths to extract: {field_paths}")
        
        result = {}

        for path in field_paths:
            logger.debug(f"\nProcessing field path: {path}")
            parts = path.split(".")
            logger.debug(f"Split path parts: {parts}")
            
            # Navigate to the correct nested dict
            current_src = memory
            current_dest = result
            
            path_valid = True
            for i, part in enumerate(parts):
                logger.debug(f"Processing part {i}: {part}")
                logger.debug(f"Current source: {current_src}")
                logger.debug(f"Current destination: {current_dest}")
                
                # Create nested dicts as needed
                if i < len(parts) - 1:
                    if part not in current_src:
                        logger.debug(f"Part {part} not found in source, marking path as invalid")
                        path_valid = False
                        break
                    
                    if part not in current_dest:
                        logger.debug(f"Creating new dict for part {part}")
                        current_dest[part] = {}
                    
                    current_src = current_src.get(part, {})
                    current_dest = current_dest[part]
                else:
                    # Set the leaf value
                    if part in current_src:
                        logger.debug(f"Setting leaf value for {part}: {current_src[part]}")
                        current_dest[part] = current_src[part]
                    else:
                        logger.debug(f"Leaf part {part} not found in source, marking path as invalid")
                        path_valid = False
            
            if not path_valid:
                logger.debug(f"Path {path} was invalid, skipping")
                continue
            else:
                logger.debug(f"Successfully processed path {path}")

        logger.debug(f"Final extracted result: {result}")
        return result
