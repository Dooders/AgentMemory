"""Step-based search strategy for the agent memory search model.

This strategy provides a specialized way to search for memories based on their simulation
step numbers rather than actual timestamps. It's particularly useful in environments
where the progression through simulation steps is more meaningful than wall-clock time.

Example usage:

```python
from memory.search import SearchModel
from memory.search.strategies.step_based import StepBasedSearchStrategy

# Create memory configuration and search model
config = MemoryConfig()
search_model = SearchModel(config)

# Create and register the step-based strategy
step_strategy = StepBasedSearchStrategy(memory_system)
search_model.register_strategy(step_strategy)

# Search for memories within a step range
step_range_memories = search_model.search(
    query={"start_step": 1000, "end_step": 2000},
    agent_id="agent-123",
    strategy_name="step_based",
    limit=10
)

# Search for memories near a specific step
nearby_memories = search_model.search(
    query="1500",  # Reference step number
    agent_id="agent-123",
    strategy_name="step_based",
    step_range=200,  # Look 200 steps in each direction
    limit=5
)
```
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from memory.search.strategies.base import SearchStrategy
from memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)


class StepBasedSearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on simulation step numbers.

    This strategy is optimized for retrieving memories based on their simulation step
    numbers rather than actual timestamps. It's ideal for scenarios where the progression
    of an agent through simulation steps is more meaningful than real-world time.

    Attributes:
        memory_system: The memory system instance
    """

    def __init__(
        self,
        memory_system: AgentMemorySystem,
    ):
        """Initialize the step-based search strategy.

        Args:
            memory_system: The memory system instance
        """
        self.memory_system = memory_system

    def name(self) -> str:
        """Return the name of the search strategy.

        Returns:
            String name of the strategy
        """
        return "step_based"

    def description(self) -> str:
        """Return a description of the search strategy.

        Returns:
            String description of the strategy
        """
        return "Searches for memories based on simulation step numbers"

    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        reference_step: Optional[int] = None,
        step_range: int = 100,
        step_weight: float = 1.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search for memories based on step numbers.

        Args:
            query: Search query (can be a step number, step range dict, or step-related data)
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            start_step: Optional start simulation step for range queries
            end_step: Optional end simulation step for range queries
            reference_step: Optional reference step for proximity-based queries
            step_range: Range around the reference step to search (if using reference_step)
            step_weight: Weight to apply for step-based scoring (higher values prioritize exact matches)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of memory entries matching the search criteria
        """
        results = []
        step_params = self._process_query(
            query, start_step, end_step, reference_step, step_range
        )
        tiers_to_search = ["stm", "im", "ltm"] if tier is None else [tier]
        agent = self.memory_system.get_memory_agent(agent_id)
        for current_tier in tiers_to_search:
            if current_tier not in ["stm", "im", "ltm"]:
                logger.warning("Unsupported memory tier: %s", current_tier)
                continue
            tier_memories = []
            if current_tier == "stm":
                tier_memories = agent.stm_store.get_all(agent_id)
            elif current_tier == "im":
                tier_memories = agent.im_store.get_all(agent_id)
            else:  # ltm
                tier_memories = agent.ltm_store.get_all(agent_id)
            filtered_memories = self._filter_memories(
                tier_memories,
                step_params,
                metadata_filter,
            )
            scored_memories = self._score_memories(
                filtered_memories,
                step_params,
                step_weight,
                current_tier,
            )
            results.extend(scored_memories)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("step_score", 0.0), reverse=True
        )
        return results[:limit]

    def _process_query(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        reference_step: Optional[int] = None,
        step_range: int = 100,
    ) -> Dict[str, Any]:
        """Process the search query to extract step parameters.

        Args:
            query: Search query
            start_step: Optional start step
            end_step: Optional end step
            reference_step: Optional reference step
            step_range: Range around reference step to search

        Returns:
            Dictionary of step parameters
        """
        params = {
            "start_step": None,
            "end_step": None,
            "reference_step": None,
            "step_range": step_range,
            "query_keys": [],  # Track keys in the query dict to identify query type
        }

        # Handle string queries
        if isinstance(query, str):
            # Try to parse as a step number
            try:
                step = int(query)
                params["reference_step"] = step
                # If we have a reference step, calculate default start and end steps
                params["start_step"] = max(0, step - step_range)
                params["end_step"] = step + step_range
                logger.debug(
                    f"Parsed query as step number: {step} with range: {params['start_step']} to {params['end_step']}"
                )
            except ValueError:
                # Not a valid step number, ignore
                pass

        # Handle dictionary queries
        elif isinstance(query, dict):
            # Track the keys in the query dictionary
            params["query_keys"] = list(query.keys())

            # Process step parameters
            if "start_step" in query:
                params["start_step"] = self._parse_int(query["start_step"])
            if "end_step" in query:
                params["end_step"] = self._parse_int(query["end_step"])
            if "reference_step" in query:
                params["reference_step"] = self._parse_int(query["reference_step"])
                # If only reference_step is provided, calculate start and end steps
                if params["start_step"] is None and params["end_step"] is None:
                    ref_step = params["reference_step"]
                    if ref_step is not None:
                        params["start_step"] = max(0, ref_step - step_range)
                        params["end_step"] = ref_step + step_range
            if "step_range" in query:
                range_value = self._parse_int(query["step_range"])
                if range_value is not None:
                    params["step_range"] = range_value

        # Override with explicitly provided parameters (these take precedence over query dict)
        if start_step is not None:
            params["start_step"] = start_step
        if end_step is not None:
            params["end_step"] = end_step
        if reference_step is not None:
            params["reference_step"] = reference_step
            # If only reference_step is provided as an override, recalculate range
            if start_step is None and end_step is None:
                params["start_step"] = max(0, reference_step - step_range)
                params["end_step"] = reference_step + step_range

        # Debug the final parameters
        logger.debug(f"Processed step parameters: {params}")
        return params

    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse an integer value.

        Args:
            value: Value to parse

        Returns:
            Parsed integer or None if parsing failed
        """
        if isinstance(value, int):
            return value

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Could not parse integer string: {value}")

        return None

    def _get_memory_step(self, memory: Dict[str, Any]) -> Optional[int]:
        """Extract step number from a memory entry.

        Args:
            memory: Memory entry

        Returns:
            Step number or None if not found
        """
        # Try to get step directly
        if "step_number" in memory:
            return memory["step_number"]

        # Try to get from contents
        if "contents" in memory and isinstance(memory["contents"], dict):
            if "step_number" in memory["contents"]:
                return memory["contents"]["step_number"]

        # Try to get from metadata
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            if "step_number" in memory["metadata"]:
                return memory["metadata"]["step_number"]
            if "step" in memory["metadata"]:
                return self._parse_int(memory["metadata"]["step"])

        # Try to get from the main memory object's step field
        if "step" in memory:
            return self._parse_int(memory["step"])

        return None

    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from a nested dictionary using a dot-separated path.

        Args:
            obj: Dictionary to extract value from
            path: Dot-separated path to the value (e.g., "content.metadata.importance")

        Returns:
            Value found at the path, or None if not found
        """
        if not path or not isinstance(obj, dict):
            return None

        parts = path.split(".")
        current = obj

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]

        return current

    def _metadata_matches(
        self, memory: Dict[str, Any], metadata_filter: Dict[str, Any]
    ) -> bool:
        """Check if a memory matches the metadata filter.

        Args:
            memory: Memory object to check
            metadata_filter: Metadata filter to apply

        Returns:
            True if memory matches filter, False otherwise
        """
        for key, filter_value in metadata_filter.items():
            memory_value = self._get_nested_value(memory, key)
            if memory_value is None:
                # Try to get from non-nested metadata
                memory_value = memory.get("metadata", {}).get(key)

            # No matching value found
            if memory_value is None:
                logger.debug(f"No value found for key {key} in memory")
                return False

            # Handle list/array values - check if filter_value is a subset of memory_value
            if isinstance(filter_value, list):
                if not isinstance(memory_value, list):
                    # Convert to list if memory_value is a single value
                    memory_value = [memory_value]

                # Check if all items in filter_value are in memory_value
                if not all(item in memory_value for item in filter_value):
                    logger.debug(
                        f"List match failed for {key}: filter={filter_value}, memory={memory_value}"
                    )
                    return False
            # For other types, do a direct comparison
            elif memory_value != filter_value:
                logger.debug(
                    f"Value mismatch for {key}: filter={filter_value}, memory={memory_value}"
                )
                return False

        return True

    def _filter_memories(
        self,
        memories: List[Dict[str, Any]],
        step_params: Dict[str, Any],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter memories based on step parameters and metadata.

        Args:
            memories: List of memories to filter
            step_params: Step parameters to filter by
            metadata_filter: Additional metadata filters

        Returns:
            Filtered list of memories
        """
        start_step = step_params.get("start_step")
        end_step = step_params.get("end_step")

        # Debug the filters
        if start_step is not None:
            logger.debug(f"Filtering with start_step: {start_step}")
        if end_step is not None:
            logger.debug(f"Filtering with end_step: {end_step}")

        filtered = []
        for memory in memories:
            memory_id = memory.get("id", "unknown")

            # Get memory step
            memory_step = self._get_memory_step(memory)

            # Skip memories without a step number
            if memory_step is None:
                logger.debug(f"Memory {memory_id} has no step number, skipping")
                continue

            # Check step constraints
            if start_step is not None and memory_step < start_step:
                logger.debug(
                    f"Memory {memory_id} is before start_step: {memory_step} < {start_step}"
                )
                continue
            if end_step is not None and memory_step > end_step:
                logger.debug(
                    f"Memory {memory_id} is after end_step: {memory_step} > {end_step}"
                )
                continue

            # Apply metadata filter
            if metadata_filter and len(metadata_filter) > 0:
                if not self._metadata_matches(memory, metadata_filter):
                    continue

            filtered.append(memory)

        logger.debug(f"Filtered down to {len(filtered)} memories from {len(memories)}")
        return filtered

    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        step_params: Dict[str, Any],
        step_weight: float,
        tier: str,
    ) -> List[Dict[str, Any]]:
        """Score memories based on step proximity.

        Args:
            memories: List of memory entries to score
            step_params: Step parameters from query processing
            step_weight: Weight for step-based scoring
            tier: Memory tier ("stm", "im", "ltm")

        Returns:
            List of scored memory entries
        """
        reference_step = step_params.get("reference_step")
        scored_memories = []

        for memory in memories:
            # Start with base score
            step_score = 0.5

            # Get memory step
            memory_step = self._get_memory_step(memory)

            # Score based on step proximity if reference step is provided
            if memory_step is not None and reference_step is not None:
                # Calculate how close this memory is to the reference step
                step_distance = abs(memory_step - reference_step)

                # Normalize step distance (closer = higher score)
                # Adjust max_distance based on your simulation scale
                max_distance = step_params.get("step_range", 100) * 2
                # Avoid division by zero
                if max_distance > 0:
                    normalized_distance = min(step_distance / max_distance, 1.0)
                else:
                    normalized_distance = 1.0 if step_distance > 0 else 0.0

                # Higher score for closer steps (1.0 for exact match, decreasing as distance increases)
                step_score = 1.0 - normalized_distance

                # Apply step weight (higher weight emphasizes proximity more)
                step_score = pow(step_score, 1.0 / max(step_weight, 0.001))

                # Log the scoring calculations for debugging
                logger.debug(
                    f"Memory {memory.get('id', 'unknown')}: step={memory_step}, "
                    f"ref={reference_step}, distance={step_distance}, "
                    f"normalized_distance={normalized_distance}, score={step_score}"
                )

            # Create a copy of the memory to avoid modifying the original
            memory_copy = memory.copy()

            # Add score to memory metadata
            if "metadata" not in memory_copy:
                memory_copy["metadata"] = {}

            memory_copy["metadata"]["step_score"] = step_score
            memory_copy["metadata"]["tier"] = tier
            memory_copy["metadata"]["step_number"] = memory_step

            # Debug scoring
            logger.debug(
                f"Memory {memory_copy.get('id', 'unknown')} with step {memory_step} scored: {step_score}"
            )

            scored_memories.append(memory_copy)

        return scored_memories
