from typing import Any, Dict, List, Optional

from .base import SearchStrategy


class CompoundQueryStrategy(SearchStrategy):
    """
    Strategy for executing complex queries with multiple conditions and logical operators,
    enabling sophisticated memory filtering across various attributes.
    """

    def __init__(self, stm_store, im_store, ltm_store):
        """
        Initialize the compound query strategy.

        Args:
            stm_store: Short-term memory store
            im_store: Intermediate memory store
            ltm_store: Long-term memory store
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store

    def name(self) -> str:
        """Return the name of the strategy."""
        return "compound"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Executes compound queries with multiple conditions using logical operators (AND/OR)"

    def search(
        self,
        query: Dict[str, Any],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute a compound query with multiple conditions.

        Args:
            query: Dict containing search parameters:
                - conditions: List of condition dictionaries, each with fields:
                    - field: Dot-notation path to the field (e.g., "metadata.importance")
                    - value: Value to compare against
                    - comparison: Comparison operator (equals, not_equals, greater_than, etc.)
                - operator: Logical operator to combine conditions ("AND" or "OR")
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters

        Returns:
            List of memory entries matching the compound query
        """
        # Check for direct field comparison
        if "field" in query and "value" in query:
            # Single field comparison
            return self._process_simple_query(
                query, agent_id, tier, limit, metadata_filter
            )

        # Extract parameters for compound queries
        operator = query.get("operator", "AND").upper()
        conditions = query.get("conditions", [])

        # Validate query structure
        if "operator" in query and not conditions:
            raise ValueError(
                "Compound query must include 'conditions' when 'operator' is specified"
            )

        if operator not in ["AND", "OR"]:
            raise ValueError(
                f"Invalid logical operator: {operator}. Must be 'AND' or 'OR'"
            )

        # Get stores for the specified tier
        store = self._get_store_for_tier(tier)

        # Retrieve all memories from appropriate store
        memories = store.list(agent_id)

        # Apply compound query filtering
        results = []
        for memory in memories:
            if self._process_compound_condition(memory, query):
                results.append(memory)

        # Apply any additional metadata filtering
        if metadata_filter:
            results = [
                memory
                for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]

        return results[:limit]

    def _process_simple_query(self, query, agent_id, tier, limit, metadata_filter=None):
        """Process a simple field-value query."""
        field = query.get("field")
        value = query.get("value")
        comparison = query.get("comparison", "equals")

        if not field or "value" not in query:
            raise ValueError("Simple query must include 'field' and 'value'")

        if comparison not in [
            "equals",
            "not_equals",
            "contains",
            "greater_than",
            "greater_than_equal",
            "less_than",
            "less_than_equal",
            "regex",
        ]:
            raise ValueError(f"Invalid comparison operator: {comparison}")

        # Get store for the tier
        store = self._get_store_for_tier(tier)
        memories = store.list(agent_id)

        # Filter memories
        results = []
        for memory in memories:
            actual_value = self._get_field_value(memory, field)
            if self._compare_values(actual_value, value, comparison):
                results.append(memory)

        # Apply any additional metadata filtering
        if metadata_filter:
            results = [
                memory
                for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]

        return results[:limit]

    def _process_compound_condition(self, memory, condition):
        """
        Process a compound condition against a memory.

        Args:
            memory: Memory to check
            condition: Condition to apply

        Returns:
            Boolean indicating if the memory matches the condition
        """
        # Handle nested compound conditions
        if "operator" in condition and "conditions" in condition:
            operator = condition["operator"].upper()
            sub_conditions = condition["conditions"]

            results = []
            for sub_condition in sub_conditions:
                match = self._process_compound_condition(memory, sub_condition)
                results.append(match)

            if operator == "AND":
                return all(results)
            elif operator == "OR":
                return any(results)
            else:
                return False

        # Handle simple field comparison
        elif "field" in condition and "value" in condition:
            field = condition["field"]
            value = condition["value"]
            comparison = condition.get("comparison", "equals")

            actual_value = self._get_field_value(memory, field)
            return self._compare_values(actual_value, value, comparison)

        return False

    def _get_field_value(self, memory, field_path):
        """
        Extract a value from a memory using dot notation for nested fields.

        Args:
            memory: The memory entry to extract from
            field_path: Dot-notation path to the field

        Returns:
            The value at the specified path or None if not found
        """
        parts = field_path.split(".")
        current = memory

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                # Handle list indexing if part is a number
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None

        return current

    def _compare_values(self, actual, expected, comparison):
        """
        Compare values using the specified operator.

        Args:
            actual: The actual value from the memory
            expected: The expected value to compare against
            comparison: The comparison type (equals, not_equals, etc.)

        Returns:
            Boolean indicating whether the comparison is satisfied
        """
        if actual is None:
            return False

        if comparison == "equals":
            return actual == expected
        elif comparison == "not_equals":
            return actual != expected
        elif comparison == "greater_than":
            return actual > expected
        elif comparison == "greater_than_equal":
            return actual >= expected
        elif comparison == "less_than":
            return actual < expected
        elif comparison == "less_than_equal":
            return actual <= expected
        elif comparison == "contains":
            if isinstance(actual, (list, tuple, set)):
                return expected in actual
            elif isinstance(actual, str) and isinstance(expected, str):
                return expected in actual
            return False
        elif comparison == "regex":
            import re

            if isinstance(actual, str) and isinstance(expected, str):
                try:
                    return bool(re.match(expected, actual))
                except:
                    return False
            return False
        else:
            return False

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

    def _matches_metadata_filters(self, memory, metadata_filter):
        """Check if a memory matches the specified metadata filters."""
        if not metadata_filter:
            return True

        memory_metadata = memory.get("metadata", {})
        for key, value in metadata_filter.items():
            if memory_metadata.get(key) != value:
                return False

        return True
