from typing import Dict, List, Any, Optional, Union
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
        return "compound_query"
    
    def description(self) -> str:
        """Return the description of the strategy."""
        return "Executes complex queries with multiple conditions using logical operators (AND/OR)"
    
    def search(
        self,
        query: Dict[str, Any],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute a compound query with multiple conditions.
        
        Args:
            query: Dict containing search parameters:
                - queries: List of condition dictionaries, each with fields:
                    - field: Dot-notation path to the field (e.g., "metadata.importance")
                    - value: Value to compare against
                    - operator: Comparison operator (==, !=, >, >=, <, <=, in, contains)
                - operator: Logical operator to combine conditions ("AND" or "OR")
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters
                
        Returns:
            List of memory entries matching the compound query
        """
        # Extract parameters
        conditions = query.get("queries", [])
        logical_operator = query.get("operator", "AND").upper()
        
        if not conditions:
            return []
            
        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)
        
        # Retrieve all memories from appropriate stores
        all_memories = []
        for store in stores:
            memories = store.get_all(agent_id)
            all_memories.extend(memories)
            
        # Apply compound query filtering
        results = []
        for memory in all_memories:
            matches = []
            
            for condition in conditions:
                field = condition.get("field")
                value = condition.get("value")
                compare_op = condition.get("operator", "==")
                
                if not field:
                    continue
                    
                # Extract the actual value from the memory
                actual_value = self._get_field_value(memory, field)
                
                # Compare using the specified operator
                match = self._compare_values(actual_value, value, compare_op)
                matches.append(match)
            
            # Apply logical operator to determine overall match
            if (logical_operator == "AND" and all(matches)) or \
               (logical_operator == "OR" and any(matches)):
                results.append(memory)
        
        # Apply additional metadata filtering
        if metadata_filter:
            results = [
                memory for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]
        
        return results[:limit]
    
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
    
    def _compare_values(self, actual, expected, operator):
        """
        Compare values using the specified operator.
        
        Args:
            actual: The actual value from the memory
            expected: The expected value to compare against
            operator: The comparison operator
            
        Returns:
            Boolean indicating whether the comparison is satisfied
        """
        if actual is None:
            return False
            
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">":
            return actual > expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<":
            return actual < expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "in":
            if isinstance(expected, (list, tuple, set)):
                return actual in expected
            return False
        elif operator == "contains":
            if isinstance(actual, (list, tuple, set)):
                return expected in actual
            elif isinstance(actual, str) and isinstance(expected, str):
                return expected in actual
            return False
        else:
            return False
    
    def _get_stores_for_tier(self, tier):
        """Get the appropriate memory stores based on the specified tier."""
        if tier == "stm":
            return [self.stm_store]
        elif tier == "im":
            return [self.im_store]
        elif tier == "ltm":
            return [self.ltm_store]
        else:
            return [self.stm_store, self.im_store, self.ltm_store]
    
    def _matches_metadata_filters(self, memory, metadata_filter):
        """Check if a memory matches the specified metadata filters."""
        if not metadata_filter:
            return True
            
        memory_metadata = memory.get("metadata", {})
        for key, value in metadata_filter.items():
            if memory_metadata.get(key) != value:
                return False
                
        return True 