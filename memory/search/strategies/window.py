from typing import Dict, List, Any, Optional, Union
import time
from .base import SearchStrategy


class TimeWindowStrategy(SearchStrategy):
    """
    Strategy for retrieving memories within a specific time window,
    such as the last N minutes or between specific timestamps.
    """
    
    def __init__(self, stm_store, im_store, ltm_store):
        """
        Initialize the time window strategy.
        
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
        return "time_window"
    
    def description(self) -> str:
        """Return the description of the strategy."""
        return "Retrieves memories from a specific time window (e.g., last N minutes)"
    
    def search(
        self,
        query: Union[int, Dict[str, Any]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for memories within a specified time window.
        
        Args:
            query: Either minutes as an integer or a dict containing search parameters
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters:
                - memory_type: Type of memory to filter by
                - sort_order: "asc" or "desc" for timestamp sorting (default: "desc")
                
        Returns:
            List of memory entries from the specified time window
        """
        # Extract parameters
        if isinstance(query, int):
            minutes = query
            start_time = None
            end_time = None
        else:
            minutes = query.get("minutes")
            start_time = query.get("start_time")
            end_time = query.get("end_time")
        
        memory_type = kwargs.get("memory_type")
        sort_order = kwargs.get("sort_order", "desc")
        
        # Calculate time thresholds
        current_time = int(time.time())
        
        if minutes is not None:
            time_threshold = current_time - (minutes * 60)
            start_timestamp = time_threshold
            end_timestamp = current_time
        elif start_time and end_time:
            # Convert datetime strings to timestamps if needed
            if isinstance(start_time, str):
                start_timestamp = self._convert_datetime_to_timestamp(start_time)
            else:
                start_timestamp = start_time
                
            if isinstance(end_time, str):
                end_timestamp = self._convert_datetime_to_timestamp(end_time)
            else:
                end_timestamp = end_time
        else:
            # Default to last 30 minutes if no time parameters specified
            start_timestamp = current_time - (30 * 60)
            end_timestamp = current_time
        
        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)
        
        # Retrieve memories from the time window
        results = []
        for store in stores:
            # Try to use optimized range query if available
            if hasattr(store, "get_range_by_time"):
                window_memories = store.get_range_by_time(
                    agent_id, 
                    start_timestamp, 
                    end_timestamp
                )
                results.extend(window_memories)
            else:
                # Fallback to filtering all memories
                memories = store.get_all(agent_id)
                for memory in memories:
                    timestamp = memory.get("timestamp", 0)
                    if start_timestamp <= timestamp <= end_timestamp:
                        results.append(memory)
        
        # Filter by memory type if specified
        if memory_type:
            results = [
                memory for memory in results
                if memory.get("metadata", {}).get("memory_type") == memory_type
            ]
        
        # Apply metadata filtering
        if metadata_filter:
            results = [
                memory for memory in results
                if self._matches_metadata_filters(memory, metadata_filter)
            ]
        
        # Sort by timestamp
        reverse_sort = sort_order.lower() == "desc"
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=reverse_sort)
        
        return results[:limit]
    
    def _convert_datetime_to_timestamp(self, datetime_str):
        """Convert a datetime string to a Unix timestamp."""
        from datetime import datetime
        try:
            # Try ISO format first
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except ValueError:
            try:
                # Try common date format
                dt = datetime.strptime(datetime_str, "%Y-%m-%d")
                return int(dt.timestamp())
            except ValueError:
                # Return current time if parsing fails
                return int(time.time())
    
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