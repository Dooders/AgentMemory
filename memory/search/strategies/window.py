from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

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
        return "window"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Retrieves memories from a specific time window (e.g., last N minutes)"

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
        Search for memories within a specified time window.

        Args:
            query: Dict containing search parameters:
                - start_time: ISO datetime string for the start of time window
                - end_time: ISO datetime string for the end of time window
                - last_minutes: Number of minutes to look back
                - last_hours: Number of hours to look back
                - last_days: Number of days to look back
                - timestamp_field: Custom field name to use for timestamp (default: "metadata.timestamp")
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of memories to return
            metadata_filter: Additional filters to apply to the results
            tier: Memory tier to search in (stm, im, ltm, or None for all)
            **kwargs: Additional parameters

        Returns:
            List of memory entries from the specified time window

        Raises:
            ValueError: If time parameters are invalid or conflicting
        """
        # Get timestamp field to use
        timestamp_field = query.get("timestamp_field", "metadata.timestamp")

        # Detect conflicting time parameters
        time_params = sum(
            1 for p in ["last_minutes", "last_hours", "last_days"] if p in query
        )
        has_range = "start_time" in query and "end_time" in query

        if time_params > 1:
            raise ValueError(
                "Can only specify one of: last_minutes, last_hours, last_days"
            )

        if time_params > 0 and has_range:
            raise ValueError("Cannot specify both time range and last_X parameters")

        # Calculate start and end times based on query parameters
        now = datetime.now()

        # Flag to track if we should use inclusive or exclusive start time comparison
        inclusive_start = False

        if "last_minutes" in query:
            minutes = query["last_minutes"]
            if minutes <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - timedelta(minutes=minutes)
            end_time = now
        elif "last_hours" in query:
            hours = query["last_hours"]
            if hours <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - timedelta(hours=hours)
            end_time = now
        elif "last_days" in query:
            days = query["last_days"]
            if days <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - timedelta(days=days)
            end_time = now
        elif has_range:
            try:
                start_time = self._parse_datetime(query["start_time"])
                end_time = self._parse_datetime(query["end_time"])
                # When explicit time range is provided, use inclusive start
                inclusive_start = True
            except ValueError:
                raise ValueError(f"Invalid time format in query")
        else:
            # Default to last 30 minutes
            start_time = now - timedelta(minutes=30)
            end_time = now

        # Get stores for the specified tier
        stores = self._get_stores_for_tier(tier)

        # Retrieve memories from the time window
        results = []
        for store in stores:
            try:
                # First try 'list' method since that's what tests expect
                if hasattr(store, "list"):
                    memories = store.list(agent_id)
                # Fall back to other methods if needed
                elif hasattr(store, "get_all"):
                    memories = store.get_all(agent_id)
                elif hasattr(store, "find_all"):
                    memories = store.find_all(agent_id)
                elif hasattr(store, "retrieve_all"):
                    memories = store.retrieve_all(agent_id)
                elif hasattr(store, "get_memories"):
                    memories = store.get_memories(agent_id)
                else:
                    raise AttributeError(
                        f"Store {type(store).__name__} has no method to list all memories"
                    )
            except Exception as e:
                # Log error and continue with next store
                print(
                    f"Error retrieving memories from {type(store).__name__}: {str(e)}"
                )
                continue

            for memory in memories:
                # Skip if metadata filter doesn't match
                if metadata_filter and not self._matches_metadata_filters(
                    memory, metadata_filter
                ):
                    continue

                # Extract timestamp based on timestamp_field
                timestamp_str = None
                if timestamp_field == "metadata.timestamp":
                    timestamp_str = memory.get("metadata", {}).get("timestamp")
                else:
                    parts = timestamp_field.split(".")
                    obj = memory
                    for part in parts:
                        if isinstance(obj, dict):
                            obj = obj.get(part, {})
                        else:
                            obj = None
                            break
                    timestamp_str = obj

                # Skip items with missing timestamp
                if not timestamp_str:
                    continue

                # Compare timestamps - different behavior based on query type
                try:
                    # Parse the timestamp to datetime for accurate comparison
                    memory_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )

                    # Check if the memory is within the time window
                    if inclusive_start:
                        # Include memories exactly at the start time (for explicit time ranges)
                        if start_time <= memory_time <= end_time:
                            results.append(memory)
                    else:
                        # Exclude memories exactly at the start time (for last_X queries)
                        if start_time < memory_time <= end_time:
                            results.append(memory)
                except (ValueError, TypeError):
                    # Skip items with invalid timestamp format
                    continue

        # Sort by timestamp (most recent first)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True
        )

        return results[:limit]

    def _parse_datetime(self, dt_str):
        """Parse a datetime string to a datetime object."""
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid datetime format: {dt_str}")

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
