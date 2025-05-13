import datetime
from typing import Any, Dict, List, Optional, Union

from memory.core import AgentMemorySystem

from .base import SearchStrategy


class TimeWindowStrategy(SearchStrategy):
    """
    Strategy for retrieving memories within a specific time window,
    such as the last N minutes or between specific timestamps.
    """

    def __init__(self, memory_system: AgentMemorySystem):
        """
        Initialize the time window strategy.

        Args:
            memory_system: The memory system instance
        """
        self.memory_system = memory_system

    def name(self) -> str:
        """Return the name of the strategy."""
        return "window"

    def description(self) -> str:
        """Return the description of the strategy."""
        return "Retrieves memories from a specific time window (e.g., last N minutes)"

    def _get_stores_for_tier(self, agent, tier):
        if tier == "stm":
            return [agent.stm_store]
        elif tier == "im":
            return [agent.im_store]
        elif tier == "ltm":
            return [agent.ltm_store]
        else:
            return [agent.stm_store, agent.im_store, agent.ltm_store]

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
        agent = self.memory_system.get_memory_agent(agent_id)
        timestamp_field = query.get("timestamp_field", "metadata.timestamp")
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
        now = datetime.datetime.now()
        inclusive_start = False
        if "last_minutes" in query:
            minutes = query["last_minutes"]
            if minutes <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - datetime.timedelta(minutes=minutes)
            end_time = now
        elif "last_hours" in query:
            hours = query["last_hours"]
            if hours <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - datetime.timedelta(hours=hours)
            end_time = now
        elif "last_days" in query:
            days = query["last_days"]
            if days <= 0:
                raise ValueError("Time window must be positive")
            start_time = now - datetime.timedelta(days=days)
            end_time = now
        elif has_range:
            try:
                start_time = self._parse_datetime(query["start_time"])
                end_time = self._parse_datetime(query["end_time"])
                inclusive_start = True
            except ValueError:
                raise ValueError(f"Invalid time format in query")
        else:
            start_time = now - datetime.timedelta(minutes=30)
            end_time = now
        stores = self._get_stores_for_tier(agent, tier)
        results = []
        for store in stores:
            try:
                if hasattr(store, "get_all"):
                    memories = store.get_all(agent_id)
                elif hasattr(store, "list"):
                    memories = store.list(agent_id)
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
                print(
                    f"Error retrieving memories from {type(store).__name__}: {str(e)}"
                )
                continue
            for memory in memories:
                if metadata_filter and not self._matches_metadata_filters(
                    memory, metadata_filter
                ):
                    continue

                # Get timestamp from memory
                timestamp_str = self._extract_timestamp(memory, timestamp_field)
                if not timestamp_str:
                    continue
                try:
                    memory_time = datetime.datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if inclusive_start:
                        if start_time <= memory_time <= end_time:
                            results.append(memory)
                    else:
                        if start_time < memory_time <= end_time:
                            results.append(memory)
                except (ValueError, TypeError):
                    continue
        results.sort(
            key=lambda x: self._extract_timestamp(x, timestamp_field) or "", reverse=True
        )
        return results[:limit]

    def _parse_datetime(self, dt_str):
        """Parse a datetime string to a datetime object."""
        try:
            return datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid datetime format: {dt_str}")

    def _matches_metadata_filters(self, memory, metadata_filter):
        """Check if a memory matches the specified metadata filters."""
        if not metadata_filter:
            return True

        memory_metadata = memory.get("metadata", {})
        for key, value in metadata_filter.items():
            if memory_metadata.get(key) != value:
                return False

        return True

    def _extract_timestamp(self, memory, timestamp_field):
        """Extract the timestamp string from a memory using the given field."""
        if timestamp_field == "metadata.timestamp":
            return memory.get("metadata", {}).get("timestamp")
        parts = timestamp_field.split(".")
        obj = memory
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
                if obj is None:
                    break
            else:
                obj = None
                break
        return obj
