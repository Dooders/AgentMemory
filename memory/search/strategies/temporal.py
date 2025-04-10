"""Temporal-based search strategy for the agent memory search model."""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from memory.search.strategies.base import SearchStrategy
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)


class TemporalSearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on temporal attributes.
    
    This strategy searches for memories based on time-related attributes such as
    creation time, last accessed time, or time references within the memory content.
    
    Attributes:
        stm_store: Short-Term Memory store
        im_store: Intermediate Memory store
        ltm_store: Long-Term Memory store
    """
    
    def __init__(
        self,
        stm_store: RedisSTMStore,
        im_store: RedisIMStore,
        ltm_store: SQLiteLTMStore,
    ):
        """Initialize the temporal search strategy.
        
        Args:
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def name(self) -> str:
        """Return the name of the search strategy.
        
        Returns:
            String name of the strategy
        """
        return "temporal"
    
    def description(self) -> str:
        """Return a description of the search strategy.
        
        Returns:
            String description of the strategy
        """
        return "Searches for memories based on temporal attributes such as creation time"
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
        recency_weight: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories based on temporal attributes.
        
        Args:
            query: Search query (can be a date/time string or a dictionary with temporal filters)
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            start_time: Optional start time for range queries
            end_time: Optional end time for range queries
            recency_weight: Weight to apply to recency in scoring (higher values favor newer memories)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Initialize results
        results = []
        
        # Process query to extract temporal parameters
        temporal_params = self._process_query(query, start_time, end_time)
        
        # Process all tiers or only the specified one
        tiers_to_search = ["stm", "im", "ltm"] if tier is None else [tier]
        
        for current_tier in tiers_to_search:
            # Skip if tier is not supported
            if current_tier not in ["stm", "im", "ltm"]:
                logger.warning("Unsupported memory tier: %s", current_tier)
                continue
            
            # Get all memories for the agent in this tier
            tier_memories = []
            if current_tier == "stm":
                tier_memories = self.stm_store.get_all_for_agent(agent_id)
            elif current_tier == "im":
                tier_memories = self.im_store.get_all_for_agent(agent_id)
            else:  # ltm
                tier_memories = self.ltm_store.get_all_for_agent(agent_id)
            
            # Filter memories by time range and metadata
            filtered_memories = self._filter_memories(
                tier_memories,
                temporal_params,
                metadata_filter,
            )
            
            # Score memories based on temporal relevance
            scored_memories = self._score_memories(
                filtered_memories,
                temporal_params,
                recency_weight,
                current_tier,
            )
            
            # Add to results
            results.extend(scored_memories)
        
        # Sort by temporal score (descending)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("temporal_score", 0.0),
            reverse=True
        )
        
        # Limit final results
        return results[:limit]
    
    def _process_query(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        start_time: Optional[Union[datetime, str]],
        end_time: Optional[Union[datetime, str]],
    ) -> Dict[str, Any]:
        """Process the search query to extract temporal parameters.
        
        Args:
            query: Search query
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Dictionary of temporal parameters
        """
        params = {
            "start_time": None,
            "end_time": None,
            "reference_time": datetime.now(),
        }
        
        # Handle string queries
        if isinstance(query, str):
            # Try to parse as a date/time string
            try:
                # Check for common date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        params["reference_time"] = datetime.strptime(query, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                # Use current time if parsing fails
                pass
        
        # Handle dictionary queries
        elif isinstance(query, dict):
            if "start_time" in query:
                params["start_time"] = self._parse_datetime(query["start_time"])
            if "end_time" in query:
                params["end_time"] = self._parse_datetime(query["end_time"])
            if "reference_time" in query:
                params["reference_time"] = self._parse_datetime(query["reference_time"])
        
        # Override with explicitly provided parameters
        if start_time:
            params["start_time"] = self._parse_datetime(start_time)
        if end_time:
            params["end_time"] = self._parse_datetime(end_time)
            
        return params
    
    def _parse_datetime(self, dt_value: Union[datetime, str]) -> Optional[datetime]:
        """Parse a datetime value from various formats.
        
        Args:
            dt_value: Datetime value to parse
            
        Returns:
            Parsed datetime or None if parsing failed
        """
        if isinstance(dt_value, datetime):
            return dt_value
        
        if isinstance(dt_value, str):
            try:
                # Try various datetime formats
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%m/%d/%Y",
                ]:
                    try:
                        return datetime.strptime(dt_value, fmt)
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning("Failed to parse datetime: %s", e)
        
        return None
    
    def _filter_memories(
        self,
        memories: List[Dict[str, Any]],
        temporal_params: Dict[str, Any],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter memories based on temporal parameters and metadata.
        
        Args:
            memories: List of memories to filter
            temporal_params: Temporal parameters to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            Filtered list of memories
        """
        start_time = temporal_params.get("start_time")
        end_time = temporal_params.get("end_time")
        
        filtered = []
        for memory in memories:
            # Get creation time
            created_at = memory.get("created_at")
            if created_at and isinstance(created_at, str):
                try:
                    created_at = self._parse_datetime(created_at)
                except Exception:
                    created_at = None
            
            # Apply time range filter
            if start_time and created_at and created_at < start_time:
                continue
            if end_time and created_at and created_at > end_time:
                continue
            
            # Apply metadata filter
            if metadata_filter:
                memory_metadata = memory.get("metadata", {})
                if not all(memory_metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            filtered.append(memory)
        
        return filtered
    
    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        temporal_params: Dict[str, Any],
        recency_weight: float,
        tier: str,
    ) -> List[Dict[str, Any]]:
        """Score memories based on temporal relevance.
        
        Args:
            memories: List of memories to score
            temporal_params: Temporal parameters for scoring
            recency_weight: Weight to apply to recency in scoring
            tier: Memory tier
            
        Returns:
            List of scored memories
        """
        reference_time = temporal_params.get("reference_time", datetime.now())
        
        for memory in memories:
            # Initialize score
            score = 0.0
            
            # Get creation time
            created_at = memory.get("created_at")
            if created_at and isinstance(created_at, str):
                try:
                    created_at = self._parse_datetime(created_at)
                except Exception:
                    created_at = None
            
            # Score based on temporal distance from reference time
            if created_at and reference_time:
                # Calculate time difference in seconds
                time_diff = abs((reference_time - created_at).total_seconds())
                # Normalize to 0-1 scale (closer to 0 means closer in time)
                # Use a logarithmic scale to handle large time differences
                max_diff = 60 * 60 * 24 * 365  # One year in seconds
                normalized_diff = min(time_diff / max_diff, 1.0)
                time_score = 1.0 - normalized_diff
                score += time_score
            
            # Add recency bonus if applicable
            if created_at and recency_weight > 0:
                # Calculate recency relative to now
                now = datetime.now()
                recency_diff = abs((now - created_at).total_seconds())
                max_recency_diff = 60 * 60 * 24 * 30  # One month in seconds
                normalized_recency = min(recency_diff / max_recency_diff, 1.0)
                recency_score = (1.0 - normalized_recency) * recency_weight
                score += recency_score
            
            # Attach score and tier information
            if "metadata" not in memory:
                memory["metadata"] = {}
            memory["metadata"]["temporal_score"] = score
            memory["metadata"]["memory_tier"] = tier
        
        return memories 