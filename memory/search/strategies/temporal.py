"""Temporal-based search strategy for the agent memory search model."""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import math

from memory.search.strategies.base import SearchStrategy
from memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)


class TemporalSearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on temporal attributes.
    
    This strategy searches for memories based on time-related attributes such as
    creation time, last accessed time, or time references within the memory content.
    
    Attributes:
        memory_system: The memory system instance
    """
    
    def __init__(
        self,
        memory_system: AgentMemorySystem,
    ):
        """Initialize the temporal search strategy.
        
        Args:
            memory_system: The memory system instance
        """
        self.memory_system = memory_system
    
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
        query: Union[str, Dict[str, Any], List[float], int],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        start_time: Optional[Union[datetime, str, int]] = None,
        end_time: Optional[Union[datetime, str, int]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        recency_weight: float = 1.0,
        step_weight: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        temporal_params = self._process_query(query, start_time, end_time, start_step, end_step)
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
                try:
                    if temporal_params.get("start_time_timestamp") is not None and temporal_params.get("end_time_timestamp") is not None:
                        tier_memories = agent.ltm_store.get_by_timerange(
                            start_time=temporal_params["start_time_timestamp"],
                            end_time=temporal_params["end_time_timestamp"],
                            limit=limit * 2
                        )
                    else:
                        tier_memories = agent.ltm_store.get_all(agent_id)
                except Exception as e:
                    logger.warning(f"Error retrieving LTM memories: {e}")
                    tier_memories = []
            filtered_memories = self._filter_memories(
                tier_memories,
                temporal_params,
                metadata_filter,
            )
            scored_memories = self._score_memories(
                filtered_memories,
                temporal_params,
                recency_weight,
                step_weight,
                current_tier,
            )
            results.extend(scored_memories)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("temporal_score", 0.0),
            reverse=True
        )
        return results[:limit]
    
    def _process_query(
        self,
        query: Union[str, Dict[str, Any], List[float], int, float],
        start_time: Optional[Union[datetime, str, int, float]],
        end_time: Optional[Union[datetime, str, int, float]],
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process the search query to extract temporal parameters.
        
        Args:
            query: Search query
            start_time: Optional start time
            end_time: Optional end time
            start_step: Optional start simulation step
            end_step: Optional end simulation step
            
        Returns:
            Dictionary of temporal parameters
        """
        params = {
            "start_time": None,
            "end_time": None,
            "start_time_timestamp": None,  # Raw integer timestamp
            "end_time_timestamp": None,    # Raw integer timestamp
            "reference_time": datetime.now(),
            "reference_timestamp": int(datetime.now().timestamp()),  # Raw timestamp
            "start_step": None,
            "end_step": None,
            "reference_step": None,
            "query_keys": [],  # Track keys in the query dict to identify query type
        }
        
        # First populate with explicit parameters as a baseline
        if start_time is not None:
            dt = self._parse_datetime(start_time)
            if dt:
                params["start_time"] = dt
                params["start_time_timestamp"] = int(dt.timestamp())
            elif isinstance(start_time, (int, float)):
                # If parsing failed but it's a number, use directly as timestamp
                params["start_time_timestamp"] = int(start_time)
        
        if end_time is not None:
            dt = self._parse_datetime(end_time)
            if dt:
                params["end_time"] = dt
                params["end_time_timestamp"] = int(dt.timestamp())
            elif isinstance(end_time, (int, float)):
                # If parsing failed but it's a number, use directly as timestamp
                params["end_time_timestamp"] = int(end_time)
        
        if start_step is not None:
            params["start_step"] = self._parse_int(start_step)
        
        if end_step is not None:
            params["end_step"] = self._parse_int(end_step)
        
        # Handle integer/float timestamp as a reference time
        if isinstance(query, (int, float)):
            # This is a timestamp - use it as reference time
            try:
                timestamp = int(query)
                params["reference_timestamp"] = timestamp
                try:
                    # Try to convert to datetime if possible
                    params["reference_time"] = datetime.fromtimestamp(timestamp)
                except (ValueError, OverflowError) as e:
                    logger.warning(f"Invalid timestamp for reference time: {e}")
                    # Keep the raw timestamp
                
                # If start_time and end_time aren't specified, set a default range
                # centered around the reference time
                if params["start_time_timestamp"] is None:
                    params["start_time_timestamp"] = timestamp - (60 * 60 * 24 * 7)  # 1 week before
                
                if params["end_time_timestamp"] is None:
                    params["end_time_timestamp"] = timestamp + (60 * 60 * 24 * 1)  # 1 day after
                
                logger.debug(f"Using timestamp reference: {timestamp}")
            except (ValueError, TypeError):
                logger.warning(f"Failed to parse query as timestamp: {query}")
        
        # Handle string timestamp
        elif isinstance(query, str):
            # Try to parse as timestamp first
            try:
                timestamp = int(float(query))
                params["reference_timestamp"] = timestamp
                try:
                    # Try to convert to datetime if possible
                    params["reference_time"] = datetime.fromtimestamp(timestamp)
                except (ValueError, OverflowError) as e:
                    logger.warning(f"Invalid timestamp for reference time: {e}")
                
                logger.debug(f"Using parsed string timestamp reference: {timestamp}")
            except (ValueError, TypeError):
                # If not a timestamp, try to parse as datetime
                dt = self._parse_datetime(query)
                if dt:
                    params["reference_time"] = dt
                    params["reference_timestamp"] = int(dt.timestamp())
                    logger.debug(f"Using parsed datetime reference: {dt}")
        
        # Handle dictionary queries
        elif isinstance(query, dict):
            # Track the keys in the query dictionary
            params["query_keys"] = list(query.keys())
            
            # Process start_time and end_time from the query (prioritize query parameters)
            if "start_time" in query:
                # Handle integer timestamp
                if isinstance(query["start_time"], (int, float)):
                    try:
                        # Store both the datetime and raw timestamp
                        timestamp = int(query["start_time"])
                        params["start_time_timestamp"] = timestamp
                        try:
                            params["start_time"] = datetime.fromtimestamp(timestamp)
                        except (ValueError, OverflowError):
                            pass  # Just use the raw timestamp
                        logger.debug(f"Using start_time from query as timestamp: {timestamp}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid start_time timestamp: {e}")
                else:
                    parsed_start = self._parse_datetime(query["start_time"])
                    if parsed_start:
                        params["start_time"] = parsed_start
                        params["start_time_timestamp"] = int(parsed_start.timestamp())
                        logger.debug(f"Using start_time from query as parsed datetime: {parsed_start}")
                
            if "end_time" in query:
                # Handle integer timestamp
                if isinstance(query["end_time"], (int, float)):
                    try:
                        # Store both the datetime and raw timestamp
                        timestamp = int(query["end_time"])
                        params["end_time_timestamp"] = timestamp
                        try:
                            params["end_time"] = datetime.fromtimestamp(timestamp)
                        except (ValueError, OverflowError):
                            pass  # Just use the raw timestamp
                        logger.debug(f"Using end_time from query as timestamp: {timestamp}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid end_time timestamp: {e}")
                else:
                    parsed_end = self._parse_datetime(query["end_time"])
                    if parsed_end:
                        params["end_time"] = parsed_end
                        params["end_time_timestamp"] = int(parsed_end.timestamp())
                        logger.debug(f"Using end_time from query as parsed datetime: {parsed_end}")
            
            # Handle timestamp for reference time
            if "timestamp" in query:
                if isinstance(query["timestamp"], (int, float)):
                    try:
                        # This is a reference timestamp
                        timestamp = int(query["timestamp"])
                        params["reference_timestamp"] = timestamp
                        try:
                            params["reference_time"] = datetime.fromtimestamp(timestamp)
                        except (ValueError, OverflowError):
                            pass  # Just use the raw timestamp
                        
                        # If start_time and end_time aren't specified, set a default range
                        if params["start_time_timestamp"] is None:
                            params["start_time_timestamp"] = timestamp - (60 * 60 * 24 * 7)  # 1 week before
                        
                        if params["end_time_timestamp"] is None:
                            params["end_time_timestamp"] = timestamp + (60 * 60 * 24 * 1)  # 1 day after
                        
                        logger.debug(f"Using timestamp reference from query: {timestamp}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid timestamp reference: {e}")
            
            # Handle datetime for reference time
            if "time" in query and isinstance(query["time"], (str, datetime)):
                dt = self._parse_datetime(query["time"])
                if dt:
                    params["reference_time"] = dt
                    params["reference_timestamp"] = int(dt.timestamp())
                    logger.debug(f"Using time reference from query: {dt}")
            
            # Handle step related parameters
            if "start_step" in query:
                params["start_step"] = self._parse_int(query["start_step"])
            
            if "end_step" in query:
                params["end_step"] = self._parse_int(query["end_step"])
            
            if "step" in query:
                params["reference_step"] = self._parse_int(query["step"])
        
        # Ensure raw timestamps are integers
        if params["start_time_timestamp"] is not None:
            params["start_time_timestamp"] = int(params["start_time_timestamp"])
        
        if params["end_time_timestamp"] is not None:
            params["end_time_timestamp"] = int(params["end_time_timestamp"])
            
        if params["reference_timestamp"] is not None:
            params["reference_timestamp"] = int(params["reference_timestamp"])
            
        logger.debug(f"Processed temporal parameters: {params}")
        return params
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse a value to int if possible.
        
        Args:
            value: Value to parse
            
        Returns:
            Parsed int or None if parsing failed
        """
        if value is None:
            return None
            
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse value as int: {value}")
            return None
    
    def _parse_datetime(self, dt_value: Union[datetime, str, int, float]) -> Optional[datetime]:
        """Parse a datetime value from various formats.
        
        Args:
            dt_value: Datetime value to parse
            
        Returns:
            Parsed datetime or None if parsing failed
        """
        if isinstance(dt_value, datetime):
            return dt_value
            
        # Handle integer or float timestamp
        if isinstance(dt_value, (int, float)):
            try:
                return datetime.fromtimestamp(dt_value)
            except (ValueError, OverflowError) as e:
                logger.warning(f"Failed to parse timestamp {dt_value}: {e}")
                return None
        
        if isinstance(dt_value, str):
            try:
                # Try to parse string as integer timestamp first
                try:
                    timestamp = int(dt_value)
                    return datetime.fromtimestamp(timestamp)
                except ValueError:
                    pass
                    
                # Try ISO format next with full timestamp
                try:
                    return datetime.fromisoformat(dt_value)
                except (ValueError, AttributeError):
                    # fromisoformat was added in Python 3.7, fall back to strptime for compatibility
                    pass
                
                # Try various datetime formats
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",  # Handle microseconds
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
                
                # All parsing attempts failed
                logger.warning(f"Failed to parse datetime string: {dt_value}")
                return None
                
            except Exception as e:
                logger.warning(f"Error parsing datetime string: {e}")
                return None
        
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
        start_timestamp = temporal_params.get("start_time_timestamp")
        end_timestamp = temporal_params.get("end_time_timestamp")
        start_step = temporal_params.get("start_step")
        end_step = temporal_params.get("end_step")
        is_dict_query = len(temporal_params.get("query_keys", [])) > 0
        
        # Debug the filters
        if start_timestamp is not None:
            logger.debug(f"Filtering with start_timestamp: {start_timestamp}")
        if end_timestamp is not None:
            logger.debug(f"Filtering with end_timestamp: {end_timestamp}")
        if start_step is not None:
            logger.debug(f"Filtering with start_step: {start_step}")
        if end_step is not None:
            logger.debug(f"Filtering with end_step: {end_step}")
        
        if not memories:
            return []
        
        filtered = []
        for memory in memories:
            # Check if memory matches the timestamp range
            timestamp = self._get_memory_timestamp(memory)
            if timestamp is not None:
                # Skip if outside timestamp range
                if start_timestamp is not None and timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and timestamp > end_timestamp:
                    continue
            
            # Check if memory matches the step range
            step = self._get_memory_step(memory)
            if step is not None:
                # Skip if outside step range
                if start_step is not None and step < start_step:
                    continue
                if end_step is not None and step > end_step:
                    continue
            
            # Check if memory matches the metadata filter
            if metadata_filter and not self._check_metadata_filter(memory, metadata_filter):
                continue
            
            # Memory passed all filters
            filtered.append(memory)
        
        # Debug the filtering results
        logger.debug(f"Filtered {len(memories)} memories to {len(filtered)} memories")
        return filtered
    
    def _check_metadata_filter(
        self, memory: Dict[str, Any], metadata_filter: Dict[str, Any]
    ) -> bool:
        """Check if memory matches a metadata filter.
        
        Args:
            memory: Memory to check
            metadata_filter: Metadata filter to apply
            
        Returns:
            True if memory matches filter, False otherwise
        """
        # For simplicity, check all top-level fields
        content = memory.get("content", {})
        if not isinstance(content, dict):
            return False
            
        metadata = memory.get("metadata", {})
        if not isinstance(metadata, dict):
            return False
        
        # Create a merged dictionary for checking
        check_dict = {
            **memory,
            "content": content,
            "metadata": metadata,
        }
        
        # Check each filter key/value
        for key, value in metadata_filter.items():
            if "." in key:
                # Handle nested paths like "content.metadata.tags"
                parts = key.split(".")
                current = check_dict
                for part in parts[:-1]:
                    if part in current and isinstance(current[part], dict):
                        current = current[part]
                    else:
                        return False
                
                last_part = parts[-1]
                if last_part not in current:
                    return False
                
                # Handle MongoDB-like operators in the value
                if isinstance(value, dict) and all(k.startswith("$") for k in value.keys()):
                    # Currently only support $gt, $lt, $gte, $lte, $eq, $ne
                    for op, op_value in value.items():
                        match op:
                            case "$gt":
                                if not current[last_part] > op_value:
                                    return False
                            case "$lt":
                                if not current[last_part] < op_value:
                                    return False
                            case "$gte":
                                if not current[last_part] >= op_value:
                                    return False
                            case "$lte":
                                if not current[last_part] <= op_value:
                                    return False
                            case "$eq":
                                if not current[last_part] == op_value:
                                    return False
                            case "$ne":
                                if not current[last_part] != op_value:
                                    return False
                            case _:
                                # Unsupported operator
                                logger.warning(f"Unsupported operator: {op}")
                                return False
                else:
                    # Direct value comparison
                    if current[last_part] != value:
                        return False
            else:
                # Handle top-level keys
                if key not in check_dict:
                    return False
                    
                # Handle MongoDB-like operators in the value
                if isinstance(value, dict) and all(k.startswith("$") for k in value.keys()):
                    # Currently only support $gt, $lt, $gte, $lte, $eq, $ne
                    for op, op_value in value.items():
                        match op:
                            case "$gt":
                                if not check_dict[key] > op_value:
                                    return False
                            case "$lt":
                                if not check_dict[key] < op_value:
                                    return False
                            case "$gte":
                                if not check_dict[key] >= op_value:
                                    return False
                            case "$lte":
                                if not check_dict[key] <= op_value:
                                    return False
                            case "$eq":
                                if not check_dict[key] == op_value:
                                    return False
                            case "$ne":
                                if not check_dict[key] != op_value:
                                    return False
                            case _:
                                # Unsupported operator
                                logger.warning(f"Unsupported operator: {op}")
                                return False
                else:
                    # Direct value comparison
                    if check_dict[key] != value:
                        return False
        
        # All filters matched
        return True
    
    def _get_memory_timestamp(self, memory: Dict[str, Any]) -> Optional[int]:
        """Extract the timestamp from a memory.
        
        Args:
            memory: Memory to extract timestamp from
            
        Returns:
            Timestamp as int or None if not found
        """
        # Check direct timestamp field
        if "timestamp" in memory:
            return self._parse_int(memory["timestamp"])
        
        # Check content timestamp
        content = memory.get("content", {})
        if isinstance(content, dict) and "timestamp" in content:
            return self._parse_int(content["timestamp"])
        
        # Check metadata timestamp
        metadata = memory.get("metadata", {})
        if isinstance(metadata, dict):
            if "creation_time" in metadata:
                return self._parse_int(metadata["creation_time"])
            if "timestamp" in metadata:
                return self._parse_int(metadata["timestamp"])
        
        return None
    
    def _get_memory_step(self, memory: Dict[str, Any]) -> Optional[int]:
        """Extract the simulation step from a memory.
        
        Args:
            memory: Memory to extract step from
            
        Returns:
            Simulation step or None if not found
        """
        # Try to find step in memory metadata
        metadata = memory.get("metadata", {})
        if "step" in metadata:
            return self._parse_int(metadata["step"])
        
        # Also check for step in the main memory object
        if "step" in memory:
            return self._parse_int(memory["step"])
            
        # Check for step number
        if "step_number" in memory:
            return self._parse_int(memory["step_number"])
        
        return None
    
    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        temporal_params: Dict[str, Any],
        recency_weight: float,
        step_weight: float,
        tier: str,
    ) -> List[Dict[str, Any]]:
        """Score memories based on temporal relevance.
        
        Args:
            memories: List of memories to score
            temporal_params: Temporal parameters for scoring
            recency_weight: Weight to apply to recency in scoring
            step_weight: Weight to apply to step closeness in scoring
            tier: Memory tier
            
        Returns:
            List of scored memories
        """
        reference_time = temporal_params.get("reference_time", datetime.now())
        reference_timestamp = temporal_params.get("reference_timestamp", int(datetime.now().timestamp()))
        now = datetime.now()
        reference_step = temporal_params.get("reference_step")
        
        for memory in memories:
            # Initialize score
            score = 0.5  # Default score
            
            # Get creation time from timestamp
            memory_timestamp = self._get_memory_timestamp(memory)
            
            # Get memory step
            memory_step = self._get_memory_step(memory)
            
            # Score based on step proximity if reference step is provided
            if memory_step is not None and reference_step is not None and step_weight > 0:
                # Calculate step difference
                step_diff = abs(reference_step - memory_step)
                # Normalize (closer to 0 means closer in step count)
                max_step_diff = 1000  # Assuming a reasonable max step difference
                normalized_step_diff = min(step_diff / max_step_diff, 1.0)
                step_score = 1.0 - normalized_step_diff
                
                # Apply step weight
                weighted_step_score = step_score * step_weight
                
                # Use step score as the primary score if step_weight is high
                if step_weight >= 1.0:
                    score = weighted_step_score
                else:
                    # Base score is weighted by steps
                    score = step_score
            
            # Score based on temporal distance from reference time
            if memory_timestamp is not None and reference_timestamp is not None:
                # Calculate time difference in seconds
                time_diff = abs(reference_timestamp - memory_timestamp)
                # Normalize (closer to 0 means closer in time)
                max_diff = 60 * 60 * 24 * 365  # One year in seconds
                normalized_diff = min(time_diff / max_diff, 1.0)
                time_score = 1.0 - normalized_diff
                
                # Apply recency weight
                weighted_time_score = time_score * recency_weight
                
                # If we don't have a step score, use time score as the base
                if memory_step is None or reference_step is None:
                    score = weighted_time_score
                # If step_weight is low, combine with step score
                elif step_weight < 1.0:
                    score = (score + weighted_time_score) / 2
            
            # Boost score for memories in STM (more recent)
            if tier == "stm":
                score *= 1.2
                score = min(score, 1.0)  # Cap at 1.0
            
            # Add the score to memory metadata for sorting
            if "metadata" not in memory:
                memory["metadata"] = {}
            memory["metadata"]["temporal_score"] = score
        
        return memories 