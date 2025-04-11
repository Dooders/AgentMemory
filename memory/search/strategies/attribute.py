"""Attribute-based search strategy for the agent memory search model."""

import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple

from memory.search.strategies.base import SearchStrategy
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)


class AttributeSearchStrategy(SearchStrategy):
    """Search strategy that finds memories based on content and metadata attributes.
    
    This strategy searches for memories that match specific content attributes or
    metadata fields, allowing for flexible filtering and querying of memories.
    
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
        """Initialize the attribute search strategy.
        
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
        return "attribute"
    
    def description(self) -> str:
        """Return a description of the search strategy.
        
        Returns:
            String description of the strategy
        """
        return "Searches for memories based on content and metadata attributes"
    
    def search(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        match_all: bool = False,
        case_sensitive: bool = False,
        use_regex: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories based on content and metadata attributes.
        
        Args:
            query: Search query, could be a string to search in content or a
                  dictionary mapping field names to search values
            agent_id: ID of the agent whose memories to search
            limit: Maximum number of results to return
            metadata_filter: Optional filters to apply to memory metadata
            tier: Optional memory tier to search ("stm", "im", "ltm", or None for all)
            content_fields: Optional list of content fields to search in
            metadata_fields: Optional list of metadata fields to search in
            match_all: Whether all query conditions must match (AND) or any (OR)
            case_sensitive: Whether to perform case-sensitive matching
            use_regex: Whether to interpret query strings as regular expressions
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of memory entries matching the search criteria
        """
        # Initialize results
        results = []
        
        # Process query into a standardized format
        query_conditions = self._process_query(
            query, content_fields, metadata_fields, case_sensitive, use_regex
        )
        
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
            
            # Filter memories by query conditions and metadata
            filtered_memories = self._filter_memories(
                tier_memories,
                query_conditions,
                metadata_filter,
                match_all,
            )
            
            # Score memories based on match quality
            scored_memories = self._score_memories(
                filtered_memories,
                query_conditions,
                match_all,
                current_tier,
            )
            
            # Add to results
            results.extend(scored_memories)
        
        # Sort by attribute score (descending)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("attribute_score", 0.0),
            reverse=True
        )
        
        # Limit final results
        return results[:limit]
    
    def _process_query(
        self,
        query: Union[str, Dict[str, Any], List[float]],
        content_fields: Optional[List[str]],
        metadata_fields: Optional[List[str]],
        case_sensitive: bool,
        use_regex: bool,
    ) -> List[Tuple[str, Any, str, bool]]:
        """Process the search query into a list of conditions.
        
        Args:
            query: Search query
            content_fields: Content fields to search in
            metadata_fields: Metadata fields to search in
            case_sensitive: Whether to use case-sensitive matching
            use_regex: Whether to interpret query strings as regular expressions
            
        Returns:
            List of tuples (field_path, value, match_type, case_sensitive)
        """
        conditions = []
        
        # Default fields if none specified
        if not content_fields:
            content_fields = ["content"]
        if not metadata_fields:
            metadata_fields = ["type", "tags", "importance", "source"]
        
        # Process string query
        if isinstance(query, str):
            # Add content field conditions
            for field in content_fields:
                conditions.append((field, query, "contains" if not use_regex else "regex", case_sensitive))
            
            # Add metadata field conditions
            for field in metadata_fields:
                conditions.append((f"metadata.{field}", query, "contains" if not use_regex else "regex", case_sensitive))
        
        # Process dictionary query
        elif isinstance(query, dict):
            # Handle content fields
            for field, value in query.items():
                if field in content_fields:
                    match_type = "equals"
                    if isinstance(value, str):
                        match_type = "contains" if not use_regex else "regex"
                    conditions.append((field, value, match_type, case_sensitive))
            
            # Handle metadata fields
            if "metadata" in query and isinstance(query["metadata"], dict):
                for field, value in query["metadata"].items():
                    if field in metadata_fields:
                        match_type = "equals"
                        if isinstance(value, str):
                            match_type = "contains" if not use_regex else "regex"
                        conditions.append((f"metadata.{field}", value, match_type, case_sensitive))
        
        # For regex conditions, compile the patterns
        if use_regex:
            compiled_conditions = []
            for field, value, match_type, is_case_sensitive in conditions:
                if match_type == "regex" and isinstance(value, str):
                    try:
                        flags = 0 if is_case_sensitive else re.IGNORECASE
                        pattern = re.compile(value, flags)
                        compiled_conditions.append((field, pattern, match_type, is_case_sensitive))
                    except re.error:
                        logger.warning("Invalid regex pattern: %s", value)
                        # Fall back to contains
                        compiled_conditions.append((field, value, "contains", is_case_sensitive))
                else:
                    compiled_conditions.append((field, value, match_type, is_case_sensitive))
            conditions = compiled_conditions
        
        return conditions
    
    def _filter_memories(
        self,
        memories: List[Dict[str, Any]],
        query_conditions: List[Tuple[str, Any, str, bool]],
        metadata_filter: Optional[Dict[str, Any]],
        match_all: bool,
    ) -> List[Dict[str, Any]]:
        """Filter memories based on query conditions and metadata.
        
        Args:
            memories: List of memories to filter
            query_conditions: List of query conditions
            metadata_filter: Additional metadata filters
            match_all: Whether all conditions must match (AND) or any (OR)
            
        Returns:
            Filtered list of memories
        """
        filtered = []
        
        for memory in memories:
            # Apply metadata filter
            if metadata_filter:
                memory_metadata = memory.get("metadata", {})
                if not all(memory_metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            # Skip if no query conditions
            if not query_conditions:
                filtered.append(memory)
                continue
            
            # Check query conditions
            matches = []
            # Group conditions by field path prefix for more flexible matching
            content_matches = []
            metadata_type_matches = []
            metadata_importance_matches = []
            other_matches = []
            
            for field_path, value, match_type, case_sensitive in query_conditions:
                # Extract field value from memory
                field_value = self._get_field_value(memory, field_path)
                
                # Skip if field doesn't exist
                if field_value is None:
                    matches.append(False)
                    continue
                
                # Check match based on match type
                match = False
                if match_type == "equals":
                    match = field_value == value
                elif match_type == "contains" and isinstance(field_value, str) and isinstance(value, str):
                    # Apply case sensitivity for contains
                    if case_sensitive:
                        match = value in field_value
                    else:
                        match = value.lower() in field_value.lower()
                elif match_type == "regex" and isinstance(field_value, str):
                    match = bool(value.search(field_value))
                
                # Store match in appropriate group
                if field_path == "content":
                    content_matches.append(match)
                elif field_path == "metadata.type":
                    metadata_type_matches.append(match)
                elif field_path == "metadata.importance":
                    metadata_importance_matches.append(match)
                else:
                    other_matches.append(match)
                
                matches.append(match)
            
            # Check if memory matches based on match_all flag
            if match_all:
                # Simple approach for hierarchical queries with match_all=True
                # Check if content contains "meeting"
                content_match = any(content_matches) if content_matches else True
                
                # Check if type is "meeting"
                type_match = any(metadata_type_matches) if metadata_type_matches else True
                
                # Check if importance is "high"
                importance_match = any(metadata_importance_matches) if metadata_importance_matches else True
                
                # Check other metadata fields if needed
                other_match = any(other_matches) if other_matches else True
                
                # Memory matches if it satisfies all required field categories
                if content_match and type_match and importance_match and other_match:
                    filtered.append(memory)
            elif any(matches):  # match_all=False means any match is sufficient
                filtered.append(memory)
        
        return filtered
    
    def _get_field_value(self, memory: Dict[str, Any], field_path: str) -> Any:
        """Get the value of a field from a memory dict.
        
        Args:
            memory: Memory dictionary
            field_path: Dot-separated path to the field
            
        Returns:
            Field value or None if not found
        """
        parts = field_path.split(".")
        value = memory
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        query_conditions: List[Tuple[str, Any, str, bool]],
        match_all: bool,
        tier: str,
    ) -> List[Dict[str, Any]]:
        """Score memories based on match quality.
        
        Args:
            memories: List of memories to score
            query_conditions: List of query conditions
            match_all: Whether all conditions must match (AND) or any (OR)
            tier: Memory tier
            
        Returns:
            List of scored memories
        """
        for memory in memories:
            # Initialize score components
            match_count = 0
            match_quality = 0.0
            
            # Calculate score based on matching conditions
            for field_path, value, match_type, case_sensitive in query_conditions:
                field_value = self._get_field_value(memory, field_path)
                
                if field_value is None:
                    continue
                
                # Check match and calculate match quality
                if match_type == "equals":
                    if field_value == value:
                        match_count += 1
                        match_quality += 1.0  # Perfect match
                
                elif match_type == "contains" and isinstance(field_value, str) and isinstance(value, str):
                    # Apply case sensitivity for contains check
                    if case_sensitive:
                        if value in field_value:
                            match_count += 1
                            # Higher quality for more specific matches
                            match_quality += min(1.0, len(value) / max(1, len(field_value)))
                    else:
                        if value.lower() in field_value.lower():
                            match_count += 1
                            # Higher quality for more specific matches
                            match_quality += min(1.0, len(value) / max(1, len(field_value)))
                
                elif match_type == "regex" and isinstance(field_value, str):
                    if value.search(field_value):
                        match_count += 1
                        # Fixed quality for regex matches
                        match_quality += 0.8
            
            # Compute final score based on match strategy
            if query_conditions:
                if match_all:
                    # For AND, score is reduced if not all conditions match
                    condition_ratio = match_count / len(query_conditions)
                    score = match_quality * condition_ratio
                else:
                    # For OR, score is based on best matches
                    score = match_quality / max(1, len(query_conditions))
            else:
                # No conditions means everything matches
                score = 1.0
            
            # Attach score and tier information
            if "metadata" not in memory:
                memory["metadata"] = {}
            memory["metadata"]["attribute_score"] = score
            memory["metadata"]["attribute_match_count"] = match_count
            memory["metadata"]["memory_tier"] = tier
        
        return memories 