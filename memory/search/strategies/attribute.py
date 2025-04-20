"""Attribute-based search strategy for the agent memory search model."""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
        scoring_method: Method used for scoring matches (default "length_ratio")
        _pattern_cache: Cache for compiled regex patterns
    """

    def __init__(
        self,
        stm_store: RedisSTMStore,
        im_store: RedisIMStore,
        ltm_store: SQLiteLTMStore,
        scoring_method: str = "length_ratio",
    ):
        """Initialize the attribute search strategy.

        Args:
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
            scoring_method: Method used for scoring matches
                Options:
                - "length_ratio": Score based on ratio of query length to field length
                - "term_frequency": Score based on term frequency
                - "bm25": Score based on BM25 ranking algorithm
                - "binary": Simple binary scoring (1.0 for match, 0.0 for no match)
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
        self.scoring_method = scoring_method
        # Initialize pattern cache dictionary: (pattern_str, is_case_sensitive) -> compiled_pattern
        self._pattern_cache = {}

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

    def clear_pattern_cache(self):
        """Clear the regex pattern cache to free memory."""
        self._pattern_cache.clear()
        logger.debug("Cleared regex pattern cache")

    def precompile_patterns(self, patterns: List[Tuple[str, bool]]):
        """Precompile and cache a list of regex patterns for better performance.

        This method is useful for precompiling commonly used patterns before
        performing a batch of searches, reducing the overhead of on-the-fly compilation.

        Args:
            patterns: List of (pattern_str, is_case_sensitive) tuples to precompile

        Returns:
            Number of successfully compiled patterns
        """
        success_count = 0
        for pattern_str, is_case_sensitive in patterns:
            if self.get_compiled_pattern(pattern_str, is_case_sensitive) is not None:
                success_count += 1

        logger.debug(f"Precompiled {success_count}/{len(patterns)} regex patterns")
        return success_count

    def get_compiled_pattern(self, pattern_str: str, is_case_sensitive: bool):
        """Get a compiled regex pattern from cache or compile it if not cached.

        Args:
            pattern_str: The regex pattern string
            is_case_sensitive: Whether the pattern is case sensitive

        Returns:
            Compiled regex pattern or None if invalid
        """
        cache_key = (pattern_str, is_case_sensitive)

        # Return from cache if available (including cached None values for invalid patterns)
        if cache_key in self._pattern_cache:
            if self._pattern_cache[cache_key] is None:
                logger.debug(f"Using cached invalid pattern result for: {pattern_str}")
            else:
                logger.debug(f"Using cached regex pattern for: {pattern_str}")
            return self._pattern_cache[cache_key]

        # Compile new pattern
        try:
            flags = 0 if is_case_sensitive else re.IGNORECASE
            compiled_pattern = re.compile(pattern_str, flags)
            # Store in cache
            self._pattern_cache[cache_key] = compiled_pattern
            logger.debug(f"Compiled and cached new regex pattern for: {pattern_str}")
            return compiled_pattern
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern_str}")
            # Cache the negative result to avoid repeated compilation attempts
            self._pattern_cache[cache_key] = None
            logger.debug(f"Cached invalid pattern result for: {pattern_str}")
            return None

    def search(
        self,
        query: Union[str, Dict[str, Any]],
        agent_id: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        match_all: bool = False,
        case_sensitive: bool = False,
        use_regex: bool = False,
        scoring_method: Optional[str] = None,
        **kwargs,
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
            scoring_method: Method used for scoring matches (override instance default)
                Options:
                - "length_ratio": Score based on ratio of query length to field length
                - "term_frequency": Score based on term frequency
                - "bm25": Score based on BM25 ranking algorithm
                - "binary": Simple binary scoring (1.0 for match, 0.0 for no match)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of memory entries matching the search criteria
        """
        # Initialize results
        results = []

        # Handle empty query case
        if (isinstance(query, str) and not query) or (
            isinstance(query, dict) and not query
        ):
            logger.debug("Empty query provided, returning empty results")
            return []

        # Process query into a standardized format
        logger.debug(
            f"Processing query: {query} with content_fields={content_fields}, metadata_fields={metadata_fields}"
        )

        query_conditions = self._process_query(
            query, content_fields, metadata_fields, case_sensitive, use_regex
        )

        logger.debug(f"Processed query conditions: {query_conditions}")

        # Process all tiers or only the specified one
        tiers_to_search = ["stm", "im", "ltm"] if tier is None else [tier]

        # Use provided scoring method or fallback to instance default
        current_scoring_method = scoring_method or self.scoring_method
        logger.debug(
            f"Using scoring method: {current_scoring_method} (instance default: {self.scoring_method})"
        )

        for current_tier in tiers_to_search:
            # Skip if tier is not supported
            if current_tier not in ["stm", "im", "ltm"]:
                logger.warning("Unsupported memory tier: %s", current_tier)
                continue

            # Get all memories for the agent in this tier
            tier_memories = []
            if current_tier == "stm":
                tier_memories = self.stm_store.get_all(agent_id)
            elif current_tier == "im":
                tier_memories = self.im_store.get_all(agent_id)
            else:  # ltm
                tier_memories = self.ltm_store.get_all()

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
                current_scoring_method,
            )

            # Add to results
            results.extend(scored_memories)

        # Sort by attribute score (descending)
        results.sort(
            key=lambda x: x.get("metadata", {}).get("attribute_score", 0.0),
            reverse=True,
        )

        # Limit final results
        return results[:limit]

    def _process_query(
        self,
        query: Union[str, Dict[str, Any]],
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
        logger.debug(
            f"Processing query: {query}, content_fields: {content_fields}, metadata_fields: {metadata_fields}"
        )

        # Handle empty or None query
        if (
            query is None
            or (isinstance(query, str) and not query)
            or (isinstance(query, dict) and not query)
        ):
            logger.debug("Empty or None query, returning empty conditions")
            return []

        # Default fields if none specified
        if not content_fields:
            content_fields = ["content.content"]
        if not metadata_fields:
            metadata_fields = [
                "content.metadata.type",
                "content.metadata.tags",
                "content.metadata.importance",
                "content.metadata.source",
            ]

        # Process string query
        if isinstance(query, str):
            # Add content field conditions
            for field in content_fields:
                conditions.append(
                    (
                        field,
                        query,
                        "contains" if not use_regex else "regex",
                        case_sensitive,
                    )
                )

            # Add metadata field conditions
            for field in metadata_fields:
                # Special handling for tags array
                if field.endswith(".tags") and isinstance(query, str):
                    # For tags, we need exact matching within array elements
                    conditions.append((field, query, "array_contains", case_sensitive))
                else:
                    conditions.append(
                        (
                            field,
                            query,
                            "contains" if not use_regex else "regex",
                            case_sensitive,
                        )
                    )

        # Process dictionary query
        elif isinstance(query, dict):
            # Process content field
            if "content" in query:
                for field in content_fields:
                    if (
                        field == "content.content"
                    ):  # Only map to the proper content field
                        match_type = "contains" if not use_regex else "regex"
                        if isinstance(query["content"], str):
                            conditions.append(
                                (field, query["content"], match_type, case_sensitive)
                            )
                        # Handle numeric or boolean content values by converting to string
                        elif isinstance(query["content"], (int, float, bool)):
                            conditions.append(
                                (
                                    field,
                                    str(query["content"]),
                                    match_type,
                                    case_sensitive,
                                )
                            )
                        logger.debug(
                            f"Added condition for content: {field}, {query['content']}, {match_type}"
                        )

            # Process metadata fields
            if "metadata" in query and isinstance(query["metadata"], dict):
                for meta_key, meta_value in query["metadata"].items():
                    # Find matching metadata field
                    for field in metadata_fields:
                        if field.endswith(f".{meta_key}"):
                            match_type = "equals"
                            if isinstance(meta_value, str):
                                match_type = "contains" if not use_regex else "regex"
                            # Handle non-string values in metadata
                            elif isinstance(meta_value, (int, float, bool)):
                                meta_value = str(meta_value)
                                match_type = "contains"
                            conditions.append(
                                (field, meta_value, match_type, case_sensitive)
                            )
                            logger.debug(
                                f"Added condition for metadata: {field}, {meta_value}, {match_type}"
                            )

        # Process numeric query (convert to string for searching)
        elif isinstance(query, (int, float)):
            str_query = str(query)
            for field in content_fields:
                conditions.append((field, str_query, "contains", case_sensitive))

            for field in metadata_fields:
                if field.endswith(".tags"):
                    conditions.append(
                        (field, str_query, "array_contains", case_sensitive)
                    )
                else:
                    conditions.append((field, str_query, "contains", case_sensitive))

        logger.debug(f"Final conditions: {conditions}")

        # For regex conditions, get compiled patterns from cache or compile new ones
        if use_regex:
            compiled_conditions = []
            for field, value, match_type, is_case_sensitive in conditions:
                if match_type == "regex" and isinstance(value, str):
                    # Get from cache or compile new pattern
                    pattern = self.get_compiled_pattern(value, is_case_sensitive)
                    if pattern:
                        compiled_conditions.append(
                            (field, pattern, match_type, is_case_sensitive)
                        )
                    else:
                        # Fall back to contains if regex compilation fails
                        compiled_conditions.append(
                            (field, value, "contains", is_case_sensitive)
                        )
                else:
                    compiled_conditions.append(
                        (field, value, match_type, is_case_sensitive)
                    )
            conditions = compiled_conditions

        return conditions

    def _filter_memories(
        self,
        memories: List[Dict[str, Any]],
        query_conditions: List[Tuple[str, Any, str, bool]],
        metadata_filter: Optional[Dict[str, Any]],
        match_all: bool,
    ) -> List[Dict[str, Any]]:
        """Filter memories based on query conditions and metadata filters.

        Args:
            memories: List of memories to filter
            query_conditions: List of query conditions
            metadata_filter: Optional filters to apply to memory metadata
            match_all: Whether all conditions must match (AND) or any (OR)

        Returns:
            List of filtered memories
        """
        # Normal filtering logic
        filtered_memories = []

        # Debug the input
        logger.debug(
            f"Filtering {len(memories)} memories with {len(query_conditions)} conditions"
        )
        for i, condition in enumerate(query_conditions):
            logger.debug(f"Condition {i}: {condition}")

        for memory in memories:
            try:
                # Make a deep copy of the memory to avoid modifying the original
                # This ensures we don't lose any existing metadata when filtering
                memory_copy = copy.deepcopy(memory)
                if "metadata" not in memory_copy and "metadata" in memory:
                    memory_copy["metadata"] = memory["metadata"].copy()

                # Debug memory info
                memory_id = memory_copy.get(
                    "memory_id", memory_copy.get("id", "unknown")
                )
                logger.debug(f"Checking memory: {memory_id}")

                # Handle different memory content structures
                memory_content = None
                if isinstance(memory_copy.get("content"), str):
                    memory_content = memory_copy.get("content")
                    logger.debug(f"Memory has string content: {memory_content[:50]}...")
                elif isinstance(
                    memory_copy.get("content"), dict
                ) and "content" in memory_copy.get("content", {}):
                    memory_content = memory_copy.get("content", {}).get("content", "")
                    logger.debug(f"Memory has dict content: {memory_content[:50]}...")
                else:
                    logger.debug(f"Memory has no recognizable content structure")

                # Log metadata if available
                if isinstance(
                    memory_copy.get("content"), dict
                ) and "metadata" in memory_copy.get("content", {}):
                    logger.debug(
                        f"Memory metadata: {memory_copy.get('content', {}).get('metadata', {})}"
                    )

                # Apply metadata filtering first if specified
                if metadata_filter:
                    try:
                        if not self._matches_metadata_filter(
                            memory_copy, metadata_filter
                        ):
                            logger.debug(
                                f"Memory {memory_id} filtered out by metadata filter"
                            )
                            continue
                    except Exception as e:
                        logger.warning(
                            f"Error in metadata filtering for memory {memory_id}: {e}"
                        )
                        # Skip this memory if metadata filtering fails
                        continue

                # Apply query conditions
                if not query_conditions:
                    # No conditions means everything matches
                    filtered_memories.append(memory_copy)
                    continue

                matches = []
                for i, (field_path, value, match_type, case_sensitive) in enumerate(
                    query_conditions
                ):
                    try:
                        field_value = self._get_field_value(memory_copy, field_path)

                        logger.debug(
                            f"Condition {i}: Field {field_path} = {field_value}, Value = {value}, Match type = {match_type}"
                        )

                        if field_value is None:
                            matches.append(False)
                            logger.debug(f"Condition {i}: No field value found")
                            continue

                        match_result = False

                        # Handle different field value types
                        if match_type == "equals":
                            # Convert values to strings for comparison if types differ
                            if type(field_value) != type(value):
                                try:
                                    # Try to convert both to strings for comparison
                                    str_field_value = str(field_value)
                                    str_value = str(value)
                                    match_result = str_field_value == str_value
                                except Exception as e:
                                    logger.warning(
                                        f"Type conversion error in equals matching: {e}"
                                    )
                                    match_result = False
                            else:
                                match_result = field_value == value

                        elif match_type == "contains":
                            try:
                                if isinstance(field_value, str) and isinstance(
                                    value, str
                                ):
                                    # Apply case sensitivity for contains check
                                    if case_sensitive:
                                        match_result = value in field_value
                                    else:
                                        match_result = (
                                            value.lower() in field_value.lower()
                                        )
                                elif isinstance(
                                    field_value, (int, float, bool)
                                ) and isinstance(value, str):
                                    # Convert numeric/boolean field value to string for matching
                                    str_field_value = str(field_value)
                                    if case_sensitive:
                                        match_result = value in str_field_value
                                    else:
                                        match_result = (
                                            value.lower() in str_field_value.lower()
                                        )
                                elif isinstance(field_value, str) and isinstance(
                                    value, (int, float, bool)
                                ):
                                    # Convert value to string for matching
                                    str_value = str(value)
                                    if case_sensitive:
                                        match_result = str_value in field_value
                                    else:
                                        match_result = (
                                            str_value.lower() in field_value.lower()
                                        )
                                elif isinstance(field_value, list):
                                    # Try to find the value in a list of items
                                    for item in field_value:
                                        try:
                                            str_item = str(item)
                                            str_val = str(value)
                                            if case_sensitive and str_val in str_item:
                                                match_result = True
                                                break
                                            elif (
                                                not case_sensitive
                                                and str_val.lower() in str_item.lower()
                                            ):
                                                match_result = True
                                                break
                                        except Exception:
                                            # Skip items that can't be converted to string
                                            continue
                            except Exception as e:
                                logger.warning(f"Error in contains matching: {e}")
                                match_result = False

                        elif match_type == "array_contains" and isinstance(
                            field_value, list
                        ):
                            try:
                                # Handle different types in arrays
                                for item in field_value:
                                    try:
                                        if isinstance(item, str) and isinstance(
                                            value, str
                                        ):
                                            # String comparison
                                            if case_sensitive and item == value:
                                                match_result = True
                                                break
                                            elif (
                                                not case_sensitive
                                                and item.lower() == value.lower()
                                            ):
                                                match_result = True
                                                break
                                        elif isinstance(item, (int, float, bool)):
                                            # Convert numeric/boolean to string
                                            str_item = str(item)
                                            str_value = str(value)
                                            if case_sensitive and str_item == str_value:
                                                match_result = True
                                                break
                                            elif (
                                                not case_sensitive
                                                and str_item.lower()
                                                == str_value.lower()
                                            ):
                                                match_result = True
                                                break
                                    except Exception as e:
                                        logger.debug(f"Error comparing array item: {e}")
                                        continue
                            except Exception as e:
                                logger.warning(f"Error in array_contains matching: {e}")
                                match_result = False

                        elif match_type == "regex":
                            try:
                                # Convert field_value to string if it's not already
                                str_field_value = (
                                    field_value
                                    if isinstance(field_value, str)
                                    else str(field_value)
                                )

                                # Apply regex matching
                                if isinstance(value, str):
                                    # Use pattern cache instead of compiling regex on the fly
                                    pattern = self.get_compiled_pattern(
                                        value, case_sensitive
                                    )
                                    match_result = (
                                        bool(pattern.search(str_field_value))
                                        if pattern
                                        else False
                                    )
                                else:
                                    # Assume already compiled pattern
                                    match_result = bool(value.search(str_field_value))
                            except (re.error, AttributeError, TypeError) as e:
                                logger.warning(
                                    f"Regex matching error: {e} for pattern: {value}"
                                )
                                match_result = False
                            except Exception as e:
                                logger.warning(
                                    f"Unexpected error in regex matching: {e}"
                                )
                                match_result = False

                        matches.append(match_result)
                        logger.debug(f"Condition {i}: Match result = {match_result}")

                    except Exception as e:
                        logger.warning(
                            f"Error processing condition {i} for memory {memory_id}: {e}"
                        )
                        matches.append(False)

                # Debug output for match_all
                logger.debug(
                    f"Memory {memory_id} matches: {matches} for conditions: {query_conditions}"
                )

                # Determine if memory should be included based on match strategy
                if not query_conditions:
                    # No conditions means everything matches (already handled above)
                    continue
                elif match_all and matches and all(matches):
                    filtered_memories.append(memory_copy)
                    logger.debug(f"Memory {memory_id} matched ALL conditions")
                elif not match_all and any(matches):
                    filtered_memories.append(memory_copy)
                    logger.debug(f"Memory {memory_id} matched SOME conditions")
                else:
                    logger.debug(f"Memory {memory_id} did not match conditions")

            except Exception as e:
                memory_id = memory.get("memory_id", memory.get("id", "unknown"))
                logger.warning(f"Error processing memory {memory_id}: {e}")
                continue

        logger.debug(f"Filtered memories count: {len(filtered_memories)}")
        return filtered_memories

    def _matches_metadata_filter(
        self, memory: Dict[str, Any], metadata_filter: Dict[str, Any]
    ) -> bool:
        """Check if a memory matches all metadata filters.

        Args:
            memory: Memory dictionary
            metadata_filter: Metadata filters to apply

        Returns:
            True if memory matches all filters
        """
        try:
            for field_path, filter_value in metadata_filter.items():
                field_value = self._get_field_value(memory, field_path)

                if field_value is None:
                    return False

                try:
                    # Handle type differences by string conversion if needed
                    if type(field_value) != type(filter_value):
                        str_field_value = str(field_value)
                        str_filter_value = str(filter_value)
                        if str_field_value != str_filter_value:
                            return False
                    elif field_value != filter_value:
                        return False
                except Exception as e:
                    logger.warning(f"Error comparing metadata values: {e}")
                    return False

            return True
        except Exception as e:
            memory_id = memory.get("memory_id", memory.get("id", "unknown"))
            logger.warning(f"Error in metadata filtering for memory {memory_id}: {e}")
            return False

    def _get_field_value(self, memory: Dict[str, Any], field_path: str) -> Any:
        """Get the value of a field from a memory dict.

        Args:
            memory: Memory dictionary
            field_path: Dot-separated path to the field

        Returns:
            Field value or None if not found
        """
        # Handle empty or None field path
        if not field_path:
            logger.warning("Empty field path provided")
            return None

        parts = field_path.split(".")
        value = memory

        # Debug the field path and initial memory ID
        memory_id = memory.get("memory_id", memory.get("id", "unknown"))
        logger.debug(
            f"Getting field value for path '{field_path}' from memory {memory_id}"
        )

        # Handle special case for content.content which might be directly in memory['content']
        if field_path == "content.content" and isinstance(memory.get("content"), str):
            return memory["content"]

        # Handle missing parts gracefully
        try:
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                    logger.debug(f"Found part '{part}', current value: {value}")
                elif isinstance(value, list) and part.isdigit():
                    # Handle array indexing if part is a digit
                    index = int(part)
                    if 0 <= index < len(value):
                        value = value[index]
                        logger.debug(
                            f"Found array index '{part}', current value: {value}"
                        )
                    else:
                        logger.debug(f"Array index '{part}' out of bounds for {value}")
                        return None
                else:
                    logger.debug(
                        f"Could not find part '{part}' in {type(value).__name__}"
                    )
                    return None
        except (TypeError, AttributeError, IndexError, ValueError) as e:
            logger.debug(
                f"Error accessing field '{field_path}' in memory {memory_id}: {e}"
            )
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error accessing field '{field_path}' in memory {memory_id}: {e}"
            )
            return None

        logger.debug(f"Final value for '{field_path}': {value}")
        return value

    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        query_conditions: List[Tuple[str, Any, str, bool]],
        match_all: bool,
        tier: str,
        scoring_method: str = "length_ratio",
    ) -> List[Dict[str, Any]]:
        """Score memories based on match quality.

        Args:
            memories: List of memories to score
            query_conditions: List of query conditions
            match_all: Whether all conditions must match (AND) or any (OR)
            tier: Memory tier
            scoring_method: Method used for scoring matches
                Options:
                - "length_ratio": Score based on ratio of query length to field length
                - "term_frequency": Score based on term frequency
                - "bm25": Score based on BM25 ranking algorithm
                - "binary": Simple binary scoring (1.0 for match, 0.0 for no match)

        Returns:
            List of scored memories
        """
        # Log the scoring method that will be used
        logger.debug(f"Scoring {len(memories)} memories using method: {scoring_method}")

        scored_memories = []
        for memory in memories:
            try:
                # Create a deep copy of memory to avoid modifying the original
                memory_copy = copy.deepcopy(memory)
                memory_id = memory_copy.get(
                    "memory_id", memory_copy.get("id", "unknown")
                )

                # Initialize score components
                match_count = 0
                match_quality = 0.0

                # Calculate score based on matching conditions
                for field_path, value, match_type, case_sensitive in query_conditions:
                    try:
                        field_value = self._get_field_value(memory_copy, field_path)

                        if field_value is None:
                            continue

                        # Check match and calculate match quality
                        if match_type == "equals":
                            if field_value == value:
                                match_count += 1
                                match_quality += 1.0  # Perfect match

                        elif match_type == "contains":
                            try:
                                # Ensure we have strings for comparison
                                str_field_value = (
                                    str(field_value)
                                    if not isinstance(field_value, str)
                                    else field_value
                                )
                                str_value = (
                                    str(value) if not isinstance(value, str) else value
                                )

                                # Apply case sensitivity for contains check
                                match_found = False
                                if case_sensitive:
                                    match_found = str_value in str_field_value
                                else:
                                    match_found = (
                                        str_value.lower() in str_field_value.lower()
                                    )

                                if match_found:
                                    match_count += 1
                                    # Calculate match quality based on selected scoring method
                                    if scoring_method == "length_ratio":
                                        # Original method: higher quality for more specific matches
                                        match_quality += min(
                                            1.0,
                                            len(str_value)
                                            / max(1, len(str_field_value)),
                                        )
                                    elif scoring_method == "term_frequency":
                                        # Term frequency: based on frequency of term in field
                                        if case_sensitive:
                                            term_count = str_field_value.count(
                                                str_value
                                            )
                                        else:
                                            term_count = str_field_value.lower().count(
                                                str_value.lower()
                                            )
                                        # Normalize by field length
                                        match_quality += min(
                                            1.0,
                                            term_count
                                            / max(1, len(str_field_value.split())),
                                        )
                                    elif scoring_method == "bm25":
                                        # Simplified BM25-inspired scoring
                                        # Constants for BM25 formula
                                        k1 = 1.2
                                        b = 0.75
                                        avg_field_len = (
                                            100  # Assume average field length
                                        )

                                        # Calculate term frequency
                                        if case_sensitive:
                                            term_count = str_field_value.count(
                                                str_value
                                            )
                                        else:
                                            term_count = str_field_value.lower().count(
                                                str_value.lower()
                                            )

                                        # Field length
                                        field_len = len(str_field_value.split())

                                        # BM25 formula (simplified)
                                        numerator = term_count * (k1 + 1)
                                        denominator = term_count + k1 * (
                                            1 - b + b * field_len / avg_field_len
                                        )
                                        bm25_score = numerator / max(0.1, denominator)

                                        # Normalize to 0-1 range
                                        match_quality += min(1.0, bm25_score / 10.0)
                                    else:  # binary or any unsupported method
                                        match_quality += 1.0  # Simple binary scoring
                            except Exception as e:
                                logger.warning(f"Error in contains scoring: {e}")
                                continue

                        elif match_type == "regex" and isinstance(field_value, str):
                            try:
                                if value.search(field_value):
                                    match_count += 1
                                    if scoring_method == "binary":
                                        match_quality += 1.0
                                    else:
                                        # Fixed quality for regex matches
                                        match_quality += 0.8
                            except Exception as e:
                                logger.warning(f"Error in regex scoring: {e}")
                                continue
                    except Exception as e:
                        logger.warning(
                            f"Error scoring condition {field_path} for memory {memory_id}: {e}"
                        )
                        continue

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

                # Ensure metadata exists
                if "metadata" not in memory_copy:
                    memory_copy["metadata"] = {}

                # Attach score and tier information
                memory_copy["metadata"]["attribute_score"] = score
                memory_copy["metadata"]["attribute_match_count"] = match_count
                memory_copy["metadata"]["memory_tier"] = tier

                # Explicitly set the scoring method
                memory_copy["metadata"]["scoring_method"] = scoring_method
                logger.debug(
                    f"Set scoring_method in memory metadata to: {scoring_method} for memory {memory_id}"
                )

                scored_memories.append(memory_copy)
            except Exception as e:
                memory_id = memory.get("memory_id", memory.get("id", "unknown"))
                logger.warning(f"Error scoring memory {memory_id}: {e}")
                continue

        logger.debug(f"Scored {len(memories)} memories using method: {scoring_method}")
        return scored_memories
