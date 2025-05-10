"""Query builder for attribute-based search strategy."""

from typing import Any, Dict, List, Optional, Tuple, Union


class AttributeQueryBuilder:
    """A builder class to simplify creating complex queries for AttributeSearchStrategy.
    
    Provides a fluent interface for constructing queries without needing to understand
    the underlying dictionary structure.
    """

    def __init__(self):
        """Initialize an empty query."""
        self._query = {}
        self._content_query = None
        self._metadata_query = {}
        self._match_all = False
        self._case_sensitive = False
        self._use_regex = False
        self._content_fields = None
        self._metadata_fields = None
        self._limit = 10
        self._metadata_filter = {}
        self._tier = None
        self._scoring_method = None

    def content(self, content_query: str):
        """Search for content matching the specified text.
        
        Args:
            content_query: The text to search for in content
            
        Returns:
            Self for method chaining
        """
        self._content_query = content_query
        return self

    def metadata(self, field: str, value: any):
        """Add a metadata field criteria to the search.
        
        Args:
            field: Metadata field name (e.g., "type", "importance")
            value: Value to match against
            
        Returns:
            Self for method chaining
        """
        self._metadata_query[field] = value
        return self
    
    def type(self, type_value: str):
        """Shorthand for setting the content type.
        
        Args:
            type_value: The content type to filter by (e.g., "meeting", "note")
            
        Returns:
            Self for method chaining
        """
        return self.metadata("type", type_value)
    
    def importance(self, importance_level: str):
        """Shorthand for setting importance level.
        
        Args:
            importance_level: Importance level (e.g., "high", "medium", "low")
            
        Returns:
            Self for method chaining
        """
        return self.metadata("importance", importance_level)
    
    def tag(self, tag_value: str):
        """Shorthand for searching by tag.
        
        Args:
            tag_value: Tag to search for
            
        Returns:
            Self for method chaining
        """
        return self.metadata("tags", tag_value)
    
    def match_all(self, match_all: bool = True):
        """Set whether all conditions must match (AND logic vs OR logic).
        
        Args:
            match_all: True for AND logic, False for OR logic
            
        Returns:
            Self for method chaining
        """
        self._match_all = match_all
        return self
    
    def case_sensitive(self, case_sensitive: bool = True):
        """Set whether searches should be case sensitive.
        
        Args:
            case_sensitive: True for case sensitive search
            
        Returns:
            Self for method chaining
        """
        self._case_sensitive = case_sensitive
        return self
    
    def use_regex(self, use_regex: bool = True):
        """Set whether to interpret query strings as regular expressions.
        
        Args:
            use_regex: True to enable regex pattern matching
            
        Returns:
            Self for method chaining
        """
        self._use_regex = use_regex
        return self
    
    def in_content_fields(self, *fields):
        """Specify which content fields to search in.
        
        Args:
            *fields: One or more content field paths
            
        Returns:
            Self for method chaining
        """
        self._content_fields = list(fields) if fields else None
        return self
    
    def in_metadata_fields(self, *fields):
        """Specify which metadata fields to search in.
        
        Args:
            *fields: One or more metadata field paths
            
        Returns:
            Self for method chaining
        """
        self._metadata_fields = list(fields) if fields else None
        return self
    
    def limit(self, limit: int):
        """Set maximum number of results to return.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            Self for method chaining
        """
        self._limit = limit
        return self
    
    def filter_metadata(self, field_path: str, value: any):
        """Add metadata filter criteria.
        
        Args:
            field_path: Full dot-notation path to metadata field
            value: Value to match exactly
            
        Returns:
            Self for method chaining
        """
        self._metadata_filter[field_path] = value
        return self
    
    def in_tier(self, tier: str):
        """Specify which memory tier to search in (stm, im, ltm).
        
        Args:
            tier: Memory tier name
            
        Returns:
            Self for method chaining
        """
        self._tier = tier
        return self
    
    def score_by(self, method: str):
        """Set the scoring method.
        
        Args:
            method: One of "length_ratio", "term_frequency", "bm25", or "binary"
            
        Returns:
            Self for method chaining
        """
        self._scoring_method = method
        return self
    
    def build(self):
        """Build the final query dictionary and search parameters.
        
        Returns:
            Tuple of (query, kwargs) ready to pass to AttributeSearchStrategy.search()
        """
        # Build the query
        query = {}
        
        if self._content_query is not None:
            if isinstance(self._content_query, (str, int, float, bool)):
                # If simple content query and no metadata, can just return the string
                if not self._metadata_query:
                    query = self._content_query
                else:
                    query["content"] = self._content_query
            else:
                query["content"] = self._content_query
                
        if self._metadata_query:
            if isinstance(query, dict):
                query["metadata"] = self._metadata_query
            else:
                # Convert to dict if it was a simple string
                content_value = query
                query = {"content": content_value, "metadata": self._metadata_query}
        
        # Build kwargs
        kwargs = {
            "limit": self._limit,
            "match_all": self._match_all,
            "case_sensitive": self._case_sensitive,
            "use_regex": self._use_regex,
        }
        
        if self._content_fields:
            kwargs["content_fields"] = self._content_fields
            
        if self._metadata_fields:
            kwargs["metadata_fields"] = self._metadata_fields
            
        if self._metadata_filter:
            kwargs["metadata_filter"] = self._metadata_filter
            
        if self._tier:
            kwargs["tier"] = self._tier
            
        if self._scoring_method:
            kwargs["scoring_method"] = self._scoring_method
            
        return query, kwargs
    
    def execute(self, strategy, agent_id: str):
        """Execute the query using the provided search strategy.
        
        Args:
            strategy: An instance of AttributeSearchStrategy
            agent_id: The agent ID to search for
            
        Returns:
            Search results from the strategy
        """
        query, kwargs = self.build()
        return strategy.search(query, agent_id, **kwargs) 