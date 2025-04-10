"""Search Strategies for Agent Memory Retrieval

This package provides a collection of search strategies that implement different approaches
for retrieving agent memories. Each strategy specializes in a particular type of memory
search and can be used individually or combined for more sophisticated retrieval.

Available strategies:

1. SimilaritySearchStrategy: Finds memories based on semantic similarity using vector
   embeddings. This strategy excels at retrieving contextually relevant information
   without requiring exact keyword matches. It uses vector embeddings to represent 
   the semantic meaning of memories and queries.

2. TemporalSearchStrategy: Retrieves memories based on time-related attributes such as
   creation time. This strategy enables time-range queries, recency-based scoring, and
   other temporal patterns. It's useful for finding memories from specific time periods
   or recent experiences.

3. AttributeSearchStrategy: Searches for memories using content and metadata attributes.
   This strategy supports exact matches, substring searches, and regex pattern matching
   across various fields. It's ideal for precise filtering based on known attributes.

4. CombinedSearchStrategy: Integrates results from multiple search strategies with
   configurable weights. This meta-strategy enables more sophisticated searches that
   blend semantic similarity, temporal relevance, and attribute matching.

All strategies implement the common SearchStrategy interface, making them interchangeable
within the SearchModel. This modular architecture allows for easy extension with new
search approaches while maintaining a consistent API.

Usage examples for each strategy can be found in their respective module docstrings.
"""

from memory.search.strategies.base import SearchStrategy
from memory.search.strategies.similarity import SimilaritySearchStrategy
from memory.search.strategies.temporal import TemporalSearchStrategy
from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.search.strategies.combined import CombinedSearchStrategy

__all__ = [
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "TemporalSearchStrategy",
    "AttributeSearchStrategy",
    "CombinedSearchStrategy",
] 