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

5. StepBasedSearchStrategy: Retrieves memories based on simulation step numbers rather
   than actual timestamps. This strategy is optimized for scenarios where the progression
   of an agent through simulation steps is more meaningful than real-world time. It
   enables step-range queries, proximity-based retrieval, and precise step matching.

6. NarrativeSequenceStrategy: Retrieves a sequence of memories surrounding a reference memory
   to form a contextual narrative. This strategy helps in understanding the context and
   sequence of events related to a specific memory.

7. ExampleMatchingStrategy: Finds memories that match a provided example pattern using
   semantic similarity. This approach allows searching for memories that are similar
   to a given example, even if they don't match exactly.

8. TimeWindowStrategy: Retrieves memories within a specific time window, such as the last N
   minutes or between specific timestamps. This strategy is useful for focusing on recent
   events or activities within a defined timeframe.

9. ContentPathStrategy: Searches memories based on content path values or patterns,
   providing precise access to nested content structures. This enables targeted searches
   through complex hierarchical memory structures.

10. ImportanceStrategy: Retrieves memories based on their importance score, allowing agents
    to focus on significant information. This strategy prioritizes memories that have been
    marked as important through scoring mechanisms.

11. CompoundQueryStrategy: Executes complex queries with multiple conditions and logical operators,
    enabling sophisticated memory filtering across various attributes. This strategy supports
    building complex search expressions with AND/OR logic.

All strategies implement the common SearchStrategy interface, making them interchangeable
within the SearchModel. This modular architecture allows for easy extension with new
search approaches while maintaining a consistent API.

Usage examples for each strategy can be found in their respective module docstrings.
"""

from .base import SearchStrategy
from .similarity import SimilaritySearchStrategy
from .temporal import TemporalSearchStrategy
from .attribute import AttributeSearchStrategy
from .step import StepBasedSearchStrategy
from .combined import CombinedSearchStrategy
from .sequence import NarrativeSequenceStrategy
from .match import ExampleMatchingStrategy
from .window import TimeWindowStrategy
from .path import ContentPathStrategy
from .importance import ImportanceStrategy
from .compound import CompoundQueryStrategy

# Export all strategy classes
__all__ = [
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "TemporalSearchStrategy",
    "AttributeSearchStrategy",
    "StepBasedSearchStrategy",
    "CombinedSearchStrategy",
    "NarrativeSequenceStrategy",
    "ExampleMatchingStrategy",
    "TimeWindowStrategy",
    "ContentPathStrategy",
    "ImportanceStrategy",
    "CompoundQueryStrategy",
]

# Registry of available search strategies
DEFAULT_STRATEGIES = {
    "similarity": SimilaritySearchStrategy,
    "temporal": TemporalSearchStrategy,
    "attribute": AttributeSearchStrategy,
    "step_based": StepBasedSearchStrategy,
    "combined": CombinedSearchStrategy,
    "narrative_sequence": NarrativeSequenceStrategy,
    "example_matching": ExampleMatchingStrategy,
    "time_window": TimeWindowStrategy,
    "content_path": ContentPathStrategy,
    "importance": ImportanceStrategy,
    "compound_query": CompoundQueryStrategy,
}

# Default strategy to use if none specified
DEFAULT_STRATEGY = "similarity" 