"""Agent Memory Search Module

This module provides a flexible search framework for retrieving agent memories using
different search strategies. It enables the agent to find relevant information using
semantic similarity, temporal patterns, attribute matching, or combinations of these approaches.

Key components:

1. SearchModel: The central search model that manages different search strategies and
   provides a unified interface for memory retrieval.

2. Search Strategies:
   - SimilaritySearchStrategy: Retrieves memories based on semantic similarity using vector
     embeddings. Ideal for finding contextually relevant information.
   - TemporalSearchStrategy: Finds memories based on time-related attributes like creation time,
     allowing for time-range queries and recency-based scoring.
   - AttributeSearchStrategy: Searches for memories based on content and metadata attributes,
     supporting exact matches, substring searches, and regex pattern matching.
   - CombinedSearchStrategy: Integrates results from multiple strategies for more sophisticated
     memory retrieval, with configurable weights for different strategies.
   - StepBasedSearchStrategy: Retrieves memories based on simulation step numbers rather than
     actual timestamps, optimized for simulation-based environments where step progression
     is more meaningful than wall-clock time.

The search model works with memory storage tiers (STM, IM, LTM) to provide efficient
and targeted access to agent memories across different contexts.

Usage example:
```python
from memory.search import SearchModel
from memory.search import SimilaritySearchStrategy, TemporalSearchStrategy
from memory.config import MemoryConfig

# Initialize search model with configuration
config = MemoryConfig()
search_model = SearchModel(config)

# Register search strategies
similarity_strategy = SimilaritySearchStrategy(vector_store, embedding_engine,
                                              stm_store, im_store, ltm_store)
temporal_strategy = TemporalSearchStrategy(stm_store, im_store, ltm_store)

search_model.register_strategy(similarity_strategy, make_default=True)
search_model.register_strategy(temporal_strategy)

# Search for memories using different strategies
similar_memories = search_model.search(
    query="meeting with client",
    agent_id="agent-1",
    limit=5
)

recent_memories = search_model.search(
    query={"start_time": "2023-06-01", "end_time": "2023-06-30"},
    agent_id="agent-1",
    strategy_name="temporal",
    limit=10
)

# Search using step numbers
step_based_memories = search_model.search(
    query={"start_step": 1000, "end_step": 2000},
    agent_id="agent-1",
    strategy_name="step_based",
    limit=10
)
```
"""

from memory.search.model import SearchModel
from memory.search.strategies.base import SearchStrategy
from memory.search.strategies.similarity import SimilaritySearchStrategy
from memory.search.strategies.temporal import TemporalSearchStrategy
from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.search.strategies.combined import CombinedSearchStrategy
from memory.search.strategies.step_based import StepBasedSearchStrategy

__all__ = [
    "SearchModel",
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "TemporalSearchStrategy",
    "AttributeSearchStrategy",
    "CombinedSearchStrategy",
    "StepBasedSearchStrategy",
] 