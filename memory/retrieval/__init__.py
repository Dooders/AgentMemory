"""Agent Memory Retrieval Module

This module provides comprehensive memory retrieval mechanisms for accessing and
filtering stored agent memories based on different criteria. It enables the agent
to recall relevant information using semantic similarity, temporal attributes, or
specific content characteristics.

Key components:

1. SimilarityRetrieval: Retrieves memories based on semantic similarity using vector
   embeddings. Supports finding memories similar to a given state, to another memory,
   or to an example pattern.

2. TemporalRetrieval: Provides time-based memory access, including retrieval by
   recency, specific time periods, or simulation step ranges. Supports retrieving
   memories from specific time windows or narrative sequences.

3. AttributeRetrieval: Enables targeted memory access based on specific attributes,
   metadata values, content patterns, and tags. Supports compound queries with
   multiple conditions for precise memory filtering.

This module works with the memory storage tiers (STM, IM, LTM) to provide flexible
and efficient access to agent memories across different time horizons and contexts.

Usage example:
```python
from memory.retrieval import SimilarityRetrieval, TemporalRetrieval, AttributeRetrieval

# Initialize retrieval mechanisms with appropriate stores
similarity_retrieval = SimilarityRetrieval(vector_store, embedding_engine, 
                                          stm_store, im_store, ltm_store)
temporal_retrieval = TemporalRetrieval(stm_store, im_store, ltm_store)
attribute_retrieval = AttributeRetrieval(stm_store, im_store, ltm_store)

# Find semantically similar memories to current state
similar_memories = similarity_retrieval.retrieve_similar_to_state(
    current_state, limit=5, min_score=0.7
)

# Get memories from the last hour
recent_memories = temporal_retrieval.retrieve_last_n_minutes(
    minutes=60, memory_type="interaction"
)

# Retrieve memories with specific metadata or content pattern
important_memories = attribute_retrieval.retrieve_by_importance(
    min_importance=0.8, tier="ltm"
)
```
"""

from memory.retrieval.attribute import AttributeRetrieval
from memory.retrieval.similarity import SimilarityRetrieval
from memory.retrieval.temporal import TemporalRetrieval

__all__ = [
    "SimilarityRetrieval",
    "TemporalRetrieval",
    "AttributeRetrieval",
]
