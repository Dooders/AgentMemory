# TASM Search System

The TASM (Tiered Adaptive Semantic Memory) Search System provides a flexible and powerful framework for retrieving relevant memories from different memory tiers. This document explains the key components of the search system, demonstrates how to use various search strategies, and shows how to extend the system for custom search requirements.

## Overview

The search system is designed to help agents effectively retrieve information from their memory stores using different search approaches:

- **Semantic similarity**: Finding memories that are conceptually similar to a query
- **Temporal patterns**: Retrieving memories based on time attributes
- **Attribute matching**: Searching for specific content characteristics or metadata
- **Combined approaches**: Integrating multiple search methods for more comprehensive results

The system's modular architecture allows for easy extension with new search strategies while maintaining a consistent interface.

## Key Components

### SearchModel

The `SearchModel` serves as the central coordinator for memory search operations. It:

- Manages a registry of available search strategies
- Provides a unified interface for querying memories
- Allows switching between strategies at runtime
- Supports setting default strategies for common search patterns

### Search Strategies

The system includes several built-in search strategies, each optimized for different retrieval scenarios:

1. **SimilaritySearchStrategy**
   - Uses vector embeddings to find semantically similar memories
   - Ideal for concept-based searches without exact keyword matching
   - Works with the vector store to perform efficient similarity comparisons

2. **TemporalSearchStrategy**
   - Retrieves memories based on time-related attributes
   - Supports time range queries and recency-based scoring
   - Useful for finding memories from specific time periods

3. **AttributeSearchStrategy**
   - Searches based on content and metadata attributes
   - Supports exact matches, substring searches, and regex patterns
   - Ideal for precise filtering based on known attributes

4. **CombinedSearchStrategy**
   - Integrates results from multiple search strategies
   - Applies configurable weights to different strategies
   - Enables sophisticated searches that blend different retrieval approaches

## Usage Guide

### Basic Setup

To start using the search system, you need to:

1. Initialize the search model with a memory configuration
2. Create and register one or more search strategies
3. Perform searches using the model's interface

Here's a basic setup example:

```python
from memory.search import SearchModel
from memory.search import SimilaritySearchStrategy, TemporalSearchStrategy
from memory.config import MemoryConfig

# Create memory configuration
config = MemoryConfig()

# Initialize search model
search_model = SearchModel(config)

# Create search strategies
similarity_strategy = SimilaritySearchStrategy(
    vector_store, embedding_engine, stm_store, im_store, ltm_store
)
temporal_strategy = TemporalSearchStrategy(stm_store, im_store, ltm_store)

# Register strategies with the model
search_model.register_strategy(similarity_strategy, make_default=True)
search_model.register_strategy(temporal_strategy)
```

### Performing Searches

Once the search model is set up, you can perform searches using different strategies:

#### Similarity Search

```python
# Find memories similar to a text query
similar_memories = search_model.search(
    query="meeting with the client about the new project",
    agent_id="agent-1",
    strategy_name="similarity",  # Optional if similarity is the default
    limit=5,
    min_score=0.7  # Minimum similarity threshold
)

# Find memories similar to a state dictionary
state_query = {
    "content": "Planning the marketing strategy",
    "source": "conversation",
    "participants": ["agent-1", "marketing-team"]
}

similar_memories = search_model.search(
    query=state_query,
    agent_id="agent-1",
    tier="ltm",  # Optionally restrict to a specific memory tier
    limit=10
)
```

#### Temporal Search

```python
# Find memories from a specific time range
time_range_memories = search_model.search(
    query={
        "start_time": "2023-06-01",
        "end_time": "2023-06-30"
    },
    agent_id="agent-1",
    strategy_name="temporal",
    limit=20
)

# Find recent memories with recency weighting
recent_memories = search_model.search(
    query="2023-07-15",  # Reference date
    agent_id="agent-1",
    strategy_name="temporal",
    recency_weight=2.0,  # Emphasize more recent memories
    limit=10
)
```

#### Step-Based Search

```python
# Find memories within a specific step range
step_range_memories = search_model.search(
    query={
        "start_step": 1000,
        "end_step": 2000
    },
    agent_id="agent-1",
    strategy_name="step_based",
    limit=20
)

# Find memories near a specific step number
nearby_step_memories = search_model.search(
    query="1500",  # Reference step number
    agent_id="agent-1",
    strategy_name="step_based",
    step_range=200,  # Search within 200 steps in each direction
    limit=10
)

# Find memories with custom step range and weighting
custom_step_memories = search_model.search(
    query={
        "reference_step": 1500,
        "step_range": 500
    },
    agent_id="agent-1",
    strategy_name="step_based", 
    step_weight=2.0,  # Emphasize closer step matches
    limit=15
)
```

#### Attribute Search

```python
# Search for memories with specific content or metadata
attribute_memories = search_model.search(
    query="error message",
    agent_id="agent-1",
    strategy_name="attribute",
    content_fields=["content", "summary"],
    metadata_fields=["type", "tags", "importance"],
    match_all=False,  # Match any field (OR logic)
    limit=15
)

# More complex attribute search with dictionary query
complex_query = {
    "content": "budget discussion",
    "metadata": {
        "type": "meeting",
        "importance": "high"
    }
}

important_meetings = search_model.search(
    query=complex_query,
    agent_id="agent-1",
    strategy_name="attribute",
    match_all=True,  # All conditions must match (AND logic)
    limit=10
)
```

#### Combined Search

```python
# Create a combined strategy
from memory.search import CombinedSearchStrategy

combined_strategy = CombinedSearchStrategy(
    strategies={
        "similarity": similarity_strategy,
        "temporal": temporal_strategy,
        "attribute": attribute_strategy
    },
    weights={
        "similarity": 1.0,
        "temporal": 0.7,
        "attribute": 0.5
    }
)

# Register the combined strategy
search_model.register_strategy(combined_strategy)

# Perform a search using the combined strategy
comprehensive_results = search_model.search(
    query="quarterly planning session",
    agent_id="agent-1",
    strategy_name="combined",
    strategy_params={
        "similarity": {"min_score": 0.6},
        "temporal": {"recency_weight": 1.5},
        "attribute": {"match_all": False}
    },
    limit=10
)
```

### Filtering Results

All search strategies support additional filtering through the `metadata_filter` parameter:

```python
# Filter results by metadata attributes
filtered_results = search_model.search(
    query="project discussion",
    agent_id="agent-1",
    metadata_filter={
        "type": "meeting",
        "project_id": "proj-123"
    },
    limit=10
)
```

## Advanced Usage

### Customizing Strategy Weights

For the `CombinedSearchStrategy`, you can adjust the weights of different strategies to emphasize certain approaches:

```python
# Update weights for the combined strategy
combined_strategy.set_weights({
    "similarity": 2.0,  # Increase similarity importance
    "temporal": 0.5,    # Decrease temporal importance
    "attribute": 1.0    # Keep attribute importance the same
})
```

### Creating Custom Search Strategies

You can extend the search system by creating custom strategies that implement the `SearchStrategy` interface:

```python
from memory.search.strategies.base import SearchStrategy

class MyCustomSearchStrategy(SearchStrategy):
    """Custom search strategy implementation."""
    
    def __init__(self, stm_store, im_store, ltm_store):
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
        # Additional initialization
    
    def name(self) -> str:
        return "custom"
    
    def description(self) -> str:
        return "My custom search strategy for specialized retrieval"
    
    def search(
        self,
        query,
        agent_id,
        limit=10,
        metadata_filter=None,
        tier=None,
        **kwargs
    ):
        # Custom search implementation
        results = []
        # ... search logic ...
        return results
```

Then register your custom strategy with the search model:

```python
custom_strategy = MyCustomSearchStrategy(stm_store, im_store, ltm_store)
search_model.register_strategy(custom_strategy)
```

## Integration with Memory Tiers

The search system works across all memory tiers (STM, IM, LTM) and allows for tier-specific searches:

```python
# Search only in short-term memory
stm_results = search_model.search(
    query="recent conversation",
    agent_id="agent-1",
    tier="stm",
    limit=5
)

# Search only in long-term memory
ltm_results = search_model.search(
    query="childhood experience",
    agent_id="agent-1",
    tier="ltm",
    limit=10
)
```

## Performance Considerations

For optimal search performance:

1. **Be specific with queries**: More targeted queries yield better results
2. **Use appropriate strategies**: Choose the strategy that best matches your search intent
3. **Limit result size**: Request only as many results as needed
4. **Use tier filtering**: Search only in relevant memory tiers
5. **Consider combined strategies**: For complex search needs, combine strategies with appropriate weights

## Example Applications

### Conversation Context Retrieval

```python
def retrieve_conversation_context(agent_id, current_topic, search_model):
    """Retrieve relevant context for an ongoing conversation."""
    return search_model.search(
        query=current_topic,
        agent_id=agent_id,
        strategy_name="combined",  # Use combined strategy for comprehensive results
        limit=5,
        metadata_filter={"type": "conversation"}
    )
```

### Decision Support

```python
def retrieve_decision_support(agent_id, decision_topic, search_model):
    """Retrieve memories to support decision-making on a topic."""
    similar_experiences = search_model.search(
        query=decision_topic,
        agent_id=agent_id,
        strategy_name="similarity",
        limit=3,
        tier="ltm"  # Focus on long-term memories for experience
    )
    
    recent_context = search_model.search(
        query=decision_topic,
        agent_id=agent_id,
        strategy_name="temporal",
        limit=2,
        tier="im"  # Recent context from intermediate memory
    )
    
    return {
        "experiences": similar_experiences,
        "recent_context": recent_context
    }
```

## Conclusion

The TASM Search System provides a versatile and powerful framework for retrieving relevant memories using different search approaches. By leveraging the appropriate search strategies and combining them as needed, agents can efficiently access the information they need to make informed decisions and maintain contextual awareness.

For more information on other components of the memory system, refer to:
- [Agent Memory Overview](agent_memory.md)
- [Memory Tiers](memory_tiers.md)
- [Embeddings](embeddings.md)
- [Memory API](memory_api.md) 