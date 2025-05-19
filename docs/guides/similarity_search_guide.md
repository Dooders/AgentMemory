# Similarity Search Strategy Guide

## Overview

The Similarity Search Strategy is a powerful search mechanism that finds memories based on semantic similarity using vector embeddings. This strategy is particularly useful when you want to find memories that are conceptually similar to a given query, even if they don't contain exact matches.

## Key Features

- Semantic similarity search using vector embeddings
- Support for multiple memory tiers (STM, IM, LTM)
- Configurable similarity threshold
- Metadata filtering capabilities
- Automatic deduplication of results
- Score-based result ranking

## Usage

### Basic Search

```python
# Basic text-based search
results = memory_system.search(
    query="your search query",
    agent_id="agent_123",
    strategy="similarity"
)
```

### Advanced Search Options

```python
# Advanced search with filters and parameters
results = memory_system.search(
    query="your search query",
    agent_id="agent_123",
    strategy="similarity",
    limit=20,                    # Maximum number of results
    min_score=0.7,              # Minimum similarity threshold (0.0-1.0)
    tier="ltm",                 # Search specific memory tier
    metadata_filter={           # Filter by metadata
        "category": "important",
        "timestamp": {"$gt": "2024-01-01"}
    }
)
```

## Query Types

The similarity search supports multiple types of queries:

1. **Text Queries**: Simple string queries that are automatically converted to embeddings
   ```python
   results = memory_system.search("What is the capital of France?")
   ```

2. **State Dictionary**: Query using a structured dictionary
   ```python
   results = memory_system.search({
       "content": "What is the capital of France?",
       "context": "geography"
   })
   ```

3. **Vector Queries**: Direct vector embeddings (advanced usage)
   ```python
   results = memory_system.search(pre_computed_vector)
   ```

## Memory Tiers

The strategy supports searching across different memory tiers:

- `stm`: Short-term memory
- `im`: Intermediate memory
- `ltm`: Long-term memory

You can search specific tiers or all tiers at once:

```python
# Search specific tier
results = memory_system.search(
    query="your query",
    agent_id="agent_123",
    strategy="similarity",
    tier="ltm"
)

# Search all tiers (default)
results = memory_system.search(
    query="your query",
    agent_id="agent_123",
    strategy="similarity"
)
```

## Result Format

Search results are returned as a list of memory entries, each containing:

```python
{
    "id": "memory_id",
    "content": "memory content",
    "metadata": {
        "similarity_score": 0.85,  # Similarity score (0.0-1.0)
        "memory_tier": "ltm",      # Source memory tier
        # ... other metadata
    }
    # ... other memory fields
}
```

Results are automatically:
- Sorted by similarity score (highest first)
- Limited to the specified number of results
- Deduplicated (keeping the highest scoring version)

## Best Practices

1. **Similarity Threshold**
   - Default threshold is 0.6
   - Adjust based on your needs:
     - Higher (0.7-0.8): More precise matches
     - Lower (0.4-0.5): More inclusive results

2. **Memory Tier Selection**
   - Use specific tiers when you know where the information should be
   - Use all tiers when searching for general information

3. **Metadata Filtering**
   - Use metadata filters to narrow down results
   - Combine with similarity search for better precision

4. **Result Limits**
   - Set appropriate limits based on your use case
   - Consider using higher limits with post-processing

## Performance Considerations

- Vector similarity search is computationally intensive
- Consider caching frequently used queries
- Use metadata filters to reduce the search space
- Balance between result quality and performance

## Error Handling

The strategy includes robust error handling for:
- Vector generation failures
- Invalid memory tiers
- Store access errors
- Invalid query types

## Example Use Cases

1. **Semantic Search**
   ```python
   results = memory_system.search(
       query="How to implement authentication?",
       agent_id="agent_123",
       strategy="similarity",
       min_score=0.7
   )
   ```

2. **Context-Aware Search**
   ```python
   results = memory_system.search(
       query={
           "content": "authentication implementation",
           "context": "security",
           "priority": "high"
       },
       agent_id="agent_123",
       strategy="similarity"
   )
   ```

3. **Filtered Search**
   ```python
   results = memory_system.search(
       query="security best practices",
       agent_id="agent_123",
       strategy="similarity",
       metadata_filter={
           "category": "security",
           "importance": "high"
       }
   )
   ``` 