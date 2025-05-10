# ImportanceSearchStrategy Guide

## Overview

The ImportanceSearchStrategy is a specialized search mechanism used in the agent memory system. It allows you to search for memories based on their importance scores, enabling agents to focus on significant information and prioritize critical memories.

## How It Works

ImportanceSearchStrategy works by matching memory importance scores against specified thresholds. It supports:

- Simple importance threshold filtering
- Range-based importance filtering (min/max)
- Top N most important memories retrieval
- Memory tier-specific searching
- Metadata filtering
- Multiple sorting options

## Basic Usage

Here's a simple example of how to use the ImportanceSearchStrategy:

```python
from memory.search.strategies.importance import ImportanceStrategy

# Initialize the strategy with your memory stores
strategy = ImportanceStrategy(
    stm_store=stm_store,
    im_store=im_store,
    ltm_store=ltm_store
)

# Basic importance threshold search
results = strategy.search(
    query=0.8,  # Find memories with importance >= 0.8
    agent_id="agent-123",
    limit=10
)

# Search with metadata filter
results = strategy.search(
    query=0.7,
    agent_id="agent-123",
    metadata_filter={"content.metadata.type": "alert"}
)
```

## Search Parameters

The strategy offers a comprehensive set of parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `float` or `Dict[str, Any]` | Importance threshold or dictionary with search parameters |
| `agent_id` | `str` | ID of the agent whose memories to search |
| `limit` | `int` | Maximum number of results to return (default: 10) |
| `metadata_filter` | `Dict[str, Any]` | Optional filters to apply to memory metadata |
| `tier` | `str` | Optional memory tier to search ("stm", "im", "ltm", or None for all) |
| `sort_order` | `str` | Sort order for results ("asc" or "desc", default: "desc") |
| `sort_by` | `List[str]` | Fields to sort by (default: ["importance"]) |

## Advanced Search Techniques

### 1. Dictionary Queries

For more targeted searches, use a dictionary to specify importance parameters:

```python
# Search for memories with importance between 0.6 and 0.8
results = strategy.search(
    query={
        "min_importance": 0.6,
        "max_importance": 0.8
    },
    agent_id="agent-123"
)

# Get top 5 most important memories
results = strategy.search(
    query={"top_n": 5},
    agent_id="agent-123"
)
```

### 2. Memory Tier Targeting

To search only specific memory tiers:

```python
# Search only short-term memory
stm_results = strategy.search(
    query=0.7,
    agent_id="agent-123",
    tier="stm"
)

# Search only long-term memory
ltm_results = strategy.search(
    query=0.7,
    agent_id="agent-123",
    tier="ltm"
)
```

### 3. String Importance Mapping

The strategy automatically maps string importance values to numeric scores:

```python
# These are equivalent:
results1 = strategy.search(query=0.9, agent_id="agent-123")
results2 = strategy.search(
    query={"min_importance": "high"},
    agent_id="agent-123"
)
```

The default importance mapping is:
- "low" → 0.3
- "medium" → 0.6
- "high" → 0.9

## Sorting Options

The strategy supports flexible sorting of results:

```python
# Sort by importance in ascending order
results = strategy.search(
    query=0.5,
    agent_id="agent-123",
    sort_order="asc"
)

# Sort by multiple fields
results = strategy.search(
    query=0.5,
    agent_id="agent-123",
    sort_by=["importance", "recency"]
)
```

## Performance Considerations

- **Tier Filtering**: Limit searches to specific tiers when possible
- **Metadata Filtering**: Apply metadata filters to narrow down results
- **Limit Results**: Use appropriate limits to avoid retrieving unnecessary memories

## Edge Case Handling

The strategy robustly handles various edge cases:

- Zero importance threshold (returns all memories)
- Very high thresholds (returns no results)
- Invalid threshold types (raises ValueError)
- Empty dictionary queries
- Invalid min/max ranges
- Non-existent tier searches
- Negative top_n values
- Zero limit searches
- Dictionary with null values
- Non-numeric importance thresholds
- Negative importance values

## Best Practices

1. **Use Appropriate Thresholds**: Choose importance thresholds that match your use case
   - Use high thresholds (0.8+) for critical information
   - Use medium thresholds (0.5-0.7) for important but not critical information
   - Use low thresholds (0.3-0.4) for supplementary information

2. **Tier-Specific Searches**: When possible, search in specific memory tiers to improve performance

3. **Metadata Filtering**: Combine importance thresholds with metadata filters for more precise results

4. **Sort Order**: Use descending sort order for most important memories first

5. **Result Limits**: Always specify appropriate limits to avoid retrieving too many memories

## Example Use Cases

1. **Find Critical Alerts**
   ```python
   # Find high-importance alerts
   results = strategy.search(
       query=0.9,
       agent_id="agent-123",
       metadata_filter={"content.metadata.type": "alert"}
   )
   ```

2. **Get Recent Important Memories**
   ```python
   # Find important memories from the last day
   results = strategy.search(
       query=0.7,
       agent_id="agent-123",
       metadata_filter={
           "creation_time": {"$gte": timestamp_24h_ago}
       }
   )
   ```

3. **Find Medium Importance Tasks**
   ```python
   # Find medium importance tasks
   results = strategy.search(
       query={
           "min_importance": 0.5,
           "max_importance": 0.7
       },
       agent_id="agent-123",
       metadata_filter={"content.metadata.type": "task"}
   )
   ```

4. **Get Top 3 Most Important Memories**
   ```python
   # Get the 3 most important memories across all tiers
   results = strategy.search(
       query={"top_n": 3},
       agent_id="agent-123"
   )
   ```

## Advanced Examples

### 1. Combining Importance with Time-Based Filtering

Find important memories from a specific time period:

```python
# Find high-importance memories from last week
results = strategy.search(
    query=0.8,
    agent_id="agent-123",
    metadata_filter={
        "creation_time": {
            "$gte": timestamp_7d_ago,
            "$lte": current_timestamp
        }
    }
)
```

### 2. Multi-Tier Search with Metadata

Search across tiers with specific metadata requirements:

```python
# Find high-importance security-related memories
results = strategy.search(
    query=0.8,
    agent_id="agent-123",
    metadata_filter={
        "content.metadata.tags": "security",
        "content.metadata.type": "note"
    }
)
```

### 3. Importance Range with Type Filtering

Find memories within a specific importance range and type:

```python
# Find medium-importance meeting notes
results = strategy.search(
    query={
        "min_importance": 0.5,
        "max_importance": 0.7
    },
    agent_id="agent-123",
    metadata_filter={"content.metadata.type": "meeting"}
)
```

## Conclusion

The ImportanceSearchStrategy provides a powerful way to retrieve memories based on their importance scores. With its support for threshold-based filtering, range queries, and metadata filtering, it enables precise control over memory retrieval based on importance.

The strategy is particularly useful for:
- Prioritizing critical information
- Filtering out low-importance memories
- Finding memories within specific importance ranges
- Combining importance with other metadata criteria

By following the best practices and using the appropriate parameters, you can effectively leverage the ImportanceSearchStrategy to enhance your agent's memory retrieval capabilities. 