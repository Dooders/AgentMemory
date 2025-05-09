# AttributeSearchStrategy Guide

## Overview

The AttributeSearchStrategy is a flexible search mechanism used in the agent memory system. It allows you to search for memories based on content attributes and metadata fields, providing advanced filtering and relevance scoring capabilities.

## How It Works

AttributeSearchStrategy works by matching content and metadata fields against search queries. It supports:

- Simple text matching within content
- Metadata field filtering
- Regular expression pattern matching
- Complex multi-criteria searches
- Multiple scoring algorithms to rank results

## Basic Usage

Here's a simple example of how to use the AttributeSearchStrategy:

```python
from memory.search.strategies.attribute import AttributeSearchStrategy

# Initialize the strategy with your memory stores
strategy = AttributeSearchStrategy(
    stm_store=stm_store,
    im_store=im_store,
    ltm_store=ltm_store
)

# Basic content search
results = strategy.search(
    query="meeting",
    agent_id="agent-123",
    limit=10
)

# Search with metadata filter
results = strategy.search(
    query="project",
    agent_id="agent-123",
    metadata_filter={"content.metadata.importance": "high"}
)
```

## Search Parameters

The strategy offers a comprehensive set of parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` or `Dict[str, Any]` | Search query - string for content search or dictionary for field-specific search |
| `agent_id` | `str` | ID of the agent whose memories to search |
| `limit` | `int` | Maximum number of results to return (default: 10) |
| `metadata_filter` | `Dict[str, Any]` | Optional filters to apply to memory metadata |
| `tier` | `str` | Optional memory tier to search ("stm", "im", "ltm", or None for all) |
| `content_fields` | `List[str]` | Optional list of content fields to search in |
| `metadata_fields` | `List[str]` | Optional list of metadata fields to search in |
| `match_all` | `bool` | Whether all query conditions must match (AND) or any (OR) |
| `case_sensitive` | `bool` | Whether to perform case-sensitive matching |
| `use_regex` | `bool` | Whether to interpret query strings as regular expressions |
| `scoring_method` | `str` | Method used for scoring matches |

## Advanced Search Techniques

### 1. Dictionary Queries

For more targeted searches, use a dictionary to specify the exact fields you want to search:

```python
results = strategy.search(
    query={
        "content": "security",
        "metadata": {
            "type": "note",
            "importance": "high"
        }
    },
    agent_id="agent-123",
    match_all=True  # All conditions must match
)
```

### 2. Regular Expression Patterns

For powerful pattern matching:

```python
results = strategy.search(
    query="secur.*patch",
    agent_id="agent-123",
    use_regex=True
)
```

### 3. Memory Tier Targeting

To search only specific memory tiers:

```python
# Search only short-term memory
stm_results = strategy.search(
    query="meeting",
    agent_id="agent-123",
    tier="stm"
)

# Search only long-term memory
ltm_results = strategy.search(
    query="meeting",
    agent_id="agent-123",
    tier="ltm"
)
```

## Scoring Methods

The strategy supports multiple scoring methods to rank search results:

1. **Length Ratio (default)**: Scoring based on the ratio of query length to field length. Prioritizes more specific matches.

2. **Term Frequency**: Scoring based on how frequently the search term appears in the content.

3. **BM25**: A more sophisticated information retrieval scoring algorithm that considers term frequency and document length.

4. **Binary**: Simple match/no-match scoring (1.0 for matches, 0.0 for non-matches).

Example of specifying a scoring method:

```python
results = strategy.search(
    query="project",
    agent_id="agent-123",
    scoring_method="bm25"
)
```

## Performance Considerations

- **Pattern Caching**: For regex searches, patterns are cached to improve performance
- **Targeted Searches**: Limit searches to specific tiers when possible
- **Field Filtering**: Use content_fields and metadata_fields to narrow the search scope

## Edge Case Handling

The strategy robustly handles various edge cases:

- Empty string queries
- Special characters in search patterns
- Type conversions between fields
- Invalid regex patterns

## Best Practices

1. **Use Targeted Fields**: Specify content_fields and metadata_fields to improve search precision and performance

2. **Appropriate Scoring**: Choose the scoring method that best fits your use case
   - Use "length_ratio" for general searches
   - Use "term_frequency" when frequency matters
   - Use "bm25" for more sophisticated relevance ranking
   - Use "binary" for simple yes/no matching

3. **Metadata Filtering**: Apply metadata filters to narrow down results before content matching

4. **Case Sensitivity**: Only use case_sensitive=True when necessary

5. **Match Strategy**: Use match_all=True for strict filtering and match_all=False for broader results

## Example Use Cases

1. **Find Meeting Notes**
   ```python
   strategy.search(
       query="meeting",
       agent_id="agent-123",
       metadata_filter={"content.metadata.type": "meeting"}
   )
   ```

2. **Search for High-Priority Tasks**
   ```python
   strategy.search(
       query={"metadata": {"type": "task", "importance": "high"}},
       agent_id="agent-123",
       match_all=True
   )
   ```

3. **Find Content with Specific Keywords**
   ```python
   strategy.search(
       query="security authentication",
       agent_id="agent-123",
       content_fields=["content.content"]
   )
   ```

4. **Search Based on Tags**
   ```python
   strategy.search(
       query="development",
       agent_id="agent-123",
       metadata_fields=["content.metadata.tags"]
   )
   ```

## Conclusion

The AttributeSearchStrategy provides a flexible, powerful way to retrieve memories based on content and metadata attributes. With its support for complex queries, multiple scoring methods, and various matching options, it can be tailored to meet diverse search requirements in agent memory systems. 