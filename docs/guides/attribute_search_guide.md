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

## Using the Query Builder

For an easier way to create complex queries, you can use the AttributeQueryBuilder:

```python
from memory.search.strategies.query_builder import AttributeQueryBuilder

# Simple content search
results = (AttributeQueryBuilder()
    .content("meeting")
    .limit(5)
    .execute(strategy, "agent-123"))

# Search for high importance security notes
results = (AttributeQueryBuilder()
    .content("security")
    .type("note")
    .importance("high")
    .match_all(True)
    .execute(strategy, "agent-123"))
    
# Regex search in specific tier
results = (AttributeQueryBuilder()
    .content("secur.*patch")
    .use_regex(True)
    .in_tier("ltm")
    .score_by("bm25")
    .execute(strategy, "agent-123"))
```

The query builder provides a fluent, intuitive interface that eliminates the need to understand the underlying query structure.

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

With the query builder, the above becomes:

```python
results = (AttributeQueryBuilder()
    .content("security")
    .type("note")
    .importance("high")
    .match_all(True)
    .execute(strategy, "agent-123"))
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

With the query builder:

```python
results = (AttributeQueryBuilder()
    .content("secur.*patch")
    .use_regex(True)
    .execute(strategy, "agent-123"))
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

With the query builder:

```python
# Search only short-term memory
stm_results = (AttributeQueryBuilder()
    .content("meeting")
    .in_tier("stm")
    .execute(strategy, "agent-123"))

# Search only long-term memory
ltm_results = (AttributeQueryBuilder()
    .content("meeting")
    .in_tier("ltm")
    .execute(strategy, "agent-123"))
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

With the query builder:

```python
results = (AttributeQueryBuilder()
    .content("project")
    .score_by("bm25")
    .execute(strategy, "agent-123"))
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

6. **Use the Query Builder**: For complex queries, use the AttributeQueryBuilder to improve code readability and avoid errors

## Example Use Cases

1. **Find Meeting Notes**
   ```python
   # Direct method
   strategy.search(
       query="meeting",
       agent_id="agent-123",
       metadata_filter={"content.metadata.type": "meeting"}
   )
   
   # Using query builder
   AttributeQueryBuilder()
       .content("meeting")
       .type("meeting")
       .execute(strategy, "agent-123")
   ```

2. **Search for High-Priority Tasks**
   ```python
   # Direct method
   strategy.search(
       query={"metadata": {"type": "task", "importance": "high"}},
       agent_id="agent-123",
       match_all=True
   )
   
   # Using query builder
   AttributeQueryBuilder()
       .type("task")
       .importance("high")
       .match_all(True)
       .execute(strategy, "agent-123")
   ```

3. **Find Content with Specific Keywords**
   ```python
   # Direct method
   strategy.search(
       query="security authentication",
       agent_id="agent-123",
       content_fields=["content.content"]
   )
   
   # Using query builder
   AttributeQueryBuilder()
       .content("security authentication")
       .in_content_fields("content.content")
       .execute(strategy, "agent-123")
   ```

4. **Search Based on Tags**
   ```python
   # Direct method
   strategy.search(
       query="development",
       agent_id="agent-123",
       metadata_fields=["content.metadata.tags"]
   )
   
   # Using query builder
   AttributeQueryBuilder()
       .tag("development")
       .execute(strategy, "agent-123")
   ```

## Advanced Examples

The following examples demonstrate more complex search scenarios that showcase the full power of AttributeQueryBuilder.

### 1. Combining Regex with Metadata Filters

Find all high-importance security notes mentioning vulnerability patterns:

```python
# Using regex to search for "CVE-" followed by digits and filtering by type and importance
results = (AttributeQueryBuilder()
    .content("CVE-\\d+")
    .use_regex(True)
    .type("note")
    .importance("high")
    .match_all(True)
    .limit(20)
    .execute(strategy, "agent-123"))
```

### 2. Field-Specific Regex Search with Multiple Metadata Constraints

Find content with email addresses in specific project-related notes:

```python
# Search for email pattern only in specific fields, with tag and type filters
results = (AttributeQueryBuilder()
    .content("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    .use_regex(True)
    .in_content_fields("content.content")
    .type("note")
    .tag("project")
    .match_all(True)
    .case_sensitive(False)
    .execute(strategy, "agent-123"))
```

### 3. Cross-Tier Search with Advanced Scoring

Search across memory tiers with BM25 scoring for more relevant results:

```python
# Search for "authentication" in all tiers with BM25 scoring
auth_results = (AttributeQueryBuilder()
    .content("authentication")
    .score_by("bm25")
    .filter_metadata("content.metadata.source", "meeting")
    .execute(strategy, "agent-123"))

# Compare with just long-term memory results
ltm_auth_results = (AttributeQueryBuilder()
    .content("authentication")
    .score_by("bm25")
    .in_tier("ltm")
    .filter_metadata("content.metadata.source", "meeting")
    .execute(strategy, "agent-123"))
```

### 4. Targeted Field Search with Regex

Find tasks with specific date patterns in their deadlines:

```python
# Search for date patterns in the deadline field using regex
results = (AttributeQueryBuilder()
    .content("\\d{4}-\\d{2}-\\d{2}")  # YYYY-MM-DD format
    .use_regex(True)
    .in_metadata_fields("content.metadata.deadline")
    .type("task")
    .importance("high")
    .match_all(True)
    .execute(strategy, "agent-123"))
```

### 5. Combining Multiple Search Approaches

This complex example combines regex pattern matching, field targeting, and metadata filtering:

```python
# Find technical discussions about API endpoints in meeting notes
results = (AttributeQueryBuilder()
    .content("/api/v[1-9]/[a-zA-Z]+")  # Match API endpoint patterns
    .use_regex(True)
    .in_content_fields("content.content", "content.summary")
    .type("meeting")
    .tag("technical")
    .filter_metadata("content.metadata.participants", "engineering-team")
    .match_all(True)
    .case_sensitive(True)
    .limit(15)
    .score_by("term_frequency")
    .execute(strategy, "agent-123"))
```

### 6. Sequential Searches for Complex Scenarios

Sometimes you need to perform sequential searches to handle complex scenarios:

```python
# First find all security-related high-importance notes
security_notes = (AttributeQueryBuilder()
    .type("note")
    .tag("security")
    .importance("high")
    .match_all(True)
    .execute(strategy, "agent-123"))

# Extract their IDs
security_note_ids = [note.get("memory_id", note.get("id")) for note in security_notes]

# Then find mentions of those notes in meeting content
if security_note_ids:
    # Build regex pattern to match any of the note IDs
    id_pattern = "|".join(security_note_ids)
    mention_results = (AttributeQueryBuilder()
        .content(f"({id_pattern})")
        .use_regex(True)
        .type("meeting")
        .execute(strategy, "agent-123"))
```

These advanced examples showcase the full potential of combining regex searches with metadata filters and other powerful features of the AttributeQueryBuilder.

## Conclusion

The AttributeSearchStrategy provides a flexible, powerful way to retrieve memories based on content and metadata attributes. With its support for complex queries, multiple scoring methods, and various matching options, it can be tailored to meet diverse search requirements in agent memory systems. 

The AttributeQueryBuilder offers a more intuitive interface for creating complex queries, making the system more accessible to developers unfamiliar with the underlying memory structure. 