# AgentMemory Release History

## v0.1.1 - Similarity Search Strategy (May 18, 2024)

### Overview
The SimilaritySearchStrategy introduces powerful semantic search capabilities to the agent memory system. It enables finding memories based on semantic similarity using vector embeddings, providing advanced search capabilities across different memory tiers with sophisticated scoring and filtering.

### Key Features
* **Semantic similarity search**: Find memories based on semantic meaning using vector embeddings
* **Multi-tier search**: Search across STM, IM, and LTM memory tiers
* **Advanced scoring system**:  
   * Cosine similarity scoring between query and memory vectors  
   * Configurable minimum score thresholds  
   * Score normalization (0.0-1.0 range)  
   * Score-based result ordering
* **Flexible query support**:  
   * Text string queries  
   * Dictionary state queries  
   * Direct vector input
* **Metadata filtering**: Combine semantic search with metadata filters
* **Memory tier targeting**: Search specific tiers or across all tiers
* **Result limiting**: Control number of returned results
* **Duplicate handling**: Smart handling of duplicate memories across tiers

### Validation
The implementation has been extensively validated with:
* Comprehensive test suite covering basic and advanced functionality
* Edge case testing for robustness
* Memory tier transition testing
* Metadata filtering validation
* Content structure testing

### Documentation
Detailed documentation is provided in `validation/search/similarity/validation.md` including:
* Basic usage examples
* Advanced search techniques
* Test results and validation methodology
* Edge case handling
* Memory tier transition scenarios

### Getting Started
```python
from memory.search.strategies.similarity import SimilaritySearchStrategy

# Initialize the strategy with your memory system
strategy = SimilaritySearchStrategy(memory_system=memory_system)

# Basic text query search
results = strategy.search(
    query="machine learning model accuracy",
    agent_id="agent-123",
    limit=10
)

# Search with metadata filter and tier specification
results = strategy.search(
    query="data processing pipeline",
    agent_id="agent-123",
    metadata_filter={"type": "process"},
    tier="stm",
    min_score=0.4
)
```

### Test Results
The validation suite has been executed successfully, with key results including:
* ✅ Basic text and dictionary query functionality
* ✅ Metadata filtering and tier-specific search
* ✅ Score threshold filtering
* ✅ Multi-tier search capabilities
* ✅ Edge case handling
* ✅ Memory tier transition scenarios
* ✅ Complex metadata filtering
* ✅ Special character handling

---

## v0.1.0 - Attribute Search Strategy (May 10, 2024)

### Overview
The AttributeSearchStrategy provides a flexible search mechanism for the agent memory system. It allows searching for memories based on content attributes and metadata fields, providing advanced filtering and relevance scoring capabilities.

### Key Features
* **Content-based search**: Find memories containing specific text or patterns
* **Metadata filtering**: Filter memories by metadata fields like type, tags, importance
* **Regular expression support**: Powerful pattern matching
* **Multiple scoring algorithms**:  
   * Length ratio scoring: Score based on ratio of query length to field length  
   * Term frequency scoring: Prioritize based on term frequency  
   * BM25 ranking algorithm: Sophisticated information retrieval scoring  
   * Binary scoring: Simple match/no-match scoring
* **Tier-specific search**: Target specific memory tiers (STM, IM, LTM)
* **Case sensitivity control**: Enable or disable case-sensitive matching
* **Conditional matching**: Support for both AND/OR logic via match_all parameter
* **Array field matching**: Search within array elements like tags

### Validation
The implementation has been extensively validated with:
* Comprehensive test suite covering basic and advanced functionality
* Edge case testing for robustness
* Performance testing under various conditions

### Documentation
Detailed documentation is provided in `attribute_search_guide.md` including:
* Basic usage examples
* Advanced search techniques
* Best practices
* Example use cases

### Getting Started
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

## Release Progression

The release progression from v0.1.0 to v0.1.1 demonstrates the evolution of the AgentMemory search capabilities:

1. **v0.1.0 (Attribute Search)**
   - Introduced basic content and metadata-based search
   - Established foundation for memory tier searching
   - Implemented multiple scoring algorithms for relevance ranking

2. **v0.1.1 (Similarity Search)**
   - Added semantic search capabilities using vector embeddings
   - Enhanced scoring with cosine similarity
   - Improved multi-tier search with duplicate handling
   - Expanded query support to include vector inputs

This progression shows how the system has evolved from traditional text-based search to more sophisticated semantic search capabilities, while maintaining backward compatibility and expanding the range of search features available to users. 