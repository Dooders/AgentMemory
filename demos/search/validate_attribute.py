"""
Validation script for the Attribute Search Strategy.

This script loads a predefined memory system and tests various scenarios
of attribute-based searching to verify the strategy works correctly.
"""

import os
import sys
from typing import Any, Dict, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from demos.demo_utils import (
    create_memory_system,
    log_print,
    pretty_print_memories,
    setup_logging,
)
from memory.search.strategies.attribute import AttributeSearchStrategy

# Constants
AGENT_ID = "test-agent-attribute-search"
MEMORY_SAMPLE = "attribute_validation_memory.json"


def run_test(
    search_strategy: AttributeSearchStrategy,
    test_name: str,
    query: Any,
    agent_id: str,
    limit: int = 10,
    metadata_filter: Dict[str, Any] = None,
    tier: str = None,
    content_fields: List[str] = None,
    metadata_fields: List[str] = None,
    match_all: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = False,
    scoring_method: str = None,
) -> List[Dict[str, Any]]:
    """Run a test case and return the results."""
    log_print(logger, f"\n=== Test: {test_name} ===")

    if isinstance(query, dict):
        log_print(logger, f"Query (dict): {query}")
    else:
        log_print(logger, f"Query: '{query}'")

    log_print(
        logger,
        f"Match All: {match_all}, Case Sensitive: {case_sensitive}, Use Regex: {use_regex}",
    )

    if metadata_filter:
        log_print(logger, f"Metadata Filter: {metadata_filter}")

    if tier:
        log_print(logger, f"Tier: {tier}")

    if content_fields:
        log_print(logger, f"Content Fields: {content_fields}")

    if metadata_fields:
        log_print(logger, f"Metadata Fields: {metadata_fields}")

    if scoring_method:
        log_print(logger, f"Scoring Method: {scoring_method}")

    results = search_strategy.search(
        query=query,
        agent_id=agent_id,
        limit=limit,
        metadata_filter=metadata_filter,
        tier=tier,
        content_fields=content_fields,
        metadata_fields=metadata_fields,
        match_all=match_all,
        case_sensitive=case_sensitive,
        use_regex=use_regex,
        scoring_method=scoring_method,
    )

    log_print(logger, f"Found {len(results)} results")
    pretty_print_memories(results, f"Results for {test_name}", logger)

    # If we have scoring method, print the scores for comparison
    if scoring_method and results:
        log_print(logger, f"\nScores using {scoring_method} scoring method:")
        for idx, result in enumerate(results[:5]):  # Show scores for top 5 results
            score = result.get("metadata", {}).get("attribute_score", 0)
            memory_id = result.get("memory_id", result.get("id", f"Result {idx+1}"))
            log_print(logger, f"  {memory_id}: {score:.4f}")

    return results


def validate_attribute_search():
    """Run validation tests for the attribute search strategy."""
    # Setup memory system
    memory_system = create_memory_system(
        logging_level="INFO",
        memory_file=MEMORY_SAMPLE,
        use_mock_redis=True,
    )

    # If memory system failed to load, exit
    if not memory_system:
        log_print(logger, "Failed to load memory system")
        return

    # Setup search strategy
    agent = memory_system.get_memory_agent(AGENT_ID)
    search_strategy = AttributeSearchStrategy(
        agent.stm_store, agent.im_store, agent.ltm_store
    )

    # Print strategy info
    log_print(logger, f"Testing search strategy: {search_strategy.name()}")
    log_print(logger, f"Description: {search_strategy.description()}")

    # Test 1: Basic content search
    run_test(
        search_strategy,
        "Basic Content Search",
        "meeting",
        AGENT_ID,
    )

    # Test 2: Case sensitive search
    run_test(
        search_strategy,
        "Case Sensitive Search",
        "Meeting",
        AGENT_ID,
        case_sensitive=True,
    )

    # Test 3: Search by metadata type
    run_test(
        search_strategy,
        "Search by Metadata Type",
        {"metadata": {"type": "meeting"}},
        AGENT_ID,
    )

    # Test 4: Search with match_all
    run_test(
        search_strategy,
        "Search with Match All",
        {"content": "meeting", "metadata": {"type": "meeting", "importance": "high"}},
        AGENT_ID,
        match_all=True,
    )

    # Test 5: Search specific memory tier
    run_test(
        search_strategy,
        "Search in STM Tier Only",
        "meeting",
        AGENT_ID,
        tier="stm",
    )

    # Test 6: Search with regex
    run_test(
        search_strategy,
        "Regex Search",
        "secur.*patch",
        AGENT_ID,
        use_regex=True,
    )

    # Test 7: Search with metadata filter
    run_test(
        search_strategy,
        "Search with Metadata Filter",
        "meeting",
        AGENT_ID,
        metadata_filter={"content.metadata.importance": "high"},
    )

    # Test 8: Search in specific content fields
    run_test(
        search_strategy,
        "Search in Specific Content Fields",
        "project",
        AGENT_ID,
        content_fields=["content.content"],
    )

    # Test 9: Search in specific metadata fields
    run_test(
        search_strategy,
        "Search in Specific Metadata Fields",
        "project",
        AGENT_ID,
        metadata_fields=["content.metadata.tags"],
    )

    # Test 10: Search with complex query and filters
    run_test(
        search_strategy,
        "Complex Search",
        {"content": "security", "metadata": {"importance": "high"}},
        AGENT_ID,
        metadata_filter={"content.metadata.source": "email"},
        match_all=True,
    )

    # Test 11: Empty query handling - string
    run_test(
        search_strategy,
        "Empty String Query",
        "",
        AGENT_ID,
    )

    # Test 12: Empty query handling - dict
    run_test(
        search_strategy,
        "Empty Dict Query",
        {},
        AGENT_ID,
    )

    # Test 13: Numeric value search
    run_test(
        search_strategy,
        "Numeric Value Search",
        42,
        AGENT_ID,
    )

    # Test 14: Boolean value search
    run_test(
        search_strategy,
        "Boolean Value Search",
        {"metadata": {"completed": True}},
        AGENT_ID,
    )

    # Test 15: Type conversion - searching string with numeric
    run_test(
        search_strategy,
        "Type Conversion - String Field with Numeric",
        123,
        AGENT_ID,
        content_fields=["content.content"],
    )

    # Test 16: Invalid regex pattern handling
    run_test(
        search_strategy,
        "Invalid Regex Pattern",
        "[unclosed-bracket",
        AGENT_ID,
        use_regex=True,
    )

    # Test 17: Array field partial matching
    run_test(
        search_strategy,
        "Array Field Partial Matching",
        "dev",
        AGENT_ID,
        metadata_fields=["content.metadata.tags"],
    )

    # Test 18: Special characters in search
    run_test(
        search_strategy,
        "Special Characters in Search",
        "meeting+notes",
        AGENT_ID,
    )

    # Test 19: Multi-tier search
    run_test(
        search_strategy,
        "Multi-Tier Search",
        "important",
        AGENT_ID,
        # No tier specified means searching all tiers
    )

    # Test 20: Large result set limiting
    run_test(
        search_strategy,
        "Large Result Set Limiting",
        "a",  # Common letter to match many memories
        AGENT_ID,
        limit=3,  # Only show top 3 results
    )

    # ===== New tests for scoring methods =====
    log_print(logger, "\n=== SCORING METHOD COMPARISON TESTS ===")

    # Test 21: Comparing scoring methods on the same query
    test_query = "meeting"
    log_print(logger, f"\nComparing scoring methods for query: '{test_query}'")

    # Try each scoring method and collect results
    length_ratio_results = run_test(
        search_strategy,
        "Default Length Ratio Scoring",
        test_query,
        AGENT_ID,
        limit=5,
        scoring_method="length_ratio",
    )

    term_freq_results = run_test(
        search_strategy,
        "Term Frequency Scoring",
        test_query,
        AGENT_ID,
        limit=5,
        scoring_method="term_frequency",
    )

    bm25_results = run_test(
        search_strategy,
        "BM25 Scoring",
        test_query,
        AGENT_ID,
        limit=5,
        scoring_method="bm25",
    )

    binary_results = run_test(
        search_strategy,
        "Binary Scoring",
        test_query,
        AGENT_ID,
        limit=5,
        scoring_method="binary",
    )

    # Test 22: Testing scoring on a document with repeated terms
    test_with_repetition_query = "security"  # Look for security-related memories
    log_print(
        logger,
        f"\nComparing scoring methods for query with potential term repetition: '{test_with_repetition_query}'",
    )

    # Test with default length ratio scoring
    run_test(
        search_strategy,
        "Default Scoring with Term Repetition",
        test_with_repetition_query,
        AGENT_ID,
        limit=5,
    )

    # Test with term frequency scoring - should favor documents with more occurrences
    run_test(
        search_strategy,
        "Term Frequency with Term Repetition",
        test_with_repetition_query,
        AGENT_ID,
        limit=5,
        scoring_method="term_frequency",
    )

    # Test with BM25 scoring - balances term frequency and document length
    run_test(
        search_strategy,
        "BM25 with Term Repetition",
        test_with_repetition_query,
        AGENT_ID,
        limit=5,
        scoring_method="bm25",
    )

    # Test 23: Testing with a specialized search strategy for each method
    log_print(
        logger,
        "\nTesting with dedicated search strategy instances for each scoring method",
    )

    # Create specialized strategy instances
    term_freq_strategy = AttributeSearchStrategy(
        agent.stm_store,
        agent.im_store,
        agent.ltm_store,
        scoring_method="term_frequency",
    )

    bm25_strategy = AttributeSearchStrategy(
        agent.stm_store,
        agent.im_store,
        agent.ltm_store,
        scoring_method="bm25",
    )

    # Run test with specialized strategies
    run_test(
        term_freq_strategy,
        "Using Term Frequency Strategy Instance",
        "project",
        AGENT_ID,
        limit=5,
    )

    run_test(
        bm25_strategy,
        "Using BM25 Strategy Instance",
        "project",
        AGENT_ID,
        limit=5,
    )

    # Test 24: Testing with a long document vs short document comparison
    # Change from "detailed" (no matches) to "authentication system" (appears in memories of different lengths)
    long_doc_query = "authentication system"
    log_print(
        logger,
        f"\nComparing scoring methods for long vs short document query: '{long_doc_query}'",
    )

    # Compare each scoring method
    run_test(
        search_strategy,
        "Length Ratio for Long Documents",
        long_doc_query,
        AGENT_ID,
        limit=5,
        scoring_method="length_ratio",
    )

    run_test(
        search_strategy,
        "Term Frequency for Long Documents",
        long_doc_query,
        AGENT_ID,
        limit=5,
        scoring_method="term_frequency",
    )

    run_test(
        search_strategy,
        "BM25 for Long Documents",
        long_doc_query,
        AGENT_ID,
        limit=5,
        scoring_method="bm25",
    )

    # Test 25: Testing with a query that matches varying document length and context
    varying_length_query = "documentation"
    log_print(
        logger,
        f"\nComparing scoring methods for documents of varying lengths: '{varying_length_query}'",
    )

    run_test(
        search_strategy,
        "Length Ratio for Documentation Query",
        varying_length_query,
        AGENT_ID,
        limit=5,
        scoring_method="length_ratio",
    )

    run_test(
        search_strategy,
        "Term Frequency for Documentation Query",
        varying_length_query,
        AGENT_ID,
        limit=5,
        scoring_method="term_frequency",
    )

    run_test(
        search_strategy,
        "BM25 for Documentation Query",
        varying_length_query,
        AGENT_ID,
        limit=5,
        scoring_method="bm25",
    )


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging("validate_attribute_search")
    log_print(logger, "Starting Attribute Search Strategy Validation")

    validate_attribute_search()

    log_print(logger, "Validation Complete")
