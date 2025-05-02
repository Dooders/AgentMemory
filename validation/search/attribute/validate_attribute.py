"""
Validation script for the Attribute Search Strategy.

This script loads a predefined memory system and tests various scenarios
of attribute-based searching to verify the strategy works correctly.
"""

import os
import sys
from typing import Any, Dict, List, Set

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from validation.demo_utils import (
    create_memory_system,
    log_print,
    pretty_print_memories,
    setup_logging,
)
from memory.search.strategies.attribute import AttributeSearchStrategy

# Constants
AGENT_ID = "test-agent-attribute-search"
MEMORY_SAMPLE = os.path.join("memory_samples", "attribute_validation_memory.json")

# Dictionary mapping memory IDs to their checksums for easier reference
MEMORY_CHECKSUMS = {
    "meeting-123456-1": "0eb0f81d07276f08e05351a604d3c994564fedee3a93329e318186da517a3c56",
    "meeting-123456-3": "f6ab36930459e74a52fdf21fb96a84241ccae3f6987365a21f9a17d84c5dae1e",
    "meeting-123456-6": "ffa0ee60ebaec5574358a02d1857823e948519244e366757235bf755c888a87f",
    "meeting-123456-9": "9214ebc2d11877665b32771bd3c080414d9519b435ec3f6c19cc5f337bb0ba90",
    "meeting-123456-11": "ad2e7c963751beb1ebc1c9b84ecb09ec3ccdef14f276cd14bbebad12d0f9b0df",
    "task-123456-2": "e0f7deb6929a17f65f56e5b03e16067c8bb65649fd2745f842aca7af701c9cac",
    "task-123456-7": "1d23b6683acd8c3863cb2f2010fe3df2c3e69a2d94c7c4757a291d4872066cfd",
    "task-123456-10": "f3c73b06d6399ed30ea9d9ad7c711a86dd58154809cc05497f8955425ec6dc67",
    "note-123456-4": "1e9e265e75c2ef678dfd0de0ab5c801f845daa48a90a48bb02ee85148ccc3470",
    "note-123456-8": "169c452e368fd62e3c0cf5ce7731769ed46ab6ae73e5048e0c3a7caaa66fba46",
    "contact-123456-5": "496d09718bbc8ae669dffdd782ed5b849fdbb1a57e3f7d07e61807b10e650092",
}


def get_checksums_for_memory_ids(memory_ids: List[str]) -> Set[str]:
    """Helper function to get checksums from memory IDs."""
    return {
        MEMORY_CHECKSUMS[memory_id]
        for memory_id in memory_ids
        if memory_id in MEMORY_CHECKSUMS
    }


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
    expected_checksums: Set[str] = None,
    expected_memory_ids: List[str] = None,
) -> Dict[str, Any]:
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

    # If expected_memory_ids is provided, convert to checksums
    if expected_memory_ids and not expected_checksums:
        expected_checksums = get_checksums_for_memory_ids(expected_memory_ids)
        log_print(
            logger,
            f"Expecting {len(expected_checksums)} memories from specified memory IDs",
        )

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

    # Track test status
    test_passed = True

    # Validate against expected checksums if provided
    if expected_checksums:
        result_checksums = {
            result.get("metadata", {}).get("checksum", "") for result in results
        }
        missing_checksums = expected_checksums - result_checksums
        unexpected_checksums = result_checksums - expected_checksums

        log_print(logger, f"\nValidation Results:")
        if not missing_checksums and not unexpected_checksums:
            log_print(logger, "All expected memories found. No unexpected memories.")
        else:
            if missing_checksums:
                log_print(logger, f"Missing expected memories: {missing_checksums}")
                test_passed = False
            if unexpected_checksums:
                log_print(logger, f"Found unexpected memories: {unexpected_checksums}")
                test_passed = False

        log_print(
            logger,
            f"Expected: {len(expected_checksums)}, Found: {len(result_checksums)}, "
            f"Missing: {len(missing_checksums)}, Unexpected: {len(unexpected_checksums)}",
        )

    return {
        "results": results,
        "test_name": test_name,
        "passed": test_passed,
        "has_validation": expected_checksums is not None
        or expected_memory_ids is not None,
    }


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

    # Track test results
    test_results = []

    # Test 1: Basic content search
    test_results.append(
        run_test(
            search_strategy,
            "Basic Content Search",
            "meeting",
            AGENT_ID,
            content_fields=["content.content"],
            metadata_filter={"content.metadata.type": "meeting"},
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    # Test 2: Case sensitive search
    test_results.append(
        run_test(
            search_strategy,
            "Case Sensitive Search",
            "Meeting",
            AGENT_ID,
            case_sensitive=True,
            content_fields=["content.content"],  # Only search in content
            metadata_filter={
                "content.metadata.type": "meeting"
            },  # Only get meeting-type memories
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
            ],
        )
    )

    # Test 3: Search by metadata type
    test_results.append(
        run_test(
            search_strategy,
            "Search by Metadata Type",
            {"metadata": {"type": "meeting"}},
            AGENT_ID,
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    # Test 4: Search with match_all
    test_results.append(
        run_test(
            search_strategy,
            "Search with Match All",
            {
                "content": "meeting",
                "metadata": {"type": "meeting", "importance": "high"},
            },
            AGENT_ID,
            match_all=True,
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    # Test 5: Search specific memory tier
    test_results.append(
        run_test(
            search_strategy,
            "Search in STM Tier Only",
            "meeting",
            AGENT_ID,
            tier="stm",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
            ],
        )
    )

    # Test 6: Search with regex
    test_results.append(
        run_test(
            search_strategy,
            "Regex Search",
            "secur.*patch",
            AGENT_ID,
            use_regex=True,
            expected_memory_ids=[
                "note-123456-4",
            ],
        )
    )

    # Test 7: Search with metadata filter
    test_results.append(
        run_test(
            search_strategy,
            "Search with Metadata Filter",
            "meeting",
            AGENT_ID,
            metadata_filter={"content.metadata.importance": "high"},
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    # Test 8: Search in specific content fields
    test_results.append(
        run_test(
            search_strategy,
            "Search in Specific Content Fields",
            "project",
            AGENT_ID,
            content_fields=["content.content"],
            expected_memory_ids=[
                "meeting-123456-1",
                "contact-123456-5",
            ],
        )
    )

    # Test 9: Search in specific metadata fields
    test_results.append(
        run_test(
            search_strategy,
            "Search in Specific Metadata Fields",
            "project",
            AGENT_ID,
            metadata_fields=["content.metadata.tags"],
            expected_memory_ids=[
                "meeting-123456-1",
                "contact-123456-5",
            ],
        )
    )

    # Test 10: Search with complex query and filters
    test_results.append(
        run_test(
            search_strategy,
            "Complex Search",
            {"content": "security", "metadata": {"importance": "high"}},
            AGENT_ID,
            metadata_filter={"content.metadata.source": "email"},
            match_all=True,
            expected_memory_ids=[
                "note-123456-4",
            ],
        )
    )

    # Test 11: Empty query handling - string
    test_results.append(
        run_test(
            search_strategy,
            "Empty String Query",
            "",
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 12: Empty query handling - dict
    test_results.append(
        run_test(
            search_strategy,
            "Empty Dict Query",
            {},
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 13: Numeric value search
    test_results.append(
        run_test(
            search_strategy,
            "Numeric Value Search",
            42,
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 14: Boolean value search
    test_results.append(
        run_test(
            search_strategy,
            "Boolean Value Search",
            {"metadata": {"completed": True}},
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 15: Type conversion - searching string with numeric
    test_results.append(
        run_test(
            search_strategy,
            "Type Conversion - String Field with Numeric",
            123,
            AGENT_ID,
            content_fields=["content.content"],
            expected_memory_ids=[],
        )
    )

    # Test 16: Invalid regex pattern handling
    test_results.append(
        run_test(
            search_strategy,
            "Invalid Regex Pattern",
            "[unclosed-bracket",
            AGENT_ID,
            use_regex=True,
            expected_memory_ids=[],
        )
    )

    # Test 17: Array field partial matching
    test_results.append(
        run_test(
            search_strategy,
            "Array Field Partial Matching",
            "dev",
            AGENT_ID,
            metadata_fields=["content.metadata.tags"],
            expected_memory_ids=[
                "meeting-123456-3",
                "task-123456-10",
            ],
        )
    )

    # Test 18: Special characters in search
    test_results.append(
        run_test(
            search_strategy,
            "Special Characters in Search",
            "meeting+notes",
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 19: Multi-tier search
    test_results.append(
        run_test(
            search_strategy,
            "Multi-Tier Search",
            "important",
            AGENT_ID,
            expected_memory_ids=[],
        )
    )

    # Test 20: Large result set limiting
    test_results.append(
        run_test(
            search_strategy,
            "Large Result Set Limiting",
            "a",  # Common letter to match many memories
            AGENT_ID,
            limit=3,  # Only show top 3 results
            expected_memory_ids=[
                "meeting-123456-1",  # Contains "about" and "allocation" - shorter content with multiple 'a's
                "meeting-123456-6",  # Contains "about" and "roadmap" - shorter content with multiple 'a's
                "meeting-123456-3",  # Contains "authentication" and "team" - longer content with fewer 'a's
            ],
        )
    )

    # ===== New tests for scoring methods =====
    log_print(logger, "\n=== SCORING METHOD COMPARISON TESTS ===")

    # Test 21: Comparing scoring methods on the same query
    test_query = "meeting"
    log_print(logger, f"\nComparing scoring methods for query: '{test_query}'")

    # Try each scoring method and collect results
    test_results.append(
        run_test(
            search_strategy,
            "Default Length Ratio Scoring",
            test_query,
            AGENT_ID,
            limit=5,
            scoring_method="length_ratio",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "Term Frequency Scoring",
            test_query,
            AGENT_ID,
            limit=5,
            scoring_method="term_frequency",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "BM25 Scoring",
            test_query,
            AGENT_ID,
            limit=5,
            scoring_method="bm25",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "Binary Scoring",
            test_query,
            AGENT_ID,
            limit=5,
            scoring_method="binary",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
        )
    )

    # Test 22: Testing scoring on a document with repeated terms
    test_with_repetition_query = "security"  # Look for security-related memories
    log_print(
        logger,
        f"\nComparing scoring methods for query with potential term repetition: '{test_with_repetition_query}'",
    )

    # Test with default length ratio scoring
    test_results.append(
        run_test(
            search_strategy,
            "Default Scoring with Term Repetition",
            test_with_repetition_query,
            AGENT_ID,
            limit=5,
            expected_memory_ids=[
                "note-123456-4",
                "note-123456-8",
                "meeting-123456-11",
            ],
        )
    )

    # Test with term frequency scoring - should favor documents with more occurrences
    test_results.append(
        run_test(
            search_strategy,
            "Term Frequency with Term Repetition",
            test_with_repetition_query,
            AGENT_ID,
            limit=5,
            scoring_method="term_frequency",
            expected_memory_ids=[
                "note-123456-4",
                "note-123456-8",
                "meeting-123456-11",
            ],
        )
    )

    # Test with BM25 scoring - balances term frequency and document length
    test_results.append(
        run_test(
            search_strategy,
            "BM25 with Term Repetition",
            test_with_repetition_query,
            AGENT_ID,
            limit=5,
            scoring_method="bm25",
            expected_memory_ids=[
                "note-123456-4",
                "note-123456-8",
                "meeting-123456-11",
            ],
        )
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
    test_results.append(
        run_test(
            term_freq_strategy,
            "Using Term Frequency Strategy Instance",
            "project",
            AGENT_ID,
            limit=5,
            expected_memory_ids=[
                "meeting-123456-1",
                "contact-123456-5",
            ],
        )
    )

    test_results.append(
        run_test(
            bm25_strategy,
            "Using BM25 Strategy Instance",
            "project",
            AGENT_ID,
            limit=5,
            expected_memory_ids=[
                "meeting-123456-1",
                "contact-123456-5",
            ],
        )
    )

    # Test 24: Testing with a long document vs short document comparison
    # Change from "detailed" (no matches) to "authentication system" (appears in memories of different lengths)
    long_doc_query = "authentication system"
    log_print(
        logger,
        f"\nComparing scoring methods for long vs short document query: '{long_doc_query}'",
    )

    # Compare each scoring method
    test_results.append(
        run_test(
            search_strategy,
            "Length Ratio for Long Documents",
            long_doc_query,
            AGENT_ID,
            limit=5,
            scoring_method="length_ratio",
            expected_memory_ids=[
                "meeting-123456-3",
                "task-123456-7",
                "task-123456-10",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "Term Frequency for Long Documents",
            long_doc_query,
            AGENT_ID,
            limit=5,
            scoring_method="term_frequency",
            expected_memory_ids=[
                "meeting-123456-3",
                "task-123456-7",
                "task-123456-10",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "BM25 for Long Documents",
            long_doc_query,
            AGENT_ID,
            limit=5,
            scoring_method="bm25",
            expected_memory_ids=[
                "meeting-123456-3",
                "task-123456-7",
                "task-123456-10",
            ],
        )
    )

    # Test 25: Testing with a query that matches varying document length and context
    varying_length_query = "documentation"
    log_print(
        logger,
        f"\nComparing scoring methods for documents of varying lengths: '{varying_length_query}'",
    )

    test_results.append(
        run_test(
            search_strategy,
            "Length Ratio for Documentation Query",
            varying_length_query,
            AGENT_ID,
            limit=5,
            scoring_method="length_ratio",
            expected_memory_ids=[
                "task-123456-2",
                "task-123456-7",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "Term Frequency for Documentation Query",
            varying_length_query,
            AGENT_ID,
            limit=5,
            scoring_method="term_frequency",
            expected_memory_ids=[
                "task-123456-2",
                "task-123456-7",
            ],
        )
    )

    test_results.append(
        run_test(
            search_strategy,
            "BM25 for Documentation Query",
            varying_length_query,
            AGENT_ID,
            limit=5,
            scoring_method="bm25",
            expected_memory_ids=[
                "task-123456-2",
                "task-123456-7",
            ],
        )
    )

    # Display validation summary
    log_print(logger, "\n\n=== VALIDATION SUMMARY ===")
    log_print(logger, "-" * 80)
    log_print(
        logger,
        "| {:<40} | {:<20} | {:<20} |".format(
            "Test Name", "Status", "Validation Status"
        ),
    )
    log_print(logger, "-" * 80)

    for result in test_results:
        status = "PASS" if result["passed"] else "FAIL"
        validation_status = status if result["has_validation"] else "N/A"
        log_print(
            logger,
            "| {:<40} | {:<20} | {:<20} |".format(
                result["test_name"][:40], status, validation_status
            ),
        )

    log_print(logger, "-" * 80)

    # Calculate overall statistics
    validated_tests = [t for t in test_results if t["has_validation"]]
    passed_tests = [t for t in validated_tests if t["passed"]]

    if validated_tests:
        success_rate = len(passed_tests) / len(validated_tests) * 100
        log_print(logger, f"\nValidated Tests: {len(validated_tests)}")
        log_print(logger, f"Passed Tests: {len(passed_tests)}")
        log_print(logger, f"Failed Tests: {len(validated_tests) - len(passed_tests)}")
        log_print(logger, f"Success Rate: {success_rate:.2f}%")
    else:
        log_print(logger, "\nNo tests with validation criteria were run.")


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging("validate_attribute_search")
    log_print(logger, "Starting Attribute Search Strategy Validation")

    validate_attribute_search()

    log_print(logger, "Validation Complete")
