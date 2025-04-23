"""
Validation script for the Importance Search Strategy.

This script loads a predefined memory system and tests various scenarios
of importance-based searching to verify the strategy works correctly.
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
from memory.search.strategies.importance import ImportanceStrategy

# Constants
AGENT_ID = "test-agent-importance-search"
MEMORY_SAMPLE = "importance_validation_memory.json"


def run_test(
    search_strategy: ImportanceStrategy,
    test_name: str,
    query: Any,
    agent_id: str,
    limit: int = 10,
    metadata_filter: Dict[str, Any] = None,
    tier: str = None,
    sort_order: str = "desc",
) -> List[Dict[str, Any]]:
    """Run a test case and return the results."""
    log_print(logger, f"\n=== Test: {test_name} ===")

    if isinstance(query, dict):
        log_print(logger, f"Query (dict): {query}")
    elif isinstance(query, (int, float)):
        log_print(logger, f"Min Importance: {query}")
    else:
        log_print(logger, f"Query: {query}")

    log_print(logger, f"Sort Order: {sort_order}")

    if metadata_filter:
        log_print(logger, f"Metadata Filter: {metadata_filter}")

    if tier:
        log_print(logger, f"Tier: {tier}")

    results = search_strategy.search(
        query=query,
        agent_id=agent_id,
        limit=limit,
        metadata_filter=metadata_filter,
        tier=tier,
        sort_order=sort_order,
    )

    log_print(logger, f"Found {len(results)} results")
    pretty_print_memories(results, f"Results for {test_name}", logger)

    # Print importance scores for top results
    if results:
        log_print(
            logger, f"\nImportance scores for top {min(5, len(results))} results:"
        )
        for idx, result in enumerate(results[:5]):
            importance = result.get("metadata", {}).get("importance_score", 0)
            memory_id = result.get("memory_id", result.get("id", f"Result {idx+1}"))
            log_print(logger, f"  {memory_id}: {importance:.4f}")

    return results


def validate_importance_search():
    """Run validation tests for the importance search strategy."""
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

    # DEBUG: Print the memory count in each store
    log_print(logger, f"Memory system loaded: {memory_system}")
    agent = memory_system.get_memory_agent(AGENT_ID)
    try:
        stm_memories = agent.stm_store.get_all(AGENT_ID)
        log_print(logger, f"STM memories loaded: {len(stm_memories)}")
        if stm_memories:
            log_print(
                logger,
                f"First STM memory: {stm_memories[0]['memory_id']} - Importance: {stm_memories[0]['metadata'].get('importance_score', 'N/A')}",
            )
    except Exception as e:
        log_print(logger, f"Error getting STM memories: {str(e)}")

    try:
        im_memories = agent.im_store.get_all(AGENT_ID)
        log_print(logger, f"IM memories loaded: {len(im_memories)}")
        if im_memories:
            log_print(
                logger,
                f"First IM memory: {im_memories[0]['memory_id']} - Importance: {im_memories[0]['metadata'].get('importance_score', 'N/A')}",
            )
    except Exception as e:
        log_print(logger, f"Error getting IM memories: {str(e)}")

    try:
        ltm_memories = agent.ltm_store.get_all(AGENT_ID)
        log_print(logger, f"LTM memories loaded: {len(ltm_memories)}")
        if ltm_memories:
            log_print(
                logger,
                f"First LTM memory: {ltm_memories[0]['memory_id']} - Importance: {ltm_memories[0]['metadata'].get('importance_score', 'N/A')}",
            )
    except Exception as e:
        log_print(logger, f"Error getting LTM memories: {str(e)}")

    # Examine the JSON file directly
    import json

    json_path = os.path.join(
        os.path.dirname(__file__), "..", "memory_samples", MEMORY_SAMPLE
    )
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            memories = data["agents"][AGENT_ID]["memories"]
            log_print(logger, f"Memories in JSON file: {len(memories)}")
            for i, mem in enumerate(memories[:3]):  # Show first 3 memories
                log_print(logger, f"Memory {i+1} from JSON: {mem['memory_id']}")
                log_print(
                    logger,
                    f"  Metadata importance: {mem['metadata'].get('importance_score', mem['metadata'].get('importance', 'N/A'))}",
                )
                log_print(
                    logger,
                    f"  Memory tier: {mem['metadata'].get('current_tier', 'N/A')}",
                )
    except Exception as e:
        log_print(logger, f"Error reading JSON file: {str(e)}")

    # Setup search strategy
    search_strategy = ImportanceStrategy(
        agent.stm_store, agent.im_store, agent.ltm_store
    )

    # Print strategy info
    log_print(logger, f"Testing search strategy: {search_strategy.name()}")
    log_print(logger, f"Description: {search_strategy.description()}")

    # Test 1: Basic importance threshold search
    run_test(
        search_strategy,
        "Basic Importance Threshold Search",
        0.7,  # Threshold value
        AGENT_ID,
    )

    # Test 2: Min/max importance range
    run_test(
        search_strategy,
        "Min/Max Importance Range Search",
        {"min_importance": 0.3, "max_importance": 0.7},
        AGENT_ID,
    )

    # Test 3: Top N results
    run_test(
        search_strategy,
        "Top N Results Search",
        {"top_n": 3},
        AGENT_ID,
    )

    # Test 4: Ascending sort order
    run_test(
        search_strategy,
        "Ascending Sort Order",
        0.5,  # Threshold value
        AGENT_ID,
        sort_order="asc",
    )

    # Test 5: Search in specific memory tier
    run_test(
        search_strategy,
        "Search in STM Tier Only",
        0.5,  # Threshold value
        AGENT_ID,
        tier="stm",
    )

    # Test 6: Search with metadata filter
    run_test(
        search_strategy,
        "Search with Metadata Filter",
        0.5,  # Threshold value
        AGENT_ID,
        metadata_filter={"tags": "important"},
    )

    # Test 7: Complex query with min/max and metadata
    run_test(
        search_strategy,
        "Complex Query with Min/Max and Metadata",
        {"min_importance": 0.6, "max_importance": 0.9},
        AGENT_ID,
        metadata_filter={"type": "meeting"},
    )

    # Test 8: Zero importance threshold
    run_test(
        search_strategy,
        "Zero Importance Threshold",
        0.0,
        AGENT_ID,
    )

    # Test 9: High importance threshold (likely no results)
    run_test(
        search_strategy,
        "High Importance Threshold",
        0.95,
        AGENT_ID,
    )

    # Test 10: Top N with min importance
    run_test(
        search_strategy,
        "Top N with Min Importance",
        {"top_n": 5, "min_importance": 0.6},
        AGENT_ID,
    )

    # Test 11: Search in multiple tiers
    run_test(
        search_strategy,
        "Search in Multiple Tiers",
        0.5,  # Threshold value
        AGENT_ID,
        # No tier specified means all tiers
    )

    # Test 12: Large limit test
    run_test(
        search_strategy,
        "Large Limit Test",
        0.1,  # Low threshold to get many results
        AGENT_ID,
        limit=20,
    )

    # Test 13: Empty results handling (too high threshold)
    run_test(
        search_strategy,
        "Empty Results Handling",
        1.1,  # Threshold above 1.0
        AGENT_ID,
    )


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging("validate_importance_search")
    log_print(logger, "Starting Importance Search Strategy Validation")

    validate_importance_search()

    log_print(logger, "Validation Complete")
