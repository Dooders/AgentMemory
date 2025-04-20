"""
Validation script for the Attribute Search Strategy.

This script loads a predefined memory system and tests various scenarios
of attribute-based searching to verify the strategy works correctly.
"""

import os
import sys
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore
from memory.core import AgentMemorySystem

from demos.demo_utils import (
    setup_logging,
    log_print,
    create_memory_system,
    pretty_print_memories,
)

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
) -> List[Dict[str, Any]]:
    """Run a test case and return the results."""
    log_print(logger, f"\n=== Test: {test_name} ===")
    
    if isinstance(query, dict):
        log_print(logger, f"Query (dict): {query}")
    else:
        log_print(logger, f"Query: '{query}'")
    
    log_print(logger, f"Match All: {match_all}, Case Sensitive: {case_sensitive}, Use Regex: {use_regex}")
    
    if metadata_filter:
        log_print(logger, f"Metadata Filter: {metadata_filter}")
    
    if tier:
        log_print(logger, f"Tier: {tier}")
    
    if content_fields:
        log_print(logger, f"Content Fields: {content_fields}")
    
    if metadata_fields:
        log_print(logger, f"Metadata Fields: {metadata_fields}")
    
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
    )
    
    log_print(logger, f"Found {len(results)} results")
    pretty_print_memories(results, f"Results for {test_name}", logger)
    
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
        agent.stm_store, 
        agent.im_store, 
        agent.ltm_store
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

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging("validate_attribute_search")
    log_print(logger, "Starting Attribute Search Strategy Validation")
    
    validate_attribute_search()
    
    log_print(logger, "Validation Complete")
