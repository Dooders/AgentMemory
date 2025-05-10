#!/usr/bin/env python
"""Example demonstrating how to use the AttributeQueryBuilder.

This example shows how the AttributeQueryBuilder can be used to create
complex queries for the AttributeSearchStrategy in a more intuitive way.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.search.strategies.query_builder import AttributeQueryBuilder
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.redis_im import RedisIMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore


def main():
    """Run query builder examples."""
    # Setup stores and strategy
    # Note: You would replace these with your actual store instances
    stm_store = RedisSTMStore(host="localhost", port=6379, db=0)
    im_store = RedisIMStore(host="localhost", port=6379, db=1)
    ltm_store = SQLiteLTMStore("memory.db")
    
    # Create search strategy
    strategy = AttributeSearchStrategy(stm_store, im_store, ltm_store)
    
    # Define agent ID for examples
    agent_id = "example-agent"
    
    print("AttributeQueryBuilder Examples\n")
    
    # Example 1: Simple content search
    print("Example 1: Simple content search")
    print("--------------------------------")
    query, kwargs = (AttributeQueryBuilder()
        .content("meeting")
        .limit(5)
        .build())
    
    print(f"Query: {query}")
    print(f"Parameters: {kwargs}")
    print("")
    
    # Example 2: Search for high importance security notes
    print("Example 2: Search for high importance security notes")
    print("-------------------------------------------------")
    query, kwargs = (AttributeQueryBuilder()
        .content("security")
        .type("note")
        .importance("high")
        .match_all(True)
        .build())
    
    print(f"Query: {query}")
    print(f"Parameters: {kwargs}")
    print("")
    
    # Example 3: Regex search in specific tier
    print("Example 3: Regex search in specific tier")
    print("------------------------------------")
    query, kwargs = (AttributeQueryBuilder()
        .content("secur.*patch")
        .use_regex(True)
        .in_tier("ltm")
        .score_by("bm25")
        .build())
    
    print(f"Query: {query}")
    print(f"Parameters: {kwargs}")
    print("")
    
    # Example 4: Search by tag with metadata filter
    print("Example 4: Search by tag with metadata filter")
    print("----------------------------------------")
    query, kwargs = (AttributeQueryBuilder()
        .tag("project")
        .filter_metadata("content.metadata.source", "email")
        .case_sensitive(True)
        .build())
    
    print(f"Query: {query}")
    print(f"Parameters: {kwargs}")
    print("")
    
    # Example 5: In-depth example with multiple conditions
    print("Example 5: In-depth example with multiple conditions")
    print("------------------------------------------------")
    query, kwargs = (AttributeQueryBuilder()
        .content("authentication system")
        .type("task")
        .tag("development")
        .importance("high")
        .in_content_fields("content.content")
        .in_metadata_fields("content.metadata.tags", "content.metadata.importance")
        .match_all(True)
        .limit(3)
        .score_by("term_frequency")
        .build())
    
    print(f"Query: {query}")
    print(f"Parameters: {kwargs}")
    print("")
    
    print("To execute these queries against your memory stores:")
    print('results = builder.execute(strategy, "your-agent-id")')


if __name__ == "__main__":
    main() 