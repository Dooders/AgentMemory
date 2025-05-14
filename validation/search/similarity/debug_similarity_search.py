"""
Debug script for testing the SimilaritySearchStrategy.

This script sets up a memory system with test data and allows interactive testing
of the similarity search strategy.
"""

import os
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.core import AgentMemorySystem
from memory.search.strategies.similarity import SimilaritySearchStrategy
from validation.demo_utils import (
    create_memory_system,
    setup_logging,
    pretty_print_memories,
    print_memory_details,
)


# Create memory system with embeddings enabled
memory_system = create_memory_system(
    use_embeddings=True,
    embedding_type="text",
    memory_file="validation/memory_samples/similarity_validation_memory.json",
    clear_db=True
)

# Initialize the similarity search strategy
search_strategy = SimilaritySearchStrategy(memory_system)

# Test agent ID
agent_id = "test-agent-similarity-search"

# Print initial memory state
print_memory_details(memory_system, agent_id, "Initial Memory State")

# Example search queries to try
test_queries = [
    "machine learning model accuracy",
    {"content": "data processing pipeline"},
    "security anomaly detection",
    "model optimization",
    "deep learning model",
]

# Run some example searches
for i, query in enumerate(test_queries, 1):
    print(f"\nTest Query {i}:")
    print(f"Query: {query}")
    
    # Perform search
    results = search_strategy.search(
        query=query,
        agent_id=agent_id,
        limit=5,
        min_score=0.6
    )
    
    # Print results
    pretty_print_memories(results, f"Search Results for Query {i}")
    
    # Print similarity scores
    if results:
        print("\nSimilarity Scores:")
        for result in results:
            score = result.get("metadata", {}).get("similarity_score", 0.0)
            print(f"Memory {result['memory_id']}: {score:.3f}")