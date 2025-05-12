"""
Debug script to analyze similarity scores across all memories in the validation dataset.
"""

import json
import logging
import os
import sys
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.vector_store import VectorStore
from memory.search.strategies.similarity import SimilaritySearchStrategy
from memory.storage.redis_im import RedisIMStore
from memory.storage.redis_stm import RedisSTMStore
from memory.storage.sqlite_ltm import SQLiteLTMStore
from validation.demo_utils import create_memory_system, setup_logging

# Configure logging
logger = setup_logging("similarity_debug")

def load_validation_memory() -> Dict:
    """Load the validation memory JSON file."""
    memory_path = os.path.join(
        "validation", "memory_samples", "similarity_validation_memory.json"
    )
    with open(memory_path, 'r') as f:
        return json.load(f)

def analyze_similarity_scores(query: str = "machine learning model accuracy"):
    """Analyze similarity scores for all memories against the given query."""
    # Create memory system with embeddings enabled
    memory_system = create_memory_system(
        use_embeddings=True,
        embedding_type="text",
        use_mock_redis=True,
        memory_file="similarity_validation_memory.json",
        clear_db=True
    )
    
    agent_id = "test-agent-similarity-search"
    
    # Create required dependencies for SimilaritySearchStrategy
    vector_store = VectorStore()
    embedding_engine = TextEmbeddingEngine(
        model_name="all-MiniLM-L6-v2"
    )  # Using a smaller model for testing
    
    # Get the storage components from the memory system
    stm_store = memory_system.get_memory_agent(agent_id).stm_store
    im_store = memory_system.get_memory_agent(agent_id).im_store
    ltm_store = memory_system.get_memory_agent(agent_id).ltm_store
    
    # Load validation memory for reference
    memory_data = load_validation_memory()
    
    # Get all memories
    memories = memory_data["agents"][agent_id]["memories"]
    
    # Store vectors for each memory
    logger.info("Storing vectors for %d memories", len(memories))
    for memory in memories:
        # Generate embeddings for the memory content
        content = memory["content"]["content"]
        embeddings = {
            "full_vector": embedding_engine.encode_stm({"content": content}),
            "compressed_vector": embedding_engine.encode_im({"content": content}),
            "abstract_vector": embedding_engine.encode_ltm({"content": content})
        }
        
        # Add embeddings to memory
        memory["embeddings"] = embeddings
        
        # Store vectors in vector store
        vector_store.store_memory_vectors(memory)
    
    # Create similarity search strategy
    similarity_strategy = SimilaritySearchStrategy(
        vector_store=vector_store,
        embedding_engine=embedding_engine,
        stm_store=stm_store,
        im_store=im_store,
        ltm_store=ltm_store,
        config=memory_system.config
    )
    
    logger.info(f"Analyzing similarity scores for query: '{query}'")
    logger.info(f"Total memories to analyze: {len(memories)}")
    
    # Use similarity strategy directly
    results = similarity_strategy.search(
        query=query,
        agent_id=agent_id,
        limit=len(memories),
        min_score=0.0
    )
    
    # Create a mapping of memory_id to content for reference
    memory_map = {m["memory_id"]: m["content"]["content"] for m in memories}
    
    # Print results in a formatted way
    logger.info("\nSimilarity Score Analysis:")
    logger.info("-" * 80)
    logger.info(f"{'Memory ID':<30} {'Score':<10} {'Content Preview':<40}")
    logger.info("-" * 80)
    
    for result in results:
        memory_id = result["memory_id"]
        score = result.get("metadata", {}).get("similarity_score", 0.0)
        content = memory_map.get(memory_id, "Content not found")
        content_preview = content[:37] + "..." if len(content) > 40 else content
        
        logger.info(f"{memory_id:<30} {score:<10.4f} {content_preview:<40}")
    
    # Print summary statistics
    if results:
        scores = [r.get("metadata", {}).get("similarity_score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        logger.info("\nSummary Statistics:")
        logger.info(f"Average Score: {avg_score:.4f}")
        logger.info(f"Highest Score: {max_score:.4f}")
        logger.info(f"Lowest Score: {min_score:.4f}")
        logger.info(f"Total Memories with Scores: {len(results)}")
    else:
        logger.info("\nNo results found with scores above threshold.")

def main():
    """Run the similarity score analysis."""
    analyze_similarity_scores()

if __name__ == "__main__":
    main() 