"""
Test Suite Implementation for Similarity Search Strategy.

This module demonstrates how to use the validation framework
to create a complete test suite for the similarity search strategy.
"""

import os
import sys
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.vector_store import VectorStore
from memory.search.strategies.similarity import SimilaritySearchStrategy
from validation.framework.test_suite import TestSuite
from memory.config import MemoryConfig


class SimilaritySearchTestSuite(TestSuite):
    """Test suite for SimilaritySearchStrategy."""

    def __init__(self, logger=None):
        """Initialize the similarity search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "similarity"
        AGENT_ID = "test-agent-similarity-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "similarity_validation_memory.json"
        )

        # Define mapping between memory IDs and checksums
        MEMORY_CHECKSUMS = {
            "test-agent-similarity-search-1": "a1b2c3d4e5f6g7h8i9j0",
            "test-agent-similarity-search-2": "b2c3d4e5f6g7h8i9j0k1",
            "test-agent-similarity-search-3": "c3d4e5f6g7h8i9j0k1l2",
            "test-agent-similarity-search-4": "d4e5f6g7h8i9j0k1l2m3",
            "test-agent-similarity-search-5": "e5f6g7h8i9j0k1l2m3n4",
            "test-agent-similarity-search-6": "f6g7h8i9j0k1l2m3n4o5",
            "test-agent-similarity-search-7": "g7h8i9j0k1l2m3n4o5p6",
            "test-agent-similarity-search-8": "h8i9j0k1l2m3n4o5p6q7",
            "test-agent-similarity-search-9": "i9j0k1l2m3n4o5p6q7r8",
            "test-agent-similarity-search-10": "j0k1l2m3n4o5p6q7r8s9",
            "test-agent-similarity-search-11": "k1l2m3n4o5p6q7r8s9t0",
            "test-agent-similarity-search-12": "l2m3n4o5p6q7r8s9t0u1",
            "test-agent-similarity-search-13": "m3n4o5p6q7r8s9t0u1v2",
            "test-agent-similarity-search-14": "n4o5p6q7r8s9t0u1v2w3",
        }

        # Create required dependencies for SimilaritySearchStrategy
        self.vector_store = VectorStore()
        self.embedding_engine = TextEmbeddingEngine(
            model_name="all-MiniLM-L6-v2"
        )  # Using a smaller model for testing

        # Create a strategy factory function that will be used by the test runner
        #! pass the memory system to the strategy
        def create_strategy(stm_store, im_store, ltm_store):
            return SimilaritySearchStrategy(
                vector_store=self.vector_store,
                embedding_engine=self.embedding_engine,
                stm_store=stm_store,
                im_store=im_store,
                ltm_store=ltm_store,
                config=MemoryConfig()
            )

        # Initialize base class with the strategy factory
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=create_strategy,  # Pass the factory function instead of the class
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for similarity search."""
        # Test 1: Basic similarity search with text query
        self.runner.run_test(
            "Basic Text Query Similarity Search",
            "machine learning model accuracy",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-6",
            ],
            min_score=0.6,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search with state dictionary
        self.runner.run_test(
            "Dictionary Query Similarity Search",
            {"content": "data processing pipeline"},
            expected_memory_ids=[
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-9",
            ],
            min_score=0.6,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search with metadata filter
        self.runner.run_test(
            "Search with Metadata Filter",
            "experiment results",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-6",
            ],
            min_score=0.6,
            metadata_filter={"type": "experiment"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Specific memory tier search - STM
        self.runner.run_test(
            "STM Tier Search",
            "performance optimization",
            expected_memory_ids=[
                "test-agent-similarity-search-12",
            ],
            min_score=0.6,
            tier="stm",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Specific memory tier search - IM
        self.runner.run_test(
            "IM Tier Search",
            "deep learning model",
            expected_memory_ids=["test-agent-similarity-search-6"],
            min_score=0.6,
            tier="im",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Result limit test
        self.runner.run_test(
            "Limited Results Search",
            "data",
            expected_memory_ids=[
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-9",
            ],
            min_score=0.6,
            limit=3,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for similarity search."""
        # Test 1: Search with minimum similarity score threshold
        self.runner.run_test(
            "Minimum Score Threshold Search",
            "security anomaly detection",
            expected_memory_ids=["test-agent-similarity-search-11"],
            min_score=0.8,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Multi-tier search (searches across all tiers)
        self.runner.run_test(
            "Multi-Tier Search",
            "machine learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-6",
                "test-agent-similarity-search-7",
                "test-agent-similarity-search-8",
            ],
            tier=None,  # Search all tiers
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Combined metadata filter with tier filtering
        self.runner.run_test(
            "Combined Filter and Tier Search",
            "data processing pipeline",
            expected_memory_ids=[
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-14",
            ],
            tier="stm",
            metadata_filter={"type": "process"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search with vector directly (instead of text or dictionary)
        # This would require getting a vector from somewhere - usually we'd mock this
        # Here we're assuming we have a test vector that matches certain memories
        test_vector = [0.1] * 384  # Mock vector for testing purposes
        self.runner.run_test(
            "Direct Vector Search",
            test_vector,
            expected_memory_ids=["test-agent-similarity-search-1"],  # Placeholder
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search with combined high threshold and limit
        self.runner.run_test(
            "High Threshold Limited Search",
            "model deployment pipeline",
            expected_memory_ids=["test-agent-similarity-search-10"],
            min_score=0.9,
            limit=1,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Search with high importance metadata filter
        self.runner.run_test(
            "High Importance Filter Search",
            "machine learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-6",
            ],
            metadata_filter={"importance": "high"},
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for similarity search."""
        # Test 1: Empty query handling
        self.runner.run_test(
            "Empty Query",
            "",
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Query with no semantic matches above threshold
        self.runner.run_test(
            "No Matches Above Threshold",
            "completely unrelated topic that has no semantic connection",
            expected_memory_ids=[],
            min_score=0.9,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Single word query
        self.runner.run_test(
            "Single Word Query",
            "security",
            expected_memory_ids=["test-agent-similarity-search-11"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Very long query
        long_query = "This is an extremely long query that contains many words and should test the system's ability to handle verbose input with many potential semantic connections to existing memories including machine learning models, data processing pipelines, performance optimization, security systems, and other technical concepts that might be found in the memory store"
        self.runner.run_test(
            "Very Long Query",
            long_query,
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-4",
                "test-agent-similarity-search-12",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Empty dictionary query
        self.runner.run_test(
            "Empty Dictionary Query",
            {},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Non-existent tier
        self.runner.run_test(
            "Non-existent Tier",
            "machine learning model",
            expected_memory_ids=[],
            tier="non_existent_tier",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Zero limit
        self.runner.run_test(
            "Zero Result Limit",
            "machine learning model",
            expected_memory_ids=[],
            limit=0,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 8: Extreme min_score values
        self.runner.run_test(
            "Perfect Match Score Threshold",
            "machine learning model",
            expected_memory_ids=[],
            min_score=1.0,  # Perfect match required
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 9: Very small min_score
        self.runner.run_test(
            "Very Low Score Threshold",
            "machine learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-6",
                "test-agent-similarity-search-7",
                "test-agent-similarity-search-8",
                "test-agent-similarity-search-12",
                "test-agent-similarity-search-13",
                "test-agent-similarity-search-14",
            ],
            min_score=0.1,  # Very low threshold
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 10: Query with special characters
        self.runner.run_test(
            "Special Characters Query",
            "model optimization & performance! @#$%^",
            expected_memory_ids=["test-agent-similarity-search-12"],
            memory_checksum_map=self.memory_checksum_map,
        )


def main():
    """Run the similarity search test suite."""
    test_suite = SimilaritySearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
