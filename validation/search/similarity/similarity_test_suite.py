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

from memory.search.strategies.similarity import SimilaritySearchStrategy
from validation.framework.test_suite import TestSuite


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
            "test-agent-similarity-search-15": "o5p6q7r8s9t0u1v2w3x4",
        }

        # Initialize base class with the strategy factory
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=SimilaritySearchStrategy,  # Pass the strategy class directly
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
            min_score=0.5,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search with state dictionary
        self.runner.run_test(
            "Dictionary Query Similarity Search",
            {"content": "data processing pipeline"},
            expected_memory_ids=[
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-14",
            ],
            min_score=0.5,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search with metadata filter
        self.runner.run_test(
            "Search with Metadata Filter",
            "experiment results",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-13",
            ],
            min_score=0.2,
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
            min_score=0.4,
            tier="stm",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Specific memory tier search - IM
        self.runner.run_test(
            "IM Tier Search",
            "deep learning model",
            expected_memory_ids=["test-agent-similarity-search-15"],
            min_score=0.4,
            tier="im",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Result limit test
        self.runner.run_test(
            "Limited Results Search",
            "machine learning model accuracy",
            expected_memory_ids=["test-agent-similarity-search-1"],
            min_score=0.4,
            limit=1,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for similarity search."""
        # Test 1: Search with minimum similarity score threshold
        self.runner.run_test(
            "Minimum Score Threshold Search",
            "security anomaly detection",
            expected_memory_ids=["test-agent-similarity-search-11"],
            min_score=0.4,
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
                "test-agent-similarity-search-12",
            ],
            tier=None,  # Search all tiers
            min_score=0.3,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Combined metadata filter with tier filtering
        self.runner.run_test(
            "Combined Filter and Tier Search",
            "data processing pipeline",
            expected_memory_ids=[
                "test-agent-similarity-search-3",
                "test-agent-similarity-search-14",
                "test-agent-similarity-search-9",
            ],
            tier="stm",
            metadata_filter={"type": "process"},
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search with combined high threshold and limit
        self.runner.run_test(
            "High Threshold Limited Search",
            "security anomaly detection",
            expected_memory_ids=["test-agent-similarity-search-11"],
            min_score=0.7,
            limit=1,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search with high importance metadata filter
        self.runner.run_test(
            "High Importance Filter Search",
            "machine learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
            ],
            metadata_filter={"importance_score": 0.9},
            min_score=0.4,
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
            min_score=0.4,
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
                "test-agent-similarity-search-12",
                "test-agent-similarity-search-13",
                "test-agent-similarity-search-7",
            ],
            memory_checksum_map=self.memory_checksum_map,
            min_score=0.265,
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
                "test-agent-similarity-search-9",
                "test-agent-similarity-search-11",
                "test-agent-similarity-search-12",
                "test-agent-similarity-search-15",
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
            min_score=0.35,  # Adjusted min_score to filter out unexpected matches
            limit=1,  # Limit results to just one
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_memory_tier_transition_tests(self) -> None:
        """Run tests for memory tier transition scenarios."""
        # Test 1: Memory in transition between tiers
        self.runner.run_test(
            "Memory in Tier Transition",
            "deep learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-15",
                "test-agent-similarity-search-6",
                "test-agent-similarity-search-2",
            ],
            tier=None,  # Search all tiers
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Memory recently moved to new tier
        self.runner.run_test(
            "Recently Moved Memory",
            "transformer model",
            expected_memory_ids=["test-agent-similarity-search-15"],
            tier="im",  # Search in new tier
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_metadata_filtering_tests(self) -> None:
        """Run tests for complex metadata filtering scenarios."""
        # Test 1: Multiple metadata conditions
        self.runner.run_test(
            "Multiple Metadata Conditions",
            "machine learning",
            expected_memory_ids=["test-agent-similarity-search-1"],
            metadata_filter={
                "type": "experiment",
                "importance": "high",
                "metrics.accuracy": {"$gt": 0.9},
            },
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Nested metadata filtering
        self.runner.run_test(
            "Nested Metadata Filter",
            "data processing",
            expected_memory_ids=["test-agent-similarity-search-3"],
            metadata_filter={"memory_type": "process", "importance_score": 0.7},
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Array metadata filtering
        self.runner.run_test(
            "Array Metadata Filter",
            "machine learning",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-6",
            ],
            metadata_filter={"memory_type": "experiment"},
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_content_structure_tests(self) -> None:
        """Run tests for different content structure scenarios."""
        # Test 1: Nested content structure
        self.runner.run_test(
            "Nested Content Structure",
            "performance metrics",
            expected_memory_ids=["test-agent-similarity-search-4"],
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Special characters in content
        self.runner.run_test(
            "Special Characters Content",
            "model optimization & performance!",
            expected_memory_ids=["test-agent-similarity-search-12"],
            min_score=0.35,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Mixed content types
        self.runner.run_test(
            "Mixed Content Types",
            "data validation pipeline",
            expected_memory_ids=["test-agent-similarity-search-14"],
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_memory_state_tests(self) -> None:
        """Run tests for different memory states."""

        # Test 1: Different importance scores
        self.runner.run_test(
            "High Importance Memory Search",
            "machine learning model",
            expected_memory_ids=[
                "test-agent-similarity-search-1",
                "test-agent-similarity-search-2",
                "test-agent-similarity-search-6",
            ],
            metadata_filter={"importance_score": {"$gt": 0.9}},
            min_score=0.4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Different retrieval counts
        self.runner.run_test(
            "Frequently Retrieved Memory",
            "deep learning model",
            expected_memory_ids=["test-agent-similarity-search-6"],
            metadata_filter={"retrieval_count": {"$gt": 0}},
            min_score=0.2,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_all_tests(self) -> None:
        """Run all test suites."""
        self.run_basic_tests()
        self.run_advanced_tests()
        self.run_edge_case_tests()
        self.run_memory_tier_transition_tests()
        self.run_metadata_filtering_tests()
        self.run_content_structure_tests()
        self.run_memory_state_tests()

        # Display summary of all test results
        self.runner.display_summary()


def main():
    """Run the similarity search test suite."""
    test_suite = SimilaritySearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
