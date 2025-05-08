"""
Test Suite Implementation for Step Search Strategy.

This module demonstrates how to use the validation framework
to create a complete test suite for the step search strategy.
"""

import os
import sys
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.step import StepBasedSearchStrategy
from validation.framework.test_suite import TestSuite


class StepSearchTestSuite(TestSuite):
    """Test suite for StepBasedSearchStrategy."""

    def __init__(self, logger=None):
        """Initialize the step search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "step_based"
        AGENT_ID = "test-agent-step-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "step_validation_memory.json"
        )

        # Define mapping between memory IDs and checksums
        MEMORY_CHECKSUMS = {
            "test-agent-step-search-stm-1": "a1b2c3d4e5f6g7h8i9j0",
            "test-agent-step-search-stm-2": "b2c3d4e5f6g7h8i9j0k1",
            "test-agent-step-search-im-1": "c3d4e5f6g7h8i9j0k1l2",
            "test-agent-step-search-im-2": "d4e5f6g7h8i9j0k1l2m3",
            "test-agent-step-search-ltm-1": "e5f6g7h8i9j0k1l2m3n4",
            "test-agent-step-search-ltm-2": "f6g7h8i9j0k1l2m3n4o5"
        }

        # Initialize base class
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=StepBasedSearchStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for step search."""
        # Test 1: Basic step range search
        self.runner.run_test(
            "Basic Step Range Search",
            {"start_step": 100, "end_step": 200},
            expected_memory_ids=[
                "test-agent-step-search-stm-1",
                "test-agent-step-search-stm-2"
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Reference step search
        self.runner.run_test(
            "Reference Step Search",
            "200",
            expected_memory_ids=[
                "test-agent-step-search-im-1",
                "test-agent-step-search-stm-2",
                "test-agent-step-search-im-2"
            ],
            step_range=50,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search by tier
        self.runner.run_test(
            "Search by Tier",
            {"start_step": 200, "end_step": 300},
            expected_memory_ids=[
                "test-agent-step-search-im-1",
                "test-agent-step-search-im-2"
            ],
            tier="im",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search with metadata filter
        self.runner.run_test(
            "Search with Metadata Filter",
            {"start_step": 100, "end_step": 400},
            expected_memory_ids=[
                "test-agent-step-search-stm-1",
                "test-agent-step-search-stm-2",
                "test-agent-step-search-ltm-1",
                "test-agent-step-search-ltm-2"
            ],
            metadata_filter={"content.metadata.importance": "high"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search with step weight
        self.runner.run_test(
            "Search with Step Weight",
            "200",
            expected_memory_ids=[
                "test-agent-step-search-im-1",
                "test-agent-step-search-stm-2",
                "test-agent-step-search-im-2"
            ],
            step_range=50,
            step_weight=2.0,
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for step search."""
        # Test 1: Multi-tier search with step range
        self.runner.run_test(
            "Multi-Tier Step Range Search",
            {"start_step": 150, "end_step": 250},
            expected_memory_ids=[
                "test-agent-step-search-stm-2",
                "test-agent-step-search-im-1",
                "test-agent-step-search-im-2"
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Step proximity scoring
        self.runner.run_test(
            "Step Proximity Scoring",
            "225",
            expected_memory_ids=[
                "test-agent-step-search-im-2",
                "test-agent-step-search-im-1",
                "test-agent-step-search-stm-2"
            ],
            step_range=50,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Combined step and metadata filtering
        self.runner.run_test(
            "Combined Step and Metadata Filtering",
            {"start_step": 200, "end_step": 350},
            expected_memory_ids=[
                "test-agent-step-search-ltm-1",
                "test-agent-step-search-ltm-2"
            ],
            metadata_filter={"content.metadata.type": "system", "content.metadata.importance": "high"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Step range with tier-specific importance
        self.runner.run_test(
            "Step Range with Tier-Specific Importance",
            {"start_step": 100, "end_step": 300},
            expected_memory_ids=[
                "test-agent-step-search-stm-1",
                "test-agent-step-search-stm-2",
                "test-agent-step-search-ltm-1"
            ],
            metadata_filter={"content.metadata.importance": "high"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Step range with content filtering
        self.runner.run_test(
            "Step Range with Content Filtering",
            {"start_step": 200, "end_step": 300},
            expected_memory_ids=[
                "test-agent-step-search-im-1",
                "test-agent-step-search-im-2"
            ],
            metadata_filter={"content.metadata.tags": ["database", "api"]},
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for step search."""
        # Test 1: Empty step range
        self.runner.run_test(
            "Empty Step Range",
            {"start_step": 100, "end_step": 100},
            expected_memory_ids=["test-agent-step-search-stm-1"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Invalid step range (end < start)
        self.runner.run_test(
            "Invalid Step Range",
            {"start_step": 200, "end_step": 100},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Negative step numbers
        self.runner.run_test(
            "Negative Step Numbers",
            {"start_step": -50, "end_step": 100},
            expected_memory_ids=["test-agent-step-search-stm-1"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Large step range
        self.runner.run_test(
            "Large Step Range",
            {"start_step": 0, "end_step": 1000},
            expected_memory_ids=[
                "test-agent-step-search-stm-1",
                "test-agent-step-search-stm-2",
                "test-agent-step-search-im-1",
                "test-agent-step-search-im-2",
                "test-agent-step-search-ltm-1",
                "test-agent-step-search-ltm-2"
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Zero step range
        self.runner.run_test(
            "Zero Step Range",
            {"start_step": 0, "end_step": 0},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Invalid tier
        self.runner.run_test(
            "Invalid Tier",
            {"start_step": 100, "end_step": 200},
            expected_memory_ids=[],
            tier="invalid_tier",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Step range with non-existent metadata
        self.runner.run_test(
            "Step Range with Non-existent Metadata",
            {"start_step": 100, "end_step": 200},
            expected_memory_ids=[],
            metadata_filter={"content.metadata.non_existent": "value"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 8: Step range with invalid metadata value
        self.runner.run_test(
            "Step Range with Invalid Metadata Value",
            {"start_step": 100, "end_step": 200},
            expected_memory_ids=[],
            metadata_filter={"content.metadata.importance": "invalid_value"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 9: Step range with empty metadata filter
        self.runner.run_test(
            "Step Range with Empty Metadata Filter",
            {"start_step": 100, "end_step": 200},
            expected_memory_ids=[
                "test-agent-step-search-stm-1",
                "test-agent-step-search-stm-2"
            ],
            metadata_filter={},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 10: Step range with null values
        self.runner.run_test(
            "Step Range with Null Values",
            {"start_step": None, "end_step": None},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )


def main():
    """Run the step search test suite."""
    test_suite = StepSearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main() 