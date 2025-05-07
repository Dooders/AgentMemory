"""
Test Suite Implementation for Sequence Search Strategy.

This module implements validation tests for the narrative sequence search strategy,
which retrieves a sequence of memories surrounding a reference memory.
"""

import os
import sys
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.sequence import NarrativeSequenceStrategy
from validation.framework.test_suite import TestSuite


class SequenceSearchTestSuite(TestSuite):
    """Test suite for NarrativeSequenceStrategy."""

    def __init__(self, logger=None):
        """Initialize the sequence search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "sequence"
        AGENT_ID = "test-agent-sequence-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "sequence_validation_memory.json"
        )

        # Memory ID to checksum mapping
        MEMORY_CHECKSUMS = {
            "test-agent-sequence-search-meeting-123456-1": "0eb0f81d07276f08e05351a604d3c994564fedee3a93329e318186da517a3c56",
            "test-agent-sequence-search-meeting-123456-2": "f6ab36930459e74a52fdf21fb96a84241ccae3f6987365a21f9a17d84c5dae1e",
            "test-agent-sequence-search-meeting-123456-3": "ffa0ee60ebaec5574358a02d1857823e948519244e366757235bf755c888a87f",
            "test-agent-sequence-search-meeting-123456-4": "9214ebc2d11877665b32771bd3c080414d9519b435ec3f6c19cc5f337bb0ba90",
            "test-agent-sequence-search-meeting-123456-5": "ad2e7c963751beb1ebc1c9b84ecb09ec3ccdef14f276cd14bbebad12d0f9b0df",
            "test-agent-sequence-search-task-123456-1": "e0f7deb6929a17f65f56e5b03e16067c8bb65649fd2745f842aca7af701c9cac",
            "test-agent-sequence-search-task-123456-2": "1d23b6683acd8c3863cb2f2010fe3df2c3e69a2d94c7c4757a291d4872066cfd",
            "test-agent-sequence-search-task-123456-3": "f3c73b06d6399ed30ea9d9ad7c711a86dd58154809cc05497f8955425ec6dc67",
            "test-agent-sequence-search-note-123456-1": "1e9e265e75c2ef678dfd0de0ab5c801f845daa48a90a48bb02ee85148ccc3470",
            "test-agent-sequence-search-note-123456-2": "169c452e368fd62e3c0cf5ce7731769ed46ab6ae73e5048e0c3a7caaa66fba46",
        }

        # Initialize base class
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=NarrativeSequenceStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for sequence search."""
        # Test 1: Basic sequence search with default parameters
        self.runner.run_test(
            "Basic Sequence Search",
            "test-agent-sequence-search-meeting-123456-3",  # Reference memory ID
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-1",
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
                "test-agent-sequence-search-meeting-123456-5",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Sequence search with custom size
        self.runner.run_test(
            "Custom Sequence Size",
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "sequence_size": 3,
            },
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Sequence search with before/after counts
        self.runner.run_test(
            "Before/After Counts",
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "before_count": 2,
                "after_count": 1,
            },
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-1",
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for sequence search."""
        # Test 1: Time window based sequence
        self.runner.run_test(
            "Time Window Sequence",
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "time_window_minutes": 30,
            },
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Multi-tier search
        self.runner.run_test(
            "Multi-Tier Search",
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "tier": "stm",
            },
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Metadata filtering
        self.runner.run_test(
            "Metadata Filtering",
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "metadata_filter": {"type": "meeting"},
            },
            expected_memory_ids=[
                "test-agent-sequence-search-meeting-123456-1",
                "test-agent-sequence-search-meeting-123456-2",
                "test-agent-sequence-search-meeting-123456-3",  # Reference memory
                "test-agent-sequence-search-meeting-123456-4",
                "test-agent-sequence-search-meeting-123456-5",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for sequence search."""
        # Test 1: Non-existent reference memory
        self.runner.run_exception_test(
            "Non-existent Reference Memory",
            ValueError,
            self.runner.strategy.search,
            "non-existent-id",
            self.agent_id,
        )

        # Test 2: Invalid sequence size
        self.runner.run_exception_test(
            "Invalid Sequence Size",
            ValueError,
            self.runner.strategy.search,
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "sequence_size": 0,
            },
            self.agent_id,
        )

        # Test 3: Invalid time window
        self.runner.run_exception_test(
            "Invalid Time Window",
            ValueError,
            self.runner.strategy.search,
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "time_window_minutes": -10,
            },
            self.agent_id,
        )

        # Test 4: Conflicting parameters
        self.runner.run_exception_test(
            "Conflicting Parameters",
            ValueError,
            self.runner.strategy.search,
            {
                "reference_id": "test-agent-sequence-search-meeting-123456-3",
                "sequence_size": 5,
                "before_count": 2,
            },
            self.agent_id,
        )

        # Test 5: Missing reference ID
        self.runner.run_exception_test(
            "Missing Reference ID",
            ValueError,
            self.runner.strategy.search,
            {},
            self.agent_id,
        )


def main():
    """Run the sequence search test suite."""
    test_suite = SequenceSearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
