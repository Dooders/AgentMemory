"""
Test Suite Implementation for Attribute Search Strategy.

This module demonstrates how to use the validation framework
to create a complete test suite for the attribute search strategy.
"""

import os
import sys
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.attribute import AttributeSearchStrategy
from validation.framework.test_suite import TestSuite


class AttributeSearchTestSuite(TestSuite):
    """Test suite for AttributeSearchStrategy."""

    def __init__(self, logger=None):
        """Initialize the attribute search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "attribute"
        AGENT_ID = "test-agent-attribute-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "attribute_validation_memory.json"
        )

        # Define mapping between memory IDs and checksums
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

        # Initialize base class
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=AttributeSearchStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for attribute search."""
        # Test 1: Basic content search
        self.runner.run_test(
            "Basic Content Search",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            content_fields=["content.content"],
            metadata_filter={"content.metadata.type": "meeting"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Case sensitive search
        self.runner.run_test(
            "Case Sensitive Search",
            "Meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
            ],
            case_sensitive=True,
            content_fields=["content.content"],
            metadata_filter={"content.metadata.type": "meeting"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search by metadata type
        self.runner.run_test(
            "Search by Metadata Type",
            {"metadata": {"type": "meeting"}},
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search in specific content fields
        self.runner.run_test(
            "Search in Specific Content Fields",
            "project",
            expected_memory_ids=["meeting-123456-1", "contact-123456-5"],
            content_fields=["content.content"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search in specific metadata fields
        self.runner.run_test(
            "Search in Specific Metadata Fields",
            "project",
            expected_memory_ids=["meeting-123456-1", "contact-123456-5"],
            metadata_fields=["content.metadata.tags"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Search with metadata filter
        self.runner.run_test(
            "Search with Metadata Filter",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            metadata_filter={"content.metadata.importance": "high"},
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for attribute search."""
        # Test 1: Search with match_all
        self.runner.run_test(
            "Search with Match All",
            {
                "content": "meeting",
                "metadata": {"type": "meeting", "importance": "high"},
            },
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            match_all=True,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search specific memory tier
        self.runner.run_test(
            "Search in STM Tier Only",
            "meeting",
            expected_memory_ids=["meeting-123456-1", "meeting-123456-3"],
            tier="stm",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search with regex
        self.runner.run_test(
            "Regex Search",
            "secur.*patch",
            expected_memory_ids=["note-123456-4"],
            use_regex=True,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Complex search with combined criteria
        self.runner.run_test(
            "Complex Search",
            {"content": "security", "metadata": {"importance": "high"}},
            expected_memory_ids=["note-123456-4"],
            metadata_filter={"content.metadata.source": "email"},
            match_all=True,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Array field partial matching
        self.runner.run_test(
            "Array Field Partial Matching",
            "dev",
            expected_memory_ids=["meeting-123456-3", "task-123456-10"],
            metadata_fields=["content.metadata.tags"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Large result set limiting
        self.runner.run_test(
            "Large Result Set Limiting",
            "a",  # Common letter to match many memories
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-6",
                "meeting-123456-3",
            ],
            limit=3,  # Only show top 3 results
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Multi-tier search
        self.runner.run_test(
            "Multi-Tier Search",
            "important",
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # === Scoring Method Tests ===

        # Test 8: Default length ratio scoring
        self.runner.run_test(
            "Default Length Ratio Scoring",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            limit=5,
            scoring_method="length_ratio",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 9: Term frequency scoring
        self.runner.run_test(
            "Term Frequency Scoring",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            limit=5,
            scoring_method="term_frequency",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 10: BM25 scoring
        self.runner.run_test(
            "BM25 Scoring",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            limit=5,
            scoring_method="bm25",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 11: Binary scoring
        self.runner.run_test(
            "Binary Scoring",
            "meeting",
            expected_memory_ids=[
                "meeting-123456-1",
                "meeting-123456-3",
                "meeting-123456-6",
                "meeting-123456-9",
                "meeting-123456-11",
            ],
            limit=5,
            scoring_method="binary",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 12: Term frequency with repetition
        self.runner.run_test(
            "Term Frequency with Term Repetition",
            "security",
            expected_memory_ids=["note-123456-4", "note-123456-8", "meeting-123456-11"],
            limit=5,
            scoring_method="term_frequency",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 13: BM25 with term repetition
        self.runner.run_test(
            "BM25 with Term Repetition",
            "security",
            expected_memory_ids=["note-123456-4", "note-123456-8", "meeting-123456-11"],
            limit=5,
            scoring_method="bm25",
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for attribute search."""
        # Test 1: Empty query handling - string
        self.runner.run_test(
            "Empty String Query",
            "",
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Empty query handling - dict
        self.runner.run_test(
            "Empty Dict Query",
            {},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Numeric value search
        self.runner.run_test(
            "Numeric Value Search",
            42,
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Invalid regex pattern handling
        self.runner.run_test(
            "Invalid Regex Pattern",
            "[unclosed-bracket",
            expected_memory_ids=[],
            use_regex=True,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Boolean value search
        self.runner.run_test(
            "Boolean Value Search",
            {"metadata": {"completed": True}},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Type conversion - searching string with numeric
        self.runner.run_test(
            "Type Conversion - String Field with Numeric",
            123,
            expected_memory_ids=[],
            content_fields=["content.content"],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Special characters in search
        self.runner.run_test(
            "Special Characters in Search",
            "meeting+notes",
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test specialized strategy instances
        # Note: We need to create new strategy instances with different scoring methods
        term_freq_strategy = AttributeSearchStrategy(
            self.runner.agent.stm_store,
            self.runner.agent.im_store,
            self.runner.agent.ltm_store,
            scoring_method="term_frequency",
        )

        bm25_strategy = AttributeSearchStrategy(
            self.runner.agent.stm_store,
            self.runner.agent.im_store,
            self.runner.agent.ltm_store,
            scoring_method="bm25",
        )

        # Test 8: Using specialized term frequency strategy
        self.runner.run_test(
            "Using Term Frequency Strategy Instance",
            "project",
            expected_memory_ids=["meeting-123456-1", "contact-123456-5"],
            limit=5,
            memory_checksum_map=self.memory_checksum_map,
            _strategy_override=term_freq_strategy,  # This would require adding support in the runner
        )

        # Test 9: Using specialized BM25 strategy
        self.runner.run_test(
            "Using BM25 Strategy Instance",
            "project",
            expected_memory_ids=["meeting-123456-1", "contact-123456-5"],
            limit=5,
            memory_checksum_map=self.memory_checksum_map,
            _strategy_override=bm25_strategy,  # This would require adding support in the runner
        )

        # Long vs short document tests
        # Test 10: Length ratio for long documents
        self.runner.run_test(
            "Length Ratio for Long Documents",
            "authentication system",
            expected_memory_ids=["meeting-123456-3", "task-123456-7", "task-123456-10"],
            limit=5,
            scoring_method="length_ratio",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 11: Term Frequency for long documents
        self.runner.run_test(
            "Term Frequency for Long Documents",
            "authentication system",
            expected_memory_ids=["meeting-123456-3", "task-123456-7", "task-123456-10"],
            limit=5,
            scoring_method="term_frequency",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 12: BM25 for long documents
        self.runner.run_test(
            "BM25 for Long Documents",
            "authentication system",
            expected_memory_ids=["meeting-123456-3", "task-123456-7", "task-123456-10"],
            limit=5,
            scoring_method="bm25",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Tests for varying document length
        # Test 13: Length ratio for documentation query
        self.runner.run_test(
            "Length Ratio for Documentation Query",
            "documentation",
            expected_memory_ids=["task-123456-2", "task-123456-7"],
            limit=5,
            scoring_method="length_ratio",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 14: Term frequency for documentation query
        self.runner.run_test(
            "Term Frequency for Documentation Query",
            "documentation",
            expected_memory_ids=["task-123456-2", "task-123456-7"],
            limit=5,
            scoring_method="term_frequency",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 15: BM25 for documentation query
        self.runner.run_test(
            "BM25 for Documentation Query",
            "documentation",
            expected_memory_ids=["task-123456-2", "task-123456-7"],
            limit=5,
            scoring_method="bm25",
            memory_checksum_map=self.memory_checksum_map,
        )


def main():
    """Run the attribute search test suite."""
    test_suite = AttributeSearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
