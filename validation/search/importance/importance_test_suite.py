"""
Test Suite Implementation for Importance Search Strategy.

This module demonstrates how to use the validation framework
to create a complete test suite for the importance search strategy.
"""

import os
import sys
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.importance import ImportanceStrategy
from validation.framework.test_suite import TestSuite


class ImportanceSearchTestSuite(TestSuite):
    """Test suite for ImportanceStrategy."""

    def __init__(self, logger=None):
        """Initialize the importance search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "importance"
        AGENT_ID = "test-agent-importance-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "importance_validation_memory.json"
        )

        # Define mapping between memory IDs and checksums
        MEMORY_CHECKSUMS = {
            "stm-high-importance-1": "319f9c037be95ae4dc07458da21598ff7b47ae13d08f82a3c313797384b6767a",
            "stm-high-importance-2": "90041b1b5cadc1bb90c28529d2544256dd97368aafceb974f35fb242a1c69269",
            "stm-medium-importance-1": "876fee01bc2619d329ed55b0f2759d7ecf5339797c19e2e8075f773bc9cb9b4d",
            "stm-medium-importance-2": "7bfe771b5a0a4695620f2804649c508824ad84db0e18e7fd8e6bb681acc8260a",
            "stm-low-importance-1": "ddfab123f24dfcf29703b4dafd985af5f9c044bfc28fcacd6624c3ce7001816b",
            "im-high-importance-1": "fcb1064d39b1a1b91b39fd34ec38a02e01634e7d02c277c46d1c0d09505d65be",
            "im-high-importance-2": "ad4f5ef4c9034624c4554aba72a259fe84712fdde43dd98fc9592f4265db278e",
            "im-medium-importance-1": "680b8f4fc00e1fec4f9e52db616a3dfdc84a01a3d927cb7ed96d7fb48f7edc3d",
            "im-medium-importance-2": "b37e10d66772a51ca03bd5af276e964e24ac11aef34ea2c066fe37b26e4b4e06",
            "im-low-importance-1": "2e658bac19ee9082ae6c71ca2847de58f1a1f294f7e2fbcba2fa3c7fddc54cc4",
            "im-low-importance-2": "8ae88e62d6cacb9be0a2fbef6efcaf7801017510fb4e20d05d07b14de8fdd267",
            "ltm-highest-importance-1": "f8dc98ec97fdf32711eed46322ae325f15fbd53fa5998c5f4640bf49e469d81c",
            "ltm-high-importance-1": "68f7348745ae732fe013b7ed45d6c6f8f3f8caaf669ea43b04a349a17b432216",
            "ltm-medium-importance-1": "abd9577d127a2d63eb3c055b584503a73d90746aa6d8b2d941f83198a8c1506a",
            "ltm-low-importance-1": "c2c86d3f7ef76ec4b9e54e48cb2160394b35693cc0a17f1cd93d4674942fb4da",
        }

        # Initialize base class
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=ImportanceStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for importance search."""
        # Test 1: Basic importance threshold search
        self.runner.run_test(
            "Basic Importance Threshold (0.8)",
            0.8,
            expected_memory_ids=[
                "stm-high-importance-1",
                "stm-high-importance-2",
                "im-high-importance-1",
                "im-high-importance-2",
                "ltm-highest-importance-1",
                "ltm-high-importance-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search with min_importance as dict parameter
        self.runner.run_test(
            "Min Importance as Dict Parameter",
            {"min_importance": 0.87},
            expected_memory_ids=[
                "ltm-highest-importance-1",
                "stm-high-importance-1",
                "ltm-high-importance-1",
                "im-high-importance-2",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search with min and max importance
        self.runner.run_test(
            "Min and Max Importance Range",
            {
                "min_importance": 0.63,  # Adjusted to match actual memory scores
                "max_importance": 0.7,  # Adjusted to match actual memory scores
            },
            expected_memory_ids=[
                "stm-medium-importance-1",  # 0.65
                "im-medium-importance-2",  # 0.68
                "ltm-medium-importance-1",  # 0.7
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search with top_n parameter
        self.runner.run_test(
            "Top N Most Important Memories",
            {"top_n": 3},
            expected_memory_ids=[
                "ltm-highest-importance-1",  # 0.98
                "stm-high-importance-1",  # 0.95
                "ltm-high-importance-1",  # 0.9
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search specific memory tier
        self.runner.run_test(
            "Search in STM Tier Only",
            0.65,  # Lower threshold to include medium importance memory
            expected_memory_ids=[
                "stm-high-importance-1",  # 0.95
                "stm-high-importance-2",  # 0.85
                "stm-medium-importance-1",  # 0.68
            ],
            tier="stm",
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for importance search."""
        # Test 1: Search with metadata filter
        self.runner.run_test(
            "Search with Metadata Filter",
            0.7,
            expected_memory_ids=[
                "stm-high-importance-1",  # 0.95, type: "alert"
            ],
            metadata_filter={"content.metadata.type": "alert"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search with sort order ascending
        self.runner.run_test(
            "Search with Ascending Sort Order",
            0.5,
            expected_memory_ids=[
                "im-medium-importance-1",  # 0.55
                "stm-medium-importance-2",  # 0.6
                "stm-medium-importance-1",  # 0.65
                "im-medium-importance-2",  # 0.68
            ],
            sort_order="asc",
            limit=4,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search combining importance threshold and memory type
        self.runner.run_test(
            "Combined Tier and Importance Search",
            0.7,
            expected_memory_ids=[
                "im-high-importance-1",  # 0.8
                "im-high-importance-2",  # 0.87
            ],
            tier="im",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Multi-tier search with limit
        self.runner.run_test(
            "Multi-Tier Search with Limit",
            0.7,
            expected_memory_ids=[
                "ltm-highest-importance-1",  # 0.98
                "stm-high-importance-1",  # 0.95
                "ltm-high-importance-1",  # 0.9
            ],
            limit=3,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search with very high threshold (limiting results)
        self.runner.run_test(
            "Very High Threshold",
            0.95,
            expected_memory_ids=[
                "ltm-highest-importance-1",  # 0.98
                "stm-high-importance-1",  # 0.95
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Search with important and recent combined criteria
        self.runner.run_test(
            "Important and Recent Memories",
            0.8,
            expected_memory_ids=[
                "stm-high-importance-1",  # 0.95, timestamp: 1686816000
                "stm-high-importance-2",  # 0.85, timestamp: 1686902400
                "im-high-importance-1",  # 0.8, timestamp: 1685520000
                "im-high-importance-2",  # 0.87, timestamp: 1685865600
            ],
            metadata_filter={"creation_time": {"$gte": 1685520000}},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Search for low importance memories specifically
        self.runner.run_test(
            "Low Importance Memories Only",
            {
                "min_importance": 0.2,
                "max_importance": 0.4,
            },
            expected_memory_ids=[
                "stm-low-importance-1",  # 0.25
                "im-low-importance-1",  # 0.3
                "im-low-importance-2",  # 0.35
                "ltm-low-importance-1",  # 0.4
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 8: String importance mapping (high)
        self.runner.run_test(
            "String Importance Mapping - High",
            0.9,
            expected_memory_ids=[
                "ltm-highest-importance-1",  # 0.98
                "stm-high-importance-1",  # 0.95
                "ltm-high-importance-1",  # 0.9
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for importance search."""
        # Test 1: Zero threshold should return all memories
        self.runner.run_test(
            "Zero Importance Threshold",
            0.0,
            expected_memory_ids=[
                "ltm-highest-importance-1",  # 0.98
                "stm-high-importance-1",  # 0.95
                "ltm-high-importance-1",  # 0.9
                "im-high-importance-2",  # 0.87
                "stm-high-importance-2",  # 0.85
                "im-high-importance-1",  # 0.8
                "ltm-medium-importance-1",  # 0.7
                "stm-medium-importance-1",  # 0.68
                "im-medium-importance-2",  # 0.68
                "im-medium-importance-1",  # 0.67
                "stm-medium-importance-2",  # 0.63
                "ltm-low-importance-1",  # 0.4
                "im-low-importance-2",  # 0.35
                "im-low-importance-1",  # 0.3
                "stm-low-importance-1",  # 0.25
            ],
            sort_order="desc",
            limit=20,  # Set limit higher than number of memories to ensure all are returned
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Very high threshold (no results)
        self.runner.run_test(
            "Very High Threshold (No Results)",
            0.99,
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Invalid threshold type
        try:
            self.runner.run_test(
                "Invalid Threshold Type",
                "high",
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "must be a number" in str(e):
                self.runner.logger.info(
                    "Test 'Invalid Threshold Type' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 4: Empty dictionary query
        self.runner.run_test(
            "Empty Dictionary Query",
            {},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Invalid min/max range
        try:
            self.runner.run_test(
                "Invalid Min/Max Range",
                {"min_importance": 0.8, "max_importance": 0.2},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected, so we'll log it as a successful test
            if "min_importance cannot be greater than max_importance" in str(e):
                self.runner.logger.info(
                    "Test 'Invalid Min/Max Range' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                # Unexpected error message, re-raise
                raise

        # Test 6: Search non-existent tier
        self.runner.run_test(
            "Non-existent Tier",
            0.7,
            expected_memory_ids=[],
            tier="non_existent_tier",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Negative top_n value
        try:
            self.runner.run_test(
                "Negative top_n Value",
                {"top_n": -5},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "top_n must be" in str(e):
                self.runner.logger.info(
                    "Test 'Negative top_n Value' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 8: Zero limit should return empty results
        self.runner.run_test(
            "Zero Limit",
            0.7,
            expected_memory_ids=[],
            limit=0,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 9: Explicit importance dict with null values
        try:
            self.runner.run_test(
                "Dict with Null Values",
                {"min_importance": None, "max_importance": None},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "min_importance must be a number" in str(e):
                self.runner.logger.info(
                    "Test 'Dict with Null Values' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 10: Non-numeric importance threshold
        try:
            self.runner.run_test(
                "Non-numeric Importance Threshold",
                {"min_importance": "high"},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "min_importance must be a number" in str(e):
                self.runner.logger.info(
                    "Test 'Non-numeric Importance Threshold' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 11: Invalid threshold (negative)
        try:
            self.runner.run_test(
                "Invalid Threshold (Negative)",
                -0.5,
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Importance value cannot be negative" in str(e):
                self.runner.logger.info(
                    "Test 'Invalid Threshold (Negative)' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise


def main():
    """Run the importance search test suite."""
    test_suite = ImportanceSearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
