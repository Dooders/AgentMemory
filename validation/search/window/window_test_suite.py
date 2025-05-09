"""
Test Suite Implementation for Time Window Search Strategy.

This module demonstrates how to use the validation framework
to create a complete test suite for the time window search strategy.
"""

import os
import sys
from typing import Dict, List, Set
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.window import TimeWindowStrategy
from validation.framework.test_suite import TestSuite


class WindowSearchTestSuite(TestSuite):
    """Test suite for TimeWindowStrategy."""

    def __init__(self, logger=None):
        """Initialize the time window search test suite.

        Args:
            logger: Optional logger to use
        """
        # Constants
        STRATEGY_NAME = "window"
        AGENT_ID = "test-agent-window-search"
        MEMORY_SAMPLE = os.path.join(
            "validation", "memory_samples", "window_validation_memory.json"
        )

        # Define mapping between memory IDs and checksums
        MEMORY_CHECKSUMS = {
            "test-agent-window-search-recent-1": "5a2f580e517d8c2e2c77fbc027d3f1a7a3e5e10c2b6b498d32efbd351d7a73d8",
            "test-agent-window-search-recent-2": "8c9d6e8af1c6f8b9e6d21e3a45f3a9c7b8a4f1c3e5d7a9f1e3c5d7a9f1e3c5d7a",
            "test-agent-window-search-recent-3": "1e9e265e75c2ef678dfd0de0ab5c801f845daa48a90a48bb02ee85148ccc3470",
            "test-agent-window-search-same-day-1": "496d09718bbc8ae669dffdd782ed5b849fdbb1a57e3f7d07e61807b10e650092",
            "test-agent-window-search-yesterday-1": "ffa0ee60ebaec5574358a02d1857823e948519244e366757235bf755c888a87f",
            "test-agent-window-search-lastweek-1": "1d23b6683acd8c3863cb2f2010fe3df2c3e69a2d94c7c4757a291d4872066cfd",
            "test-agent-window-search-lastweek-2": "169c452e368fd62e3c0cf5ce7731769ed46ab6ae73e5048e0c3a7caaa66fba46",
            "test-agent-window-search-lastmonth-1": "9214ebc2d11877665b32771bd3c080414d9519b435ec3f6c19cc5f337bb0ba90",
            "test-agent-window-search-lastmonth-2": "f3c73b06d6399ed30ea9d9ad7c711a86dd58154809cc05497f8955425ec6dc67",
            "test-agent-window-search-custom-timestamp-1": "ad2e7c963751beb1ebc1c9b84ecb09ec3ccdef14f276cd14bbebad12d0f9b0df",
            "test-agent-window-search-recent-nested-1": "b5e6f972a93651cb6734e4eeff16c743a7e3f7bd2f3b2b2e67b4e4a6c3d4f8a9",
        }

        # Initialize base class
        super().__init__(
            strategy_name=STRATEGY_NAME,
            strategy_class=TimeWindowStrategy,
            agent_id=AGENT_ID,
            memory_sample_path=MEMORY_SAMPLE,
            memory_checksum_map=MEMORY_CHECKSUMS,
            logger=logger,
        )

    def run_basic_tests(self) -> None:
        """Run basic functionality tests for time window search."""
        # Get reference dates from the memory sample
        today = datetime.fromisoformat("2024-07-17T12:00:00Z".replace("Z", "+00:00"))
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=6)

        # Test 1: Last 3 hours window
        self.runner.run_test(
            "Last 3 Hours Window",
            {"last_hours": 3},
            expected_memory_ids=[
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Last 24 hours window
        self.runner.run_test(
            "Last 24 Hours Window",
            {"last_hours": 24},
            expected_memory_ids=[
                "test-agent-window-search-same-day-1",
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
                "test-agent-window-search-yesterday-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Last 7 days window
        self.runner.run_test(
            "Last 7 Days Window",
            {"last_days": 7},
            expected_memory_ids=[
                "test-agent-window-search-same-day-1",
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
                "test-agent-window-search-yesterday-1",
                "test-agent-window-search-lastweek-1",
                "test-agent-window-search-lastweek-2",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Explicit time range (today only)
        self.runner.run_test(
            "Explicit Time Range (Today Only)",
            {
                "start_time": today.replace(hour=0, minute=0, second=0).isoformat(),
                "end_time": today.replace(hour=23, minute=59, second=59).isoformat(),
            },
            expected_memory_ids=[
                "test-agent-window-search-same-day-1",
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Search specific memory tier with time window
        self.runner.run_test(
            "STM Tier with Last 24 Hours Window",
            {"last_hours": 24},
            expected_memory_ids=[
                "test-agent-window-search-same-day-1",
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
                "test-agent-window-search-yesterday-1",
            ],
            tier="stm",
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_advanced_tests(self) -> None:
        """Run advanced functionality tests for time window search."""
        # Get reference dates from the memory sample
        today = datetime.fromisoformat("2024-07-17T12:00:00Z".replace("Z", "+00:00"))
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=6)
        last_month = today - timedelta(days=30)

        # Test 1: Search with metadata filter
        self.runner.run_test(
            "Time Window with Metadata Filter",
            {"last_days": 7},
            expected_memory_ids=[
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-yesterday-1",
                "test-agent-window-search-lastweek-2",
            ],
            metadata_filter={"content.metadata.tags": "auth"},
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Search with limited results
        self.runner.run_test(
            "Time Window with Limited Results",
            {"last_days": 7},
            expected_memory_ids=[
                "test-agent-window-search-recent-nested-1",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-3",
            ],
            limit=3,
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 3: Search with specific timestamp format
        self.runner.run_test(
            "Search with Custom Timestamp Field",
            {
                "start_time": "2024-07-17T12:30:00Z",
                "end_time": "2024-07-17T14:30:00Z",
                "timestamp_field": "content.metadata.details.task_info.started_at",
            },
            expected_memory_ids=[
                "test-agent-window-search-recent-nested-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 4: Search with combined tier and time window
        self.runner.run_test(
            "Combined Tier and Time Window",
            {"last_days": 30},
            expected_memory_ids=[
                "test-agent-window-search-lastweek-1",
                "test-agent-window-search-lastweek-2",
            ],
            tier="im",
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 5: Time range across multiple months
        self.runner.run_test(
            "Time Range Across Multiple Months",
            {
                "start_time": "2024-06-10T00:00:00Z",
                "end_time": "2024-07-20T23:59:59Z",
            },
            expected_memory_ids=[
                "test-agent-window-search-same-day-1",
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
                "test-agent-window-search-yesterday-1",
                "test-agent-window-search-lastweek-1",
                "test-agent-window-search-lastweek-2",
                "test-agent-window-search-lastmonth-1",
                "test-agent-window-search-lastmonth-2",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 6: Specific time on a specific day
        self.runner.run_test(
            "Specific Time Range on Today",
            {
                "start_time": "2024-07-17T10:00:00Z",
                "end_time": "2024-07-17T12:00:00Z",
            },
            expected_memory_ids=[
                "test-agent-window-search-recent-1",
                "test-agent-window-search-recent-2",
                "test-agent-window-search-recent-3",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 7: Custom timestamp field in nested structure
        self.runner.run_test(
            "Custom Timestamp in Nested Structure",
            {"last_hours": 3, "timestamp_field": "content.metadata.custom_date"},
            expected_memory_ids=[
                "test-agent-window-search-custom-timestamp-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

    def run_edge_case_tests(self) -> None:
        """Run edge case tests for time window search."""
        # Get reference dates
        today = datetime.fromisoformat("2024-07-17T12:00:00Z".replace("Z", "+00:00"))

        # Test 1: Default 30 minute window when no parameters provided
        self.runner.run_test(
            "Default 30 Minute Window (No Parameters)",
            {},
            expected_memory_ids=[
                "test-agent-window-search-recent-3",
                "test-agent-window-search-custom-timestamp-1",
                "test-agent-window-search-recent-nested-1",
            ],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 2: Zero time window
        try:
            self.runner.run_test(
                "Zero Time Window",
                {"last_minutes": 0},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Time window must be positive" in str(e):
                self.runner.logger.info(
                    "Test 'Zero Time Window' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 3: Negative time window
        try:
            self.runner.run_test(
                "Negative Time Window",
                {"last_hours": -5},
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Time window must be positive" in str(e):
                self.runner.logger.info(
                    "Test 'Negative Time Window' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 4: End time before start time
        try:
            self.runner.run_test(
                "End Time Before Start Time",
                {
                    "start_time": "2024-07-17T12:00:00Z",
                    "end_time": "2024-07-16T12:00:00Z",
                },
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # Depending on implementation, this might not raise an error
            # but instead return empty results, so we handle both cases
            self.runner.logger.info(
                "Test 'End Time Before Start Time' passed with error: %s",
                str(e),
            )

        # Test 5: Invalid time format
        try:
            self.runner.run_test(
                "Invalid Time Format",
                {
                    "start_time": "2024/07/17 12:00:00",
                    "end_time": "2024/07/17 14:00:00",
                },
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Invalid time format" in str(e) or "Invalid datetime format" in str(e):
                self.runner.logger.info(
                    "Test 'Invalid Time Format' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 6: Multiple conflicting time parameters
        try:
            self.runner.run_test(
                "Multiple Conflicting Time Parameters",
                {
                    "last_minutes": 30,
                    "last_hours": 2,
                    "last_days": 1,
                },
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Can only specify one of" in str(e):
                self.runner.logger.info(
                    "Test 'Multiple Conflicting Time Parameters' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 7: Conflicting time range and last_X parameters
        try:
            self.runner.run_test(
                "Conflicting Time Range and last_X Parameters",
                {
                    "start_time": "2024-07-17T10:00:00Z",
                    "end_time": "2024-07-17T12:00:00Z",
                    "last_hours": 2,
                },
                expected_memory_ids=[],
                memory_checksum_map=self.memory_checksum_map,
            )
        except ValueError as e:
            # This exception is expected
            if "Cannot specify both time range and last_X parameters" in str(e):
                self.runner.logger.info(
                    "Test 'Conflicting Time Range and last_X Parameters' passed: Expected ValueError caught: %s",
                    str(e),
                )
            else:
                raise

        # Test 8: Non-existent timestamp field
        self.runner.run_test(
            "Non-existent Timestamp Field",
            {
                "last_hours": 24,
                "timestamp_field": "content.metadata.non_existent_field",
            },
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 9: Far future time window (no results)
        self.runner.run_test(
            "Far Future Time Window",
            {
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-02T00:00:00Z",
            },
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 10: Far past time window (no results)
        self.runner.run_test(
            "Far Past Time Window",
            {
                "start_time": "2020-01-01T00:00:00Z",
                "end_time": "2020-01-02T00:00:00Z",
            },
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )

        # Test 11: Very short time window (1 minute)
        self.runner.run_test(
            "Very Short Time Window",
            {"last_minutes": 1},
            expected_memory_ids=[],
            memory_checksum_map=self.memory_checksum_map,
        )


def main():
    """Run the time window search test suite."""
    test_suite = WindowSearchTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
