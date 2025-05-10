"""
Test Runner for AgentMemory System Validations.

This module provides a standardized way to run validation tests
across different strategies and test types.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set, Type, Union

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from validation.demo_utils import (
    create_memory_system,
    log_print,
    pretty_print_memories,
    setup_logging,
)
from validation.framework.validation_framework import PerformanceTest, ValidationTest


class ValidationTestRunner:
    """Class to run and manage validation tests."""

    def __init__(
        self,
        strategy_name: str,
        strategy_class: type,
        agent_id: str,
        memory_sample_path: Optional[str] = None,
        use_mock_redis: bool = True,
        logging_level: str = "INFO",
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the validation test runner.

        Args:
            strategy_name: Name of the strategy being tested
            strategy_class: Class implementing the strategy
            agent_id: ID of the agent to use for testing
            memory_sample_path: Path to memory sample JSON file
            use_mock_redis: Whether to use mock Redis
            logging_level: Logging level for the memory system
            logger: Optional logger to use
        """
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.agent_id = agent_id
        self.memory_sample_path = memory_sample_path
        self.test_results = []

        # Set up logger if not provided
        self.logger = logger or setup_logging(f"validate_{strategy_name}")
        log_print(self.logger, f"Starting {strategy_name} Validation")

        # Set up memory system
        self.memory_system = create_memory_system(
            logging_level=logging_level,
            memory_file=memory_sample_path,
            use_mock_redis=use_mock_redis,
            use_embeddings=False,  # Disable embeddings for attribute search
            embedding_type="text",  # This won't be used since embeddings are disabled
        )

        if not self.memory_system:
            log_print(self.logger, "Failed to load memory system")
            return

        # Get agent and create strategy
        self.agent = self.memory_system.get_memory_agent(agent_id)
        self.strategy = strategy_class(
            stm_store=self.agent.stm_store,
            im_store=self.agent.im_store,
            ltm_store=self.agent.ltm_store
        )

        # Print strategy info
        log_print(self.logger, f"Testing search strategy: {self.strategy.name()}")
        log_print(self.logger, f"Description: {self.strategy.description()}")

    def get_checksums_for_memory_ids(
        self, memory_ids: List[str], memory_checksum_map: Dict[str, str]
    ) -> Set[str]:
        """Helper function to get checksums from memory IDs.

        Args:
            memory_ids: List of memory IDs
            memory_checksum_map: Dictionary mapping memory IDs to checksums

        Returns:
            Set of checksums
        """
        return {
            memory_checksum_map[memory_id]
            for memory_id in memory_ids
            if memory_id in memory_checksum_map
        }

    def run_test(
        self,
        test_name: str,
        query: Any,
        expected_checksums: Optional[Set[str]] = None,
        expected_memory_ids: Optional[List[str]] = None,
        memory_checksum_map: Optional[Dict[str, str]] = None,
        _strategy_override: Any = None,
        **search_params,
    ) -> Dict[str, Any]:
        """Run a single test case and log the results.

        Args:
            test_name: Name of the test
            query: Search query
            expected_checksums: Set of expected memory checksums
            expected_memory_ids: List of expected memory IDs
            memory_checksum_map: Mapping of memory IDs to checksums
            _strategy_override: Optional strategy instance to use for this test
            **search_params: Additional search parameters for the strategy

        Returns:
            Dictionary containing test results
        """
        log_print(self.logger, f"\n=== Test: {test_name} ===")

        # Log query
        if isinstance(query, dict):
            log_print(self.logger, f"Query (dict): {query}")
        else:
            log_print(self.logger, f"Query: '{query}'")

        # Log search parameters
        for param, value in search_params.items():
            log_print(self.logger, f"{param}: {value}")

        # If expected_memory_ids is provided, convert to checksums
        if expected_memory_ids and not expected_checksums and memory_checksum_map:
            expected_checksums = self.get_checksums_for_memory_ids(
                expected_memory_ids, memory_checksum_map
            )
            log_print(
                self.logger,
                f"Expecting {len(expected_checksums)} memories from specified memory IDs",
            )

        # Use the provided strategy override if specified, otherwise use the default
        strategy = (
            _strategy_override if _strategy_override is not None else self.strategy
        )

        # If using a strategy override, log information about it
        if _strategy_override is not None:
            log_print(self.logger, f"Using custom strategy: {strategy.name()}")
            if hasattr(strategy, "description"):
                log_print(
                    self.logger,
                    f"Custom strategy description: {strategy.description()}",
                )

        # Run search with strategy
        results = strategy.search(query=query, agent_id=self.agent_id, **search_params)

        log_print(self.logger, f"Found {len(results)} results")
        pretty_print_memories(results, f"Results for {test_name}", self.logger)

        # Track test status
        test_passed = True

        # Validate against expected checksums if provided
        if expected_checksums:
            result_checksums = {
                result.get("metadata", {}).get("checksum", "") for result in results
            }
            missing_checksums = expected_checksums - result_checksums
            unexpected_checksums = result_checksums - expected_checksums

            log_print(self.logger, f"\nValidation Results:")
            if not missing_checksums and not unexpected_checksums:
                log_print(
                    self.logger, "All expected memories found. No unexpected memories."
                )
            else:
                if missing_checksums:
                    log_print(
                        self.logger, f"Missing expected memories: {missing_checksums}"
                    )
                    test_passed = False
                if unexpected_checksums:
                    log_print(
                        self.logger,
                        f"Found unexpected memories: {unexpected_checksums}",
                    )
                    test_passed = False

            log_print(
                self.logger,
                f"Expected: {len(expected_checksums)}, Found: {len(result_checksums)}, "
                f"Missing: {len(missing_checksums)}, Unexpected: {len(unexpected_checksums)}",
            )

        # Store test result
        test_result = {
            "results": results,
            "test_name": test_name,
            "passed": test_passed,
            "has_validation": expected_checksums is not None
            or expected_memory_ids is not None,
        }

        self.test_results.append(test_result)
        return test_result

    def run_exception_test(
        self,
        test_name: str,
        expected_exception: Type[Exception],
        func_to_test: callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a test that is expected to raise an exception.

        Args:
            test_name: Name of the test
            expected_exception: Exception type that should be raised
            func_to_test: Function to call that should raise the exception
            *args: Positional arguments to pass to func_to_test
            **kwargs: Keyword arguments to pass to func_to_test

        Returns:
            Dictionary containing test results
        """
        log_print(self.logger, f"\n=== Test: {test_name} ===")

        try:
            func_to_test(*args, **kwargs)
            log_print(self.logger, f"Test {test_name}: Failed - Exception not raised")
            test_result = {
                "test_name": test_name,
                "passed": False,
                "has_validation": True,
                "exception_raised": False,
            }
        except expected_exception:
            log_print(
                self.logger, f"Test {test_name}: Passed - Expected exception raised"
            )
            test_result = {
                "test_name": test_name,
                "passed": True,
                "has_validation": True,
                "exception_raised": True,
            }
        except Exception as e:
            log_print(
                self.logger, f"Test {test_name}: Failed - Unexpected exception: {e}"
            )
            test_result = {
                "test_name": test_name,
                "passed": False,
                "has_validation": True,
                "exception_raised": True,
                "wrong_exception": True,
            }

        self.test_results.append(test_result)
        return test_result

    def display_summary(self) -> None:
        """Display summary of test results."""
        log_print(self.logger, "\n\n=== VALIDATION SUMMARY ===")
        log_print(self.logger, "-" * 80)
        log_print(
            self.logger,
            "| {:<40} | {:<20} | {:<20} |".format(
                "Test Name", "Status", "Validation Status"
            ),
        )
        log_print(self.logger, "-" * 80)

        for result in self.test_results:
            status = "PASS" if result["passed"] else "FAIL"
            validation_status = status if result["has_validation"] else "N/A"
            log_print(
                self.logger,
                "| {:<40} | {:<20} | {:<20} |".format(
                    result["test_name"][:40], status, validation_status
                ),
            )

        log_print(self.logger, "-" * 80)

        # Calculate overall statistics
        validated_tests = [t for t in self.test_results if t["has_validation"]]
        passed_tests = [t for t in validated_tests if t["passed"]]

        if validated_tests:
            success_rate = len(passed_tests) / len(validated_tests) * 100
            log_print(self.logger, f"\nValidated Tests: {len(validated_tests)}")
            log_print(self.logger, f"Passed Tests: {len(passed_tests)}")
            log_print(
                self.logger, f"Failed Tests: {len(validated_tests) - len(passed_tests)}"
            )
            log_print(self.logger, f"Success Rate: {success_rate:.2f}%")
        else:
            log_print(self.logger, "\nNo tests with validation criteria were run.")

        log_print(self.logger, "\nValidation Complete")
