"""
Command-line Interface for AgentMemory Validation Framework.

This module provides a CLI for running validations against different
components of the AgentMemory system.
"""

import argparse
import importlib
import inspect
import logging
import os
import sys
from typing import Dict, List, Type

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from validation.framework.test_suite import TestSuite, PerformanceTestSuite
from validation.demo_utils import setup_logging


def discover_test_suites() -> Dict[str, Type[TestSuite]]:
    """Discover all available test suites in the validation directory.
    
    Returns:
        Dictionary mapping test suite names to test suite classes
    """
    test_suites = {}
    validation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Search for test suites in the validation directory structure
    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            if file.endswith(".py") and file.startswith("validate_"):
                # Get relative module path
                module_path = os.path.relpath(os.path.join(root, file), os.path.dirname(validation_dir))
                module_name = os.path.splitext(module_path)[0].replace(os.path.sep, ".")
                
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Find test suite classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, TestSuite) and 
                            obj != TestSuite and 
                            obj != PerformanceTestSuite):
                            
                            # Use class name as key
                            key = name.lower().replace("testsuite", "")
                            test_suites[key] = obj
                except (ImportError, AttributeError) as e:
                    print(f"Error importing {module_name}: {e}")
    
    return test_suites


def main():
    """Main entry point for the validation CLI."""
    # Discover available test suites
    test_suites = discover_test_suites()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="AgentMemory Validation Framework CLI")
    
    # Add arguments
    parser.add_argument(
        "suite",
        choices=list(test_suites.keys()) + ["all"],
        help="Test suite to run (use 'all' to run all suites)"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["basic", "advanced", "edge", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Run performance tests if available"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--sample-path",
        help="Path to custom memory sample file"
    )
    
    parser.add_argument(
        "--mock-redis",
        action="store_true",
        default=True,
        help="Use mock Redis instead of real Redis"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("validation_cli", log_level=log_level)
    
    # Determine which suites to run
    suites_to_run = test_suites.values() if args.suite == "all" else [test_suites[args.suite]]
    
    # Run selected test suites
    for suite_class in suites_to_run:
        # Create test suite instance
        suite = suite_class(logger=logger)
        
        # Determine which tests to run
        if args.test_type == "all":
            suite.run_all_tests()
        elif args.test_type == "basic":
            suite.run_basic_tests()
            suite.runner.display_summary()
        elif args.test_type == "advanced":
            suite.run_advanced_tests()
            suite.runner.display_summary()
        elif args.test_type == "edge":
            suite.run_edge_case_tests()
            suite.runner.display_summary()
        
        # Run performance tests if requested and available
        if args.perf and isinstance(suite, PerformanceTestSuite):
            suite.run_scaling_tests([1000, 5000, 10000])
            suite.run_concurrent_tests([1, 5, 10])


if __name__ == "__main__":
    main() 