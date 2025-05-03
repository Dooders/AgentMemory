"""
Validation Framework for Testing Memory Search Strategies.

This framework provides a standardized way to create validation tests
for memory search strategies, including both functional and performance testing.
"""

import os
import sys
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from validation.demo_utils import create_memory_system, log_print, setup_logging

class ValidationTest(ABC):
    """Base class for validation tests."""
    
    def __init__(self, logger: logging.Logger, strategy_name: str):
        """Initialize the validation test.
        
        Args:
            logger: Logger instance for output
            strategy_name: Name of the search strategy being tested
        """
        self.logger = logger
        self.strategy_name = strategy_name
        self.results = []
        self.memory_system = None
        self.search_strategy = None
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), "results", strategy_name)
        os.makedirs(self.results_dir, exist_ok=True)
    
    @abstractmethod
    def setup_memory_system(self, memory_count: int = 1000) -> None:
        """Initialize memory system with test data.
        
        Args:
            memory_count: Number of test memories to create
        """
        pass
    
    @abstractmethod
    def run_test(
        self,
        test_name: str,
        query: Any,
        expected_checksums: Optional[Set[str]] = None,
        expected_memory_ids: Optional[List[str]] = None,
        **search_params
    ) -> Dict[str, Any]:
        """Run a single test case.
        
        Args:
            test_name: Name of the test
            query: Search query
            expected_checksums: Set of expected memory checksums
            expected_memory_ids: List of expected memory IDs
            **search_params: Additional search parameters
            
        Returns:
            Dictionary containing test results
        """
        pass
    
    def validate_results(
        self,
        results: List[Dict[str, Any]],
        expected_checksums: Optional[Set[str]] = None,
        expected_memory_ids: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate test results against expectations.
        
        Args:
            results: List of result dictionaries
            expected_checksums: Set of expected memory checksums
            expected_memory_ids: List of expected memory IDs
            
        Returns:
            Tuple of (test_passed, validation_details)
        """
        validation_details = {
            "missing_checksums": set(),
            "unexpected_checksums": set(),
            "result_count": len(results)
        }
        
        if expected_checksums:
            result_checksums = {
                result.get("metadata", {}).get("checksum", "")
                for result in results
            }
            validation_details["missing_checksums"] = expected_checksums - result_checksums
            validation_details["unexpected_checksums"] = result_checksums - expected_checksums
            
        test_passed = (
            not validation_details["missing_checksums"] and
            not validation_details["unexpected_checksums"]
        )
        
        return test_passed, validation_details
    
    def save_results(self) -> None:
        """Save test results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.results_dir,
            f"validation_results_{timestamp}.json"
        )
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        log_print(self.logger, f"Results saved to {results_file}")

class PerformanceTest(ValidationTest):
    """Class for performance testing."""
    
    def __init__(self, logger: logging.Logger, strategy_name: str):
        super().__init__(logger, strategy_name)
        self._first_run_complete = False
    
    def run_scaling_test(
        self,
        query: Any,
        memory_sizes: List[int],
        **search_params
    ) -> List[Dict[str, Any]]:
        """Test performance at different memory sizes.
        
        Args:
            query: Search query
            memory_sizes: List of memory counts to test
            **search_params: Additional search parameters
            
        Returns:
            List of test results
        """
        results = []
        
        for size in memory_sizes:
            self.setup_memory_system(size)
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            results = self.search_strategy.search(query, **search_params)
            
            duration = time.time() - start_time
            memory_usage = process.memory_info().rss - start_memory
            
            test_result = self.generate_test_summary(
                f"Scaling Test - {size} memories",
                duration,
                process.cpu_percent(),
                memory_usage,
                len(results),
                {"memory_size": size, **search_params}
            )
            
            results.append(test_result)
        
        return results
    
    def run_concurrent_test(
        self,
        query: Any,
        concurrency_levels: List[int],
        **search_params
    ) -> List[Dict[str, Any]]:
        """Test performance under concurrent load.
        
        Args:
            query: Search query
            concurrency_levels: List of concurrent user counts to test
            **search_params: Additional search parameters
            
        Returns:
            List of test results
        """
        results = []
        
        def search_task():
            return self.search_strategy.search(query, **search_params)
        
        for level in concurrency_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(search_task) for _ in range(level)]
                concurrent_results = [f.result() for f in futures]
            
            duration = time.time() - start_time
            
            test_result = self.generate_test_summary(
                f"Concurrent Test - {level} users",
                duration,
                psutil.Process().cpu_percent(),
                psutil.Process().memory_info().rss,
                sum(len(r) for r in concurrent_results),
                {"concurrency_level": level, **search_params}
            )
            
            results.append(test_result)
        
        return results
    
    def generate_test_summary(
        self,
        test_name: str,
        duration: float,
        cpu_usage: float,
        memory_usage: int,
        result_count: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of test results.
        
        Args:
            test_name: Name of the test
            duration: Test duration in seconds
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in bytes
            result_count: Number of results
            params: Test parameters
            
        Returns:
            Dictionary containing test summary
        """
        return {
            "test_name": test_name,
            "duration": duration,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "result_count": result_count,
            "params": params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_plots(
        self,
        scaling_results: List[Dict[str, Any]],
        concurrent_results: List[Dict[str, Any]]
    ) -> None:
        """Generate performance plots.
        
        Args:
            scaling_results: Results from scaling tests
            concurrent_results: Results from concurrent tests
        """
        # Scaling plot
        if scaling_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                [r["params"]["memory_size"] for r in scaling_results],
                [r["duration"] for r in scaling_results],
                marker="o"
            )
            plt.title(f"{self.strategy_name} - Scaling Performance")
            plt.xlabel("Memory Size")
            plt.ylabel("Duration (seconds)")
            plt.savefig(os.path.join(self.results_dir, "scaling_performance.png"))
            plt.close()
        
        # Concurrency plot
        if concurrent_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                [r["params"]["concurrency_level"] for r in concurrent_results],
                [r["duration"] for r in concurrent_results],
                marker="o"
            )
            plt.title(f"{self.strategy_name} - Concurrency Performance")
            plt.xlabel("Concurrent Users")
            plt.ylabel("Duration (seconds)")
            plt.savefig(os.path.join(self.results_dir, "concurrency_performance.png"))
            plt.close()

def create_validation_test(
    strategy_name: str,
    strategy_class: type,
    is_performance: bool = False
) -> ValidationTest:
    """Factory function to create validation tests.
    
    Args:
        strategy_name: Name of the search strategy
        strategy_class: Class implementing the search strategy
        is_performance: Whether to create a performance test
        
    Returns:
        ValidationTest instance
    """
    logger = setup_logging(f"validate_{strategy_name}")
    
    if is_performance:
        return PerformanceTest(logger, strategy_name)
    else:
        return ValidationTest(logger, strategy_name) 