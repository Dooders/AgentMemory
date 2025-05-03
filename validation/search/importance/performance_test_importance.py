"""
Performance Testing script for the Importance Search Strategy.

This script evaluates the performance of the importance-based search strategy
across various scaling and concurrency scenarios.
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from validation.demo_utils import (
    create_memory_system,
    log_print,
    setup_logging,
)
from memory.search.strategies.importance import ImportanceStrategy
from validation.framework.validation_framework import PerformanceTest

# Constants
AGENT_ID = "test-agent-importance-search"
MEMORY_SAMPLE = os.path.join("validation", "memory_samples", "importance_validation_memory.json")

class ImportancePerformanceTest(PerformanceTest):
    """Performance testing implementation for the ImportanceStrategy."""
    
    def setup_memory_system(self, memory_count: int = 1000) -> None:
        """Initialize memory system with test data.
        
        Args:
            memory_count: Number of test memories to create
        """
        self.logger.info(f"Setting up memory system with {memory_count} memories")
        
        # First load the base memory system from the sample file
        self.memory_system = create_memory_system(
            logging_level="INFO",
            memory_file=MEMORY_SAMPLE,
            use_mock_redis=True,
        )
        
        if not self.memory_system:
            self.logger.error("Failed to load memory system")
            return
        
        # Get the memory agent
        agent = self.memory_system.get_memory_agent(AGENT_ID)
        
        # Load the ImportanceStrategy
        self.search_strategy = ImportanceStrategy(
            agent.stm_store, agent.im_store, agent.ltm_store
        )
        
        # If testing with more than the sample memories, generate additional test memories
        base_memories_count = len(agent.stm_store.get_all(AGENT_ID)) + \
                             len(agent.im_store.get_all(AGENT_ID)) + \
                             len(agent.ltm_store.get_all(AGENT_ID))
        
        if memory_count > base_memories_count:
            self.logger.info(f"Generating {memory_count - base_memories_count} additional test memories")
            self._generate_test_memories(agent, memory_count - base_memories_count)
    
    def _generate_test_memories(self, agent, count: int) -> None:
        """Generate additional test memories with varying importance scores.
        
        Args:
            agent: The memory agent to add memories to
            count: Number of additional memories to create
        """
        # Lists of sample content and tags to create varied test data
        contents = [
            "Important business meeting with client.",
            "Reminder to complete weekly report.",
            "Office supplies need to be restocked.",
            "Critical system error detected.",
            "New feature request from customer.",
            "Team lunch scheduled for Friday.",
            "Security vulnerability patched.",
            "Quarterly financial review.",
            "Product roadmap planning session.",
            "Customer feedback survey results."
        ]
        
        tags = [
            ["business", "client", "meeting"],
            ["report", "deadline", "weekly"],
            ["office", "supplies", "routine"],
            ["system", "error", "critical"],
            ["feature", "customer", "development"],
            ["team", "social", "schedule"],
            ["security", "patch", "maintenance"],
            ["financial", "quarterly", "review"],
            ["product", "planning", "roadmap"],
            ["customer", "feedback", "survey"]
        ]
        
        # Generate memories with random importance
        for i in range(count):
            # Select random content and tags
            content_idx = random.randint(0, len(contents) - 1)
            
            # Determine memory tier - bias towards STM for performance testing
            tier_rand = random.random()
            if tier_rand < 0.6:
                tier = "stm"
            elif tier_rand < 0.9:
                tier = "im"
            else:
                tier = "ltm"
            
            # Generate random importance score between 0.1 and 1.0
            importance_score = round(random.random(), 2)
            
            # Create memory
            memory = {
                "memory_id": f"{tier}-test-memory-{i}",
                "agent_id": AGENT_ID,
                "step_number": i + 1000,  # Start high to avoid conflicts
                "timestamp": int(time.time()) - random.randint(0, 10000),
                "content": {
                    "content": f"{contents[content_idx]} (Test memory {i})",
                    "metadata": {
                        "type": "test",
                        "tags": tags[content_idx % len(tags)]
                    }
                },
                "metadata": {
                    "creation_time": int(time.time()) - random.randint(0, 10000),
                    "last_access_time": int(time.time()) - random.randint(0, 1000),
                    "compression_level": 0 if tier == "stm" else (1 if tier == "im" else 2),
                    "importance_score": importance_score,
                    "retrieval_count": random.randint(0, 10),
                    "memory_type": "generic",
                    "current_tier": tier
                },
                "type": "generic",
                "embeddings": {}
            }
            
            # Add to appropriate store
            if tier == "stm":
                agent.stm_store.add(memory)
            elif tier == "im":
                agent.im_store.add(memory)
            else:
                agent.ltm_store.add(memory)
    
    def run_test(
        self,
        test_name: str,
        query: Any,
        expected_checksums=None,
        expected_memory_ids=None,
        **search_params
    ) -> Dict[str, Any]:
        """Run a single performance test case.
        
        Args:
            test_name: Name of the test
            query: Search query
            expected_checksums: Set of expected memory checksums (not used in perf tests)
            expected_memory_ids: List of expected memory IDs (not used in perf tests)
            **search_params: Additional search parameters
            
        Returns:
            Dictionary containing test results
        """
        self.logger.info(f"\n=== Performance Test: {test_name} ===")
        
        # Start measuring performance
        start_time = time.time()
        
        # Execute the search
        results = self.search_strategy.search(
            query=query,
            agent_id=AGENT_ID,
            **search_params
        )
        
        # Calculate performance metrics
        execution_time = time.time() - start_time
        
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Found {len(results)} results")
        self.logger.info(f"Execution time: {execution_time:.4f} seconds")
        
        # Return performance metrics
        return {
            "test_name": test_name,
            "execution_time": execution_time,
            "result_count": len(results),
            "query": query,
            "search_params": search_params
        }
    
    def run_importance_specific_tests(self):
        """Run performance tests specific to the importance search strategy."""
        self.logger.info("\n\n=== IMPORTANCE-SPECIFIC PERFORMANCE TESTS ===")
        
        results = []
        
        # Test 1: Varying importance thresholds
        importance_thresholds = [0.0, 0.25, 0.5, 0.75, 0.9]
        for threshold in importance_thresholds:
            test_result = self.run_test(
                f"Importance Threshold {threshold}",
                threshold,
                limit=100
            )
            test_result["threshold"] = threshold
            results.append(test_result)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(
            [r["threshold"] for r in results],
            [r["execution_time"] for r in results],
            marker="o"
        )
        plt.title("Execution Time vs. Importance Threshold")
        plt.xlabel("Importance Threshold")
        plt.ylabel("Execution Time (seconds)")
        plt.savefig(os.path.join(self.results_dir, "importance_threshold_performance.png"))
        plt.close()
        
        # Test 2: Comparing min/max range vs single threshold
        range_results = []
        
        # Wide range
        range_results.append(
            self.run_test(
                "Wide Range (0.2-0.8)",
                {"min_importance": 0.2, "max_importance": 0.8},
                limit=100
            )
        )
        
        # Medium range
        range_results.append(
            self.run_test(
                "Medium Range (0.4-0.6)",
                {"min_importance": 0.4, "max_importance": 0.6},
                limit=100
            )
        )
        
        # Narrow range
        range_results.append(
            self.run_test(
                "Narrow Range (0.45-0.55)",
                {"min_importance": 0.45, "max_importance": 0.55},
                limit=100
            )
        )
        
        # Single value
        range_results.append(
            self.run_test(
                "Single Value (0.5)",
                0.5,
                limit=100
            )
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r["test_name"] for r in range_results],
            [r["execution_time"] for r in range_results]
        )
        plt.title("Execution Time for Different Query Types")
        plt.xlabel("Query Type")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "query_type_performance.png"))
        plt.close()
        
        # Test 3: Impact of sort_order
        sort_results = []
        
        # Descending (default)
        sort_results.append(
            self.run_test(
                "Descending Sort",
                0.3,
                limit=100,
                sort_order="desc"
            )
        )
        
        # Ascending
        sort_results.append(
            self.run_test(
                "Ascending Sort",
                0.3,
                limit=100,
                sort_order="asc"
            )
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(
            [r["test_name"] for r in sort_results],
            [r["execution_time"] for r in sort_results]
        )
        plt.title("Execution Time vs. Sort Order")
        plt.xlabel("Sort Order")
        plt.ylabel("Execution Time (seconds)")
        plt.savefig(os.path.join(self.results_dir, "sort_order_performance.png"))
        plt.close()
        
        return results + range_results + sort_results


def run_performance_tests():
    """Run the full suite of performance tests for ImportanceStrategy."""
    # Setup logging
    logger = setup_logging("performance_test_importance_search")
    logger.info("Starting Importance Search Strategy Performance Tests")
    
    # Create performance test instance
    perf_test = ImportancePerformanceTest(logger, "importance")
    
    # Run scaling tests
    memory_sizes = [100, 500, 1000, 5000, 10000]
    scaling_results = perf_test.run_scaling_test(
        0.5,  # Threshold value
        memory_sizes,
        limit=50
    )
    
    # Run concurrency tests
    concurrency_levels = [1, 5, 10, 20, 50]
    concurrent_results = perf_test.run_concurrent_test(
        0.5,  # Threshold value
        concurrency_levels,
        limit=50
    )
    
    # Run importance-specific tests
    specific_results = perf_test.run_importance_specific_tests()
    
    # Generate performance plots
    perf_test.generate_plots(scaling_results, concurrent_results)
    
    # Save all results
    all_results = {
        "scaling_tests": scaling_results,
        "concurrency_tests": concurrent_results,
        "specific_tests": specific_results
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        perf_test.results_dir,
        f"performance_results_{timestamp}.json"
    )
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Performance test results saved to {results_file}")
    logger.info("Performance Testing Complete")


if __name__ == "__main__":
    run_performance_tests() 