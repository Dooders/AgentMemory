"""
Performance Testing script for the Sequence Search Strategy.

This script evaluates the performance of the narrative sequence search strategy
across various scaling and concurrency scenarios.
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import psutil

# Add project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.sequence import NarrativeSequenceStrategy
from validation.demo_utils import create_memory_system, log_print, setup_logging
from validation.framework.validation_framework import PerformanceTest

# Constants
AGENT_ID = "test-agent-sequence-search"
MEMORY_SAMPLE = os.path.join(
    "validation", "memory_samples", "sequence_validation_memory.json"
)


class SequencePerformanceTest(PerformanceTest):
    """Performance test suite for NarrativeSequenceStrategy."""

    def __init__(self, logger=None):
        """Initialize the sequence performance test suite.

        Args:
            logger: Optional logger to use
        """
        super().__init__(logger, "sequence")
        self.memory_system = None
        self.search_strategy = None
        self._first_run_complete = False

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

        # Load the NarrativeSequenceStrategy
        self.search_strategy = NarrativeSequenceStrategy(
            agent.stm_store, agent.im_store, agent.ltm_store
        )

        # If testing with more than the sample memories, generate additional test memories
        base_memories_count = (
            len(agent.stm_store.get_all(AGENT_ID))
            + len(agent.im_store.get_all(AGENT_ID))
            + len(agent.ltm_store.get_all(AGENT_ID))
        )

        if memory_count > base_memories_count:
            self.logger.info(
                f"Generating {memory_count - base_memories_count} additional test memories"
            )
            self._generate_test_memories(agent, memory_count - base_memories_count)

    def _generate_test_memories(self, agent, count: int) -> None:
        """Generate additional test memories with sequential timestamps.

        Args:
            agent: The memory agent to add memories to
            count: Number of additional memories to create
        """
        # Lists of sample content and types to create varied test data
        contents = [
            "Team meeting to discuss project progress.",
            "Client presentation preparation.",
            "Code review session with the team.",
            "Weekly status update meeting.",
            "Planning session for next sprint.",
            "Bug triage meeting.",
            "Architecture review discussion.",
            "User feedback analysis meeting.",
            "Release planning session.",
            "Technical debt review meeting.",
        ]

        types = ["meeting", "task", "note"]
        base_timestamp = time.time()

        for i in range(count):
            # Create a memory with sequential timestamp
            memory = {
                "content": {
                    "content": random.choice(contents),
                    "metadata": {
                        "type": random.choice(types),
                        "timestamp": base_timestamp + i * 300,  # 5 minutes apart
                    },
                }
            }

            # Store in appropriate tier based on timestamp
            if i < count // 3:
                agent.stm_store.add(memory, AGENT_ID)
            elif i < 2 * count // 3:
                agent.im_store.add(memory, AGENT_ID)
            else:
                agent.ltm_store.add(memory, AGENT_ID)

    def run_scaling_test(self, memory_sizes: List[int] = [100, 1000, 10000]) -> None:
        """Run scaling tests with different memory sizes.

        Args:
            memory_sizes: List of memory counts to test
        """
        self.logger.info("Running scaling tests...")

        # Reference memory ID to use for sequence search
        reference_id = "meeting-123456-3"

        results = []
        for size in memory_sizes:
            self.setup_memory_system(size)

            # Test different sequence sizes
            for seq_size in [3, 5, 10]:
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss

                # Run sequence search
                self.search_strategy.search(
                    query={
                        "reference_id": reference_id,
                        "sequence_size": seq_size,
                    },
                    agent_id=AGENT_ID,
                )

                duration = time.time() - start_time
                memory_usage = process.memory_info().rss - start_memory

                results.append(
                    {
                        "memory_size": size,
                        "sequence_size": seq_size,
                        "duration": duration,
                        "memory_usage": memory_usage,
                    }
                )

        # Plot results
        self._plot_scaling_results(results)

    def run_concurrent_test(
        self, concurrency_levels: List[int] = [1, 5, 10, 20]
    ) -> None:
        """Run concurrent load tests.

        Args:
            concurrency_levels: List of concurrent user counts to test
        """
        self.logger.info("Running concurrent load tests...")

        # Set up memory system with fixed size
        self.setup_memory_system(1000)

        # Reference memory ID to use for sequence search
        reference_id = "meeting-123456-3"

        results = []
        for level in concurrency_levels:
            start_time = time.time()

            # Create thread pool for concurrent requests
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = []
                for _ in range(level):
                    futures.append(
                        executor.submit(
                            self.search_strategy.search,
                            query={
                                "reference_id": reference_id,
                                "sequence_size": 5,
                            },
                            agent_id=AGENT_ID,
                        )
                    )

                # Wait for all requests to complete
                concurrent_results = [f.result() for f in futures]

            duration = time.time() - start_time
            results.append(
                {
                    "concurrency_level": level,
                    "duration": duration,
                    "total_results": sum(len(r) for r in concurrent_results),
                }
            )

        # Plot results
        self._plot_concurrent_results(results)

    def _plot_scaling_results(self, results: List[Dict[str, Any]]) -> None:
        """Plot scaling test results.

        Args:
            results: List of test results
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot duration by memory size and sequence size
        for seq_size in sorted(set(r["sequence_size"] for r in results)):
            data = [r for r in results if r["sequence_size"] == seq_size]
            ax1.plot(
                [r["memory_size"] for r in data],
                [r["duration"] for r in data],
                marker="o",
                label=f"Sequence Size: {seq_size}",
            )

        ax1.set_xlabel("Memory Size")
        ax1.set_ylabel("Duration (seconds)")
        ax1.set_title("Search Duration vs Memory Size")
        ax1.legend()
        ax1.grid(True)

        # Plot memory usage by memory size and sequence size
        for seq_size in sorted(set(r["sequence_size"] for r in results)):
            data = [r for r in results if r["sequence_size"] == seq_size]
            ax2.plot(
                [r["memory_size"] for r in data],
                [r["memory_usage"] / 1024 / 1024 for r in data],  # Convert to MB
                marker="o",
                label=f"Sequence Size: {seq_size}",
            )

        ax2.set_xlabel("Memory Size")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_title("Memory Usage vs Memory Size")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join("validation", "results", "sequence", "scaling_results.png")
        )
        plt.close()

    def _plot_concurrent_results(self, results: List[Dict[str, Any]]) -> None:
        """Plot concurrent test results.

        Args:
            results: List of test results
        """
        plt.figure(figsize=(10, 5))

        # Plot duration by concurrency level
        plt.plot(
            [r["concurrency_level"] for r in results],
            [r["duration"] for r in results],
            marker="o",
            label="Duration",
        )

        plt.xlabel("Concurrency Level")
        plt.ylabel("Duration (seconds)")
        plt.title("Search Duration vs Concurrency Level")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join("validation", "results", "sequence", "concurrent_results.png")
        )
        plt.close()


def main():
    """Run the sequence performance tests."""
    logger = setup_logging("validate_sequence_performance")
    test_suite = SequencePerformanceTest(logger)

    # Run scaling tests
    test_suite.run_scaling_test()

    # Run concurrent tests
    test_suite.run_concurrent_test()


if __name__ == "__main__":
    main()
