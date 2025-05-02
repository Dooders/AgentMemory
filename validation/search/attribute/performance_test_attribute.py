"""
Performance Testing for the Attribute Search Strategy.

This script conducts performance tests under various load conditions
to evaluate speed, resource usage, and scalability.
"""

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import psutil
from memory_profiler import profile
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Suppress checksum validation warnings - important to set this before imports
logging.getLogger('memory.storage.redis_im').setLevel(logging.ERROR)
logging.getLogger('memory.storage.redis_stm').setLevel(logging.ERROR)
logging.getLogger('memory.storage.sqlite_ltm').setLevel(logging.ERROR)

from memory.search.strategies.attribute import AttributeSearchStrategy
from validation.demo_utils import create_memory_system, log_print, setup_logging

# Constants
AGENT_ID = "perf-test-agent"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class PerformanceTest:
    def __init__(self, logger):
        self.logger = logger
        self.results = []
        self.memory_system = None
        self.search_strategy = None
        self._first_run_complete = False

        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def setup_memory_system(self, memory_count=1000):
        """Initialize memory system with specified number of test memories."""
        log_print(
            self.logger, f"Setting up memory system with {memory_count} test memories"
        )

        # Create memory system with clean database for first run
        if not hasattr(self, "_first_run_complete"):
            # First run - create with clean database
            self.memory_system = create_memory_system(
                logging_level="ERROR",  # Reduce logging noise during tests
                use_mock_redis=True,
                clear_db=True,  # Clear DB only on first run
            )
            self._first_run_complete = True
        else:
            # Subsequent runs - use existing database
            self.memory_system = create_memory_system(
                logging_level="ERROR",
                use_mock_redis=True,
                clear_db=False,  # Keep existing tables
            )

        # Get agent reference
        agent = self.memory_system.get_memory_agent(AGENT_ID)

        # Generate test memories
        self._generate_test_memories(memory_count)

        # Setup search strategy with skip_validation enabled for performance tests
        log_print(self.logger, "Setting up search strategy with skip_validation=True")
        
        # Make absolutely sure we're skipping validation
        agent.stm_store.skip_validation = True
        agent.im_store.skip_validation = True
        
        self.search_strategy = AttributeSearchStrategy(
            agent.stm_store, 
            agent.im_store, 
            agent.ltm_store,
            skip_validation=True  # Skip checksum validation for performance tests
        )
        log_print(self.logger, "Memory system and search strategy initialized")

    def _generate_test_memories(self, count):
        """Generate test memories with varied content and distribute across tiers."""
        log_print(self.logger, f"Generating {count} test memories")
        agent = self.memory_system.get_memory_agent(AGENT_ID)

        # Define templates for content variety
        templates = [
            "Meeting about {topic} with {people}",
            "Task: {action} the {item} by {deadline}",
            "Note regarding {subject}: {detail}",
            "Email from {sender} about {topic}",
            "Report on {project}: {status}",
        ]

        topics = [
            "project",
            "sales",
            "marketing",
            "development",
            "design",
            "budgeting",
            "planning",
        ]
        people = [
            "team",
            "John",
            "Sarah",
            "management",
            "clients",
            "stakeholders",
            "department",
        ]
        actions = ["review", "update", "create", "finalize", "analyze", "present"]
        items = ["document", "proposal", "code", "design", "report", "plan", "strategy"]
        deadlines = ["tomorrow", "next week", "end of month", "Q3", "Friday", "ASAP"]
        subjects = [
            "meeting",
            "project",
            "issue",
            "update",
            "status",
            "concern",
            "opportunity",
        ]
        details = [
            "needs attention",
            "completed successfully",
            "requires review",
            "on track",
            "delayed",
        ]
        senders = ["boss", "client", "team", "partner", "vendor", "colleague"]
        projects = [
            "Alpha",
            "Beta",
            "Phoenix",
            "Horizon",
            "Foundation",
            "Revolution",
            "Discovery",
        ]
        statuses = [
            "on track",
            "delayed",
            "completed",
            "at risk",
            "needs resources",
            "ahead of schedule",
        ]

        # Track memory counts per tier
        stm_count = 0
        im_count = 0
        ltm_count = 0

        # Create memories with varied distribution
        for i in range(count):
            # Status update every 1000 items
            if i > 0 and i % 1000 == 0:
                log_print(self.logger, f"  Generated {i} memories...")

            # Select template
            template = random.choice(templates)

            # Fill template with random values
            content = template.format(
                topic=random.choice(topics),
                people=random.choice(people),
                action=random.choice(actions),
                item=random.choice(items),
                deadline=random.choice(deadlines),
                subject=random.choice(subjects),
                detail=random.choice(details),
                sender=random.choice(senders),
                project=random.choice(projects),
                status=random.choice(statuses),
            )

            # Create varied metadata
            metadata = {
                "type": random.choice(["meeting", "task", "note", "email", "report"]),
                "importance": random.choice(["high", "medium", "low"]),
                "tags": random.sample(
                    topics + subjects + projects, random.randint(1, 4)
                ),
            }

            # Create a unique memory ID
            memory_id = f"{AGENT_ID}-mem-{i}"

            # Determine tier (distribute 20% STM, 30% IM, 50% LTM)
            tier_rand = random.random()

            # Create memory data structure
            memory_data = {
                "agent_id": AGENT_ID,
                "content": {"text": content, "metadata": metadata},
                "metadata": {
                    "importance_score": random.random(),
                    "creation_time": time.time(),
                },
                "step_number": i,
                "type": random.choice(["state", "interaction", "action"]),
                "memory_id": memory_id,
            }

            # Calculate checksum
            content_str = json.dumps(memory_data["content"], sort_keys=True)
            checksum = hashlib.md5(content_str.encode()).hexdigest()
            memory_data["metadata"]["checksum"] = checksum

            # Debug log the memory data we're generating
            if i < 3:  # Just log a few examples to avoid log spam
                log_print(self.logger, f"Generated test memory {i}: {memory_id}")
                log_print(self.logger, f"  Content hash: {checksum}")
                log_print(self.logger, f"  Content: {content_str[:50]}...")

            # Add memory to the appropriate tier based on random distribution
            if tier_rand < 0.2:
                # Store in STM (default)
                self.memory_system.add_memory(memory_data)
                stm_count += 1
            elif tier_rand < 0.5:
                # Store directly in IM
                memory_data["metadata"]["compression_level"] = 1
                agent.im_store.store(AGENT_ID, memory_data)
                im_count += 1
            else:
                # Store directly in LTM
                memory_data["metadata"]["compression_level"] = 2
                agent.ltm_store.store(memory_data)
                ltm_count += 1

        # Log memory distribution
        log_print(self.logger, f"Memory generation complete")
        log_print(self.logger, f"  STM Count: {stm_count} ({stm_count/count:.1%})")
        log_print(self.logger, f"  IM Count: {im_count} ({im_count/count:.1%})")
        log_print(self.logger, f"  LTM Count: {ltm_count} ({ltm_count/count:.1%})")

    def run_test(self, test_name: str, query: Any, **search_params):
        """Run a performance test with the given parameters and record metrics."""
        if not self.search_strategy:
            raise ValueError(
                "Search strategy not initialized. Call setup_memory_system first."
            )

        log_print(self.logger, f"\n{'-'*50}")
        log_print(self.logger, f"RUNNING TEST: {test_name}")
        log_print(self.logger, f"  Query: {query}")
        log_print(self.logger, f"  Parameters: {search_params}")
        log_print(self.logger, f"{'-'*50}")

        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Record start time and CPU usage
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)

        # Run search
        results = self.search_strategy.search(
            query=query, 
            agent_id=AGENT_ID, 
            skip_validation=True,  # Always skip validation for performance tests
            **search_params
        )

        # Record end metrics
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        duration = end_time - start_time
        cpu_usage = end_cpu - start_cpu if end_cpu > start_cpu else end_cpu
        memory_usage = end_memory - start_memory
        result_count = len(results)

        # Log results
        log_print(self.logger, f"{'-'*50}")
        log_print(self.logger, f"TEST RESULTS: {test_name}")
        log_print(self.logger, f"  Duration: {duration:.6f} seconds")
        log_print(self.logger, f"  CPU Usage: {cpu_usage:.2f}%")
        log_print(self.logger, f"  Memory Usage: {memory_usage:.2f} MB")
        log_print(self.logger, f"  Results Count: {result_count}")
        log_print(self.logger, f"{'-'*50}")
        
        # Generate and log test summary
        self.generate_test_summary(test_name, duration, cpu_usage, memory_usage, result_count, search_params)

        # Store results
        self.results.append(
            {
                "test_name": test_name,
                "query": str(query),
                "duration": duration,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "result_count": result_count,
                "params": search_params,
            }
        )

        return duration, results
        
    def generate_test_summary(self, test_name, duration, cpu_usage, memory_usage, result_count, params):
        """Generate and log a summary for a completed test."""
        log_print(self.logger, f"\n{'-'*20} SUMMARY: {test_name} {'-'*20}")
        
        # Performance rating based on duration
        if duration < 0.1:
            perf_rating = "Excellent"
        elif duration < 0.5:
            perf_rating = "Good"
        elif duration < 1.0:
            perf_rating = "Acceptable"
        else:
            perf_rating = "Needs improvement"
            
        # Memory usage assessment
        if memory_usage < 5:
            mem_rating = "Minimal"
        elif memory_usage < 20:
            mem_rating = "Moderate"
        else:
            mem_rating = "High"
            
        log_print(self.logger, f"Performance rating: {perf_rating} ({duration:.6f}s)")
        log_print(self.logger, f"Memory consumption: {mem_rating} ({memory_usage:.2f} MB)")
        log_print(self.logger, f"Results found: {result_count}")
        
        # Summary statement
        log_print(self.logger, f"SUMMARY: {test_name} completed with {perf_rating.lower()} performance and {mem_rating.lower()} memory usage.")
        
        # Include any notable parameter effects
        if params:
            notable_params = []
            for key, value in params.items():
                if key in ['limit', 'scoring_method', 'use_regex', 'case_sensitive', 'match_all']:
                    notable_params.append(f"{key}={value}")
            
            if notable_params:
                log_print(self.logger, f"Notable parameters: {', '.join(notable_params)}")
                
        log_print(self.logger, f"{'-'*60}")

    def run_scaling_test(self, query, memory_sizes):
        """Test performance scaling with increasing memory counts."""
        log_print(self.logger, f"{'='*80}")
        log_print(self.logger, f"RUNNING SCALING TEST WITH QUERY: {query}")
        log_print(self.logger, f"Memory sizes to test: {memory_sizes}")
        log_print(self.logger, f"{'='*80}")

        scaling_results = []

        for size in memory_sizes:
            log_print(self.logger, f"\n{'-'*50}")
            log_print(self.logger, f"TESTING WITH {size} MEMORIES...")
            log_print(self.logger, f"{'-'*50}")
            self.setup_memory_system(memory_count=size)

            # Run the test
            test_name = f"Scaling test - {size} memories"
            duration, _ = self.run_test(test_name, query, limit=10)

            scaling_results.append({"memory_count": size, "duration": duration})

        log_print(self.logger, f"\n{'='*80}")
        log_print(self.logger, "SCALING TEST COMPLETE")
        
        # Add summary for the entire scaling test
        log_print(self.logger, f"\n{'-'*20} SCALING TEST SUMMARY {'-'*20}")
        min_size = min(memory_sizes)
        max_size = max(memory_sizes)
        min_duration = min([r["duration"] for r in scaling_results])
        max_duration = max([r["duration"] for r in scaling_results])
        scaling_factor = max_duration / min_duration if min_duration > 0 else "N/A"
        
        log_print(self.logger, f"Memory size range tested: {min_size} to {max_size} memories")
        log_print(self.logger, f"Duration range: {min_duration:.6f}s to {max_duration:.6f}s")
        log_print(self.logger, f"Scaling factor: {scaling_factor if isinstance(scaling_factor, str) else f'{scaling_factor:.2f}x'}")
        
        # Analyze scaling characteristics
        if isinstance(scaling_factor, str) or scaling_factor < len(memory_sizes):
            log_print(self.logger, "Scaling characteristics: Sub-linear (good)")
        elif scaling_factor < 2 * len(memory_sizes):
            log_print(self.logger, "Scaling characteristics: Roughly linear")
        else:
            log_print(self.logger, "Scaling characteristics: Super-linear (may need optimization)")
        
        log_print(self.logger, f"{'-'*60}")
        log_print(self.logger, f"{'='*80}")
        
        return scaling_results

    def run_concurrent_test(self, query, concurrency_levels):
        """Test performance under concurrent load."""
        log_print(self.logger, f"{'='*80}")
        log_print(self.logger, f"RUNNING CONCURRENT TEST WITH QUERY: {query}")
        log_print(self.logger, f"Concurrency levels to test: {concurrency_levels}")
        log_print(self.logger, f"{'='*80}")

        concurrent_results = []

        # Setup with a fixed size
        log_print(
            self.logger, "\nSetting up system with 5000 memories for concurrent testing"
        )
        self.setup_memory_system(memory_count=5000)

        for concurrency in concurrency_levels:
            log_print(self.logger, f"\n{'-'*50}")
            log_print(self.logger, f"TESTING WITH CONCURRENCY LEVEL {concurrency}...")
            log_print(self.logger, f"{'-'*50}")
            start_time = time.time()

            # Function for thread pool
            def search_task():
                return self.search_strategy.search(
                    query=query, agent_id=AGENT_ID, limit=10
                )

            # Execute concurrent searches
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(search_task) for _ in range(concurrency)]
                results = [future.result() for future in futures]

            end_time = time.time()
            total_duration = end_time - start_time
            avg_duration = total_duration / concurrency

            log_print(self.logger, f"  Total duration: {total_duration:.6f} seconds")
            log_print(self.logger, f"  Average duration: {avg_duration:.6f} seconds")

            concurrent_results.append(
                {
                    "concurrency": concurrency,
                    "total_duration": total_duration,
                    "avg_duration": avg_duration,
                }
            )

        log_print(self.logger, f"\n{'='*80}")
        log_print(self.logger, "CONCURRENT TEST COMPLETE")
        
        # Add summary for concurrent test
        log_print(self.logger, f"\n{'-'*20} CONCURRENT TEST SUMMARY {'-'*20}")
        min_concurrency = min(concurrency_levels)
        max_concurrency = max(concurrency_levels)
        baseline_duration = next((r["total_duration"] for r in concurrent_results if r["concurrency"] == min_concurrency), 0)
        max_concurrency_duration = next((r["total_duration"] for r in concurrent_results if r["concurrency"] == max_concurrency), 0)
        
        log_print(self.logger, f"Concurrency range tested: {min_concurrency} to {max_concurrency}")
        log_print(self.logger, f"Baseline duration (concurrency={min_concurrency}): {baseline_duration:.6f}s")
        log_print(self.logger, f"Max concurrency duration: {max_concurrency_duration:.6f}s")
        
        # Calculate overhead
        if baseline_duration > 0:
            overhead = (max_concurrency_duration / baseline_duration) / max_concurrency
            log_print(self.logger, f"Concurrent processing overhead: {overhead:.2%}")
            
            if overhead < 0.2:
                log_print(self.logger, "Concurrency efficiency: Excellent")
            elif overhead < 0.5:
                log_print(self.logger, "Concurrency efficiency: Good")
            elif overhead < 1.0:
                log_print(self.logger, "Concurrency efficiency: Fair")
            else:
                log_print(self.logger, "Concurrency efficiency: Poor - may need optimization")
        
        log_print(self.logger, f"{'-'*60}")
        log_print(self.logger, f"{'='*80}")
        
        return concurrent_results

    def run_scoring_method_comparison(self, query):
        """Compare performance of different scoring methods."""
        scoring_methods = ["length_ratio", "term_frequency", "bm25", "binary"]
        log_print(self.logger, f"{'='*80}")
        log_print(self.logger, f"RUNNING SCORING METHOD COMPARISON WITH QUERY: {query}")
        log_print(self.logger, f"Methods to test: {scoring_methods}")
        log_print(self.logger, f"{'='*80}")

        scoring_results = []

        # Setup with a fixed size
        log_print(
            self.logger,
            "\nSetting up system with 5000 memories for scoring method testing",
        )
        self.setup_memory_system(memory_count=5000)

        for method in scoring_methods:
            log_print(self.logger, f"\n{'-'*50}")
            log_print(self.logger, f"TESTING SCORING METHOD: {method}")
            log_print(self.logger, f"{'-'*50}")
            test_name = f"Scoring method - {method}"
            duration, results = self.run_test(
                test_name, query, limit=100, scoring_method=method
            )

            scoring_results.append(
                {
                    "scoring_method": method,
                    "duration": duration,
                    "result_count": len(results),
                }
            )

        log_print(self.logger, f"\n{'='*80}")
        log_print(self.logger, "SCORING METHOD COMPARISON COMPLETE")
        
        # Add summary for scoring method comparison
        log_print(self.logger, f"\n{'-'*20} SCORING METHOD COMPARISON SUMMARY {'-'*20}")
        
        # Find fastest and slowest methods
        fastest_method = min(scoring_results, key=lambda x: x["duration"])
        slowest_method = max(scoring_results, key=lambda x: x["duration"])
        
        log_print(self.logger, f"Fastest scoring method: {fastest_method['scoring_method']} ({fastest_method['duration']:.6f}s)")
        log_print(self.logger, f"Slowest scoring method: {slowest_method['scoring_method']} ({slowest_method['duration']:.6f}s)")
        log_print(self.logger, f"Speed difference factor: {slowest_method['duration']/fastest_method['duration']:.2f}x")
        
        # Analyze result counts
        result_counts = {r["scoring_method"]: r["result_count"] for r in scoring_results}
        log_print(self.logger, f"Result counts by method:")
        for method, count in result_counts.items():
            log_print(self.logger, f"  {method}: {count} results")
            
        # Method recommendations
        log_print(self.logger, "RECOMMENDATIONS:")
        if fastest_method["duration"] < 0.1:
            log_print(self.logger, f"- For best performance, use the {fastest_method['scoring_method']} method")
        else:
            log_print(self.logger, f"- For balanced performance, consider {fastest_method['scoring_method']}")
            
        # Check for any outliers in result count
        avg_count = sum(result_counts.values()) / len(result_counts)
        for method, count in result_counts.items():
            if count > avg_count * 1.5 or count < avg_count * 0.5:
                log_print(self.logger, f"- Note: {method} returned significantly {'more' if count > avg_count else 'fewer'} results")
        
        log_print(self.logger, f"{'-'*60}")
        log_print(self.logger, f"{'='*80}")
        
        return scoring_results

    def run_regex_comparison(self, std_query, regex_query):
        """Compare performance between standard and regex queries."""
        log_print(self.logger, f"{'='*80}")
        log_print(self.logger, f"RUNNING REGEX COMPARISON")
        log_print(self.logger, f"  Standard query: {std_query}")
        log_print(self.logger, f"  Regex query: {regex_query}")
        log_print(self.logger, f"{'='*80}")

        # Setup with a fixed size
        log_print(
            self.logger, "\nSetting up system with 5000 memories for regex comparison"
        )
        self.setup_memory_system(memory_count=5000)

        # Standard query
        log_print(self.logger, f"\n{'-'*50}")
        log_print(self.logger, "TESTING STANDARD QUERY")
        log_print(self.logger, f"{'-'*50}")
        std_duration, std_results = self.run_test(
            "Standard query", std_query, limit=100
        )

        # Regex query
        log_print(self.logger, f"\n{'-'*50}")
        log_print(self.logger, "TESTING REGEX QUERY")
        log_print(self.logger, f"{'-'*50}")
        regex_duration, regex_results = self.run_test(
            "Regex query", regex_query, limit=100, use_regex=True
        )

        log_print(self.logger, f"\n{'='*80}")
        log_print(self.logger, "REGEX COMPARISON COMPLETE")
        log_print(
            self.logger,
            f"  Standard query time: {std_duration:.6f}s, results: {len(std_results)}",
        )
        log_print(
            self.logger,
            f"  Regex query time: {regex_duration:.6f}s, results: {len(regex_results)}",
        )
        
        # Add summary for regex comparison
        log_print(self.logger, f"\n{'-'*20} REGEX COMPARISON SUMMARY {'-'*20}")
        
        # Calculate performance difference
        speed_ratio = regex_duration / std_duration if std_duration > 0 else float('inf')
        result_ratio = len(regex_results) / len(std_results) if len(std_results) > 0 else float('inf')
        
        log_print(self.logger, f"Performance comparison:")
        log_print(self.logger, f"  Standard query: {std_duration:.6f}s for {len(std_results)} results")
        log_print(self.logger, f"  Regex query: {regex_duration:.6f}s for {len(regex_results)} results")
        log_print(self.logger, f"  Speed ratio: Regex is {speed_ratio:.2f}x {'slower' if speed_ratio > 1 else 'faster'} than standard query")
        
        # Analyze result differences
        common_result_count = 0
        if len(std_results) > 0 and len(regex_results) > 0:
            # Just hypothetical - we'd need to compare actual results
            # This is just a placeholder for the concept
            log_print(self.logger, f"  Result difference ratio: Regex returned {result_ratio:.2f}x {'more' if result_ratio > 1 else 'fewer'} results")
        
        # Recommendations
        log_print(self.logger, "RECOMMENDATIONS:")
        if speed_ratio > 3:
            log_print(self.logger, "- Standard queries are significantly faster for this workload")
            log_print(self.logger, "- Use regex only when pattern matching is essential")
        elif speed_ratio < 1.5:
            log_print(self.logger, "- Both standard and regex queries perform similarly")
            log_print(self.logger, "- Choose based on query flexibility needs rather than performance")
        else:
            log_print(self.logger, "- Regex queries have moderate performance impact")
            log_print(self.logger, "- Consider query complexity and result precision when choosing method")
        
        log_print(self.logger, f"{'-'*60}")
        log_print(self.logger, f"{'='*80}")

        return {
            "standard": {
                "query": std_query,
                "duration": std_duration,
                "result_count": len(std_results),
            },
            "regex": {
                "query": regex_query,
                "duration": regex_duration,
                "result_count": len(regex_results),
            },
        }

    def run_all_tests(self):
        """Run a comprehensive set of performance tests."""
        log_print(self.logger, "Starting comprehensive performance test suite")

        # Test 1: Memory scaling test
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*30} TEST 1: MEMORY SCALING {'#'*30}")
        log_print(self.logger, f"{'#'*80}")
        memory_sizes = [100, 500, 1000, 5000, 10000]
        scaling_results = self.run_scaling_test("meeting", memory_sizes)

        # Calculate scaling factor for later use
        if scaling_results and len(scaling_results) > 1:
            min_duration = min([r["duration"] for r in scaling_results])
            max_duration = max([r["duration"] for r in scaling_results])
            scaling_factor = max_duration / min_duration if min_duration > 0 else float('inf')
        else:
            scaling_factor = 1.0
            
        # Test 2: Concurrent load test
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*30} TEST 2: CONCURRENT LOAD {'#'*30}")
        log_print(self.logger, f"{'#'*80}")
        concurrency_levels = [1, 5, 10, 20, 50]
        concurrent_results = self.run_concurrent_test("project", concurrency_levels)

        # Test 3: Scoring method comparison
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*29} TEST 3: SCORING METHOD COMPARISON {'#'*29}")
        log_print(self.logger, f"{'#'*80}")
        scoring_results = self.run_scoring_method_comparison("report")

        # Test 4: Regex vs standard comparison
        log_print(self.logger, f"\n{'#'*80}")
        log_print(
            self.logger, f"{'#'*29} TEST 4: REGEX VS STANDARD COMPARISON {'#'*29}"
        )
        log_print(self.logger, f"{'#'*80}")
        regex_comparison = self.run_regex_comparison(
            "meeting with team", "meeting.*team"
        )

        # Test 5: Complex query performance
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*29} TEST 5: COMPLEX QUERY PERFORMANCE {'#'*29}")
        log_print(self.logger, f"{'#'*80}")
        log_print(
            self.logger,
            "Setting up system with 5000 memories for complex query testing",
        )
        self.setup_memory_system(memory_count=5000)
        complex_query = {
            "content": "project",
            "metadata": {"importance": "high", "type": "meeting"},
        }
        log_print(self.logger, f"Testing complex query: {complex_query}")
        complex_duration, complex_results = self.run_test("Complex query test", complex_query, match_all=True, limit=100)
        
        # Complex query summary
        log_print(self.logger, f"\n{'-'*20} COMPLEX QUERY SUMMARY {'-'*20}")
        log_print(self.logger, f"Query structure: Nested with metadata filters")
        log_print(self.logger, f"Performance: {complex_duration:.6f}s for {len(complex_results)} results")
        
        if complex_duration > 0.5:
            log_print(self.logger, "Complex queries show significant performance impact")
            log_print(self.logger, "RECOMMENDATION: Consider optimizing metadata indexing for frequent query patterns")
        else:
            log_print(self.logger, "Complex queries perform adequately")
            log_print(self.logger, "RECOMMENDATION: No optimization needed at current load levels")
        log_print(self.logger, f"{'-'*60}")

        # Test 6: Filter performance
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*30} TEST 6: FILTER PERFORMANCE {'#'*30}")
        log_print(self.logger, f"{'#'*80}")
        log_print(self.logger, "Testing metadata filter performance")
        filter_duration, filter_results = self.run_test(
            "Filter test",
            "project",
            metadata_filter={"content.metadata.importance": "high"},
            limit=100,
        )
        
        # Filter test summary
        log_print(self.logger, f"\n{'-'*20} FILTER TEST SUMMARY {'-'*20}")
        log_print(self.logger, f"Filter applied: content.metadata.importance=high")
        log_print(self.logger, f"Performance: {filter_duration:.6f}s for {len(filter_results)} results")
        
        # Compare to complex query performance
        if complex_duration > 0:
            filter_vs_complex = filter_duration / complex_duration
            log_print(self.logger, f"Filter vs complex query: {filter_vs_complex:.2f}x {'slower' if filter_vs_complex > 1 else 'faster'}")
            
            if filter_vs_complex < 0.7:
                log_print(self.logger, "FINDING: Metadata filters are more efficient than complex nested queries")
            elif filter_vs_complex > 1.3:
                log_print(self.logger, "FINDING: Metadata filters have overhead compared to complex queries")
            else:
                log_print(self.logger, "FINDING: Metadata filters and complex queries have similar performance")
        log_print(self.logger, f"{'-'*60}")

        # Test 7: Case sensitivity performance impact
        log_print(self.logger, f"\n{'#'*80}")
        log_print(self.logger, f"{'#'*28} TEST 7: CASE SENSITIVITY IMPACT {'#'*28}")
        log_print(self.logger, f"{'#'*80}")
        log_print(self.logger, "Testing case sensitivity impact")
        sensitive_duration, sensitive_results = self.run_test("Case sensitive test", "Project", case_sensitive=True, limit=100)
        insensitive_duration, insensitive_results = self.run_test(
            "Case insensitive test", "Project", case_sensitive=False, limit=100
        )
        
        # Case sensitivity summary
        log_print(self.logger, f"\n{'-'*20} CASE SENSITIVITY SUMMARY {'-'*20}")
        log_print(self.logger, f"Case sensitive search: {sensitive_duration:.6f}s for {len(sensitive_results)} results")
        log_print(self.logger, f"Case insensitive search: {insensitive_duration:.6f}s for {len(insensitive_results)} results")
        
        # Compare performance
        if sensitive_duration > 0:
            case_ratio = insensitive_duration / sensitive_duration
            log_print(self.logger, f"Performance ratio: Case insensitive is {case_ratio:.2f}x {'slower' if case_ratio > 1 else 'faster'}")
        
        # Compare result counts
        result_diff = abs(len(sensitive_results) - len(insensitive_results))
        if result_diff > 0:
            log_print(self.logger, f"Result difference: {result_diff} more results with case {'sensitive' if len(sensitive_results) > len(insensitive_results) else 'insensitive'} search")
        
        # Recommendation
        if insensitive_duration > sensitive_duration * 1.5:
            log_print(self.logger, "RECOMMENDATION: Use case sensitive search when possible for better performance")
        else:
            log_print(self.logger, "RECOMMENDATION: Case insensitivity has minimal performance impact - use based on UX needs")
        log_print(self.logger, f"{'-'*60}")

        # Save all results
        self.save_results()

        # Generate plots
        self.generate_plots(
            scaling_results, concurrent_results, scoring_results, regex_comparison
        )
        
        # Final comprehensive test summary
        log_print(self.logger, f"\n{'='*80}")
        log_print(self.logger, "ALL PERFORMANCE TESTS COMPLETE")
        log_print(self.logger, f"\n{'#'*20} COMPREHENSIVE TEST SUMMARY {'#'*20}")
        
        # Collect key findings
        all_test_durations = [test["duration"] for test in self.results]
        avg_duration = sum(all_test_durations) / len(all_test_durations) if all_test_durations else 0
        max_duration = max(all_test_durations) if all_test_durations else 0
        min_duration = min(all_test_durations) if all_test_durations else 0
        
        log_print(self.logger, f"Tests executed: {len(self.results)}")
        log_print(self.logger, f"Average query duration: {avg_duration:.6f}s")
        log_print(self.logger, f"Duration range: {min_duration:.6f}s - {max_duration:.6f}s")
        
        # Overall performance assessment
        if avg_duration < 0.1:
            performance_rating = "Excellent"
        elif avg_duration < 0.3:
            performance_rating = "Good"
        elif avg_duration < 0.5:
            performance_rating = "Acceptable"
        else:
            performance_rating = "Needs optimization"
            
        log_print(self.logger, f"Overall performance rating: {performance_rating}")
        
        # Key recommendations
        log_print(self.logger, "KEY RECOMMENDATIONS:")
        
        # Add specific recommendations based on test results
        if any(test["duration"] > 1.0 for test in self.results):
            slow_tests = [test["test_name"] for test in self.results if test["duration"] > 1.0]
            log_print(self.logger, f"- Consider optimizing slow operations: {', '.join(slow_tests[:3])}")
            
        # Add scaling recommendation
        if scaling_factor > 2:
            log_print(self.logger, f"- Improve scaling characteristics for larger memory sets")
            
        log_print(self.logger, f"{'#'*60}")
        log_print(self.logger, f"{'='*80}")

    def save_results(self):
        """Save test results to CSV file."""
        csv_path = os.path.join(RESULTS_DIR, "attribute_search_performance.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        log_print(self.logger, f"Results saved to {csv_path}")

        # Also save as JSON for easier analysis
        json_path = os.path.join(RESULTS_DIR, "attribute_search_performance.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        log_print(self.logger, f"Results also saved as JSON to {json_path}")

    def generate_plots(
        self, scaling_results, concurrent_results, scoring_results, regex_comparison
    ):
        """Generate performance visualization plots."""
        log_print(self.logger, "Generating performance visualization plots")

        # Scaling plot
        plt.figure(figsize=(10, 6))
        scaling_df = pd.DataFrame(scaling_results)
        plt.plot(scaling_df["memory_count"], scaling_df["duration"], marker="o")
        plt.title("Search Performance Scaling with Memory Size")
        plt.xlabel("Number of Memories")
        plt.ylabel("Duration (seconds)")
        plt.grid(True)
        scaling_plot = os.path.join(RESULTS_DIR, "scaling_performance.png")
        plt.savefig(scaling_plot)
        log_print(self.logger, f"Saved scaling plot to {scaling_plot}")

        # Concurrency plot
        plt.figure(figsize=(10, 6))
        concurrent_df = pd.DataFrame(concurrent_results)
        plt.plot(
            concurrent_df["concurrency"],
            concurrent_df["avg_duration"],
            marker="o",
            label="Avg Duration",
        )
        plt.plot(
            concurrent_df["concurrency"],
            concurrent_df["total_duration"],
            marker="s",
            label="Total Duration",
        )
        plt.title("Search Performance under Concurrent Load")
        plt.xlabel("Concurrency Level")
        plt.ylabel("Duration (seconds)")
        plt.legend()
        plt.grid(True)
        concurrency_plot = os.path.join(RESULTS_DIR, "concurrent_performance.png")
        plt.savefig(concurrency_plot)
        log_print(self.logger, f"Saved concurrency plot to {concurrency_plot}")

        # Scoring methods comparison
        plt.figure(figsize=(10, 6))
        scoring_df = pd.DataFrame(scoring_results)
        plt.bar(scoring_df["scoring_method"], scoring_df["duration"])
        plt.title("Performance by Scoring Method")
        plt.xlabel("Scoring Method")
        plt.ylabel("Duration (seconds)")
        scoring_plot = os.path.join(RESULTS_DIR, "scoring_method_performance.png")
        plt.savefig(scoring_plot)
        log_print(self.logger, f"Saved scoring method plot to {scoring_plot}")

        # Regex vs standard comparison
        plt.figure(figsize=(8, 5))
        regex_data = [
            regex_comparison["standard"]["duration"],
            regex_comparison["regex"]["duration"],
        ]
        plt.bar(["Standard", "Regex"], regex_data)
        plt.title("Standard vs Regex Query Performance")
        plt.ylabel("Duration (seconds)")
        regex_plot = os.path.join(RESULTS_DIR, "regex_comparison.png")
        plt.savefig(regex_plot)
        log_print(self.logger, f"Saved regex comparison plot to {regex_plot}")

        log_print(self.logger, "Performance plot generation complete")


def main():
    """Main entry point for performance testing."""
    # Setup logging
    logger = setup_logging("attribute_search_performance")
    log_print(logger, "Starting Attribute Search Performance Testing")

    try:
        perf_test = PerformanceTest(logger)
        perf_test.run_all_tests()
        log_print(logger, "Performance testing completed successfully")
    except Exception as e:
        log_print(logger, f"ERROR: Performance testing failed: {e}")
        import traceback

        log_print(logger, traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
