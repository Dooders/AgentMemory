"""
Benchmark runner for executing benchmarks and collecting results.

This module provides functionality for running various benchmarks on the
Agent Memory System and collecting their results.
"""

import os
import time
import importlib
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from memory.benchmarking.config import BenchmarkConfig
from memory.benchmarking.results import BenchmarkResults


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_runner')


class BenchmarkRunner:
    """Class for running benchmarks on the AgentMemory system."""
    
    # Mapping of benchmark categories to their module paths
    BENCHMARK_MODULES = {
        "storage": "agent_memory.benchmarking.benchmarks.storage",
        "compression": "agent_memory.benchmarking.benchmarks.compression",
        "memory_transition": "agent_memory.benchmarking.benchmarks.memory_transition",
        "retrieval": "agent_memory.benchmarking.benchmarks.retrieval",
        "scalability": "agent_memory.benchmarking.benchmarks.scalability",
        "integration": "agent_memory.benchmarking.benchmarks.integration"
    }
    
    def __init__(self, config: Optional[BenchmarkConfig] = None,
                results_manager: Optional[BenchmarkResults] = None):
        """Initialize the benchmark runner.
        
        Args:
            config: Configuration for benchmarks
            results_manager: Results manager for storing results
        """
        self.config = config or BenchmarkConfig()
        self.results_manager = results_manager or BenchmarkResults(self.config.output_dir)
        self.running_benchmarks: Set[str] = set()
        self.completed_benchmarks: Dict[str, str] = {}  # Maps benchmark ID to result file path
    
    def discover_benchmarks(self, category: Optional[str] = None) -> Dict[str, Dict[str, Callable]]:
        """Discover available benchmarks.
        
        Args:
            category: Optional category to limit discovery to
            
        Returns:
            Dictionary mapping categories to their available benchmarks
        """
        available_benchmarks = {}
        
        categories = [category] if category else self.BENCHMARK_MODULES.keys()
        
        for cat in categories:
            if cat not in self.BENCHMARK_MODULES:
                logger.warning(f"Unknown benchmark category: {cat}")
                continue
            
            module_path = self.BENCHMARK_MODULES[cat]
            try:
                module = importlib.import_module(module_path)
                
                # Find all benchmark functions in the module
                benchmarks = {}
                for name in dir(module):
                    if name.startswith("benchmark_") and callable(getattr(module, name)):
                        benchmarks[name[10:]] = getattr(module, name)
                
                if benchmarks:
                    available_benchmarks[cat] = benchmarks
                else:
                    logger.warning(f"No benchmarks found in module: {module_path}")
                
            except ImportError:
                logger.warning(f"Benchmark module not found: {module_path}")
                continue
        
        return available_benchmarks
    
    def run_benchmark(self, category: str, benchmark_name: str,
                     params: Optional[Dict[str, Any]] = None) -> str:
        """Run a specific benchmark.
        
        Args:
            category: Category of the benchmark
            benchmark_name: Name of the benchmark
            params: Optional parameters to pass to the benchmark
            
        Returns:
            Path to the results file
            
        Raises:
            ValueError: If the benchmark cannot be found
        """
        # Add to running benchmarks
        benchmark_id = f"{category}.{benchmark_name}"
        self.running_benchmarks.add(benchmark_id)
        
        try:
            # Import the module containing the benchmark
            if category not in self.BENCHMARK_MODULES:
                raise ValueError(f"Unknown benchmark category: {category}")
            
            module_path = self.BENCHMARK_MODULES[category]
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                raise ValueError(f"Benchmark module not found: {module_path}")
            
            # Look for the benchmark function
            func_name = f"benchmark_{benchmark_name}"
            if not hasattr(module, func_name) or not callable(getattr(module, func_name)):
                raise ValueError(f"Benchmark function not found: {func_name} in {module_path}")
            
            benchmark_func = getattr(module, func_name)
            
            # Get configuration for this category
            category_config = getattr(self.config, category, {})
            
            # Merge provided params with configuration
            run_params = {}
            if category_config:
                # Extract relevant params from category config
                for param_name, param_value in vars(category_config).items():
                    run_params[param_name] = param_value
            
            # Override with explicit params
            if params:
                run_params.update(params)
            
            # Run the benchmark and measure execution time
            logger.info(f"Running benchmark: {benchmark_id}")
            start_time = time.time()
            results = benchmark_func(**run_params)
            end_time = time.time()
            
            # Add execution time to results metadata
            execution_time = end_time - start_time
            logger.info(f"Benchmark {benchmark_id} completed in {execution_time:.2f} seconds")
            
            # Save results
            metadata = {
                "execution_time": execution_time,
                "params": run_params,
                "config": self.config.to_dict()
            }
            
            result_path = self.results_manager.save_result(
                category=category,
                benchmark_name=benchmark_name,
                results=results,
                metadata=metadata
            )
            
            self.completed_benchmarks[benchmark_id] = result_path
            return result_path
            
        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Remove from running benchmarks
            self.running_benchmarks.discard(benchmark_id)
    
    def run_category(self, category: str, 
                    params: Optional[Dict[str, Any]] = None,
                    parallel: bool = False,
                    max_workers: int = 4) -> List[str]:
        """Run all benchmarks in a category.
        
        Args:
            category: Category of benchmarks to run
            params: Optional parameters to pass to the benchmarks
            parallel: Whether to run benchmarks in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to result files
        """
        benchmarks = self.discover_benchmarks(category)
        if not benchmarks or category not in benchmarks:
            logger.warning(f"No benchmarks found for category: {category}")
            return []
        
        benchmark_funcs = benchmarks[category]
        result_paths = []
        
        if parallel and len(benchmark_funcs) > 1:
            logger.info(f"Running {len(benchmark_funcs)} benchmarks in category {category} in parallel")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_benchmark = {
                    executor.submit(self.run_benchmark, category, name, params): name
                    for name in benchmark_funcs
                }
                
                for future in as_completed(future_to_benchmark):
                    benchmark_name = future_to_benchmark[future]
                    try:
                        result_path = future.result()
                        result_paths.append(result_path)
                    except Exception as e:
                        logger.error(f"Benchmark {category}.{benchmark_name} failed: {str(e)}")
        else:
            logger.info(f"Running {len(benchmark_funcs)} benchmarks in category {category} sequentially")
            for benchmark_name in benchmark_funcs:
                try:
                    result_path = self.run_benchmark(category, benchmark_name, params)
                    result_paths.append(result_path)
                except Exception as e:
                    logger.error(f"Benchmark {category}.{benchmark_name} failed: {str(e)}")
        
        return result_paths
    
    def run_all(self, params: Optional[Dict[str, Dict[str, Any]]] = None,
               parallel_categories: bool = False,
               parallel_benchmarks: bool = False,
               max_workers: int = 4) -> Dict[str, List[str]]:
        """Run all available benchmarks.
        
        Args:
            params: Optional parameters to pass to benchmarks, by category
            parallel_categories: Whether to run categories in parallel
            parallel_benchmarks: Whether to run benchmarks within a category in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping categories to lists of result file paths
        """
        all_benchmarks = self.discover_benchmarks()
        if not all_benchmarks:
            logger.warning("No benchmarks discovered")
            return {}
        
        results = {}
        params = params or {}
        
        if parallel_categories and len(all_benchmarks) > 1:
            logger.info(f"Running {len(all_benchmarks)} benchmark categories in parallel")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_category = {
                    executor.submit(
                        self.run_category, 
                        category, 
                        params.get(category), 
                        parallel_benchmarks,
                        max_workers
                    ): category
                    for category in all_benchmarks
                }
                
                for future in as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        category_results = future.result()
                        results[category] = category_results
                    except Exception as e:
                        logger.error(f"Category {category} failed: {str(e)}")
        else:
            logger.info(f"Running {len(all_benchmarks)} benchmark categories sequentially")
            for category in all_benchmarks:
                try:
                    category_results = self.run_category(
                        category, 
                        params.get(category),
                        parallel_benchmarks,
                        max_workers
                    )
                    results[category] = category_results
                except Exception as e:
                    logger.error(f"Category {category} failed: {str(e)}")
        
        return results
    
    def compare_with_baseline(self, current_results: Dict[str, List[str]],
                             baseline_dir: str,
                             threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """Compare current results with a baseline.
        
        Args:
            current_results: Dictionary of current result files by category
            baseline_dir: Directory containing baseline results
            threshold: Threshold for performance regression (fraction)
            
        Returns:
            Dictionary with comparison results
        """
        # Create a baseline results manager
        baseline_manager = BenchmarkResults(baseline_dir)
        
        comparison = {}
        
        for category, result_files in current_results.items():
            category_comparison = {}
            
            # Find baseline results for this category
            baseline_files = baseline_manager.list_results(category)
            if not baseline_files:
                logger.warning(f"No baseline results found for category: {category}")
                continue
            
            # For each benchmark in this category
            for result_file in result_files:
                result_data = self.results_manager.load_result(result_file)
                benchmark_name = result_data.get("benchmark")
                
                # Find matching baseline result
                matching_baselines = [f for f in baseline_files 
                                    if os.path.basename(f).startswith(f"{benchmark_name}_")]
                
                if not matching_baselines:
                    logger.warning(f"No baseline found for benchmark: {benchmark_name}")
                    continue
                
                # Use the most recent baseline
                baseline_file = max(matching_baselines)
                baseline_data = baseline_manager.load_result(baseline_file)
                
                # Compare key metrics
                metrics_comparison = {}
                current_results = result_data.get("results", {})
                baseline_results = baseline_data.get("results", {})
                
                # Flatten results for comparison
                flat_current = self.results_manager._flatten_dict(current_results)
                flat_baseline = self.results_manager._flatten_dict(baseline_results)
                
                # Compare each metric
                for metric, current_value in flat_current.items():
                    if metric in flat_baseline and isinstance(current_value, (int, float)):
                        baseline_value = flat_baseline[metric]
                        if baseline_value != 0:
                            # Calculate relative change
                            rel_change = (current_value - baseline_value) / baseline_value
                            
                            # Determine if this is a regression
                            is_regression = False
                            if "latency" in metric or "time" in metric:
                                # For timing metrics, higher is worse
                                is_regression = rel_change > threshold
                            else:
                                # For throughput and other metrics, lower is worse
                                is_regression = rel_change < -threshold
                            
                            metrics_comparison[metric] = {
                                "current": current_value,
                                "baseline": baseline_value,
                                "relative_change": rel_change,
                                "is_regression": is_regression
                            }
                
                category_comparison[benchmark_name] = metrics_comparison
            
            comparison[category] = category_comparison
        
        return comparison
    
    def generate_comparison_report(self, comparison: Dict[str, Dict[str, Any]],
                                 output_dir: Optional[str] = None) -> str:
        """Generate an HTML report for baseline comparison.
        
        Args:
            comparison: Comparison data from compare_with_baseline
            output_dir: Directory to save the report
            
        Returns:
            Path to the report file
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"baseline_comparison_{timestamp}.html")
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Benchmark Baseline Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .regression {{ background-color: #ffdddd; }}
                .improvement {{ background-color: #ddffdd; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Benchmark Baseline Comparison</h1>
                <div class="summary">
                    <p>Report generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
        """
        
        # Add sections for each category
        for category, benchmarks in comparison.items():
            html += f"<h2>Category: {category}</h2>"
            
            for benchmark, metrics in benchmarks.items():
                html += f"<h3>Benchmark: {benchmark}</h3>"
                
                # Count regressions
                regressions = sum(1 for m in metrics.values() if m.get("is_regression", False))
                if regressions > 0:
                    html += f"<p><strong>Warning: {regressions} metric(s) show performance regression!</strong></p>"
                
                # Create table for metrics
                html += """
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Current</th>
                        <th>Baseline</th>
                        <th>Change (%)</th>
                    </tr>
                """
                
                for metric, values in metrics.items():
                    current = values.get("current")
                    baseline = values.get("baseline")
                    rel_change = values.get("relative_change", 0) * 100  # to percentage
                    is_regression = values.get("is_regression", False)
                    
                    row_class = "regression" if is_regression else (
                        "improvement" if abs(rel_change) > 5 else ""
                    )
                    
                    html += f"""
                    <tr class="{row_class}">
                        <td>{metric}</td>
                        <td>{current:.4f}</td>
                        <td>{baseline:.4f}</td>
                        <td>{rel_change:+.2f}%</td>
                    </tr>
                    """
                
                html += "</table>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, "w") as f:
            f.write(html)
        
        return report_path 