"""
Command-line interface for running benchmarks.

This module provides a command-line interface for running benchmarks 
on the Agent Memory System.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, Optional, List

from memory.benchmarking.config import BenchmarkConfig
from memory.benchmarking.runner import BenchmarkRunner
from memory.benchmarking.results import BenchmarkResults


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run benchmarks for the Agent Memory System"
    )
    
    # Define command types
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--category", 
        help="Benchmark category to run (e.g., 'storage', 'retrieval')"
    )
    run_parser.add_argument(
        "--benchmark", 
        help="Specific benchmark to run"
    )
    run_parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    run_parser.add_argument(
        "--output-dir", 
        help="Directory to store results"
    )
    run_parser.add_argument(
        "--params", 
        help="JSON string or file path with parameters to override"
    )
    run_parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run benchmarks in parallel"
    )
    run_parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of parallel workers"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available benchmarks")
    list_parser.add_argument(
        "--category", 
        help="Benchmark category to list"
    )
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Work with benchmark results")
    results_subparsers = results_parser.add_subparsers(dest="results_command", help="Results command")
    
    # Results list command
    results_list_parser = results_subparsers.add_parser("list", help="List available results")
    results_list_parser.add_argument(
        "--category", 
        help="Filter by category"
    )
    results_list_parser.add_argument(
        "--benchmark", 
        help="Filter by benchmark name"
    )
    results_list_parser.add_argument(
        "--results-dir", 
        help="Directory containing results"
    )
    
    # Results report command
    results_report_parser = results_subparsers.add_parser("report", help="Generate a results report")
    results_report_parser.add_argument(
        "--category", 
        help="Filter by category"
    )
    results_report_parser.add_argument(
        "--benchmark", 
        help="Filter by benchmark name"
    )
    results_report_parser.add_argument(
        "--results-dir", 
        help="Directory containing results"
    )
    results_report_parser.add_argument(
        "--output-dir", 
        help="Directory to save the report"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "--current-dir", 
        required=True, 
        help="Directory containing current results"
    )
    compare_parser.add_argument(
        "--baseline-dir", 
        required=True, 
        help="Directory containing baseline results"
    )
    compare_parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.1, 
        help="Threshold for detecting regressions (default: 0.1 or 10%%)"
    )
    compare_parser.add_argument(
        "--output-dir", 
        help="Directory to save the comparison report"
    )
    
    return parser.parse_args()


def load_params(params_arg: str) -> Dict[str, Any]:
    """Load parameters from a JSON string or file.
    
    Args:
        params_arg: JSON string or file path
        
    Returns:
        Dictionary with parameters
    """
    if os.path.isfile(params_arg):
        with open(params_arg, 'r') as f:
            return json.load(f)
    else:
        try:
            return json.loads(params_arg)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in params: {params_arg}")


def run_command(args: argparse.Namespace) -> int:
    """Run the specified benchmarks.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    # Load configuration
    config = None
    if args.config:
        config = BenchmarkConfig.load(args.config)
    else:
        config = BenchmarkConfig()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Create benchmark runner
    runner = BenchmarkRunner(config)
    
    # Load parameters if specified
    params = None
    if args.params:
        params = load_params(args.params)
    
    # Determine what to run
    if args.benchmark and not args.category:
        print("Error: --benchmark requires --category to be specified")
        return 1
    
    try:
        if args.category and args.benchmark:
            # Run a specific benchmark
            print(f"Running benchmark: {args.category}.{args.benchmark}")
            result_path = runner.run_benchmark(args.category, args.benchmark, params)
            print(f"Results saved to: {result_path}")
            
        elif args.category:
            # Run all benchmarks in a category
            print(f"Running all benchmarks in category: {args.category}")
            result_paths = runner.run_category(
                args.category, 
                params, 
                args.parallel, 
                args.workers
            )
            print(f"Completed {len(result_paths)} benchmarks")
            for path in result_paths:
                print(f"- {path}")
            
        else:
            # Run all benchmarks
            print("Running all benchmarks")
            parallel_categories = args.parallel
            parallel_benchmarks = args.parallel
            
            category_params = {}
            if params:
                for cat in runner.BENCHMARK_MODULES:
                    if cat in params:
                        category_params[cat] = params.get(cat)
            
            results = runner.run_all(
                category_params, 
                parallel_categories, 
                parallel_benchmarks, 
                args.workers
            )
            
            total_benchmarks = sum(len(paths) for paths in results.values())
            print(f"Completed {total_benchmarks} benchmarks across {len(results)} categories")
        
        return 0
    
    except Exception as e:
        print(f"Error running benchmarks: {str(e)}")
        return 1


def list_command(args: argparse.Namespace) -> int:
    """List available benchmarks.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    runner = BenchmarkRunner()
    
    try:
        if args.category:
            # List benchmarks in a specific category
            benchmarks = runner.discover_benchmarks(args.category)
            if not benchmarks or args.category not in benchmarks:
                print(f"No benchmarks found in category: {args.category}")
                return 1
            
            print(f"Benchmarks in category '{args.category}':")
            for benchmark in sorted(benchmarks[args.category].keys()):
                print(f"- {benchmark}")
        else:
            # List all benchmarks by category
            all_benchmarks = runner.discover_benchmarks()
            if not all_benchmarks:
                print("No benchmarks found")
                return 1
            
            for category, benchmarks in sorted(all_benchmarks.items()):
                print(f"\nCategory: {category}")
                for benchmark in sorted(benchmarks.keys()):
                    print(f"- {benchmark}")
        
        return 0
    
    except Exception as e:
        print(f"Error listing benchmarks: {str(e)}")
        return 1


def results_command(args: argparse.Namespace) -> int:
    """Handle results-related commands.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    if not args.results_command:
        print("Error: Must specify a results command")
        return 1
    
    # Create results manager
    results_dir = args.results_dir or "benchmark_results"
    results_manager = BenchmarkResults(results_dir)
    
    try:
        if args.results_command == "list":
            # List available results
            result_files = results_manager.list_results(args.category, args.benchmark)
            
            if not result_files:
                if args.category and args.benchmark:
                    print(f"No results found for category '{args.category}' and benchmark '{args.benchmark}'")
                elif args.category:
                    print(f"No results found for category '{args.category}'")
                elif args.benchmark:
                    print(f"No results found for benchmark '{args.benchmark}'")
                else:
                    print("No benchmark results found")
                return 1
            
            print(f"Found {len(result_files)} result files:")
            for filepath in result_files:
                try:
                    result = results_manager.load_result(filepath)
                    cat = result.get("category", "unknown")
                    bench = result.get("benchmark", "unknown")
                    timestamp = result.get("timestamp", "unknown")
                    print(f"- [{cat}.{bench}] {filepath} ({timestamp})")
                except:
                    print(f"- {filepath} (error loading)")
            
            return 0
            
        elif args.results_command == "report":
            # Generate a report
            output_dir = args.output_dir
            try:
                report_path = results_manager.generate_report(
                    args.category, 
                    args.benchmark,
                    output_dir
                )
                print(f"Report generated: {report_path}")
                return 0
            except ValueError as e:
                print(f"Error generating report: {str(e)}")
                return 1
        
        else:
            print(f"Unknown results command: {args.results_command}")
            return 1
            
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return 1


def compare_command(args: argparse.Namespace) -> int:
    """Compare benchmark results.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    # Validate directories
    if not os.path.isdir(args.current_dir):
        print(f"Error: Current results directory not found: {args.current_dir}")
        return 1
    
    if not os.path.isdir(args.baseline_dir):
        print(f"Error: Baseline results directory not found: {args.baseline_dir}")
        return 1
    
    try:
        # Create results managers
        current_manager = BenchmarkResults(args.current_dir)
        
        # Create a benchmark runner for comparison
        runner = BenchmarkRunner()
        runner.results_manager = current_manager
        
        # Get all current results
        current_results = {}
        for category in runner.BENCHMARK_MODULES:
            result_files = current_manager.list_results(category)
            if result_files:
                current_results[category] = result_files
        
        if not current_results:
            print("No results found in current directory")
            return 1
        
        # Compare with baseline
        comparison = runner.compare_with_baseline(
            current_results,
            args.baseline_dir,
            args.threshold
        )
        
        if not comparison:
            print("No comparable results found")
            return 1
        
        # Generate comparison report
        report_path = runner.generate_comparison_report(
            comparison,
            args.output_dir
        )
        
        print(f"Comparison report generated: {report_path}")
        
        # Print summary of regressions
        has_regressions = False
        for category, benchmarks in comparison.items():
            for benchmark, metrics in benchmarks.items():
                regressions = sum(1 for m in metrics.values() if m.get("is_regression", False))
                if regressions > 0:
                    has_regressions = True
                    print(f"WARNING: {category}.{benchmark} has {regressions} regression(s)")
        
        if has_regressions:
            print(f"\nRegressions detected! See report for details: {report_path}")
            return 2  # Special code for regressions
        else:
            print("No performance regressions detected")
            return 0
        
    except Exception as e:
        print(f"Error comparing results: {str(e)}")
        return 1


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    if args.command == "run":
        return run_command(args)
    elif args.command == "list":
        return list_command(args)
    elif args.command == "results":
        return results_command(args)
    elif args.command == "compare":
        return compare_command(args)
    else:
        print("Error: Must specify a command")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 