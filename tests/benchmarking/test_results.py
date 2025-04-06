"""
Unit tests for the BenchmarkResults class.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from memory.benchmarking.results import BenchmarkResults


class TestBenchmarkResults:
    """Test cases for the BenchmarkResults class."""

    def test_initialization(self):
        """Test initialization with default and custom paths."""
        # Test with default path
        results = BenchmarkResults()
        assert results.results_dir == "benchmark_results"
        
        # Test with custom path
        custom_path = "tests/benchmarking/test_results"
        results = BenchmarkResults(custom_path)
        assert results.results_dir == custom_path

    def test_save_and_load_result(self):
        """Test saving and loading benchmark results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = BenchmarkResults(temp_dir)
            
            # Sample benchmark data
            category = "storage"
            benchmark_name = "write_speed"
            benchmark_results = {
                "throughput": 1500,
                "latency_ms": 5.2,
                "batch_sizes": [10, 100, 1000],
                "metrics": {
                    "min_latency": 2.1,
                    "max_latency": 10.5,
                    "avg_latency": 5.2
                }
            }
            metadata = {
                "device": "test_device",
                "version": "1.0.0"
            }
            
            # Save the result
            result_path = results.save_result(
                category=category,
                benchmark_name=benchmark_name,
                results=benchmark_results,
                metadata=metadata
            )
            
            # Verify the file exists
            assert os.path.exists(result_path)
            
            # Load the result
            loaded_result = results.load_result(result_path)
            
            # Verify loaded content
            assert loaded_result["benchmark"] == benchmark_name
            assert loaded_result["category"] == category
            assert "timestamp" in loaded_result
            assert loaded_result["results"] == benchmark_results
            assert loaded_result["metadata"] == metadata

    def test_list_results(self):
        """Test listing benchmark results with different filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = BenchmarkResults(temp_dir)
            
            # Create sample results for different categories and benchmarks
            categories = ["storage", "retrieval"]
            benchmarks = ["speed_test", "accuracy_test"]
            
            created_files = []
            
            for category in categories:
                for benchmark in benchmarks:
                    # Create sample result
                    result_path = results.save_result(
                        category=category,
                        benchmark_name=benchmark,
                        results={"sample": "data"}
                    )
                    created_files.append(result_path)
            
            # List all results
            all_results = results.list_results()
            assert len(all_results) == 4
            
            # List results by category
            storage_results = results.list_results(category="storage")
            assert len(storage_results) == 2
            
            # List results by benchmark name
            speed_results = results.list_results(benchmark_name="speed_test")
            assert len(speed_results) == 2
            
            # List results by category and benchmark name
            specific_results = results.list_results(
                category="retrieval", 
                benchmark_name="accuracy_test"
            )
            assert len(specific_results) == 1
            
            # Test with non-existent category
            empty_results = results.list_results(category="non_existent")
            assert len(empty_results) == 0

    def test_compare_results(self):
        """Test comparing results from multiple benchmark runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = BenchmarkResults(temp_dir)
            
            # Create sample results for comparison
            filepaths = []
            
            for i in range(3):
                result_path = results.save_result(
                    category="storage",
                    benchmark_name="write_speed",
                    results={
                        "throughput": 1000 + i * 200,
                        "latency_ms": 10 - i,
                        "nested": {
                            "value": 5 + i
                        }
                    }
                )
                filepaths.append(result_path)
            
            # Compare the results
            comparison_df = results.compare_results(filepaths)
            
            # Check that the DataFrame has the expected structure
            assert isinstance(comparison_df, pd.DataFrame)
            assert "run" in comparison_df.columns
            assert "benchmark" in comparison_df.columns
            assert "category" in comparison_df.columns
            assert "timestamp" in comparison_df.columns
            assert "throughput" in comparison_df.columns
            assert "latency_ms" in comparison_df.columns
            assert "nested_value" in comparison_df.columns
            
            # Check that there are 3 rows (one for each result)
            assert len(comparison_df) == 3
            
            # Compare only specific metrics
            metrics_df = results.compare_results(filepaths, metrics=["throughput"])
            assert "throughput" in metrics_df.columns
            assert "latency_ms" not in metrics_df.columns

    @pytest.mark.parametrize("x_axis", [None, "throughput"])
    def test_plot_comparison(self, x_axis):
        """Test plotting comparison of benchmark results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = BenchmarkResults(temp_dir)
            
            # Create sample results with numeric data for plotting
            filepaths = []
            
            for i in range(3):
                result_path = results.save_result(
                    category="storage",
                    benchmark_name="write_speed",
                    results={
                        "throughput": 1000 + i * 200,
                        "latency_ms": 10 - i
                    }
                )
                filepaths.append(result_path)
            
            # Plot the comparison
            fig = results.plot_comparison(
                filepaths=filepaths,
                metric="latency_ms",
                x_axis=x_axis,
                title="Test Plot"
            )
            
            # Check that a valid figure was created
            assert isinstance(fig, plt.Figure)
            
            # Test with non-existent metric (should raise ValueError)
            with pytest.raises(ValueError):
                results.plot_comparison(
                    filepaths=filepaths,
                    metric="non_existent_metric"
                )

    def test_flatten_dict(self):
        """Test the _flatten_dict helper method."""
        results = BenchmarkResults()
        
        # Test with a nested dictionary
        nested_dict = {
            "top_level": "value",
            "nested": {
                "level1": "value1",
                "deeper": {
                    "level2": "value2"
                }
            },
            "list_data": [1, 2, 3]
        }
        
        flat_dict = results._flatten_dict(nested_dict)
        
        # Check flattened structure
        assert flat_dict["top_level"] == "value"
        assert flat_dict["nested_level1"] == "value1"
        assert flat_dict["nested_deeper_level2"] == "value2"
        
        # For lists of numbers, the implementation creates stats instead of preserving the list
        assert "list_data_avg" in flat_dict
        assert flat_dict["list_data_avg"] == 2.0  # (1+2+3)/3
        assert flat_dict["list_data_max"] == 3
        assert flat_dict["list_data_min"] == 1 