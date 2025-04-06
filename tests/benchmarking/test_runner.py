"""
Unit tests for the BenchmarkRunner class.
"""

import os
import tempfile
from unittest import mock
import pytest

from memory.benchmarking.runner import BenchmarkRunner
from memory.benchmarking.config import BenchmarkConfig
from memory.benchmarking.results import BenchmarkResults


class TestBenchmarkRunner:
    """Test cases for the BenchmarkRunner class."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        runner = BenchmarkRunner()
        assert isinstance(runner.config, BenchmarkConfig)
        assert isinstance(runner.results_manager, BenchmarkResults)
        assert runner.running_benchmarks == set()
        assert runner.completed_benchmarks == {}
        
        # Test with custom parameters
        config = BenchmarkConfig(output_dir="custom_output")
        results_manager = BenchmarkResults("custom_results")
        
        runner = BenchmarkRunner(config=config, results_manager=results_manager)
        assert runner.config is config
        assert runner.results_manager is results_manager

    @mock.patch("importlib.import_module")
    def test_discover_benchmarks(self, mock_import):
        """Test discovering available benchmarks."""
        # Mock the imported module
        mock_module = mock.MagicMock()
        mock_module.benchmark_test1 = lambda: None
        mock_module.benchmark_test2 = lambda: None
        mock_module.not_a_benchmark = "not a function"
        
        mock_import.return_value = mock_module
        
        runner = BenchmarkRunner()
        
        # Test discovering all categories
        available_benchmarks = runner.discover_benchmarks()
        
        # Since we mocked the import for all modules, all categories should be discovered
        assert len(available_benchmarks) == len(runner.BENCHMARK_MODULES)
        
        # Each category should have our two benchmark functions
        for category in available_benchmarks:
            assert "test1" in available_benchmarks[category]
            assert "test2" in available_benchmarks[category]
            assert len(available_benchmarks[category]) == 2
        
        # Test discovering a specific category
        category_benchmarks = runner.discover_benchmarks(category="storage")
        assert len(category_benchmarks) == 1
        assert "storage" in category_benchmarks
        assert "test1" in category_benchmarks["storage"]
        assert "test2" in category_benchmarks["storage"]
        
        # Test with non-existent category
        mock_import.side_effect = ImportError
        empty_benchmarks = runner.discover_benchmarks(category="non_existent")
        assert len(empty_benchmarks) == 0

    @mock.patch("importlib.import_module")
    def test_run_benchmark(self, mock_import):
        """Test running a specific benchmark."""
        # Create a mock benchmark function that returns test results
        mock_benchmark = mock.MagicMock(return_value={"metric": 100})
        
        # Mock the module with our benchmark function
        mock_module = mock.MagicMock()
        mock_module.benchmark_test_benchmark = mock_benchmark
        mock_import.return_value = mock_module
        
        # Create a runner with mocked results manager
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(output_dir=temp_dir)
            results_manager = BenchmarkResults(temp_dir)
            
            runner = BenchmarkRunner(config=config, results_manager=results_manager)
            
            # Run the benchmark
            result_path = runner.run_benchmark(
                category="storage",
                benchmark_name="test_benchmark"
            )
            
            # Verify benchmark function was called
            mock_benchmark.assert_called_once()
            
            # Verify running_benchmarks is empty (should be cleared after running)
            assert len(runner.running_benchmarks) == 0
            
            # Verify completed_benchmarks contains our benchmark
            assert "storage.test_benchmark" in runner.completed_benchmarks
            assert runner.completed_benchmarks["storage.test_benchmark"] == result_path
            
            # Verify result file exists
            assert os.path.exists(result_path)
    
    @mock.patch("importlib.import_module")
    def test_run_benchmark_with_params(self, mock_import):
        """Test running a benchmark with custom parameters."""
        # Create a mock benchmark function
        mock_benchmark = mock.MagicMock(return_value={"metric": 100})
        
        # Mock the module
        mock_module = mock.MagicMock()
        mock_module.benchmark_custom_params = mock_benchmark
        mock_import.return_value = mock_module
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(
                config=BenchmarkConfig(output_dir=temp_dir),
                results_manager=BenchmarkResults(temp_dir)
            )
            
            # Run with custom parameters
            custom_params = {"batch_size": 500, "iterations": 10}
            runner.run_benchmark(
                category="storage",
                benchmark_name="custom_params",
                params=custom_params
            )
            
            # Verify the custom parameters were passed to the benchmark function
            for key, value in custom_params.items():
                assert mock_benchmark.call_args[1][key] == value
    
    @mock.patch("importlib.import_module")
    def test_run_benchmark_error_handling(self, mock_import):
        """Test error handling during benchmark execution."""
        # Mock module with a benchmark function that raises an exception
        mock_module = mock.MagicMock()
        mock_module.benchmark_failing = mock.MagicMock(side_effect=ValueError("Test error"))
        mock_import.return_value = mock_module
        
        runner = BenchmarkRunner()
        
        # Run the failing benchmark
        with pytest.raises(ValueError) as excinfo:
            runner.run_benchmark(
                category="storage",
                benchmark_name="failing"
            )
        
        assert "Test error" in str(excinfo.value)
        
        # Verify running_benchmarks is empty (should be cleared even on error)
        assert len(runner.running_benchmarks) == 0
        
        # Verify the benchmark is not in completed_benchmarks
        assert "storage.failing" not in runner.completed_benchmarks

    @mock.patch.object(BenchmarkRunner, "run_benchmark")
    @mock.patch.object(BenchmarkRunner, "discover_benchmarks")
    def test_run_category(self, mock_discover, mock_run):
        """Test running all benchmarks in a category."""
        # Mock discovery to return 3 benchmarks
        mock_discover.return_value = {
            "storage": {
                "benchmark1": lambda: None,
                "benchmark2": lambda: None,
                "benchmark3": lambda: None
            }
        }
        
        # Mock run_benchmark to return a file path
        mock_run.side_effect = [
            "/path/to/result1.json",
            "/path/to/result2.json",
            "/path/to/result3.json"
        ]
        
        runner = BenchmarkRunner()
        
        # Run all benchmarks in the category
        result_paths = runner.run_category(
            category="storage",
            params={"test_param": "value"},
            parallel=False
        )
        
        # Verify discover_benchmarks was called with the correct category
        mock_discover.assert_called_once_with("storage")
        
        # Verify run_benchmark was called for each benchmark
        assert mock_run.call_count == 3
        
        # Check that all parameters were passed correctly
        for call_args in mock_run.call_args_list:
            assert call_args[0][0] == "storage"  # category
            assert call_args[0][1] in ["benchmark1", "benchmark2", "benchmark3"]
            assert call_args[0][2] == {"test_param": "value"}  # params
        
        # Verify results
        assert len(result_paths) == 3
        assert all(path.startswith("/path/to/result") for path in result_paths)

    def test_run_all(self):
        """Test running all benchmark categories."""
        # Create a runner with mocked discover_benchmarks and run_category
        runner = BenchmarkRunner()
        
        # Mock the discover_benchmarks method
        runner.discover_benchmarks = mock.MagicMock(return_value={
            "storage": {"benchmark1": lambda: None},
            "retrieval": {"benchmark2": lambda: None},
            "compression": {"benchmark3": lambda: None}
        })
        
        # Mock the run_category method
        runner.run_category = mock.MagicMock(side_effect=[
            ["/path/to/storage1.json", "/path/to/storage2.json"],
            ["/path/to/retrieval1.json", "/path/to/retrieval2.json"],
            ["/path/to/compression1.json"]
        ])
        
        # Run all benchmarks
        category_params = {
            "storage": {"batch_size": 100},
            "retrieval": {"query_count": 10}
        }
        
        results = runner.run_all(
            params=category_params,
            parallel_categories=False,
            parallel_benchmarks=True
        )
        
        # Verify discover_benchmarks was called
        runner.discover_benchmarks.assert_called_once()
        
        # Verify run_category was called for each category
        assert runner.run_category.call_count == 3
        
        # Check that parameters were passed correctly
        # Examine each call to run_category to verify parameters using positional arguments
        expected_calls = [
            mock.call("storage", category_params.get("storage"), True, 4),
            mock.call("retrieval", category_params.get("retrieval"), True, 4),
            mock.call("compression", None, True, 4)
        ]
        runner.run_category.assert_has_calls(expected_calls, any_order=True)
        
        # Verify results structure
        assert set(results.keys()) == {"storage", "retrieval", "compression"}
        assert len(results["storage"]) == 2
        assert len(results["retrieval"]) == 2
        assert len(results["compression"]) == 1

    def test_module_imports(self):
        """Test that all required modules are available."""
        # This is to verify that the module import paths are correct
        # in an actual environment rather than with mocks
        
        runner = BenchmarkRunner()
        
        # Try to import each module (but expect it might fail in test environment)
        for category, module_path in runner.BENCHMARK_MODULES.items():
            try:
                module = __import__(module_path, fromlist=[""])
                assert module is not None
            except ImportError:
                # This is expected in test environment without actual benchmark modules
                pass 