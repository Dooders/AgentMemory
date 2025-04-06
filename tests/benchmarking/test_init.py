"""
Unit tests for the benchmarking module's __init__.py.
"""

import pytest
from memory.benchmarking import BenchmarkRunner, BenchmarkResults, BenchmarkConfig


def test_module_imports():
    """Test that the module exports the expected classes."""
    # Verify that the main classes are properly exported
    assert BenchmarkRunner is not None
    assert BenchmarkResults is not None
    assert BenchmarkConfig is not None
    
    # Verify that the classes are the correct types
    from memory.benchmarking.runner import BenchmarkRunner as DirectRunner
    from memory.benchmarking.results import BenchmarkResults as DirectResults
    from memory.benchmarking.config import BenchmarkConfig as DirectConfig
    
    assert BenchmarkRunner is DirectRunner
    assert BenchmarkResults is DirectResults
    assert BenchmarkConfig is DirectConfig 