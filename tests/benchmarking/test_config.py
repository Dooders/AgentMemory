"""
Unit tests for the BenchmarkConfig class.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest

from memory.benchmarking.config import (
    BenchmarkConfig, 
    StorageConfig, 
    CompressionConfig, 
    MemoryTransitionConfig,
    RetrievalConfig,
    ScalabilityConfig,
    IntegrationConfig
)


class TestBenchmarkConfig:
    """Test cases for the BenchmarkConfig class."""

    def test_default_initialization(self):
        """Test that default initialization works correctly."""
        config = BenchmarkConfig()
        
        # Verify default output directory
        assert config.output_dir == "benchmark_results"
        
        # Verify that all configuration components are initialized
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.compression, CompressionConfig)
        assert isinstance(config.memory_transition, MemoryTransitionConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.scalability, ScalabilityConfig)
        assert isinstance(config.integration, IntegrationConfig)

    def test_to_dict(self):
        """Test the to_dict method for serialization."""
        config = BenchmarkConfig()
        config_dict = config.to_dict()
        
        # Check that top-level fields are correct
        assert "output_dir" in config_dict
        assert config_dict["output_dir"] == "benchmark_results"
        
        # Check that all configuration components are included
        assert "storage" in config_dict
        assert "compression" in config_dict
        assert "memory_transition" in config_dict
        assert "retrieval" in config_dict
        assert "scalability" in config_dict
        assert "integration" in config_dict
        
        # Validate structure of one component
        storage_dict = config_dict["storage"]
        assert "batch_sizes" in storage_dict
        assert "data_complexity_levels" in storage_dict
        assert "num_samples" in storage_dict

    def test_save_and_load(self):
        """Test saving and loading configuration to/from file."""
        config = BenchmarkConfig()
        
        # Modify a property to verify persistence
        config.output_dir = "custom_output_dir"
        config.storage.num_samples = 2000
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            
            # Save the configuration
            saved_path = config.save(config_path)
            assert saved_path == config_path
            assert os.path.exists(config_path)
            
            # Verify the file contains the expected JSON
            with open(config_path, "r") as f:
                saved_data = json.load(f)
                assert saved_data["output_dir"] == "custom_output_dir"
                assert saved_data["storage"]["num_samples"] == 2000
            
            # Load the configuration
            loaded_config = BenchmarkConfig.load(config_path)
            
            # Verify loaded configuration matches original
            assert loaded_config.output_dir == "custom_output_dir"
            assert loaded_config.storage.num_samples == 2000
            assert isinstance(loaded_config.retrieval, RetrievalConfig)

    def test_default_save_location(self):
        """Test default save location is created correctly."""
        config = BenchmarkConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the output directory to the temp directory
            config.output_dir = temp_dir
            
            # Save with default path
            saved_path = config.save()
            
            # Verify the file was created at the expected location
            expected_path = os.path.join(temp_dir, "benchmark_config.json")
            assert saved_path == expected_path
            assert os.path.exists(expected_path)

    def test_custom_config_values(self):
        """Test configuration with custom values."""
        custom_storage = StorageConfig(
            batch_sizes=[5, 50, 500],
            data_complexity_levels=[2, 4],
            num_samples=500
        )
        
        custom_retrieval = RetrievalConfig(
            dataset_size=5000,
            query_counts=[5, 25],
            vector_dimensions=[64, 128]
        )
        
        config = BenchmarkConfig(
            storage=custom_storage,
            retrieval=custom_retrieval,
            output_dir="custom_benchmarks"
        )
        
        # Verify custom values were set
        assert config.output_dir == "custom_benchmarks"
        assert config.storage.batch_sizes == [5, 50, 500]
        assert config.storage.data_complexity_levels == [2, 4]
        assert config.storage.num_samples == 500
        assert config.retrieval.dataset_size == 5000
        assert config.retrieval.query_counts == [5, 25]
        assert config.retrieval.vector_dimensions == [64, 128]
        
        # Verify other configurations still have defaults
        assert config.compression.dataset_sizes == [100, 1000, 10000] 