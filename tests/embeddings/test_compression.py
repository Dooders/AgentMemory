"""Unit tests for memory.embeddings.compression module.

This test suite covers the vector compression functionality for memory embeddings.
"""

import numpy as np
import pytest
import json
import base64
import zlib

from memory.embeddings.vector_compression import (
    CompressionConfig,
    compress_vector_rp,
    decompress_vector_rp,
    dequantize_vector,
    quantize_vector,
)
from memory.embeddings.compression import CompressionEngine, AbstractionEngine
from memory.config import AutoencoderConfig

#################################
# Quantization Tests
#################################


def test_quantize_vector():
    """Test vector quantization."""
    # Create a test vector with values between -1 and 1
    vector = np.array([-0.9, -0.5, 0.0, 0.3, 0.7])

    # Quantize the vector to 8 bits
    quantized = quantize_vector(vector, bits=8)

    # Check that the quantized values are in the expected range for 8-bit
    assert all(isinstance(x, int) for x in quantized)
    assert all(0 <= x < 256 for x in quantized)
    assert len(quantized) == len(vector)


def test_dequantize_vector():
    """Test vector dequantization."""
    # Create a quantized vector (8-bit)
    quantized = np.array([0, 64, 128, 192, 255])

    # Dequantize back to float
    dequantized = dequantize_vector(quantized, bits=8)

    # Check the values are in the expected range
    assert all(isinstance(x, float) for x in dequantized)
    assert all(-1.0 <= x <= 1.0 for x in dequantized)
    assert len(dequantized) == len(quantized)

    # The following pairs should be approximate:
    # 0 -> -1.0, 64 -> -0.5, 128 -> 0.0, 192 -> 0.5, 255 -> 1.0
    assert abs(dequantized[0] + 1.0) < 0.05  # Should be close to -1.0
    assert abs(dequantized[1] + 0.5) < 0.05  # Should be close to -0.5
    assert abs(dequantized[2] - 0.0) < 0.05  # Should be close to 0.0
    assert abs(dequantized[3] - 0.5) < 0.05  # Should be close to 0.5
    assert abs(dequantized[4] - 1.0) < 0.05  # Should be close to 1.0


def test_quantize_dequantize_roundtrip():
    """Test roundtrip of quantization and dequantization."""
    # Create a test vector
    original = np.array([-0.8, -0.4, 0.0, 0.3, 0.9])

    # Roundtrip: quantize and then dequantize
    quantized = quantize_vector(original, bits=8)
    dequantized = dequantize_vector(quantized, bits=8)

    # Values should be approximately preserved
    assert len(dequantized) == len(original)
    for orig, deq in zip(original, dequantized):
        # With 8-bit quantization, we expect some loss in precision
        assert abs(orig - deq) < 0.05


#################################
# Random Projection Tests
#################################


def test_compress_vector_rp():
    """Test vector compression using random projection."""
    # Create a test vector
    vector = np.random.rand(100) * 2 - 1  # Values between -1 and 1

    # Compress to a lower dimension
    compressed = compress_vector_rp(vector, target_dim=20)

    # Check the compressed vector has the expected dimension
    assert len(compressed) == 20

    # Check values are still in a reasonable range
    assert all(-5.0 <= x <= 5.0 for x in compressed)


def test_decompress_vector_rp():
    """Test vector decompression using random projection."""
    # Create a test vector and compress it
    original = np.random.rand(100) * 2 - 1
    compressed = compress_vector_rp(original, target_dim=20)

    # Decompress back to original dimension
    decompressed = decompress_vector_rp(compressed, original_dim=100)

    # Check the decompressed vector has the original dimension
    assert len(decompressed) == 100

    # Random projection is lossy, but should preserve some similarity
    similarity = np.dot(original, decompressed) / (
        np.linalg.norm(original) * np.linalg.norm(decompressed)
    )
    assert similarity > 0  # Should have some positive correlation


def test_compress_decompress_roundtrip_preservation():
    """Test that compression-decompression preserves some information."""
    # Create test vectors with clear patterns
    v1 = np.array([1.0] * 50 + [0.0] * 50)  # First half ones, second half zeros
    v2 = np.array([0.0] * 50 + [1.0] * 50)  # First half zeros, second half ones

    # Compress both vectors
    compressed_v1 = compress_vector_rp(v1, target_dim=20)
    compressed_v2 = compress_vector_rp(v2, target_dim=20)

    # Decompress both vectors
    decompressed_v1 = decompress_vector_rp(compressed_v1, original_dim=100)
    decompressed_v2 = decompress_vector_rp(compressed_v2, original_dim=100)

    # Check that the vectors are still distinguishable after compression/decompression
    # The similarity between each vector and its round-trip version should be higher
    # than the similarity between the two different vectors

    # Calculate similarities
    sim_v1_dec_v1 = np.dot(v1, decompressed_v1) / (
        np.linalg.norm(v1) * np.linalg.norm(decompressed_v1)
    )
    sim_v2_dec_v2 = np.dot(v2, decompressed_v2) / (
        np.linalg.norm(v2) * np.linalg.norm(decompressed_v2)
    )
    sim_v1_dec_v2 = np.dot(v1, decompressed_v2) / (
        np.linalg.norm(v1) * np.linalg.norm(decompressed_v2)
    )

    # Self-similarity should be higher than cross-similarity
    assert sim_v1_dec_v1 > sim_v1_dec_v2
    assert sim_v2_dec_v2 > sim_v1_dec_v2


#################################
# CompressionConfig Tests
#################################


def test_compression_config_init():
    """Test initialization of CompressionConfig."""
    config = CompressionConfig(
        enabled=True,
        method="random_projection",
        stm_dimension=200,
        im_dimension=64,
        ltm_dimension=32,
    )

    assert config.enabled is True
    assert config.method == "random_projection"
    assert config.stm_dimension == 200
    assert config.im_dimension == 64
    assert config.ltm_dimension == 32


def test_compression_config_validate():
    """Test validation of CompressionConfig."""
    # Valid config
    valid_config = CompressionConfig(
        enabled=True,
        method="random_projection",
        stm_dimension=200,
        im_dimension=64,
        ltm_dimension=32,
    )

    # Should not raise exceptions
    valid_config.validate()

    # Invalid config with method
    invalid_method_config = CompressionConfig(
        enabled=True,
        method="invalid_method",
        stm_dimension=200,
        im_dimension=64,
        ltm_dimension=32,
    )

    # Should raise ValueError for invalid method
    with pytest.raises(ValueError):
        invalid_method_config.validate()

    # Invalid config with dimensions
    invalid_dim_config = CompressionConfig(
        enabled=True,
        method="random_projection",
        stm_dimension=0,  # Invalid dimension
        im_dimension=64,
        ltm_dimension=32,
    )

    # Should raise ValueError for invalid dimension
    with pytest.raises(ValueError):
        invalid_dim_config.validate()


#################################
# CompressionEngine Tests
#################################

@pytest.fixture
def mock_autoencoder_config():
    """Create a mock AutoencoderConfig for testing."""
    return AutoencoderConfig(
        enabled=True, 
        embedding_dim=512, 
        compression_dim=64
    )


@pytest.fixture
def sample_memory_entry():
    """Create a sample memory entry for testing."""
    return {
        "id": "mem123",
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "metadata": {
            "timestamp": 1234567890,
            "source": "user_interaction",
        },
        "content": {
            "memory_type": "interaction",
            "interaction_type": "conversation",
            "entities": ["user", "agent"],
            "dialog": "Hello, how can I help you today?",
            "outcome": "user_satisfied",
            "raw_observation": "User entered text via keyboard",
            "full_observation": "Detailed capture of user's typing behavior",
            "detailed_state": "Complex state representation",
            "importance": 0.75321,
            "timestamp": 1234567890,
            "step_number": 42
        }
    }


def test_compression_engine_init(mock_autoencoder_config):
    """Test initialization of CompressionEngine."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    assert engine.config == mock_autoencoder_config
    assert len(engine.level_configs) == 3
    assert 0 in engine.level_configs
    assert 1 in engine.level_configs
    assert 2 in engine.level_configs


def test_compress_level0(mock_autoencoder_config, sample_memory_entry):
    """Test compression level 0 (no compression)."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    compressed = engine.compress(sample_memory_entry, 0)
    
    # Should be a copy but not the same object
    assert compressed is not sample_memory_entry
    assert compressed == sample_memory_entry


def test_compress_level1(mock_autoencoder_config, sample_memory_entry):
    """Test compression level 1 (moderate compression)."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    compressed = engine.compress(sample_memory_entry, 1)
    
    # Check metadata is updated
    assert compressed["metadata"]["compression_level"] == 1
    
    # Check numeric precision is reduced
    assert compressed["content"]["importance"] == round(sample_memory_entry["content"]["importance"], 3)
    
    # Check filtered keys
    filter_keys = engine.level_configs[1]["content_filter_keys"]
    for key in filter_keys:
        assert key not in compressed["content"]


def test_compress_level2(mock_autoencoder_config, sample_memory_entry):
    """Test compression level 2 (high compression)."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    compressed = engine.compress(sample_memory_entry, 2)
    
    # Check metadata is updated
    assert compressed["metadata"]["compression_level"] == 2
    
    # Check binary compression
    assert "_compressed" in compressed["content"]
    assert "_compression_info" in compressed["content"]
    assert compressed["content"]["_compression_info"]["algorithm"] == "zlib"


def test_decompress(mock_autoencoder_config, sample_memory_entry):
    """Test decompression of compressed memory entries."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    # Compress to level 2
    compressed = engine.compress(sample_memory_entry, 2)
    
    # Now decompress
    decompressed = engine.decompress(compressed)
    
    # Check key fields are restored
    assert "memory_type" in decompressed["content"]
    assert decompressed["content"]["memory_type"] == "interaction"
    assert decompressed["content"]["interaction_type"] == "conversation"
    
    # Note: Some information loss is expected due to compression


def test_reduce_numeric_precision(mock_autoencoder_config):
    """Test numeric precision reduction."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    # Test with dict
    test_dict = {"a": 1.23456, "b": {"c": 7.89012}, "d": [3.14159, {"e": 2.71828}]}
    engine._reduce_numeric_precision(test_dict, 2)
    
    assert test_dict["a"] == 1.23
    assert test_dict["b"]["c"] == 7.89
    assert test_dict["d"][0] == 3.14
    assert test_dict["d"][1]["e"] == 2.72


def test_binary_compression_roundtrip(mock_autoencoder_config):
    """Test binary compression and decompression."""
    engine = CompressionEngine(mock_autoencoder_config)
    
    test_data = {"key1": "value1", "key2": 123.456, "key3": [1, 2, 3]}
    
    # Compress
    compressed = engine._binary_compress(test_data)
    assert isinstance(compressed, str)
    
    # Decompress
    decompressed = engine._binary_decompress(compressed)
    assert decompressed == test_data


#################################
# AbstractionEngine Tests
#################################

@pytest.fixture
def sample_state_memory():
    """Create a sample state memory entry for testing."""
    return {
        "content": {
            "memory_type": "state",
            "location": "office",
            "status": "active",
            "goals": ["complete_task", "respond_to_user"],
            "resources": {"cpu": 0.5, "memory": 0.3},
            "relationships": {"user123": "positive"},
            "timestamp": 1234567890,
            "step_number": 42,
            "detailed_info": "lots of details that should be abstracted away"
        }
    }


@pytest.fixture
def sample_action_memory():
    """Create a sample action memory entry for testing."""
    return {
        "content": {
            "memory_type": "action",
            "action_type": "response",
            "target": "user_query",
            "outcome": "success",
            "success": True,
            "timestamp": 1234567890,
            "step_number": 42,
            "detailed_steps": "step1, step2, step3",
            "raw_data": "lots of raw data"
        }
    }


@pytest.fixture
def sample_interaction_memory():
    """Create a sample interaction memory entry for testing."""
    return {
        "content": {
            "memory_type": "interaction",
            "interaction_type": "conversation",
            "entities": ["user", "agent"],
            "dialog": "Hello, how can I help you today?",
            "outcome": "user_satisfied",
            "timestamp": 1234567890,
            "step_number": 42,
            "detailed_context": "complex contextual information"
        }
    }


def test_abstraction_engine_init():
    """Test initialization of AbstractionEngine."""
    engine = AbstractionEngine()
    # Currently, initialization doesn't do much, but test it anyway
    assert isinstance(engine, AbstractionEngine)


def test_light_abstraction_state(sample_state_memory):
    """Test light abstraction of state memory."""
    engine = AbstractionEngine()
    
    result = engine._light_abstraction(sample_state_memory["content"])
    
    # Check that key fields are preserved
    assert result["memory_type"] == "state"
    assert result["location"] == "office"
    assert result["status"] == "active"
    assert "goals" in result
    assert "resources" in result
    assert "relationships" in result
    
    # Check that non-essential fields are removed
    assert "detailed_info" not in result


def test_light_abstraction_action(sample_action_memory):
    """Test light abstraction of action memory."""
    engine = AbstractionEngine()
    
    result = engine._light_abstraction(sample_action_memory["content"])
    
    # Check that key fields are preserved
    assert result["memory_type"] == "action"
    assert result["action_type"] == "response"
    assert result["target"] == "user_query"
    assert result["outcome"] == "success"
    assert result["success"] is True
    
    # Check that non-essential fields are removed
    assert "detailed_steps" not in result
    assert "raw_data" not in result


def test_light_abstraction_interaction(sample_interaction_memory):
    """Test light abstraction of interaction memory."""
    engine = AbstractionEngine()
    
    result = engine._light_abstraction(sample_interaction_memory["content"])
    
    # Check that key fields are preserved
    assert result["memory_type"] == "interaction"
    assert result["interaction_type"] == "conversation"
    assert result["entities"] == ["user", "agent"]
    assert result["dialog"] == "Hello, how can I help you today?"
    assert result["outcome"] == "user_satisfied"
    
    # Check that non-essential fields are removed
    assert "detailed_context" not in result


def test_heavy_abstraction_state(sample_state_memory):
    """Test heavy abstraction of state memory."""
    engine = AbstractionEngine()
    
    result = engine._heavy_abstraction(sample_state_memory["content"])
    
    # Check that only essential fields are preserved
    assert result["memory_type"] == "state"
    assert result["location"] == "office"
    assert result["status"] == "active"
    assert "timestamp" in result
    assert "step_number" in result
    
    # Check that other fields are removed
    assert "goals" not in result
    assert "resources" not in result
    assert "relationships" not in result
    assert "detailed_info" not in result


def test_heavy_abstraction_action(sample_action_memory):
    """Test heavy abstraction of action memory."""
    engine = AbstractionEngine()
    
    result = engine._heavy_abstraction(sample_action_memory["content"])
    
    # Check that only essential fields are preserved
    assert result["memory_type"] == "action"
    assert result["action_type"] == "response"
    assert result["outcome"] == "success"
    assert "timestamp" in result
    assert "step_number" in result
    
    # Check that other fields are removed
    assert "target" not in result
    assert "success" not in result
    assert "detailed_steps" not in result
    assert "raw_data" not in result


def test_heavy_abstraction_interaction(sample_interaction_memory):
    """Test heavy abstraction of interaction memory."""
    engine = AbstractionEngine()
    
    result = engine._heavy_abstraction(sample_interaction_memory["content"])
    
    # Check that only essential fields are preserved
    assert result["memory_type"] == "interaction"
    assert result["interaction_type"] == "conversation"
    assert result["outcome"] == "user_satisfied"
    assert "timestamp" in result
    assert "step_number" in result
    
    # Check that other fields are removed
    assert "entities" not in result
    assert "dialog" not in result
    assert "detailed_context" not in result


def test_abstract_memory_level1(sample_state_memory):
    """Test abstract_memory with level 1."""
    engine = AbstractionEngine()
    
    result = engine.abstract_memory(sample_state_memory, level=1)
    
    # Check that metadata is updated
    assert "metadata" in result
    assert result["metadata"]["abstraction_level"] == 1
    
    # Content should be lightly abstracted
    assert result["content"]["memory_type"] == "state"
    assert "location" in result["content"]
    assert "detailed_info" not in result["content"]


def test_abstract_memory_level2(sample_interaction_memory):
    """Test abstract_memory with level 2."""
    engine = AbstractionEngine()
    
    result = engine.abstract_memory(sample_interaction_memory, level=2)
    
    # Check that metadata is updated
    assert "metadata" in result
    assert result["metadata"]["abstraction_level"] == 2
    
    # Content should be heavily abstracted
    assert result["content"]["memory_type"] == "interaction"
    assert "interaction_type" in result["content"]
    assert "outcome" in result["content"]
    assert "entities" not in result["content"]
    assert "dialog" not in result["content"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
