"""Unit tests for agent_memory.embeddings.compression module.

This test suite covers the vector compression functionality for memory embeddings.
"""

import numpy as np
import pytest

from agent_memory.embeddings.vector_compression import (
    CompressionConfig,
    compress_vector_rp,
    decompress_vector_rp,
    dequantize_vector,
    quantize_vector,
)

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


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
