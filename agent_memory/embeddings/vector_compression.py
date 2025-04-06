"""Vector compression utilities for agent memory.

This module provides functionality for compressing and decompressing vectors
used in embeddings, supporting both quantization and random projection methods.
"""

from typing import Any

import numpy as np


def quantize_vector(vector: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize a vector to the specified number of bits.

    Args:
        vector: Input vector to quantize (assumed to be in range [-1, 1])
        bits: Bit depth for quantization (default: 8)

    Returns:
        Quantized vector with integer values
    """
    # Calculate the range of quantized values
    levels = 2**bits

    # Scale the vector from [-1, 1] to [0, levels-1]
    scaled = (vector + 1) * (levels - 1) / 2

    # Convert to integers
    quantized = np.round(scaled).astype(int)

    # Clip to ensure values are within valid range
    return np.clip(quantized, 0, levels - 1).tolist()


def dequantize_vector(quantized: np.ndarray, bits: int = 8) -> np.ndarray:
    """Dequantize a vector from integer values back to float range [-1, 1].

    Args:
        quantized: Quantized vector with integer values
        bits: Bit depth used for quantization (default: 8)

    Returns:
        Dequantized vector with float values in range [-1, 1]
    """
    # Convert list to numpy array if needed
    if isinstance(quantized, list):
        quantized = np.array(quantized)
        
    # Calculate the range of quantized values
    levels = 2**bits

    # Scale back from [0, levels-1] to [-1, 1]
    return (quantized / (levels - 1)) * 2 - 1


def compress_vector_rp(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Compress a vector using random projection.

    Args:
        vector: Input vector to compress
        target_dim: Target dimension for compression

    Returns:
        Compressed vector of dimension target_dim
    """
    # Create a random projection matrix
    # For reproducibility, we use a fixed seed
    np.random.seed(42)
    projection_matrix = np.random.randn(len(vector), target_dim) / np.sqrt(target_dim)

    # Project the vector
    return np.dot(vector, projection_matrix)


def decompress_vector_rp(compressed: np.ndarray, original_dim: int) -> np.ndarray:
    """Decompress a vector using random projection (approximate reconstruction).

    Args:
        compressed: Compressed vector
        original_dim: Original dimension to reconstruct to

    Returns:
        Decompressed vector of dimension original_dim
    """
    # Create the same random projection matrix used for compression
    np.random.seed(42)
    projection_matrix = np.random.randn(original_dim, len(compressed)) / np.sqrt(
        len(compressed)
    )

    # Approximate reconstruction using the pseudo-inverse
    return np.dot(compressed, projection_matrix.T)


class CompressionConfig:
    """Configuration for vector compression settings.

    Attributes:
        enabled: Whether compression is enabled
        method: Compression method ("quantization" or "random_projection")
        stm_dimension: Dimension for short-term memory vectors
        im_dimension: Dimension for intermediate memory vectors
        ltm_dimension: Dimension for long-term memory vectors
    """

    def __init__(
        self,
        enabled: bool = True,
        method: str = "random_projection",
        stm_dimension: int = 768,
        im_dimension: int = 256,
        ltm_dimension: int = 64,
    ):
        """Initialize compression configuration.

        Args:
            enabled: Whether compression is enabled
            method: Compression method ("quantization" or "random_projection")
            stm_dimension: Dimension for short-term memory vectors
            im_dimension: Dimension for intermediate memory vectors
            ltm_dimension: Dimension for long-term memory vectors
        """
        self.enabled = enabled
        self.method = method
        self.stm_dimension = stm_dimension
        self.im_dimension = im_dimension
        self.ltm_dimension = ltm_dimension

    def validate(self):
        """Validate the compression configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.method not in ["quantization", "random_projection"]:
            raise ValueError(f"Unsupported compression method: {self.method}")

        if any(
            d <= 0 for d in [self.stm_dimension, self.im_dimension, self.ltm_dimension]
        ):
            raise ValueError("Vector dimensions must be positive")
