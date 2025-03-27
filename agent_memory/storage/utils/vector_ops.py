"""Vector operations utilities for memory storage.

This module provides common vector operations used across different
storage backends for similarity search and vector manipulations.
"""

import logging
import math
import numpy as np
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")
    
    # Handle edge case of zero vectors
    if not any(a) or not any(b):
        return 0.0
        
    try:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        similarity = dot_product / (norm_a * norm_b)
        
        # Numerical stability: ensure result is between -1 and 1
        return max(-1.0, min(1.0, similarity))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector with unit length
    """
    try:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            return [x / norm for x in vector]
        return vector
    except Exception as e:
        logger.error(f"Error normalizing vector: {e}")
        return vector


def compress_vector(
    vector: List[float], target_dim: int, method: str = "average"
) -> List[float]:
    """Compress a vector to a lower dimension.
    
    Args:
        vector: Input vector to compress
        target_dim: Target dimension (must be smaller than original)
        method: Compression method ('average', 'sample', 'pca')
        
    Returns:
        Compressed vector
        
    Raises:
        ValueError: If target dimension is larger than original
    """
    if target_dim >= len(vector):
        return vector
    
    if method == "average":
        # Average pooling: group elements and average them
        np_vector = np.array(vector)
        chunks = np.array_split(np_vector, target_dim)
        return [float(chunk.mean()) for chunk in chunks]
    
    elif method == "sample":
        # Uniform sampling
        indices = np.linspace(0, len(vector) - 1, target_dim, dtype=int)
        return [vector[i] for i in indices]
    
    elif method == "pca":
        # This would require scikit-learn; simplified PCA approach:
        if len(vector) < 100:  # For small vectors, fallback to average
            return compress_vector(vector, target_dim, "average")
            
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            compressed = pca.fit_transform(np.array(vector).reshape(1, -1))
            return list(compressed.flatten())
        except ImportError:
            logger.warning("scikit-learn not available, falling back to average pooling")
            return compress_vector(vector, target_dim, "average")
    else:
        logger.warning(f"Unknown compression method: {method}, using average")
        return compress_vector(vector, target_dim, "average")


def dot_product(a: List[float], b: List[float]) -> float:
    """Calculate dot product between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product value
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")
    
    return sum(x * y for x, y in zip(a, b)) 