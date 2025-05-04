"""Utility functions for the embeddings module.

This module provides common utility functions used across the embeddings
package to avoid code duplication.
"""

import logging
from typing import Any, Dict, Set

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        prefix: Key prefix for nested dictionaries

    Returns:
        Flattened dictionary
    """
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if not v:  # Handle empty dictionaries
                result[key] = v
            else:
                result.update(flatten_dict(v, key))
        else:
            result[key] = v
    return result


def object_to_text(obj: Any) -> str:
    """Convert any object to an enhanced text representation.

    Args:
        obj: Any Python object to convert to text

    Returns:
        String representation of the object
    """
    if obj is None:
        return ""
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        if not obj:
            return "empty"

        # Special handling for memory content
        if "content" in obj:
            content = obj["content"]
            if isinstance(content, dict):
                parts = []
                # Add content text if available
                if "content" in content:
                    parts.append(str(content["content"]))
                # Add metadata if available
                if "metadata" in content:
                    metadata = content["metadata"]
                    if isinstance(metadata, dict):
                        metadata_parts = []
                        for key, value in metadata.items():
                            metadata_parts.append(f"{key}: {value}")
                        if metadata_parts:
                            parts.append("metadata: " + ", ".join(metadata_parts))
                return " | ".join(parts)
            else:
                return str(content)

        # Default dictionary handling
        parts = []
        for key, value in obj.items():
            if isinstance(value, dict):
                formatted = f"{key}: " + object_to_text(value)
            elif isinstance(value, list):
                formatted = f"{key}: " + ", ".join(str(item) for item in value)
            else:
                formatted = f"{key}: {value}"
            parts.append(formatted)
        return " | ".join(parts)
    elif isinstance(obj, list):
        if not obj:
            return "empty"
        return "items: " + ", ".join(object_to_text(item) for item in obj)
    else:
        return str(obj)


def filter_dict_keys(content: Dict[str, Any], filter_keys: Set[str]) -> Dict[str, Any]:
    """Remove specified keys from a dictionary.

    Args:
        content: Dictionary to filter
        filter_keys: Set of keys to remove

    Returns:
        Filtered dictionary
    """
    if not isinstance(content, dict):
        return content

    # Get keys to remove (cannot modify during iteration)
    keys_to_remove = [key for key in content if key in filter_keys]

    # Create a filtered copy
    result = {k: v for k, v in content.items() if k not in keys_to_remove}

    return result
