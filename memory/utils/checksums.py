"""Checksum Utilities for Agent Memory

This module provides functions for generating and validating checksums for memory entries,
ensuring data integrity throughout the memory lifecycle. Checksums are computed based on
memory content and select metadata fields, providing a way to detect data corruption or
unauthorized modifications.

These utilities support the memory integrity features of the agent memory system by:
1. Computing consistent checksums for memory entries
2. Validating checksums when memories are retrieved or transferred
3. Supporting different hashing algorithms for various security requirements
"""

import copy
import hashlib
import json
from typing import Any, Dict


def generate_checksum(memory_content: Dict[str, Any], algorithm: str = "sha256") -> str:
    """Generate a checksum for memory content.

    Creates a reproducible hash of memory content to detect data corruption or tampering.

    Args:
        memory_content: Dictionary containing the memory contents
        algorithm: Hashing algorithm to use (default: sha256)

    Returns:
        String representation of the content hash

    Raises:
        ValueError: If an unsupported hashing algorithm is specified
    """
    if algorithm not in hashlib.algorithms_guaranteed:
        raise ValueError(f"Unsupported hashing algorithm: {algorithm}")

    # Sort keys to ensure consistent serialization
    serialized = json.dumps(memory_content, sort_keys=True)

    # Create hash object with selected algorithm
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(serialized.encode("utf-8"))

    return hash_obj.hexdigest()


def validate_checksum(memory_entry: Dict[str, Any], strict: bool = False) -> bool:
    """Validate the checksum of a memory entry.

    Verifies that a memory entry's content matches its stored checksum,
    ensuring data integrity.

    Args:
        memory_entry: Complete memory entry including content and metadata
        strict: If True, raises an exception on validation failure;
               if False, returns False (default: False)

    Returns:
        True if checksum is valid or missing, False if invalid

    Raises:
        ValueError: If strict=True and checksum validation fails
    """
    # Extract the stored checksum if present
    metadata = memory_entry.get("metadata", {})
    stored_checksum = metadata.get("checksum")

    # If no checksum exists, validation passes by default
    if not stored_checksum:
        return True

    # Generate a new checksum from the content
    content = memory_entry.get("contents", {})
    if not content:
        content = memory_entry.get("content", {})

    # Use same algorithm as stored checksum if detectable
    algorithm = "sha256"  # Default
    if stored_checksum and len(stored_checksum) == 64:
        algorithm = "sha256"
    elif stored_checksum and len(stored_checksum) == 128:
        algorithm = "sha512"

    computed_checksum = generate_checksum(content, algorithm)

    # Compare checksums
    is_valid = computed_checksum == stored_checksum

    if not is_valid and strict:
        raise ValueError(
            f"Checksum validation failed. Stored: {stored_checksum}, Computed: {computed_checksum}"
        )

    return is_valid


def add_checksum_to_memory(
    memory_entry: Dict[str, Any], algorithm: str = "sha256"
) -> Dict[str, Any]:
    """Add or update a checksum in a memory entry.

    Computes and adds a checksum to the metadata of a memory entry.

    Args:
        memory_entry: Memory entry to update
        algorithm: Hashing algorithm to use (default: sha256)

    Returns:
        Updated memory entry with checksum in metadata

    Raises:
        ValueError: If memory entry lacks required content field
    """
    # Create a deep copy to avoid modifying the original
    memory_copy = copy.deepcopy(memory_entry)

    # Ensure metadata exists
    if "metadata" not in memory_copy:
        memory_copy["metadata"] = {}

    # Get content field (could be named "content" or "contents")
    content = memory_copy.get("contents", {})
    if not content:
        content = memory_copy.get("content", {})

    if not content:
        raise ValueError("Memory entry must contain 'content' or 'contents' field")

    # Generate and store checksum
    checksum = generate_checksum(content, algorithm)
    memory_copy["metadata"]["checksum"] = checksum

    return memory_copy
