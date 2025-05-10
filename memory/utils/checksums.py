"""Checksum Utilities for Agent Memory

This module provides functions for generating and validating checksums for memory entries,
ensuring data integrity throughout the memory lifecycle. Checksums are computed based on
memory content and select metadata fields, providing a way to detect data corruption or
unauthorized modifications.

These utilities support the memory integrity features of the agent memory system by:
1. Computing consistent checksums for memory entries
2. Validating checksums when memories are retrieved or transferred
3. Supporting different hashing algorithms for various security requirements
4. Providing incremental checksum updates for large memory entries
5. Supporting checksum versioning for backward compatibility
"""

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ChecksumVersion(Enum):
    """Supported checksum versions."""

    V1 = "v1"  # Original version
    V2 = "v2"  # Added versioning and incremental support


@dataclass
class ChecksumConfig:
    """Configuration for checksum generation and validation."""

    algorithm: str = "sha256"
    version: ChecksumVersion = ChecksumVersion.V2
    include_metadata: bool = False
    chunk_size: int = 1024 * 1024  # 1MB chunks for incremental processing


def generate_checksum(
    memory_content: Dict[str, Any],
    algorithm: str = "sha256",
    version: Union[str, ChecksumVersion] = ChecksumVersion.V2,
    include_metadata: bool = False,
    chunk_size: Optional[int] = None,
) -> str:
    """Generate a checksum for memory content.

    Creates a reproducible hash of memory content to detect data corruption or tampering.
    Supports incremental processing for large memory entries.

    Args:
        memory_content: Dictionary containing the memory contents
        algorithm: Hashing algorithm to use (default: sha256)
        version: Checksum version to use (default: V2)
        include_metadata: Whether to include metadata in checksum (default: False)
        chunk_size: Size of chunks for incremental processing (default: 1MB)

    Returns:
        String representation of the content hash

    Raises:
        ValueError: If an unsupported hashing algorithm is specified
        TypeError: If memory_content is not a dictionary
    """
    if not isinstance(memory_content, dict):
        raise TypeError("memory_content must be a dictionary")

    if algorithm not in hashlib.algorithms_guaranteed:
        raise ValueError(f"Unsupported hashing algorithm: {algorithm}")

    # Convert version string to enum if needed
    if isinstance(version, str):
        try:
            version = ChecksumVersion(version)
        except ValueError:
            raise ValueError(f"Unsupported checksum version: {version}")

    # Validate chunk_size if provided
    if chunk_size is not None:
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    # Create hash object with selected algorithm
    hash_obj = hashlib.new(algorithm)

    # Prepare content for hashing
    if version == ChecksumVersion.V1:
        # Original version - simple JSON serialization
        serialized = json.dumps(memory_content, sort_keys=True)
        hash_obj.update(serialized.encode("utf-8"))
    else:
        # V2 - Support incremental processing and metadata inclusion
        if include_metadata:
            content_to_hash = memory_content
        else:
            content_to_hash = memory_content.get(
                "content", memory_content.get("contents", {})
            )

        # Sort keys to ensure consistent serialization
        serialized = json.dumps(content_to_hash, sort_keys=True)

        # Process in chunks if specified
        if chunk_size:
            for i in range(0, len(serialized), chunk_size):
                chunk = serialized[i : i + chunk_size]
                hash_obj.update(chunk.encode("utf-8"))
        else:
            hash_obj.update(serialized.encode("utf-8"))

    return hash_obj.hexdigest()


def validate_checksum(
    memory_entry: Dict[str, Any],
    strict: bool = False,
    config: Optional[ChecksumConfig] = None,
) -> bool:
    """Validate the checksum of a memory entry.

    Verifies that a memory entry's content matches its stored checksum,
    ensuring data integrity.

    Args:
        memory_entry: Complete memory entry including content and metadata
        strict: If True, raises an exception on validation failure;
               if False, returns False (default: False)
        config: Optional configuration for validation

    Returns:
        True if checksum is valid or missing, False if invalid

    Raises:
        ValueError: If strict=True and checksum validation fails
        TypeError: If memory_entry is not a dictionary
    """
    if not isinstance(memory_entry, dict):
        raise TypeError("memory_entry must be a dictionary")

    # Use default config if none provided
    if config is None:
        config = ChecksumConfig()

    # Extract the stored checksum if present
    metadata = memory_entry.get("metadata", {})
    stored_checksum = metadata.get("checksum")

    # If no checksum exists, validation passes by default
    if not stored_checksum:
        return True

    # Get content field (could be named "content" or "contents")
    content = memory_entry.get("contents", {})
    if not content:
        content = memory_entry.get("content", {})

    # Use same algorithm as stored checksum if detectable
    algorithm = metadata.get("algorithm", config.algorithm)
    version = metadata.get("version", config.version)
    include_metadata = metadata.get("include_metadata", config.include_metadata)

    if not algorithm and stored_checksum:
        # Fallback to length-based inference for backward compatibility
        if len(stored_checksum) == 64:
            algorithm = "sha256"
        elif len(stored_checksum) == 128:
            algorithm = "sha512"

    try:
        computed_checksum = generate_checksum(
            content,
            algorithm=algorithm,
            version=version,
            include_metadata=include_metadata,
            chunk_size=config.chunk_size,
        )
    except Exception as e:
        logger.error(f"Error computing checksum: {str(e)}")
        if strict:
            raise ValueError(f"Failed to compute checksum: {str(e)}")
        return False

    # Compare checksums
    is_valid = computed_checksum == stored_checksum

    if not is_valid:
        logger.warning(
            f"Checksum validation failed. Stored: {stored_checksum}, Computed: {computed_checksum}"
        )
        if strict:
            raise ValueError(
                f"Checksum validation failed. Stored: {stored_checksum}, Computed: {computed_checksum}"
            )

    return is_valid


def add_checksum_to_memory(
    memory_entry: Dict[str, Any], config: Optional[ChecksumConfig] = None
) -> Dict[str, Any]:
    """Add or update a checksum in a memory entry.

    Computes and adds a checksum to the metadata of a memory entry.

    Args:
        memory_entry: Memory entry to update
        config: Optional configuration for checksum generation

    Returns:
        Updated memory entry with checksum in metadata

    Raises:
        ValueError: If memory entry lacks required content field
        TypeError: If memory_entry is not a dictionary
    """
    if not isinstance(memory_entry, dict):
        raise TypeError("memory_entry must be a dictionary")

    # Use default config if none provided
    if config is None:
        config = ChecksumConfig()

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

    try:
        # Generate and store checksum
        checksum = generate_checksum(
            content,
            algorithm=config.algorithm,
            version=config.version,
            include_metadata=config.include_metadata,
            chunk_size=config.chunk_size,
        )

        # Store checksum and configuration in metadata
        memory_copy["metadata"]["checksum"] = checksum
        memory_copy["metadata"]["algorithm"] = config.algorithm
        memory_copy["metadata"]["version"] = config.version.value
        memory_copy["metadata"]["include_metadata"] = config.include_metadata

        return memory_copy
    except Exception as e:
        logger.error(f"Error adding checksum to memory: {str(e)}")
        raise ValueError(f"Failed to add checksum: {str(e)}")
