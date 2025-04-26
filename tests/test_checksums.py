"""Tests for memory checksum utilities."""

import json
import pytest
from memory.utils.checksums import (
    generate_checksum,
    validate_checksum,
    add_checksum_to_memory,
)


def test_generate_checksum():
    """Test generating checksums for memory content."""
    # Test with dictionary content
    content1 = {"key1": "value1", "key2": 42}
    checksum1 = generate_checksum(content1)
    assert isinstance(checksum1, str)
    assert len(checksum1) > 0

    # Test with same content but different order (should be the same checksum)
    content2 = {"key2": 42, "key1": "value1"}
    checksum2 = generate_checksum(content2)
    assert checksum1 == checksum2

    # Test with different content
    content3 = {"key1": "value1", "key2": 43}
    checksum3 = generate_checksum(content3)
    assert checksum1 != checksum3

    # Test with nested content
    content4 = {"key1": "value1", "nested": {"a": 1, "b": 2}}
    checksum4 = generate_checksum(content4)
    assert isinstance(checksum4, str)
    assert checksum1 != checksum4


def test_validate_checksum():
    """Test validating checksums for memory entries."""
    # Create a memory entry with content
    content = {"observation": "User asked about the weather", "response": "It's sunny"}
    checksum = generate_checksum(content)
    memory_entry = {
        "memory_id": "mem123",
        "agent_id": "agent001",
        "content": content,
        "metadata": {
            "checksum": checksum,
            "creation_time": 1649879872.123,
        },
    }

    # Validate the checksum (should pass)
    assert validate_checksum(memory_entry) is True

    # Modify the content and validate (should fail)
    modified_entry = memory_entry.copy()
    modified_entry["content"] = {
        "observation": "User asked about something else",
        "response": "It's sunny",
    }
    assert validate_checksum(modified_entry) is False

    # Test with strict=True (should raise exception)
    with pytest.raises(ValueError):
        validate_checksum(modified_entry, strict=True)

    # Test with missing checksum
    no_checksum_entry = memory_entry.copy()
    no_checksum_entry["metadata"] = {"creation_time": 1649879872.123}
    assert validate_checksum(no_checksum_entry) is True  # Missing checksum passes


def test_add_checksum_to_memory():
    """Test adding checksums to memory entries."""
    # Create a memory entry without checksum
    memory_entry = {
        "memory_id": "mem456",
        "agent_id": "agent001",
        "content": {"key1": "value1", "key2": 42},
        "metadata": {
            "creation_time": 1649879872.123,
            "importance_score": 0.8,
        },
    }

    # Add checksum
    updated_entry = add_checksum_to_memory(memory_entry)

    # Verify checksum was added
    assert "checksum" in updated_entry["metadata"]
    assert isinstance(updated_entry["metadata"]["checksum"], str)

    # Verify original entry wasn't modified
    assert "checksum" not in memory_entry["metadata"]

    # Verify checksum is valid
    assert validate_checksum(updated_entry) is True

    # Test with empty content
    empty_content_entry = {
        "memory_id": "mem789",
        "agent_id": "agent001",
        "metadata": {
            "creation_time": 1649879872.123,
        },
    }

    # Should raise ValueError
    with pytest.raises(ValueError):
        add_checksum_to_memory(empty_content_entry)


def test_checksum_different_algorithms():
    """Test checksums with different hashing algorithms."""
    content = {"key1": "value1", "key2": 42}

    # Generate checksums with different algorithms
    sha256_checksum = generate_checksum(content, algorithm="sha256")
    sha512_checksum = generate_checksum(content, algorithm="sha512")

    # Verify they're different
    assert sha256_checksum != sha512_checksum
    assert len(sha512_checksum) > len(sha256_checksum)

    # Test with invalid algorithm
    with pytest.raises(ValueError):
        generate_checksum(content, algorithm="invalid_algorithm")


def test_checksum_serialization():
    """Test that checksums remain valid after serialization/deserialization."""
    # Create memory entry with checksum
    content = {"observation": "User asked about the weather"}
    memory_entry = {
        "memory_id": "mem123",
        "agent_id": "agent001",
        "content": content,
        "metadata": {
            "creation_time": 1649879872.123,
            "importance_score": 0.8,
        },
    }

    # Add checksum
    memory_entry = add_checksum_to_memory(memory_entry)

    # Serialize and deserialize
    serialized = json.dumps(memory_entry)
    deserialized = json.loads(serialized)

    # Validate checksum still passes
    assert validate_checksum(deserialized) is True
