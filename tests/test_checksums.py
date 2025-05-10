"""Tests for memory checksum utilities."""

import json

import pytest

from memory.utils.checksums import (
    ChecksumConfig,
    ChecksumVersion,
    add_checksum_to_memory,
    generate_checksum,
    validate_checksum,
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

    # Test type validation
    with pytest.raises(TypeError):
        generate_checksum("not a dict")

    # Test invalid version
    with pytest.raises(ValueError):
        generate_checksum(content1, version="invalid_version")


def test_checksum_versions():
    """Test different checksum versions."""
    content = {"key1": "value1", "key2": 42}

    # Test V1 version
    v1_checksum = generate_checksum(content, version=ChecksumVersion.V1)
    assert isinstance(v1_checksum, str)

    # Test V2 version
    v2_checksum = generate_checksum(content, version=ChecksumVersion.V2)
    assert isinstance(v2_checksum, str)

    # V1 and V2 should produce different checksums
    assert v1_checksum != v2_checksum


def test_checksum_with_metadata():
    """Test checksum generation with metadata inclusion."""
    content = {"key1": "value1", "key2": 42}
    metadata = {"importance": 0.8, "timestamp": 1234567890}

    # Test without metadata
    no_metadata_checksum = generate_checksum(content)

    # Test with metadata
    with_metadata_checksum = generate_checksum(
        {"content": content, "metadata": metadata}, include_metadata=True
    )

    # Checksums should be different
    assert no_metadata_checksum != with_metadata_checksum


def test_incremental_checksum():
    """Test incremental checksum generation."""
    # Create a large content dictionary
    large_content = {"key" + str(i): "value" + str(i) * 1000 for i in range(1000)}

    # Test with different chunk sizes
    chunk_sizes = [1024, 1024 * 1024, None]  # 1KB, 1MB, and no chunking

    checksums = []
    for chunk_size in chunk_sizes:
        checksum = generate_checksum(large_content, chunk_size=chunk_size)
        checksums.append(checksum)

    # All checksums should be identical regardless of chunk size
    assert len(set(checksums)) == 1


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

    # Test with custom config
    config = ChecksumConfig(
        algorithm="sha256", version=ChecksumVersion.V2, include_metadata=True
    )
    assert validate_checksum(memory_entry, config=config) is True

    # Test type validation
    with pytest.raises(TypeError):
        validate_checksum("not a dict")


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

    # Add checksum with default config
    updated_entry = add_checksum_to_memory(memory_entry)

    # Verify checksum was added
    assert "checksum" in updated_entry["metadata"]
    assert isinstance(updated_entry["metadata"]["checksum"], str)
    assert "version" in updated_entry["metadata"]
    assert "algorithm" in updated_entry["metadata"]
    assert "include_metadata" in updated_entry["metadata"]

    # Verify original entry wasn't modified
    assert "checksum" not in memory_entry["metadata"]

    # Verify checksum is valid
    assert validate_checksum(updated_entry) is True

    # Test with custom config
    config = ChecksumConfig(
        algorithm="sha512",
        version=ChecksumVersion.V2,
        include_metadata=True,
        chunk_size=1024,
    )
    custom_entry = add_checksum_to_memory(memory_entry, config=config)
    assert custom_entry["metadata"]["algorithm"] == "sha512"
    assert custom_entry["metadata"]["version"] == ChecksumVersion.V2.value
    assert custom_entry["metadata"]["include_metadata"] is True

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

    # Test type validation
    with pytest.raises(TypeError):
        add_checksum_to_memory("not a dict")


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

    # Test with different versions
    for version in ChecksumVersion:
        memory_entry = add_checksum_to_memory(
            memory_entry, config=ChecksumConfig(version=version)
        )
        serialized = json.dumps(memory_entry)
        deserialized = json.loads(serialized)
        assert validate_checksum(deserialized) is True
