"""Unit tests for schema.py functionality.

This module tests the JSON schema validation functions in the schema module,
including validation of memory system structures and handling of invalid data.
"""

import json
import os
import tempfile
from pathlib import Path

import jsonschema
import pytest

from memory.schema import (
    MEMORY_SYSTEM_SCHEMA,
    get_schema_as_json,
    load_and_validate,
    save_schema_to_file,
    validate_memory_system_json,
)

# Path to the memory samples directory
SAMPLES_DIR = Path("validation/memory_samples")

# Sample files to test
ATTRIBUTE_VALIDATION_SAMPLE = SAMPLES_DIR / "attribute_validation_memory.json"
IMPORTANCE_VALIDATION_SAMPLE = SAMPLES_DIR / "importance_validation_memory.json"


def test_schema_completeness():
    """Test that the schema contains all required fields."""
    schema = MEMORY_SYSTEM_SCHEMA

    # Verify top-level structure
    assert schema["type"] == "object"
    assert "required" in schema
    assert "properties" in schema
    assert "config" in schema["properties"]
    assert "agents" in schema["properties"]

    # Verify memory entry definition
    assert "definitions" in schema
    assert "memory_entry" in schema["definitions"]
    memory_entry = schema["definitions"]["memory_entry"]
    assert "required" in memory_entry
    assert "properties" in memory_entry

    # Check for critical memory entry properties
    critical_properties = ["memory_id", "agent_id", "content", "metadata", "type"]
    for prop in critical_properties:
        assert prop in memory_entry["properties"]
        assert prop in memory_entry["required"]


def test_valid_sample_json_files():
    """Test validating the sample JSON files against the schema."""
    sample_files = [
        ATTRIBUTE_VALIDATION_SAMPLE,
        IMPORTANCE_VALIDATION_SAMPLE,
    ]

    for sample_file in sample_files:
        # Load the sample JSON
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate against schema
        assert validate_memory_system_json(
            data
        ), f"Sample file {sample_file} failed schema validation"


def test_invalid_json_validation():
    """Test validating invalid JSON structures."""
    # Test with a completely invalid structure
    invalid_data = {"data": "This is not a valid memory system"}
    assert not validate_memory_system_json(invalid_data)

    # Test with missing required fields
    missing_config = {"agents": {"agent1": {"agent_id": "agent1", "memories": []}}}
    assert not validate_memory_system_json(missing_config)

    # Test with invalid agent structure
    invalid_agent = {"config": {}, "agents": {"agent1": "not an object"}}
    assert not validate_memory_system_json(invalid_agent)

    # Test with invalid memory entry
    invalid_memory = {
        "config": {},
        "agents": {
            "agent1": {
                "agent_id": "agent1",
                "memories": [{"memory_id": "123"}],  # Missing required fields
            }
        },
    }
    assert not validate_memory_system_json(invalid_memory)


def test_get_schema_as_json():
    """Test getting the schema as a JSON string."""
    schema_json = get_schema_as_json()
    assert isinstance(schema_json, str)

    # Verify the JSON can be parsed back to a dictionary
    parsed = json.loads(schema_json)
    assert parsed == MEMORY_SYSTEM_SCHEMA


def test_save_schema_to_file():
    """Test saving the schema to a file."""
    # Create a temporary file to save the schema
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save the schema to the temporary file
        assert save_schema_to_file(temp_path)

        # Verify the file exists and contains valid JSON
        assert os.path.exists(temp_path)
        with open(temp_path, "r", encoding="utf-8") as f:
            saved_schema = json.load(f)

        # Verify the saved schema matches the original
        assert saved_schema == MEMORY_SYSTEM_SCHEMA
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_and_validate():
    """Test loading and validating a JSON file."""
    # Test with a valid file
    valid_data = load_and_validate(str(ATTRIBUTE_VALIDATION_SAMPLE))
    assert valid_data is not None

    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(b'{"data": "This is not a valid memory system"}')
        temp_path = temp_file.name

    try:
        # Test with an invalid file
        invalid_data = load_and_validate(temp_path)
        assert invalid_data is None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_validation_error_details():
    """Test that validation errors provide detailed information."""
    # Create an invalid memory system with a specific validation error
    invalid_data = {
        "config": {},
        "agents": {
            "agent1": {
                "agent_id": "agent1",
                "memories": [
                    {
                        "memory_id": "test1",
                        "agent_id": "agent1",
                        "content": {},
                        "metadata": {
                            "importance_score": 1.5  # Invalid: exceeds maximum of 1.0
                        },
                        "type": "generic",
                    }
                ],
            }
        },
    }

    # Test validation fails
    assert not validate_memory_system_json(invalid_data)

    # Test with direct jsonschema validation to verify the exact error
    with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
        jsonschema.validate(instance=invalid_data, schema=MEMORY_SYSTEM_SCHEMA)

    # The error should mention importance_score
    assert "importance_score" in str(exc_info.value)
