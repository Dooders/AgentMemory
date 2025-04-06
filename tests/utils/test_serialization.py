"""Unit tests for serialization utilities.

This module contains tests for the serialization module in agent_memory/utils.
"""

import datetime
import json
import pickle
import base64
import unittest
from typing import Dict, Any, List

import pytest

from memory.utils.serialization import (
    MemorySerializer,
    MemoryJSONEncoder,
    memory_json_decoder,
    to_json,
    from_json,
    serialize_memory,
    deserialize_memory,
)


class TestMemorySerializer(unittest.TestCase):
    """Test the MemorySerializer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_memory = {
            "id": "mem_123",
            "agent_id": "agent_1",
            "content": "This is a test memory",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": {
                "timestamp": datetime.datetime(2023, 1, 1, 12, 0, 0),
                "tags": {"test", "memory"},
                "priority": 1,
            },
        }

    def test_serialize_deserialize_memory_json(self):
        """Test serializing and deserializing memory entries with JSON format."""
        # Serialize
        serialized = MemorySerializer.serialize_memory(self.test_memory, format="json")
        self.assertIsInstance(serialized, str)
        
        # Deserialize
        deserialized = MemorySerializer.deserialize_memory(serialized, format="json")
        
        # Check basic fields
        self.assertEqual(deserialized["id"], self.test_memory["id"])
        self.assertEqual(deserialized["agent_id"], self.test_memory["agent_id"])
        self.assertEqual(deserialized["content"], self.test_memory["content"])
        self.assertEqual(deserialized["embedding"], self.test_memory["embedding"])
        
        # Check metadata fields
        metadata = deserialized["metadata"]
        orig_metadata = self.test_memory["metadata"]
        
        # Set should be deserialized properly
        self.assertIsInstance(metadata["tags"], set)
        self.assertEqual(metadata["tags"], orig_metadata["tags"])
        
        # Datetime should be deserialized properly
        self.assertIsInstance(metadata["timestamp"], datetime.datetime)
        self.assertEqual(metadata["timestamp"], orig_metadata["timestamp"])

    def test_serialize_deserialize_memory_pickle(self):
        """Test serializing and deserializing memory entries with pickle format."""
        # Serialize
        serialized = MemorySerializer.serialize_memory(self.test_memory, format="pickle")
        self.assertIsInstance(serialized, str)
        
        # Deserialize
        deserialized = MemorySerializer.deserialize_memory(serialized, format="pickle")
        
        # Check that the deserialized object is equal to the original
        self.assertEqual(deserialized, self.test_memory)

    def test_serialize_deserialize_memory_invalid_format(self):
        """Test serializing with an invalid format."""
        with self.assertRaises(ValueError):
            MemorySerializer.serialize_memory(self.test_memory, format="invalid")
        
        with self.assertRaises(ValueError):
            MemorySerializer.deserialize_memory("{}", format="invalid")

    def test_to_json_from_json(self):
        """Test to_json and from_json methods."""
        # Test with a simple object
        obj = {"name": "test", "value": 123}
        json_str = MemorySerializer.to_json(obj)
        deserialized = MemorySerializer.from_json(json_str)
        self.assertEqual(deserialized, obj)
        
        # Test with a complex object containing datetime
        complex_obj = {
            "time": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "items": [1, 2, 3],
            "flags": {"enabled", "active"}
        }
        json_str = MemorySerializer.to_json(complex_obj)
        deserialized = MemorySerializer.from_json(json_str)
        
        self.assertEqual(deserialized["time"], complex_obj["time"])
        self.assertEqual(deserialized["items"], complex_obj["items"])
        self.assertEqual(deserialized["flags"], complex_obj["flags"])

    def test_to_pickle_from_pickle(self):
        """Test to_pickle and from_pickle methods."""
        obj = {"name": "test", "value": 123, "complex": {"set": {1, 2, 3}}}
        pickle_str = MemorySerializer.to_pickle(obj)
        deserialized = MemorySerializer.from_pickle(pickle_str)
        self.assertEqual(deserialized, obj)

    def test_serialize_deserialize_vector(self):
        """Test serializing and deserializing vectors."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        serialized = MemorySerializer.serialize_vector(vector)
        self.assertIsInstance(serialized, str)
        
        # Should be JSON serialized
        parsed = json.loads(serialized)
        self.assertEqual(parsed, vector)
        
        # Deserialize
        deserialized = MemorySerializer.deserialize_vector(serialized)
        self.assertEqual(deserialized, vector)


class TestMemoryJSONEncoder(unittest.TestCase):
    """Test the custom JSON encoder for memory entries."""

    def test_datetime_encoding_decoding(self):
        """Test encoding and decoding datetime objects."""
        dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
        encoded = json.dumps(dt, cls=MemoryJSONEncoder)
        decoded = json.loads(encoded, object_hook=memory_json_decoder)
        self.assertEqual(decoded, dt)

    def test_set_encoding_decoding(self):
        """Test encoding and decoding set objects."""
        s = {"a", "b", "c"}
        encoded = json.dumps(s, cls=MemoryJSONEncoder)
        decoded = json.loads(encoded, object_hook=memory_json_decoder)
        self.assertEqual(decoded, s)

    def test_bytes_encoding_decoding(self):
        """Test encoding and decoding bytes objects."""
        b = b"hello world"
        encoded = json.dumps(b, cls=MemoryJSONEncoder)
        decoded = json.loads(encoded, object_hook=memory_json_decoder)
        self.assertEqual(decoded, b)


@pytest.mark.parametrize(
    "obj",
    [
        {"simple": "object"},
        {"with_date": datetime.datetime(2023, 1, 1, 12, 0, 0)},
        {"with_set": {"a", "b", "c"}},
        {"with_bytes": b"hello world"},
        {"nested": {"date": datetime.datetime(2023, 1, 1), "set": {1, 2, 3}}},
        [1, 2, 3, {"with_date": datetime.datetime(2023, 1, 1)}],
    ],
)
def test_serialize_deserialize_roundtrip(obj):
    """Test roundtrip serialization and deserialization for various objects."""
    serialized = serialize_memory(obj)
    deserialized = deserialize_memory(serialized)
    
    # Test equality - this is tricky because sets and datetimes need special handling
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, set):
                assert isinstance(deserialized[k], set)
                assert deserialized[k] == v
            elif isinstance(v, datetime.datetime):
                assert isinstance(deserialized[k], datetime.datetime)
                assert deserialized[k] == v
            elif isinstance(v, bytes):
                assert isinstance(deserialized[k], bytes)
                assert deserialized[k] == v
            else:
                assert deserialized[k] == v
    else:
        assert deserialized == obj


# Test the module-level functions
def test_to_json_from_json():
    """Test the to_json and from_json module-level functions."""
    obj = {"name": "test", "time": datetime.datetime(2023, 1, 1)}
    json_str = to_json(obj)
    deserialized = from_json(json_str)
    assert deserialized["name"] == obj["name"]
    assert deserialized["time"] == obj["time"]


if __name__ == "__main__":
    unittest.main() 