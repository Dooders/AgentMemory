"""Unit tests for serialization utilities.

This module contains tests for the serialization module in memory/utils.
"""

import base64
import datetime
import json
import os
import pickle
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, mock_open, patch

import pytest

from memory.utils.serialization import (
    MemoryJSONEncoder,
    MemorySerializer,
    deserialize_memory,
    from_json,
    load_memory_system_from_json,
    memory_json_decoder,
    save_memory_system_to_json,
    serialize_memory,
    to_json,
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
        serialized = MemorySerializer.serialize_memory(
            self.test_memory, format="pickle"
        )
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
            "flags": {"enabled", "active"},
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


class TestMemorySystemSerialization(unittest.TestCase):
    """Test the memory system serialization functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_filepath = os.path.join(self.temp_dir.name, "memory_system.json")

        # Create plain objects instead of magic mocks to avoid circular reference issues
        self.config_dict = {
            "agent_ids": ["agent_1", "agent_2"],
            "embedding_dimensions": 384,
            "stm_capacity": 100,
            "im_capacity": 1000,
        }

        # Sample memory data
        self.stm_memories = [
            {
                "memory_id": "mem_stm_1",
                "agent_id": "agent_1",
                "content": {"text": "STM test memory"},
                "type": "state",
                "metadata": {
                    "creation_time": int(datetime.datetime.now().timestamp()),
                    "importance_score": 0.9,
                    "memory_type": "state",
                    "current_tier": "stm",
                },
                "step_number": 1,
                "timestamp": int(datetime.datetime.now().timestamp()),
                "embeddings": {"text": [0.1, 0.2, 0.3]},
            }
        ]

        self.im_memories = [
            {
                "memory_id": "mem_im_1",
                "agent_id": "agent_1",
                "content": {"text": "IM test memory"},
                "type": "interaction",
                "metadata": {
                    "creation_time": int(datetime.datetime.now().timestamp()),
                    "importance_score": 0.7,
                    "memory_type": "interaction",
                    "current_tier": "im",
                },
                "step_number": 2,
                "timestamp": int(datetime.datetime.now().timestamp()),
                "embeddings": {"text": [0.4, 0.5, 0.6]},
            }
        ]

        self.ltm_memories = [
            {
                "memory_id": "mem_ltm_1",
                "agent_id": "agent_1",
                "content": {"text": "LTM test memory"},
                "type": "action",
                "metadata": {
                    "creation_time": int(datetime.datetime.now().timestamp()),
                    "importance_score": 0.5,
                    "memory_type": "action",
                    "current_tier": "ltm",
                },
                "step_number": 3,
                "timestamp": int(datetime.datetime.now().timestamp()),
                "embeddings": {"text": [0.7, 0.8, 0.9]},
            }
        ]

    def tearDown(self):
        """Clean up temporary files after tests."""
        self.temp_dir.cleanup()

    @patch("memory.schema.validate_memory_system_json")
    def test_save_memory_system_to_json(self, mock_validate):
        """Test saving a memory system to a JSON file."""
        # Set the validator to return True
        mock_validate.return_value = True

        # Create mocks with proper configuration to avoid circular references
        mock_stm_store = MagicMock()
        mock_stm_store.get_all.return_value = self.stm_memories

        mock_im_store = MagicMock()
        mock_im_store.get_all.return_value = self.im_memories

        mock_ltm_store = MagicMock()
        mock_ltm_store.get_all.return_value = self.ltm_memories

        # Create a mock agent that uses the store mocks
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_1"
        mock_agent.stm_store = mock_stm_store
        mock_agent.im_store = mock_im_store
        mock_agent.ltm_store = mock_ltm_store

        # Create a plain object for config to avoid MagicMock issues
        class Config:
            pass

        config = Config()
        for key, value in self.config_dict.items():
            setattr(config, key, value)

        # Create the memory system mock with manually set attributes
        mock_memory_system = MagicMock()
        mock_memory_system.config = config
        mock_memory_system.agents = {"agent_1": mock_agent}

        # Call the function
        with patch("builtins.open", mock_open()) as m:
            result = save_memory_system_to_json(mock_memory_system, self.test_filepath)

        # Check the result
        self.assertTrue(result)

        # Verify the validation was called
        mock_validate.assert_called_once()

        # Verify file was opened for writing
        m.assert_called_once_with(self.test_filepath, "w", encoding="utf-8")

        # Verify all stores were queried
        mock_stm_store.get_all.assert_called_once_with("agent_1")
        mock_im_store.get_all.assert_called_once_with("agent_1")
        mock_ltm_store.get_all.assert_called_once_with("agent_1")

    @patch("memory.schema.validate_memory_system_json")
    def test_save_memory_system_validation_failure(self, mock_validate):
        """Test saving with schema validation failure."""
        # Set the validator to return False
        mock_validate.return_value = False

        # Create a minimal mock system for this test
        # Use a simple class instead of MagicMock to avoid circular reference issues
        class SimpleConfig:
            pass

        config = SimpleConfig()

        mock_memory_system = MagicMock()
        mock_memory_system.config = config
        mock_memory_system.agents = {}

        # Call the function
        result = save_memory_system_to_json(mock_memory_system, self.test_filepath)

        # Check the result
        self.assertFalse(result)

        # Verify the validation was called
        mock_validate.assert_called_once()

    @patch("memory.schema.validate_memory_system_json")
    @patch("memory.core.AgentMemorySystem")
    @patch("memory.config.MemoryConfig")
    @patch("memory.config.RedisSTMConfig")
    @patch("memory.config.RedisIMConfig")
    @patch("memory.config.SQLiteLTMConfig")
    @patch("memory.config.AutoencoderConfig")
    @patch("memory.embeddings.vector_store.VectorStore")
    @patch("memory.embeddings.text_embeddings.TextEmbeddingEngine")
    def test_load_memory_system_from_json(
        self,
        mock_embedding_engine,
        mock_vector_store,
        mock_autoencoder_config,
        mock_sqlite_config,
        mock_redis_im_config,
        mock_redis_stm_config,
        mock_memory_config,
        mock_memory_system_class,
        mock_validate,
    ):
        """Test loading a memory system from a JSON file."""
        # Create test data
        test_data = {
            "config": {
                "agent_ids": ["agent_1"],
                "embedding_dimensions": 384,
                "cleanup_interval": 100,
                "memory_priority_decay": 0.95,
                "enable_memory_hooks": True,
                "logging_level": "INFO",
                "stm_config": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "ttl": 86400,
                    "memory_limit": 1000,
                    "namespace": "memory:stm",
                },
                "im_config": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 1,
                    "ttl": 604800,
                    "memory_limit": 10000,
                    "namespace": "memory:im",
                },
                "ltm_config": {"db_path": ":memory:", "memory_limit": 100000},
                "autoencoder_config": {"stm_dim": 384, "im_dim": 384, "ltm_dim": 384},
            },
            "agents": {
                "agent_1": {
                    "agent_id": "agent_1",
                    "memories": [
                        {
                            "memory_id": "mem_1",
                            "agent_id": "agent_1",
                            "type": "state",
                            "content": {"text": "Test state memory"},
                            "metadata": {
                                "memory_type": "state",
                                "importance_score": 0.8,
                                "creation_time": int(
                                    datetime.datetime.now().timestamp()
                                ),
                                "current_tier": "stm",
                            },
                            "step_number": 1,
                        },
                        {
                            "memory_id": "mem_2",
                            "agent_id": "agent_1",
                            "type": "interaction",
                            "content": {"text": "Test interaction memory"},
                            "metadata": {
                                "memory_type": "interaction",
                                "importance_score": 0.6,
                                "creation_time": int(
                                    datetime.datetime.now().timestamp()
                                ),
                                "current_tier": "im",
                            },
                            "step_number": 2,
                        },
                    ],
                }
            },
        }

        # Set the validator to return True
        mock_validate.return_value = True

        # Create a mock memory agent with mock store attributes
        mock_memory_agent = MagicMock()
        mock_memory_agent.stm_store = MagicMock()
        mock_memory_agent.im_store = MagicMock()
        mock_memory_agent.ltm_store = MagicMock()

        # Configure the mock memory system to return the mock agent
        mock_memory_system = MagicMock()
        mock_memory_system.get_memory_agent.return_value = mock_memory_agent
        mock_memory_system_class.return_value = mock_memory_system

        # Mock configurations
        mock_memory_config.return_value = MagicMock()

        # Configure Redis STM config
        mock_stm_config = MagicMock()
        mock_stm_config.connection_params = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
        }
        mock_stm_config.namespace = "memory:stm"
        mock_redis_stm_config.return_value = mock_stm_config

        # Configure Redis IM config
        mock_im_config = MagicMock()
        mock_im_config.connection_params = {
            "host": "localhost",
            "port": 6379,
            "db": 1,
            "password": None,
        }
        mock_im_config.namespace = "memory:im"
        mock_redis_im_config.return_value = mock_im_config

        # Configure SQLite LTM config
        mock_ltm_config = MagicMock()
        mock_sqlite_config.return_value = mock_ltm_config

        # Set the configs on the memory config
        mock_memory_config.return_value.stm_config = mock_stm_config
        mock_memory_config.return_value.im_config = mock_im_config
        mock_memory_config.return_value.ltm_config = mock_ltm_config

        mock_vector_store.return_value = MagicMock()
        mock_embedding_engine.return_value = MagicMock()

        # Configure autoencoder config mock
        mock_autoencoder = MagicMock()
        mock_autoencoder.stm_dim = 384
        mock_autoencoder.im_dim = 384
        mock_autoencoder.ltm_dim = 384
        mock_autoencoder_config.return_value = mock_autoencoder

        # Set the autoencoder config on the memory config
        mock_memory_config.return_value.autoencoder_config = mock_autoencoder

        # Call the function
        with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
            with patch("logging.Logger.error") as mock_logger:
                result = load_memory_system_from_json(self.test_filepath)
                if result is None:
                    # Print the error message that was logged
                    error_calls = mock_logger.call_args_list
                    for call in error_calls:
                        print(f"Error logged: {call[0][0]}")
                        if len(call[0]) > 1:
                            print(f"Error args: {call[0][1]}")

        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(result, mock_memory_system)

        # Verify the validation was called
        mock_validate.assert_called_once()

        # Verify memory agent was created and memories were added to the correct stores
        mock_memory_system.get_memory_agent.assert_called_once_with("agent_1")

        # Verify that the stm_store.store was called for the state memory
        self.assertEqual(mock_memory_agent.stm_store.store.call_count, 1)

        # Verify that the im_store.store was called for the interaction memory
        self.assertEqual(mock_memory_agent.im_store.store.call_count, 1)

    @patch("memory.schema.validate_memory_system_json")
    def test_load_memory_system_validation_failure(self, mock_validate):
        """Test loading with schema validation failure."""
        # Set the validator to return False
        mock_validate.return_value = False

        # Call the function
        with patch("builtins.open", mock_open(read_data="{}")):
            result = load_memory_system_from_json(self.test_filepath)

        # Check the result
        self.assertIsNone(result)

        # Verify the validation was called
        mock_validate.assert_called_once()

    def test_load_memory_system_file_not_found(self):
        """Test loading from a non-existent file."""
        # Call the function with a non-existent file
        non_existent_file = os.path.join(self.temp_dir.name, "does_not_exist.json")
        result = load_memory_system_from_json(non_existent_file)

        # Check the result
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
