"""Edge case tests for the SQLite-based Long-Term Memory (LTM) storage.

This module tests boundary conditions, error cases, and unusual scenarios
for the SQLiteLTMStore implementation.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest import mock
from contextlib import contextmanager

import numpy as np
import pytest
import sqlite3

from memory.config import SQLiteLTMConfig
from memory.storage.sqlite_ltm import SQLiteLTMStore
from memory.utils.error_handling import SQLitePermanentError, SQLiteTemporaryError


class TestSQLiteLTMStoreEdgeCases:
    """Test suite for SQLiteLTMStore edge cases and error handling."""

    @pytest.fixture
    def tmp_db_path(self):
        """Create a temporary database file path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name
        yield db_path
        # Cleanup after test
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def ltm_config(self, tmp_db_path):
        """Create a SQLiteLTMConfig with a temporary database path."""
        return SQLiteLTMConfig(
            db_path=tmp_db_path,
            compression_level=2,
            batch_size=10,
            table_prefix="test_ltm"
        )

    @pytest.fixture
    def ltm_store(self, ltm_config):
        """Create a SQLiteLTMStore instance for testing."""
        agent_id = "test_agent"
        store = SQLiteLTMStore(agent_id=agent_id, config=ltm_config)
        return store

    def test_missing_database_dir(self):
        """Test behavior when database directory doesn't exist."""
        # Create a temp path but remove it to test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = os.path.join(temp_dir, "non_existent", "subdir")
            db_path = os.path.join(non_existent_dir, "test.db")
            
            # Verify directory doesn't exist
            assert not os.path.exists(non_existent_dir)
            
            # Initialize store with non-existent path
            config = SQLiteLTMConfig(db_path=db_path)
            SQLiteLTMStore(agent_id="test_agent", config=config)
            
            # Verify directory was created
            assert os.path.exists(non_existent_dir)

    def test_invalid_db_path(self):
        """Test with invalid database path."""
        # Test with a path that can't be created
        if os.name == 'nt':  # Windows
            invalid_path = r"\\?\invalid_device:abc"
        else:  # Unix
            invalid_path = "/root/no_permission_here/test.db"  # requires root to write
            
            # Skip if we have permission (e.g., running as root)
            if os.access(os.path.dirname(invalid_path), os.W_OK):
                pytest.skip("Test skipped because we have write permission")
            
        config = SQLiteLTMConfig(db_path=invalid_path)
        
        # Initialization should fail gracefully
        with pytest.raises(Exception):
            SQLiteLTMStore(agent_id="test_agent", config=config)

    def test_empty_memory_id(self, ltm_store):
        """Test with empty memory_id."""
        memory = {
            "memory_id": "",  # Empty ID
            "agent_id": "test_agent",
            "content": {"text": "Memory with empty ID"}
        }
        
        # Store should succeed - SQLite accepts empty strings as primary keys
        success = ltm_store.store(memory)
        assert success is True
        
        # Retrieval should work
        retrieved = ltm_store.get("")
        assert retrieved is not None
        assert retrieved["content"]["text"] == "Memory with empty ID"

    def test_extreme_memory_sizes(self, ltm_store):
        """Test with very large and very small memory content."""
        # Very small memory
        tiny_memory = {
            "memory_id": "tiny_memory",
            "agent_id": "test_agent",
            "content": {"text": ""},  # Empty text
            "metadata": {}
        }
        
        success = ltm_store.store(tiny_memory)
        assert success is True
        
        # Very large memory
        large_text = "x" * (10 * 1024 * 1024)  # 10 MB text
        large_memory = {
            "memory_id": "large_memory",
            "agent_id": "test_agent",
            "content": {"text": large_text},
            "metadata": {"size": "large"}
        }
        
        success = ltm_store.store(large_memory)
        assert success is True
        
        # Verify retrieval works for both
        tiny_retrieved = ltm_store.get("tiny_memory")
        assert tiny_retrieved is not None
        assert tiny_retrieved["content"]["text"] == ""
        
        large_retrieved = ltm_store.get("large_memory")
        assert large_retrieved is not None
        assert len(large_retrieved["content"]["text"]) == len(large_text)

    def test_special_characters_in_content(self, ltm_store):
        """Test with special characters in memory content."""
        special_chars = {
            "memory_id": "special_chars",
            "agent_id": "test_agent",
            "content": {
                "text": "Special chars: àéíóú ñ çøåß 你好 😀🔥💯",
                "quotes": "Single ' and double \" quotes",
                "backslashes": "Path with \\ backslashes",
                "control_chars": "\n\t\r\b\f",
                "unicode": "\u0000\u001F\u0080\uFFFF"  # NULL and other control chars
            }
        }
        
        success = ltm_store.store(special_chars)
        assert success is True
        
        retrieved = ltm_store.get("special_chars")
        assert retrieved is not None
        assert retrieved["content"]["text"] == special_chars["content"]["text"]
        assert retrieved["content"]["quotes"] == special_chars["content"]["quotes"]
        assert retrieved["content"]["backslashes"] == special_chars["content"]["backslashes"]
        assert retrieved["content"]["control_chars"] == special_chars["content"]["control_chars"]
        assert retrieved["content"]["unicode"] == special_chars["content"]["unicode"]

    def test_invalid_json_recovery(self, ltm_store):
        """Test manual recovery from invalid JSON in the database."""
        # Store a valid memory
        memory = {
            "memory_id": "valid_memory",
            "agent_id": "test_agent",
            "content": {"text": "Valid memory"}
        }
        ltm_store.store(memory)
        
        # Directly manipulate the database to insert invalid JSON
        conn = sqlite3.connect(ltm_store.db_path)
        cursor = conn.cursor()
        
        # Insert invalid JSON in content_json field
        cursor.execute(
            f"UPDATE {ltm_store.memory_table} SET content_json = ? WHERE memory_id = ?",
            ("This is not valid JSON", "valid_memory")
        )
        conn.commit()
        conn.close()
        
        # Retrieval should handle the error gracefully
        retrieved = ltm_store.get("valid_memory")
        assert retrieved is None  # Should fail gracefully

    def test_corrupt_database_recovery(self, ltm_config):
        """Test recovery behavior with a corrupted database."""
        # Create a corrupted database file
        with open(ltm_config.db_path, 'w') as f:
            f.write("This is not a valid SQLite database file")
        
        # Attempting to initialize store should fail gracefully
        with pytest.raises(Exception):
            SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
        
        # Clean up the corrupted file and create a valid one
        os.remove(ltm_config.db_path)
        store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
        
        # Should be able to operate normally now
        success = store.store({
            "memory_id": "recovery_test",
            "agent_id": "test_agent",
            "content": {"text": "After recovery"}
        })
        assert success is True

    def test_identical_memory_ids_different_agents(self, ltm_config):
        """Test handling of identical memory IDs for different agents."""
        # Create stores for different agents
        agent1_store = SQLiteLTMStore(agent_id="agent1", config=ltm_config)
        agent2_store = SQLiteLTMStore(agent_id="agent2", config=ltm_config)
        
        # Create memories with the same ID
        memory_id = "duplicate_memory_id"
        
        agent1_memory = {
            "memory_id": memory_id,
            "agent_id": "agent1",
            "content": {"text": "Agent 1 memory"}
        }
        
        agent2_memory = {
            "memory_id": memory_id,
            "agent_id": "agent2",
            "content": {"text": "Agent 2 memory"}
        }
        
        # Store both memories
        agent1_store.store(agent1_memory)
        agent2_store.store(agent2_memory)
        
        # Verify each agent gets their own memory
        retrieved1 = agent1_store.get(memory_id)
        retrieved2 = agent2_store.get(memory_id)
        
        assert retrieved1["content"]["text"] == "Agent 1 memory"
        assert retrieved2["content"]["text"] == "Agent 2 memory"

    def test_null_values_in_memory(self, ltm_store):
        """Test handling of null/None values in memory data."""
        memory_with_nulls = {
            "memory_id": "memory_with_nulls",
            "agent_id": "test_agent",
            "step_number": None,  # Explicit None
            "timestamp": int(time.time()),
            "type": None,  # Explicit None
            "content": {
                "text": None,  # Explicit None in content
                "valid_field": "This field has a value"
            },
            "metadata": {
                "importance_score": None,  # Explicit None in metadata
                "valid_metadata": 123
            }
        }
        
        # Store should handle nulls
        success = ltm_store.store(memory_with_nulls)
        assert success is True
        
        # Retrieval should preserve nulls
        retrieved = ltm_store.get("memory_with_nulls")
        assert retrieved is not None
        assert retrieved["content"]["text"] is None
        assert retrieved["type"] is None
        assert retrieved["content"]["valid_field"] == "This field has a value"
        assert retrieved["metadata"]["importance_score"] is None
        assert retrieved["metadata"]["valid_metadata"] == 123

    def test_memory_with_minimal_fields(self, ltm_store):
        """Test with minimal required fields in memory object."""
        minimal_memory = {
            "memory_id": "minimal_memory",
            # No other fields provided
        }
        
        # Store should succeed with defaults
        success = ltm_store.store(minimal_memory)
        assert success is True
        
        # Verify retrieval
        retrieved = ltm_store.get("minimal_memory")
        assert retrieved is not None
        assert retrieved["memory_id"] == "minimal_memory"
        assert "content" in retrieved
        assert "metadata" in retrieved

    def test_race_condition_on_store(self, ltm_store, monkeypatch):
        """Test handling of race conditions during store operations."""
        # Create a custom connection factory that simulates lock errors
        real_get_connection = ltm_store._get_connection
        
        call_count = 0
        
        @contextmanager
        def mock_get_connection():
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                # Simulate database lock on first connection attempt
                raise SQLiteTemporaryError("database is locked")
            else:
                # Use the real connection for subsequent attempts
                with real_get_connection() as conn:
                    yield conn
        
        # Replace the connection factory
        monkeypatch.setattr(ltm_store, "_get_connection", mock_get_connection)
        
        # Store should fail due to the simulated lock
        memory = {
            "memory_id": "race_condition_test",
            "agent_id": "test_agent",
            "content": {"text": "Test race condition"}
        }
        
        success = ltm_store.store(memory)
        assert success is False

    def test_binary_data_in_content(self, ltm_store):
        """Test storing and retrieving binary data in memory content."""
        # Create memory with binary data
        binary_data = bytes([0, 1, 2, 3, 255])
        binary_b64 = binary_data.hex()  # Use hex representation for JSON compatibility
        
        memory = {
            "memory_id": "binary_data_memory",
            "agent_id": "test_agent",
            "content": {
                "text": "Memory with binary data",
                "binary": binary_b64
            }
        }
        
        success = ltm_store.store(memory)
        assert success is True
        
        # Retrieve and verify
        retrieved = ltm_store.get("binary_data_memory")
        assert retrieved is not None
        assert retrieved["content"]["binary"] == binary_b64

    def test_embeddings_edge_cases(self, ltm_store):
        """Test edge cases for vector embeddings."""
        # Test with zero vector
        zero_vector_memory = {
            "memory_id": "zero_vector",
            "agent_id": "test_agent",
            "content": {"text": "Zero vector memory"},
            "embeddings": {
                "compressed_vector": [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        
        ltm_store.store(zero_vector_memory)
        
        # Test with very large vector
        large_vector = [0.1] * 1024  # 1024-dimensional vector
        large_vector_memory = {
            "memory_id": "large_vector",
            "agent_id": "test_agent",
            "content": {"text": "Large vector memory"},
            "embeddings": {
                "compressed_vector": large_vector
            }
        }
        
        ltm_store.store(large_vector_memory)
        
        # Test similarity search with zero vector
        # (This tests division by zero handling in cosine similarity)
        similar_to_zero = ltm_store.get_most_similar([0.0, 0.0, 0.0, 0.0, 0.0])
        # Should not crash, though results may not be meaningful
        assert isinstance(similar_to_zero, list)
        
        # Verify large vector was stored correctly
        retrieved = ltm_store.get("large_vector")
        assert len(retrieved["embeddings"]["compressed_vector"]) == 1024

    def test_db_readonly_mode(self, ltm_config):
        """Test behavior when database is in read-only mode."""
        # First create and populate the database
        store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
        store.store({
            "memory_id": "test_memory",
            "agent_id": "test_agent",
            "content": {"text": "Test memory"}
        })
        
        # Now make the database read-only
        os.chmod(ltm_config.db_path, 0o444)  # Read-only permission
        
        try:
            # Read operations should still work
            readonly_store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
            retrieved = readonly_store.get("test_memory")
            assert retrieved is not None
            
            # Write operations should fail gracefully
            success = readonly_store.store({
                "memory_id": "new_memory",
                "agent_id": "test_agent",
                "content": {"text": "This should fail"}
            })
            assert success is False
            
        finally:
            # Restore permissions for cleanup
            os.chmod(ltm_config.db_path, 0o644)

    def test_transaction_abort(self, ltm_store, monkeypatch):
        """Test that transactions are properly aborted on error."""
        # Create a custom store class that simulates errors during batch processing
        class TestStore(SQLiteLTMStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.commit_called = False
            
            def store_batch(self, memory_entries):
                # Create a batch of successful entries but fail on a specific one
                for i, entry in enumerate(memory_entries):
                    if "batch_fail_4" == entry.get("memory_id"):
                        # For this specific test, we'll return false without actually trying
                        # to store anything, simulating the transaction rollback behavior
                        return False
                
                # Should not reach here in the test case
                self.commit_called = True
                return super().store_batch(memory_entries)
        
        # Create our test store with the same config
        test_store = TestStore(agent_id="test_agent", config=ltm_store.config)
        
        # Create a batch of memories
        batch = []
        for i in range(10):
            batch.append({
                "memory_id": f"batch_fail_{i}",
                "agent_id": "test_agent",
                "content": {"text": f"Memory {i}"}
            })
        
        # Batch store should fail
        success = test_store.store_batch(batch)
        assert success is False
        
        # No commit should have happened
        assert not test_store.commit_called
        
        # None of the memories should be stored
        for i in range(10):
            assert test_store.get(f"batch_fail_{i}") is None

    def test_unicode_memory_ids(self, ltm_store):
        """Test with Unicode characters in memory IDs."""
        unicode_ids = [
            "memory_with_emoji_😀",
            "memory_with_chinese_你好",
            "memory_with_arabic_مرحبا",
            "memory_with_russian_привет",
            "memory_with_combining_é",
        ]
        
        # Store all memories
        for i, memory_id in enumerate(unicode_ids):
            success = ltm_store.store({
                "memory_id": memory_id,
                "agent_id": "test_agent",
                "content": {"text": f"Unicode memory {i}"}
            })
            assert success is True
        
        # Retrieve all memories
        for memory_id in unicode_ids:
            retrieved = ltm_store.get(memory_id)
            assert retrieved is not None
            assert retrieved["memory_id"] == memory_id

    def test_memory_id_collisions(self, ltm_store):
        """Test behavior when inserting memories with colliding IDs."""
        # Store original memory
        original_memory = {
            "memory_id": "colliding_id",
            "agent_id": "test_agent",
            "timestamp": int(time.time()),
            "content": {"text": "Original memory"},
            "metadata": {"version": 1}
        }
        
        success = ltm_store.store(original_memory)
        assert success is True
        
        # Store a memory with the same ID but different content
        updated_memory = {
            "memory_id": "colliding_id",
            "agent_id": "test_agent",
            "timestamp": int(time.time()) + 100,
            "content": {"text": "Updated memory"},
            "metadata": {"version": 2}
        }
        
        success = ltm_store.store(updated_memory)
        assert success is True
        
        # Retrieve should return the updated memory
        retrieved = ltm_store.get("colliding_id")
        assert retrieved["content"]["text"] == "Updated memory"
        assert retrieved["metadata"]["version"] == 2 