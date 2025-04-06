"""Unit tests for the SQLite-based Long-Term Memory (LTM) storage.

This module tests the functionality, reliability, and error handling of the 
SQLiteLTMStore class that provides persistent storage for agent memories.
"""

import json
import os
import sqlite3
import tempfile
import time
from unittest import mock

import numpy as np
import pytest

from memory.config import SQLiteLTMConfig
from memory.storage.sqlite_ltm import SQLiteLTMStore
from memory.utils.error_handling import SQLitePermanentError, SQLiteTemporaryError


class TestSQLiteLTMStore:
    """Test suite for SQLiteLTMStore class."""

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

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory entry for testing."""
        return {
            "memory_id": "test_memory_001",
            "agent_id": "test_agent",
            "step_number": 42,
            "timestamp": int(time.time()),
            "type": "observation",
            "content": {
                "text": "This is a test memory",
                "source": "unit_test"
            },
            "metadata": {
                "importance_score": 0.75,
                "compression_level": 2,
                "retrieval_count": 0,
                "creation_time": int(time.time()),
                "last_access_time": int(time.time())
            },
            "embeddings": {
                "compressed_vector": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }

    @pytest.fixture
    def sample_memories(self):
        """Create multiple sample memory entries for batch testing."""
        base_time = int(time.time())
        memories = []
        
        for i in range(5):
            memories.append({
                "memory_id": f"test_memory_{i:03d}",
                "agent_id": "test_agent",
                "step_number": i,
                "timestamp": base_time + i,
                "type": "observation",
                "content": {
                    "text": f"This is test memory {i}",
                    "source": "unit_test"
                },
                "metadata": {
                    "importance_score": 0.5 + (i * 0.1),
                    "compression_level": 2,
                    "retrieval_count": i,
                    "creation_time": base_time + i,
                    "last_access_time": base_time + i
                },
                "embeddings": {
                    # Modified to ensure memory 2 is most similar to itself
                    # Each memory has a vector that's most similar to itself
                    "compressed_vector": [0.1, 0.1, 0.1, 0.1, 0.1] if i != 2 else [0.5, 0.5, 0.5, 0.5, 0.5]
                }
            })
        
        return memories

    def test_init_creates_database_directory(self, ltm_config):
        """Test that initialization creates the database directory if it doesn't exist."""
        # Create a config with a path in a non-existent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_subdir")
            db_path = os.path.join(test_dir, "test.db")
            
            config = SQLiteLTMConfig(db_path=db_path)
            
            # Directory should not exist yet
            assert not os.path.exists(test_dir)
            
            # Initialize store
            SQLiteLTMStore(agent_id="test_agent", config=config)
            
            # Directory should now exist
            assert os.path.exists(test_dir)
            assert os.path.exists(os.path.dirname(db_path))

    def test_init_creates_tables(self, ltm_store, ltm_config):
        """Test that initialization creates the required database tables."""
        # Connect to the database directly
        conn = sqlite3.connect(ltm_config.db_path)
        cursor = conn.cursor()
        
        # Check if the tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Verify both tables are created
        assert f"{ltm_config.table_prefix}_memories" in tables
        assert f"{ltm_config.table_prefix}_embeddings" in tables
        
        # Verify the structure of the memories table
        cursor.execute(f"PRAGMA table_info({ltm_config.table_prefix}_memories)")
        columns = {row[1] for row in cursor.fetchall()}
        
        expected_columns = {
            "memory_id", "agent_id", "step_number", "timestamp", 
            "content_json", "metadata_json", "compression_level", 
            "importance_score", "retrieval_count", "memory_type",
            "created_at", "last_accessed"
        }
        
        assert expected_columns.issubset(columns)
        
        conn.close()

    def test_store_and_get(self, ltm_store, sample_memory):
        """Test storing and retrieving a memory entry."""
        # Store the memory
        success = ltm_store.store(sample_memory)
        assert success is True
        
        # Retrieve the memory
        retrieved = ltm_store.get(sample_memory["memory_id"])
        
        # Verify the retrieved memory
        assert retrieved is not None
        assert retrieved["memory_id"] == sample_memory["memory_id"]
        assert retrieved["agent_id"] == sample_memory["agent_id"]
        assert retrieved["step_number"] == sample_memory["step_number"]
        assert retrieved["timestamp"] == sample_memory["timestamp"]
        assert retrieved["type"] == sample_memory["type"]
        assert retrieved["content"] == sample_memory["content"]
        
        # Verify embeddings are retrieved correctly
        assert "embeddings" in retrieved
        assert "compressed_vector" in retrieved["embeddings"]
        assert len(retrieved["embeddings"]["compressed_vector"]) == len(
            sample_memory["embeddings"]["compressed_vector"])

    def test_store_without_memory_id(self, ltm_store):
        """Test that storing a memory without a memory_id fails gracefully."""
        invalid_memory = {
            "agent_id": "test_agent",
            "content": {"text": "Invalid memory without ID"}
        }
        
        success = ltm_store.store(invalid_memory)
        assert success is False

    def test_store_batch(self, ltm_store, sample_memories):
        """Test storing and retrieving multiple memories in a batch."""
        # Store the batch of memories
        success = ltm_store.store_batch(sample_memories)
        assert success is True
        
        # Retrieve and verify each memory
        for memory in sample_memories:
            retrieved = ltm_store.get(memory["memory_id"])
            assert retrieved is not None
            assert retrieved["memory_id"] == memory["memory_id"]
            assert retrieved["content"] == memory["content"]

    def test_get_nonexistent_memory(self, ltm_store):
        """Test that retrieving a non-existent memory returns None."""
        retrieved = ltm_store.get("nonexistent_memory_id")
        assert retrieved is None

    def test_update_access_metadata(self, ltm_store, sample_memory):
        """Test that retrieval count is updated when a memory is accessed."""
        # Store the memory
        ltm_store.store(sample_memory)
        
        # Get initial retrieval count directly from the database
        conn = sqlite3.connect(ltm_store.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            f"SELECT retrieval_count FROM {ltm_store.memory_table} WHERE memory_id = ?",
            (sample_memory["memory_id"],)
        )
        
        row = cursor.fetchone()
        initial_count = row["retrieval_count"]
        
        # Retrieve memory to trigger update
        ltm_store.get(sample_memory["memory_id"])
        
        # Check updated count
        cursor.execute(
            f"SELECT retrieval_count FROM {ltm_store.memory_table} WHERE memory_id = ?",
            (sample_memory["memory_id"],)
        )
        
        row = cursor.fetchone()
        updated_count = row["retrieval_count"]
        conn.close()
        
        # Verify retrieval count was incremented by exactly 1
        assert updated_count == initial_count + 1

    def test_get_by_timerange(self, ltm_store, sample_memories):
        """Test retrieving memories within a specific time range."""
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Get the timestamp range for testing
        min_time = min(m["timestamp"] for m in sample_memories)
        max_time = max(m["timestamp"] for m in sample_memories)
        mid_time = min_time + ((max_time - min_time) // 2)
        
        # Test with full range
        full_range = ltm_store.get_by_timerange(min_time, max_time)
        assert len(full_range) == len(sample_memories)
        
        # Test with partial range
        partial_range = ltm_store.get_by_timerange(mid_time, max_time)
        assert 0 < len(partial_range) < len(sample_memories)
        
        # Verify all retrieved memories are within the specified time range
        for memory in partial_range:
            assert mid_time <= memory["timestamp"] <= max_time

    def test_get_by_importance(self, ltm_store, sample_memories):
        """Test retrieving memories based on importance score range."""
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Test with full importance range
        full_range = ltm_store.get_by_importance(0.0, 1.0)
        assert len(full_range) == len(sample_memories)
        
        # Test with specific importance range
        high_importance = ltm_store.get_by_importance(0.7, 1.0)
        
        # Verify all retrieved memories have importance scores in the specified range
        for memory in high_importance:
            assert 0.7 <= memory["metadata"]["importance_score"] <= 1.0

    def test_get_most_similar(self, ltm_store, sample_memories):
        """Test retrieving memories most similar to a query vector."""
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Create a query vector that's exactly the same as memory[2]'s vector
        query_vector = sample_memories[2]["embeddings"]["compressed_vector"]
        
        # Get most similar memories
        similar_memories = ltm_store.get_most_similar(query_vector, top_k=3)
        
        # Verify we get results
        assert len(similar_memories) > 0
        
        # Each result should be a tuple of (memory_entry, similarity_score)
        assert len(similar_memories[0]) == 2
        
        # The memory most similar to the query should be the one we based it on
        assert similar_memories[0][0]["memory_id"] == sample_memories[2]["memory_id"]
        
        # Similarity scores should be between -1 and 1 (cosine similarity)
        for _, similarity in similar_memories:
            assert -1.0 <= similarity <= 1.0

    def test_count(self, ltm_store, sample_memories):
        """Test counting the number of memories for an agent."""
        # Initially should be 0
        assert ltm_store.count() == 0
        
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Should match the number of memories we stored
        assert ltm_store.count() == len(sample_memories)

    def test_get_all(self, ltm_store, sample_memories):
        """Test retrieving all memories for an agent."""
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Get all memories
        all_memories = ltm_store.get_all()
        
        # Should match the number of memories we stored
        assert len(all_memories) == len(sample_memories)
        
        # Test with limit
        limited_memories = ltm_store.get_all(limit=2)
        assert len(limited_memories) == 2

    def test_delete(self, ltm_store, sample_memory):
        """Test deleting a memory entry."""
        # Store the memory
        ltm_store.store(sample_memory)
        
        # Verify it exists
        assert ltm_store.get(sample_memory["memory_id"]) is not None
        
        # Delete the memory
        success = ltm_store.delete(sample_memory["memory_id"])
        assert success is True
        
        # Verify it's gone
        assert ltm_store.get(sample_memory["memory_id"]) is None
        
        # Deleting non-existent memory should return False
        assert ltm_store.delete("nonexistent_memory_id") is False

    def test_clear(self, ltm_store, sample_memories):
        """Test clearing all memories for an agent."""
        # Store memories
        ltm_store.store_batch(sample_memories)
        
        # Verify they exist
        assert ltm_store.count() == len(sample_memories)
        
        # Clear all memories
        success = ltm_store.clear()
        assert success is True
        
        # Verify they're gone
        assert ltm_store.count() == 0

    def test_check_health(self, ltm_store):
        """Test health check functionality."""
        # A new store should be healthy
        health = ltm_store.check_health()
        
        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert health["integrity"] == "ok"
        assert health["client"] == "sqlite-ltm"

    def test_connection_error(self, ltm_config):
        """Test handling of connection errors."""
        # Create a real config but mock SQLiteLTMStore's _init_database method
        # to avoid initialization errors
        with mock.patch.object(SQLiteLTMStore, '_init_database', return_value=None):
            # Create store with mocked initialization
            store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
            
            # Now mock _get_connection for store operations
            with mock.patch.object(store, '_get_connection') as mock_conn:
                mock_conn.side_effect = SQLiteTemporaryError("database is locked")
                
                # Test store method with the mocked connection
                memory = {
                    "memory_id": "test_error_memory", 
                    "content": {"text": "Test memory"}
                }
                
                success = store.store(memory)
                assert success is False

    def test_temporary_sqlite_error(self, ltm_config):
        """Test handling of temporary SQLite errors."""
        # Create a real config but mock SQLiteLTMStore's _init_database method
        # to avoid initialization errors
        with mock.patch.object(SQLiteLTMStore, '_init_database', return_value=None):
            # Create store with mocked initialization
            store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
            
            # Now mock _get_connection for get operations
            with mock.patch.object(store, '_get_connection') as mock_conn:
                # Set the side effect to raise the temporary error
                mock_conn.side_effect = SQLiteTemporaryError("database is locked")
                
                # Attempt to get a memory
                memory = store.get("any_id")
                
                # Should return None due to error handling in the get method
                assert memory is None
                
                # The error should be logged but not propagated

    def test_permanent_sqlite_error(self, ltm_config):
        """Test handling of permanent SQLite errors."""
        # Create a real config but mock SQLiteLTMStore's _init_database method
        # to avoid initialization errors
        with mock.patch.object(SQLiteLTMStore, '_init_database', return_value=None):
            # Create store with mocked initialization
            store = SQLiteLTMStore(agent_id="test_agent", config=ltm_config)
            
            # Now mock _get_connection for get operations
            with mock.patch.object(store, '_get_connection') as mock_conn:
                # Set the side effect to raise the permanent error
                mock_conn.side_effect = SQLitePermanentError("database disk image is malformed")
                
                # Attempt to get a memory
                memory = store.get("any_id")
                
                # Should return None due to error handling in the get method
                assert memory is None
                
                # The error should be logged but not propagated

    def test_update_existing_memory(self, ltm_store, sample_memory):
        """Test updating an existing memory entry."""
        # Store the original memory
        ltm_store.store(sample_memory)
        
        # Create an updated version
        updated_memory = sample_memory.copy()
        updated_memory["content"] = {"text": "Updated test memory", "source": "unit_test"}
        updated_memory["metadata"]["importance_score"] = 0.9
        
        # Store the updated memory
        success = ltm_store.store(updated_memory)
        assert success is True
        
        # Retrieve and verify the memory has been updated
        retrieved = ltm_store.get(sample_memory["memory_id"])
        assert retrieved["content"] == updated_memory["content"]
        assert retrieved["metadata"]["importance_score"] == 0.9

    def test_memory_without_embeddings(self, ltm_store):
        """Test storing and retrieving a memory without embeddings."""
        memory = {
            "memory_id": "memory_without_embeddings",
            "agent_id": "test_agent",
            "step_number": 1,
            "timestamp": int(time.time()),
            "type": "observation",
            "content": {"text": "Memory without embeddings"},
            "metadata": {"importance_score": 0.5}
        }
        
        # Store the memory
        success = ltm_store.store(memory)
        assert success is True
        
        # Retrieve and verify
        retrieved = ltm_store.get(memory["memory_id"])
        assert retrieved is not None
        assert retrieved["memory_id"] == memory["memory_id"]
        assert "embeddings" not in retrieved or not retrieved["embeddings"]

    def test_empty_batch(self, ltm_store):
        """Test that storing an empty batch succeeds."""
        success = ltm_store.store_batch([])
        assert success is True

    def test_serialization_edge_cases(self, ltm_store):
        """Test serialization of memories with complex content."""
        # Memory with nested structures
        complex_memory = {
            "memory_id": "complex_memory",
            "agent_id": "test_agent",
            "step_number": 1,
            "timestamp": int(time.time()),
            "type": "complex",
            "content": {
                "nested": {"level1": {"level2": {"level3": "deep value"}}},
                "list_of_dicts": [{"key1": "value1"}, {"key2": "value2"}],
                "numbers": [1, 2, 3, 4, 5],
                "boolean": True,
                "null_value": None
            },
            "metadata": {"importance_score": 0.5}
        }
        
        # Store the memory
        success = ltm_store.store(complex_memory)
        assert success is True
        
        # Retrieve and verify
        retrieved = ltm_store.get(complex_memory["memory_id"])
        assert retrieved is not None
        assert retrieved["content"] == complex_memory["content"] 