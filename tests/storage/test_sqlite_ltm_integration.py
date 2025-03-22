"""Integration tests for the SQLite-based Long-Term Memory (LTM) storage.

This module tests how SQLiteLTMStore integrates with the rest of the memory system,
focusing on real-world usage patterns and interactions with other components.
"""

import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from agent_memory.config import SQLiteLTMConfig
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore


class TestSQLiteLTMStoreIntegration:
    """Integration tests for SQLiteLTMStore class."""

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

    def test_concurrent_access(self, ltm_config):
        """Test concurrent access to the database from multiple threads."""
        agent_id = "concurrent_test_agent"
        store = SQLiteLTMStore(agent_id=agent_id, config=ltm_config)
        
        # Create unique memories for each thread
        def create_memory(i):
            return {
                "memory_id": f"concurrent_memory_{i}",
                "agent_id": agent_id,
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "observation",
                "content": {"text": f"Concurrent test memory {i}"},
                "metadata": {"importance_score": 0.5},
                "embeddings": {
                    "compressed_vector": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
        
        # Function to store a memory and immediately retrieve it
        def store_and_retrieve(i):
            memory = create_memory(i)
            success = store.store(memory)
            retrieved = store.get(memory["memory_id"])
            return success, retrieved is not None and retrieved["memory_id"] == memory["memory_id"]
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(store_and_retrieve, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all operations succeeded
        assert all(store_success and retrieve_success for store_success, retrieve_success in results)
        
        # Verify we can count the correct number of entries
        assert store.count() == 20

    def test_large_batch_operations(self, ltm_store):
        """Test storing and retrieving large batches of memories."""
        # Create a large batch of memories
        large_batch = []
        for i in range(100):
            large_batch.append({
                "memory_id": f"large_batch_{i:03d}",
                "agent_id": "test_agent",
                "step_number": i,
                "timestamp": int(time.time()) + i,
                "type": "observation",
                "content": {"text": f"Test memory {i} in large batch"},
                "metadata": {"importance_score": 0.1 + (i % 10) / 10},
                "embeddings": {
                    "compressed_vector": np.random.rand(5).tolist()
                }
            })
        
        # Store the batch
        success = ltm_store.store_batch(large_batch)
        assert success is True
        
        # Verify count
        assert ltm_store.count() == 100
        
        # Test retrieving by importance in several ranges
        high_importance = ltm_store.get_by_importance(0.7, 1.0)
        medium_importance = ltm_store.get_by_importance(0.4, 0.6)
        low_importance = ltm_store.get_by_importance(0.0, 0.3)
        
        # Verify each memory is in the correct range
        for memory in high_importance:
            assert 0.7 <= memory["metadata"]["importance_score"] <= 1.0
            
        for memory in medium_importance:
            assert 0.4 <= memory["metadata"]["importance_score"] <= 0.6
            
        for memory in low_importance:
            assert 0.0 <= memory["metadata"]["importance_score"] <= 0.3

    def test_memory_timespan_retrieval(self, ltm_store):
        """Test retrieving memories across different time spans."""
        # Create memories at specific times
        base_time = int(time.time()) - 86400  # 1 day ago
        time_spans = {
            "hour": base_time + 3600,  # 1 hour after base
            "day": base_time + 86400,  # 1 day after base
            "week": base_time + 604800,  # 1 week after base
            "month": base_time + 2592000,  # 30 days after base
        }
        
        # Store memories at each time point
        for span, timestamp in time_spans.items():
            ltm_store.store({
                "memory_id": f"time_span_{span}",
                "agent_id": "test_agent",
                "step_number": 1,
                "timestamp": timestamp,
                "type": "observation",
                "content": {"text": f"Memory from {span} timespan"},
                "metadata": {"importance_score": 0.5}
            })
        
        # Test retrieval across different time ranges
        hour_memories = ltm_store.get_by_timerange(
            base_time, base_time + 3600
        )
        assert len(hour_memories) == 1
        assert hour_memories[0]["memory_id"] == "time_span_hour"
        
        day_memories = ltm_store.get_by_timerange(
            base_time, base_time + 86400
        )
        assert len(day_memories) == 2  # hour and day
        
        all_memories = ltm_store.get_by_timerange(
            base_time, base_time + 2592000
        )
        assert len(all_memories) == len(time_spans)

    def test_vector_similarity_search(self, ltm_store):
        """Test similarity search with realistic vector embeddings."""
        # Create seeds for different topics
        topic_seeds = {
            "weather": [0.9, 0.1, 0.1, 0.1, 0.1],
            "politics": [0.1, 0.9, 0.1, 0.1, 0.1],
            "sports": [0.1, 0.1, 0.9, 0.1, 0.1],
            "technology": [0.1, 0.1, 0.1, 0.9, 0.1],
            "entertainment": [0.1, 0.1, 0.1, 0.1, 0.9]
        }
        
        # Create variations of each topic
        memories = []
        for topic, seed in topic_seeds.items():
            for i in range(3):  # 3 variations of each topic
                # Create variations by adding small amount of noise
                vector = [v + (np.random.random() * 0.1 - 0.05) for v in seed]
                
                # Normalize to unit length
                magnitude = sum(v**2 for v in vector) ** 0.5
                vector = [v / magnitude for v in vector]
                
                memories.append({
                    "memory_id": f"{topic}_{i}",
                    "agent_id": "test_agent",
                    "step_number": i,
                    "timestamp": int(time.time()),
                    "type": "observation",
                    "content": {"text": f"{topic} memory {i}"},
                    "metadata": {"importance_score": 0.5, "topic": topic},
                    "embeddings": {"compressed_vector": vector}
                })
        
        # Store all memories
        ltm_store.store_batch(memories)
        
        # Test similarity search with the original seed vectors
        for topic, seed in topic_seeds.items():
            similar = ltm_store.get_most_similar(seed, top_k=5)
            
            # The top 3 results should be the variations of this topic
            topic_count = sum(1 for mem, _ in similar[:3] 
                              if mem["metadata"]["topic"] == topic)
            
            # At least 2 of the top 3 should be the correct topic
            assert topic_count >= 2
            
            # The top result should definitely be the correct topic
            assert similar[0][0]["metadata"]["topic"] == topic

    def test_realistic_data_shapes(self, ltm_store):
        """Test with realistic memory data shapes and sizes."""
        # Create memories with varying content sizes and formats
        memories = []
        
        # Text memories of different lengths
        for i, size in enumerate(["small", "medium", "large"]):
            text_length = 10 if size == "small" else 100 if size == "medium" else 1000
            text = f"Text memory {i} " * text_length
            
            memories.append({
                "memory_id": f"text_{size}_{i}",
                "agent_id": "test_agent",
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "text",
                "content": {"text": text},
                "metadata": {"size": size, "length": len(text)}
            })
        
        # Structured data with different shapes
        structured_samples = [
            # Simple key-value pairs
            {"name": "John", "age": 30, "city": "New York"},
            
            # Nested structures
            {
                "user": {
                    "profile": {
                        "name": "Jane",
                        "preferences": {
                            "colors": ["blue", "green"],
                            "foods": ["pizza", "pasta"]
                        }
                    },
                    "history": [
                        {"date": "2023-01-01", "action": "login"},
                        {"date": "2023-01-02", "action": "purchase"}
                    ]
                }
            },
            
            # Array data
            {
                "readings": [1.2, 3.4, 5.6, 7.8] * 10,
                "timestamps": [int(time.time()) + i for i in range(40)]
            }
        ]
        
        for i, data in enumerate(structured_samples):
            memories.append({
                "memory_id": f"structured_{i}",
                "agent_id": "test_agent",
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "structured",
                "content": data,
                "metadata": {"complexity": i}
            })
            
        # Store all memories
        ltm_store.store_batch(memories)
        
        # Verify we can retrieve each memory with its content intact
        for memory in memories:
            retrieved = ltm_store.get(memory["memory_id"])
            assert retrieved is not None
            assert retrieved["memory_id"] == memory["memory_id"]
            assert retrieved["content"] == memory["content"]

    def test_transaction_resilience(self, ltm_store):
        """Test the resilience of SQLite transactions."""
        # Create a batch of memories
        batch = []
        for i in range(10):
            batch.append({
                "memory_id": f"transaction_test_{i}",
                "agent_id": "test_agent",
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "observation",
                "content": {"text": f"Transaction test memory {i}"},
                "metadata": {"position": i}
            })
        
        # Test stores with deliberate invalid entries
        # Add an invalid entry (missing memory_id)
        invalid_batch = batch.copy()
        invalid_batch.append({
            "agent_id": "test_agent",
            "content": {"text": "Invalid memory"}
        })
        
        # The batch should still succeed even with the invalid entry
        # (SQLite transactions still commit valid entries)
        success = ltm_store.store_batch(invalid_batch)
        assert success is False  # Overall operation fails
        
        # Check that valid entries were still stored
        count = ltm_store.count()
        assert count == 10  # All valid entries should be stored
        
        # Verify we can retrieve valid entries
        for i in range(10):
            retrieved = ltm_store.get(f"transaction_test_{i}")
            assert retrieved is not None
            assert retrieved["memory_id"] == f"transaction_test_{i}"
            
    def test_multiple_agent_isolation(self, ltm_config):
        """Test that memories from different agents are isolated."""
        # Create stores for different agents using the same database
        agent1_store = SQLiteLTMStore(agent_id="agent1", config=ltm_config)
        agent2_store = SQLiteLTMStore(agent_id="agent2", config=ltm_config)
        
        # Create and store memories for each agent
        for i in range(5):
            # Agent 1 memory
            agent1_store.store({
                "memory_id": f"agent1_memory_{i}",
                "agent_id": "agent1",
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "observation",
                "content": {"text": f"Agent 1 memory {i}"},
                "metadata": {"agent": "agent1"}
            })
            
            # Agent 2 memory
            agent2_store.store({
                "memory_id": f"agent2_memory_{i}",
                "agent_id": "agent2",
                "step_number": i,
                "timestamp": int(time.time()),
                "type": "observation",
                "content": {"text": f"Agent 2 memory {i}"},
                "metadata": {"agent": "agent2"}
            })
        
        # Verify counts for each agent
        assert agent1_store.count() == 5
        assert agent2_store.count() == 5
        
        # Verify agent1 cannot access agent2's memories
        for i in range(5):
            assert agent1_store.get(f"agent2_memory_{i}") is None
            assert agent2_store.get(f"agent1_memory_{i}") is None
        
        # Retrieve all memories for each agent
        agent1_memories = agent1_store.get_all()
        agent2_memories = agent2_store.get_all()
        
        # Verify each agent only sees their own memories
        assert all(memory["agent_id"] == "agent1" for memory in agent1_memories)
        assert all(memory["agent_id"] == "agent2" for memory in agent2_memories)

    def test_store_update_delete_cycle(self, ltm_store):
        """Test the full lifecycle of creating, updating, and deleting memories."""
        # Generate a unique memory ID
        memory_id = f"lifecycle_test_{uuid.uuid4()}"
        
        # Stage 1: Create
        original_memory = {
            "memory_id": memory_id,
            "agent_id": "test_agent",
            "step_number": 1,
            "timestamp": int(time.time()),
            "type": "note",
            "content": {"text": "Original note content"},
            "metadata": {
                "importance_score": 0.3,
                "status": "draft"
            }
        }
        
        success = ltm_store.store(original_memory)
        assert success is True
        
        # Verify it exists
        retrieved = ltm_store.get(memory_id)
        assert retrieved is not None
        assert retrieved["content"]["text"] == "Original note content"
        assert retrieved["metadata"]["status"] == "draft"
        
        # Stage 2: Update content
        updated_memory = original_memory.copy()
        updated_memory["content"] = {"text": "Updated note content"}
        updated_memory["metadata"] = {
            "importance_score": 0.7,
            "status": "reviewed"
        }
        
        success = ltm_store.store(updated_memory)
        assert success is True
        
        # Verify update took effect
        retrieved = ltm_store.get(memory_id)
        assert retrieved["content"]["text"] == "Updated note content"
        assert retrieved["metadata"]["status"] == "reviewed"
        assert retrieved["metadata"]["importance_score"] == 0.7
        
        # Stage 3: Add embeddings
        embedding_memory = updated_memory.copy()
        embedding_memory["embeddings"] = {
            "compressed_vector": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        success = ltm_store.store(embedding_memory)
        assert success is True
        
        # Verify embeddings are stored
        retrieved = ltm_store.get(memory_id)
        assert "embeddings" in retrieved
        assert "compressed_vector" in retrieved["embeddings"]
        
        # Stage 4: Delete
        success = ltm_store.delete(memory_id)
        assert success is True
        
        # Verify it's gone
        retrieved = ltm_store.get(memory_id)
        assert retrieved is None 