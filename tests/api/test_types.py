"""Unit tests for memory API types."""

import pytest
from typing import Dict, List, Any, Optional
from memory.api.types import (
    MemoryMetadata,
    MemoryEmbeddings,
    MemoryEntry,
    MemoryChangeRecord,
    MemoryTypeDistribution,
    MemoryStatistics,
    SimilaritySearchResult,
    ConfigUpdate,
    QueryResult,
    MemoryStore
)


class TestMemoryTypesStructures:
    """Test suite for memory type structures."""

    def test_memory_metadata_structure(self):
        """Test MemoryMetadata structure with all fields."""
        metadata: MemoryMetadata = {
            "creation_time": 1649879872.123,
            "last_access_time": 1649879900.456,
            "compression_level": 0,
            "importance_score": 0.8,
            "retrieval_count": 5,
            "memory_type": "state",
            "current_tier": "stm",
            "checksum": "abc123"
        }
        
        assert metadata["creation_time"] == 1649879872.123
        assert metadata["last_access_time"] == 1649879900.456
        assert metadata["compression_level"] == 0
        assert metadata["importance_score"] == 0.8
        assert metadata["retrieval_count"] == 5
        assert metadata["memory_type"] == "state"
        assert metadata["current_tier"] == "stm"
        assert metadata["checksum"] == "abc123"

    def test_memory_metadata_partial(self):
        """Test MemoryMetadata with partial fields (total=False)."""
        # This should work because MemoryMetadata has total=False
        metadata: MemoryMetadata = {
            "creation_time": 1649879872.123,
            "importance_score": 0.8,
            "memory_type": "state",
        }
        
        assert metadata["creation_time"] == 1649879872.123
        assert metadata["importance_score"] == 0.8
        assert metadata["memory_type"] == "state"
        assert "compression_level" not in metadata

    def test_memory_embeddings_structure(self):
        """Test MemoryEmbeddings structure with all fields."""
        embeddings: MemoryEmbeddings = {
            "full_vector": [0.1, 0.2, 0.3, 0.4],
            "compressed_vector": [0.15, 0.25, 0.35],
            "abstract_vector": [0.2, 0.3]
        }
        
        assert embeddings["full_vector"] == [0.1, 0.2, 0.3, 0.4]
        assert embeddings["compressed_vector"] == [0.15, 0.25, 0.35]
        assert embeddings["abstract_vector"] == [0.2, 0.3]

    def test_memory_embeddings_partial(self):
        """Test MemoryEmbeddings with partial fields (total=False)."""
        # This should work because MemoryEmbeddings has total=False
        embeddings: MemoryEmbeddings = {
            "full_vector": [0.1, 0.2, 0.3, 0.4],
        }
        
        assert embeddings["full_vector"] == [0.1, 0.2, 0.3, 0.4]
        assert "compressed_vector" not in embeddings
        assert "abstract_vector" not in embeddings

    def test_memory_entry_structure(self):
        """Test MemoryEntry structure with all fields."""
        memory: MemoryEntry = {
            "memory_id": "mem_12345",
            "agent_id": "agent_001",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "contents": {"observation": "User asked about weather"},
            "metadata": {
                "importance_score": 0.8,
                "memory_type": "interaction",
                "creation_time": 1649879872.123,
                "retrieval_count": 0,
                "compression_level": 0,
                "current_tier": "stm",
                "last_access_time": 1649879872.123,
                "checksum": "abc123"
            },
            "embeddings": {
                "full_vector": [0.1, 0.2, 0.3, 0.4]
            }
        }
        
        assert memory["memory_id"] == "mem_12345"
        assert memory["agent_id"] == "agent_001"
        assert memory["step_number"] == 42
        assert memory["timestamp"] == 1649879872.123
        assert memory["contents"]["observation"] == "User asked about weather"
        assert memory["metadata"]["importance_score"] == 0.8
        assert memory["embeddings"]["full_vector"] == [0.1, 0.2, 0.3, 0.4]

    def test_memory_entry_without_embeddings(self):
        """Test MemoryEntry without embeddings (embeddings is Optional)."""
        memory: MemoryEntry = {
            "memory_id": "mem_12345",
            "agent_id": "agent_001",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "contents": {"observation": "User asked about weather"},
            "metadata": {
                "importance_score": 0.8,
                "memory_type": "interaction",
                "creation_time": 1649879872.123,
                "retrieval_count": 0,
                "compression_level": 0,
                "current_tier": "stm",
                "last_access_time": 1649879872.123,
                "checksum": "abc123"
            },
            "embeddings": None
        }
        
        assert memory["memory_id"] == "mem_12345"
        assert memory["embeddings"] is None

    def test_memory_change_record_structure(self):
        """Test MemoryChangeRecord structure."""
        change_record: MemoryChangeRecord = {
            "memory_id": "mem_12345",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "previous_value": 10,
            "new_value": 20
        }
        
        assert change_record["memory_id"] == "mem_12345"
        assert change_record["step_number"] == 42
        assert change_record["timestamp"] == 1649879872.123
        assert change_record["previous_value"] == 10
        assert change_record["new_value"] == 20

    def test_memory_change_record_with_none_previous(self):
        """Test MemoryChangeRecord with None previous value."""
        change_record: MemoryChangeRecord = {
            "memory_id": "mem_12345",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "previous_value": None,
            "new_value": 20
        }
        
        assert change_record["memory_id"] == "mem_12345"
        assert change_record["previous_value"] is None
        assert change_record["new_value"] == 20

    def test_memory_type_distribution_structure(self):
        """Test MemoryTypeDistribution structure with all fields."""
        distribution: MemoryTypeDistribution = {
            "state": 10,
            "action": 20,
            "interaction": 30
        }
        
        assert distribution["state"] == 10
        assert distribution["action"] == 20
        assert distribution["interaction"] == 30

    def test_memory_type_distribution_partial(self):
        """Test MemoryTypeDistribution with partial fields (total=False)."""
        # This should work because MemoryTypeDistribution has total=False
        distribution: MemoryTypeDistribution = {
            "state": 10,
            "action": 20,
        }
        
        assert distribution["state"] == 10
        assert distribution["action"] == 20
        assert "interaction" not in distribution

    def test_memory_statistics_structure(self):
        """Test MemoryStatistics structure."""
        stats: MemoryStatistics = {
            "total_memories": 100,
            "stm_count": 20,
            "im_count": 30,
            "ltm_count": 50,
            "memory_type_distribution": {
                "state": 40,
                "action": 30,
                "interaction": 30
            },
            "last_maintenance_time": 1649879872.123,
            "insert_count_since_maintenance": 5
        }
        
        assert stats["total_memories"] == 100
        assert stats["stm_count"] == 20
        assert stats["im_count"] == 30
        assert stats["ltm_count"] == 50
        assert stats["memory_type_distribution"]["state"] == 40
        assert stats["last_maintenance_time"] == 1649879872.123
        assert stats["insert_count_since_maintenance"] == 5

    def test_memory_statistics_with_none_time(self):
        """Test MemoryStatistics with None for last_maintenance_time."""
        stats: MemoryStatistics = {
            "total_memories": 100,
            "stm_count": 20,
            "im_count": 30,
            "ltm_count": 50,
            "memory_type_distribution": {
                "state": 40,
                "action": 30,
                "interaction": 30
            },
            "last_maintenance_time": None,
            "insert_count_since_maintenance": 5
        }
        
        assert stats["total_memories"] == 100
        assert stats["last_maintenance_time"] is None

    def test_similarity_search_result_structure(self):
        """Test SimilaritySearchResult structure."""
        search_result: SimilaritySearchResult = {
            "memory_id": "mem_12345",
            "agent_id": "agent_001",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "contents": {"observation": "User asked about weather"},
            "metadata": {
                "importance_score": 0.8,
                "memory_type": "interaction",
                "creation_time": 1649879872.123,
                "retrieval_count": 0,
                "compression_level": 0,
                "current_tier": "stm",
                "last_access_time": 1649879872.123,
                "checksum": "abc123"
            },
            "embeddings": {
                "full_vector": [0.1, 0.2, 0.3, 0.4]
            },
            "_similarity_score": 0.95
        }
        
        assert search_result["memory_id"] == "mem_12345"
        assert search_result["contents"]["observation"] == "User asked about weather"
        assert search_result["_similarity_score"] == 0.95

    def test_config_update_structure(self):
        """Test ConfigUpdate structure (Dict[str, Any])."""
        config_update: ConfigUpdate = {
            "stm_config": {"memory_limit": 1000},
            "im_config": {"memory_limit": 5000},
            "enable_memory_hooks": True
        }
        
        assert config_update["stm_config"]["memory_limit"] == 1000
        assert config_update["im_config"]["memory_limit"] == 5000
        assert config_update["enable_memory_hooks"] is True

    def test_query_result_structure(self):
        """Test QueryResult structure."""
        query_result: QueryResult = {
            "memory_id": "mem_12345",
            "agent_id": "agent_001",
            "step_number": 42,
            "timestamp": 1649879872.123,
            "contents": {"observation": "User asked about weather"},
            "metadata": {
                "importance_score": 0.8,
                "memory_type": "interaction",
                "creation_time": 1649879872.123,
                "retrieval_count": 0,
                "compression_level": 0,
                "current_tier": "stm",
                "last_access_time": 1649879872.123,
                "checksum": "abc123"
            }
        }
        
        assert query_result["memory_id"] == "mem_12345"
        assert query_result["agent_id"] == "agent_001"
        assert query_result["step_number"] == 42
        assert query_result["contents"]["observation"] == "User asked about weather"
        assert query_result["metadata"]["importance_score"] == 0.8


class TestMemoryStoreProtocol:
    """Test suite for MemoryStore protocol implementation."""
    
    class MockMemoryStore:
        """Mock implementation of MemoryStore protocol for testing."""
        
        def __init__(self):
            self.memories = {}
            
        def get(self, memory_id: str) -> Optional[MemoryEntry]:
            """Get a memory by ID."""
            return self.memories.get(memory_id)
            
        def get_recent(self, count: int, memory_type: Optional[str] = None) -> List[MemoryEntry]:
            """Get recent memories."""
            return []
            
        def get_by_step_range(
            self, start_step: int, end_step: int, memory_type: Optional[str] = None
        ) -> List[MemoryEntry]:
            """Get memories in a step range."""
            return []
            
        def get_by_attributes(
            self, attributes: Dict[str, Any], memory_type: Optional[str] = None
        ) -> List[MemoryEntry]:
            """Get memories matching attributes."""
            return []
            
        def search_by_vector(
            self, vector: List[float], k: int = 5, memory_type: Optional[str] = None
        ) -> List[MemoryEntry]:
            """Search memories by vector similarity."""
            return []
            
        def search_by_content(
            self, content_query: Dict[str, Any], k: int = 5
        ) -> List[MemoryEntry]:
            """Search memories by content."""
            return []
            
        def contains(self, memory_id: str) -> bool:
            """Check if a memory exists."""
            return memory_id in self.memories
            
        def update(self, memory: MemoryEntry) -> bool:
            """Update a memory."""
            self.memories[memory["memory_id"]] = memory
            return True
            
        def count(self) -> int:
            """Count memories."""
            return len(self.memories)
            
        def count_by_type(self) -> Dict[str, int]:
            """Count memories by type."""
            return {"state": 0, "action": 0, "interaction": 0}
            
        def clear(self) -> bool:
            """Clear all memories."""
            self.memories.clear()
            return True
    
    def test_memory_store_protocol_compliance(self):
        """Test that MockMemoryStore complies with MemoryStore protocol."""
        store = self.MockMemoryStore()
        
        # This will fail if the MockMemoryStore doesn't properly implement MemoryStore
        assert isinstance(store, MemoryStore)
        
        # Test basic functionality
        test_memory: MemoryEntry = {
            "memory_id": "test-mem",
            "agent_id": "test-agent",
            "step_number": 1,
            "timestamp": 1000.0,
            "contents": {"data": "test content"},
            "metadata": {
                "importance_score": 0.5,
                "memory_type": "state",
                "creation_time": 1000.0,
                "compression_level": 0,
                "current_tier": "stm",
                "last_access_time": 1000.0,
                "retrieval_count": 0,
                "checksum": "test"
            },
            "embeddings": None
        }
        
        assert store.update(test_memory) is True
        assert store.contains("test-mem") is True
        assert store.count() == 1
        retrieved = store.get("test-mem")
        assert retrieved is not None
        assert retrieved["memory_id"] == "test-mem"
        assert store.clear() is True
        assert store.count() == 0 