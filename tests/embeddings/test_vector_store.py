"""Unit tests for memory.embeddings.vector_store module.

This test suite covers:
1. RedisVectorIndex - Redis-based vector storage and retrieval
2. Comprehensive VectorStore tests - additional vector store functionality
3. Edge cases for vector operations
"""

import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
from memory.embeddings.vector_store import (
    VectorIndex, 
    InMemoryVectorIndex, 
    RedisVectorIndex,
    VectorStore
)

#################################
# RedisVectorIndex Tests
#################################

@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    client = MagicMock()
    
    # Mock index existence check
    client.execute_command.side_effect = lambda *args, **kwargs: {
        ('FT._LIST',): [], 
        ('FT.CREATE',): 'OK',
        ('FT.SEARCH',): [1, b'index:test1', [b'score', b'1.0', b'metadata', b'{"name":"test1"}', b'timestamp', b'1234567890']],
        ('FT.DROPINDEX',): 'OK'
    }.get(args[:1], 'OK')
    
    # Mock scan
    client.scan.return_value = (0, [b'index:test1', b'index:test2'])
    
    return client

def test_redis_vector_index_init(mock_redis_client):
    """Test initialization of RedisVectorIndex."""
    index = RedisVectorIndex(
        mock_redis_client,
        index_name="test_index",
        dimension=128
    )
    
    assert index.redis is mock_redis_client
    assert index.index_name == "test_index"
    assert index.dimension == 128

def test_redis_vector_index_ensure_index_creates_new(mock_redis_client):
    """Test index creation when it doesn't exist."""
    index = RedisVectorIndex(mock_redis_client, "test_index")
    
    # Should call FT._LIST and then FT.CREATE
    assert mock_redis_client.execute_command.call_count >= 2
    create_args = mock_redis_client.execute_command.call_args_list[1][0]
    assert create_args[0] == "FT.CREATE"
    assert create_args[1] == "test_index"

def test_redis_vector_index_add(mock_redis_client):
    """Test adding a vector to the Redis index."""
    index = RedisVectorIndex(mock_redis_client, "test_index", dimension=3)
    result = index.add("test1", [0.1, 0.2, 0.3], {"name": "test"})
    
    assert result is True
    # Check that hset was called with the correct arguments
    mock_redis_client.hset.assert_called_once()
    args = mock_redis_client.hset.call_args[0]
    assert args[0] == "test_index:test1"
    
    # Check that mapping includes the vector field and metadata
    mapping = mock_redis_client.hset.call_args[1]["mapping"]
    assert "embedding" in mapping
    assert "metadata" in mapping
    assert "timestamp" in mapping
    
    # Verify metadata was serialized
    assert json.loads(mapping["metadata"]) == {"name": "test"}

@patch('struct.pack')
def test_redis_vector_index_float_to_bytes(mock_pack, mock_redis_client):
    """Test conversion of float list to bytes."""
    mock_pack.return_value = b'packed'
    
    index = RedisVectorIndex(mock_redis_client, "test_index", dimension=3)
    result = index._float_list_to_bytes([0.1, 0.2, 0.3])
    
    # Each float should be packed once
    assert mock_pack.call_count == 3
    
    # Shorter vector should be padded
    result = index._float_list_to_bytes([0.1, 0.2])
    assert mock_pack.call_count == 6  # 3 more calls
    
    # Longer vector should be truncated
    result = index._float_list_to_bytes([0.1, 0.2, 0.3, 0.4])
    assert mock_pack.call_count == 9  # 3 more calls

def test_redis_vector_index_search(mock_redis_client):
    """Test searching for similar vectors in the Redis index."""
    index = RedisVectorIndex(mock_redis_client, "test_index", dimension=3)
    results = index.search([0.1, 0.2, 0.3], limit=5)
    
    # Verify search executed correctly
    assert len(results) == 1
    assert results[0]["id"] == "test1"
    assert results[0]["score"] == 1.0
    assert results[0]["metadata"] == {"name": "test1"}
    
    # Check search parameters
    search_args = mock_redis_client.execute_command.call_args[0]
    assert search_args[0] == "FT.SEARCH"
    assert search_args[1] == "test_index"
    
    # Check limit is passed correctly
    assert "LIMIT" in search_args
    limit_index = search_args.index("LIMIT")
    assert search_args[limit_index + 2] == "5"

def test_redis_vector_index_delete(mock_redis_client):
    """Test deleting a vector from the Redis index."""
    index = RedisVectorIndex(mock_redis_client, "test_index")
    result = index.delete("test1")
    
    assert result is True
    mock_redis_client.delete.assert_called_once_with("test_index:test1")

def test_redis_vector_index_clear(mock_redis_client):
    """Test clearing all vectors from the Redis index."""
    index = RedisVectorIndex(mock_redis_client, "test_index")
    result = index.clear()
    
    assert result is True
    # Should scan for keys
    mock_redis_client.scan.assert_called_once()
    # Should delete the found keys
    mock_redis_client.delete.assert_called_once()
    # Should drop and recreate the index
    assert mock_redis_client.execute_command.call_count >= 3


#################################
# Comprehensive VectorStore Tests
#################################

@pytest.fixture
def mock_indices():
    """Create mock vector indices for testing."""
    stm_index = MagicMock(spec=VectorIndex)
    im_index = MagicMock(spec=VectorIndex)
    ltm_index = MagicMock(spec=VectorIndex)
    
    # Set up return values
    for index in [stm_index, im_index, ltm_index]:
        index.add.return_value = True
        index.search.return_value = [{"id": "test1", "score": 0.9, "metadata": {"type": "test"}}]
        index.delete.return_value = True
        index.clear.return_value = True
    
    return stm_index, im_index, ltm_index

def test_vector_store_init_with_redis(mock_redis_client):
    """Test initializing VectorStore with Redis."""
    with patch('memory.embeddings.vector_store.RedisVectorIndex') as mock_redis_index:
        store = VectorStore(redis_client=mock_redis_client, namespace="test")
        
        # Should create three Redis indices
        assert mock_redis_index.call_count == 3
        
        # Check index names include namespace
        assert mock_redis_index.call_args_list[0][0][1] == "test:stm_vectors"
        assert mock_redis_index.call_args_list[1][0][1] == "test:im_vectors"
        assert mock_redis_index.call_args_list[2][0][1] == "test:ltm_vectors"
        
        # Check dimensions
        assert mock_redis_index.call_args_list[0][1]["dimension"] == 384
        assert mock_redis_index.call_args_list[1][1]["dimension"] == 128
        assert mock_redis_index.call_args_list[2][1]["dimension"] == 32

def test_vector_store_init_in_memory():
    """Test initializing VectorStore with in-memory indices."""
    store = VectorStore(namespace="test")
    
    assert isinstance(store.stm_index, InMemoryVectorIndex)
    assert isinstance(store.im_index, InMemoryVectorIndex)
    assert isinstance(store.ltm_index, InMemoryVectorIndex)

def test_vector_store_store_memory_vectors():
    """Test storing memory vectors with different embedding types."""
    store = VectorStore(namespace="test")
    
    # Replace indices with mocks
    store.stm_index = MagicMock(spec=VectorIndex)
    store.im_index = MagicMock(spec=VectorIndex)
    store.ltm_index = MagicMock(spec=VectorIndex)
    store.stm_index.add.return_value = True
    store.im_index.add.return_value = True
    store.ltm_index.add.return_value = True
    
    # Test memory entry with all vector types
    memory_entry = {
        "memory_id": "test1",
        "embeddings": {
            "full_vector": [0.1] * 384,
            "compressed_vector": [0.2] * 128,
            "abstract_vector": [0.3] * 32
        },
        "metadata": {"type": "test"}
    }
    
    result = store.store_memory_vectors(memory_entry)
    
    assert result is True
    store.stm_index.add.assert_called_once_with("test1", [0.1] * 384, {"type": "test"})
    store.im_index.add.assert_called_once_with("test1", [0.2] * 128, {"type": "test"})
    store.ltm_index.add.assert_called_once_with("test1", [0.3] * 32, {"type": "test"})
    
    # Test with partial vectors
    store.stm_index.reset_mock()
    store.im_index.reset_mock()
    store.ltm_index.reset_mock()
    
    partial_memory = {
        "memory_id": "test2",
        "embeddings": {
            "full_vector": [0.1] * 384,
            # Missing compressed_vector
            "abstract_vector": [0.3] * 32
        },
        "metadata": {"type": "test"}
    }
    
    store.store_memory_vectors(partial_memory)
    
    store.stm_index.add.assert_called_once()
    store.im_index.add.assert_not_called()  # Should skip missing vector
    store.ltm_index.add.assert_called_once()

def test_vector_store_find_similar_memories(mock_indices):
    """Test finding similar memories in different tiers."""
    stm_index, im_index, ltm_index = mock_indices
    
    store = VectorStore()
    store.stm_index = stm_index
    store.im_index = im_index
    store.ltm_index = ltm_index
    
    # Query vector
    query = [0.1] * 384
    
    # Test STM tier
    results = store.find_similar_memories(query, tier="stm", limit=5)
    stm_index.search.assert_called_once()
    assert stm_index.search.call_args[0][0] == query
    assert stm_index.search.call_args[0][1] == 5
    
    # Test IM tier
    store.find_similar_memories(query, tier="im", limit=10)
    im_index.search.assert_called_once()
    assert im_index.search.call_args[0][1] == 10
    
    # Test LTM tier
    store.find_similar_memories(query, tier="ltm")
    ltm_index.search.assert_called_once()

def test_vector_store_find_with_metadata_filter(mock_indices):
    """Test finding memories with metadata filtering."""
    stm_index, _, _ = mock_indices
    
    store = VectorStore()
    store.stm_index = stm_index
    
    query = [0.1] * 384
    metadata_filter = {"category": "important"}
    
    store.find_similar_memories(query, metadata_filter=metadata_filter)
    
    # Should pass a filter function
    assert stm_index.search.call_args[0][2] is not None
    filter_fn = stm_index.search.call_args[0][2]
    
    # Test the filter function
    assert filter_fn({"category": "important"}) is True
    assert filter_fn({"category": "normal"}) is False
    assert filter_fn({"other": "value"}) is False

def test_vector_store_delete_memory_vectors(mock_indices):
    """Test deleting memory vectors from all indices."""
    stm_index, im_index, ltm_index = mock_indices
    
    store = VectorStore()
    store.stm_index = stm_index
    store.im_index = im_index
    store.ltm_index = ltm_index
    
    result = store.delete_memory_vectors("test1")
    
    assert result is True
    stm_index.delete.assert_called_once_with("test1")
    im_index.delete.assert_called_once_with("test1")
    ltm_index.delete.assert_called_once_with("test1")

def test_vector_store_clear_all(mock_indices):
    """Test clearing all vectors from all indices."""
    stm_index, im_index, ltm_index = mock_indices
    
    store = VectorStore()
    store.stm_index = stm_index
    store.im_index = im_index
    store.ltm_index = ltm_index
    
    result = store.clear_all()
    
    assert result is True
    stm_index.clear.assert_called_once()
    im_index.clear.assert_called_once()
    ltm_index.clear.assert_called_once()

#################################
# Edge Cases and Error Handling
#################################

def test_vector_store_empty_memory_id():
    """Test handling of memory entries with missing IDs."""
    store = VectorStore()
    
    # Memory without ID
    bad_memory = {
        "embeddings": {"full_vector": [0.1] * 384},
        "metadata": {"type": "test"}
    }
    
    result = store.store_memory_vectors(bad_memory)
    assert result is False

def test_redis_vector_index_add_error(mock_redis_client):
    """Test error handling when adding vectors fails."""
    index = RedisVectorIndex(mock_redis_client, "test_index")
    
    # Make hset raise an exception
    mock_redis_client.hset.side_effect = Exception("Test error")
    
    result = index.add("test1", [0.1, 0.2, 0.3])
    assert result is False

def test_redis_vector_index_search_error(mock_redis_client):
    """Test error handling when search fails."""
    index = RedisVectorIndex(mock_redis_client, "test_index")
    
    # Make execute_command raise an exception during search
    def side_effect(*args, **kwargs):
        if args[0] == "FT.SEARCH":
            raise Exception("Search error")
        return []
    
    mock_redis_client.execute_command.side_effect = side_effect
    
    results = index.search([0.1, 0.2, 0.3])
    assert results == []

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 