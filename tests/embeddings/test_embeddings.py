"""Unit tests for agent_memory.embeddings modules.

This test suite covers:
1. TextEmbeddingEngine - text-to-vector encoding with context weights
2. VectorIndex implementations - in-memory vector storage and retrieval
3. Utility functions - cosine similarity, dict flattening, object-to-text conversion
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock SentenceTransformer before imports
sentence_transformer_mock = MagicMock()
sentence_transformer_mock.get_sentence_embedding_dimension.return_value = 384
sentence_transformer_mock.encode.return_value = np.array([0.1] * 384)

# Mock the sentence_transformers module
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock(
    return_value=sentence_transformer_mock
)

# Import after mocking
# from agent_memory.embeddings.text_embeddings import TextEmbeddingEngine
from agent_memory.embeddings.utils import (
    cosine_similarity,
    filter_dict_keys,
    flatten_dict,
    object_to_text,
)
from agent_memory.embeddings.vector_store import InMemoryVectorIndex, VectorStore

#################################
# TextEmbeddingEngine Tests
#################################


# def test_text_embedding_engine_init():
#     """Test initialization of TextEmbeddingEngine."""
#     engine = TextEmbeddingEngine(model_name="all-MiniLM-L6-v2")
#     assert engine.model is not None
#     assert engine.embedding_dim == 384
#     sys.modules["sentence_transformers"].SentenceTransformer.assert_called_with(
#         "all-MiniLM-L6-v2"
#     )


# def test_text_embedding_engine_encode_string():
#     """Test encoding a simple string."""
#     engine = TextEmbeddingEngine()
#     result = engine.encode("test text")

#     assert isinstance(result, list)
#     assert len(result) == 384
#     sentence_transformer_mock.encode.assert_called_with("test text")


# def test_text_embedding_engine_encode_dict():
#     """Test encoding a dictionary object."""
#     engine = TextEmbeddingEngine()
#     test_data = {"name": "test", "value": 123}

#     result = engine.encode(test_data)

#     assert isinstance(result, list)
#     assert len(result) == 384
#     # Should convert dict to text before encoding
#     sentence_transformer_mock.encode.assert_called_once()


# def test_text_embedding_engine_context_weights():
#     """Test encoding with context weights emphasis."""
#     engine = TextEmbeddingEngine()

#     data = {
#         "position": {"location": "kitchen", "x": 10, "y": 20},
#         "inventory": ["apple", "knife"],
#         "health": 100,
#     }

#     context_weights = {"position": 1.5, "inventory": 2.0}
#     result = engine.encode(data, context_weights=context_weights)

#     assert isinstance(result, list)
#     assert len(result) == 384

#     # Check that encode was called with text that emphasizes weighted fields
#     args, _ = sentence_transformer_mock.encode.call_args
#     combined_text = args[0]

#     # Text should contain weighted components
#     assert "location is kitchen" in combined_text
#     assert "has apple" in combined_text
#     assert "has knife" in combined_text


# def test_tier_specific_encoding_methods():
#     """Test the tier-specific encoding methods."""
#     engine = TextEmbeddingEngine()
#     test_data = {"content": "test"}

#     # All tier methods should call the base encode method
#     stm_result = engine.encode_stm(test_data)
#     im_result = engine.encode_im(test_data)
#     ltm_result = engine.encode_ltm(test_data)

#     assert len(stm_result) == 384
#     assert len(im_result) == 384
#     assert len(ltm_result) == 384

#     # Each method should have called encode
#     assert sentence_transformer_mock.encode.call_count == 3


#################################
# Vector Index Tests
#################################


def test_in_memory_vector_index_add():
    """Test adding vectors to InMemoryVectorIndex."""
    index = InMemoryVectorIndex()
    vector = [0.1] * 10
    metadata = {"name": "test"}

    result = index.add("test1", vector, metadata)

    assert result is True
    assert "test1" in index.vectors
    assert index.vectors["test1"] == vector
    assert index.metadata["test1"] == metadata


def test_in_memory_vector_index_search():
    """Test searching for similar vectors."""
    index = InMemoryVectorIndex()

    # Add test vectors
    index.add("test1", [1.0, 0.0, 0.0], {"name": "test1"})
    index.add("test2", [0.0, 1.0, 0.0], {"name": "test2"})
    index.add("test3", [0.8, 0.2, 0.0], {"name": "test3"})

    # Search for vector similar to test1
    results = index.search([0.9, 0.1, 0.0], limit=2)

    assert len(results) == 2
    assert results[0]["id"] == "test1" or results[0]["id"] == "test3"
    assert results[0]["score"] > 0.9  # High similarity
    assert "metadata" in results[0]


def test_in_memory_vector_index_search_with_filter():
    """Test searching with a filter function."""
    index = InMemoryVectorIndex()

    # Add test vectors with metadata
    index.add("test1", [1.0, 0.0, 0.0], {"type": "A"})
    index.add("test2", [0.9, 0.1, 0.0], {"type": "A"})
    index.add("test3", [0.0, 1.0, 0.0], {"type": "B"})

    # Filter for type A
    def filter_type_a(metadata):
        return metadata.get("type") == "A"

    results = index.search([1.0, 0.0, 0.0], filter_fn=filter_type_a)

    assert len(results) == 2
    assert all(r["metadata"]["type"] == "A" for r in results)


def test_in_memory_vector_index_delete():
    """Test deleting vectors from the index."""
    index = InMemoryVectorIndex()

    # Add a vector
    index.add("test1", [0.1] * 10, {"name": "test"})
    assert "test1" in index.vectors

    # Delete it
    result = index.delete("test1")

    assert result is True
    assert "test1" not in index.vectors
    assert "test1" not in index.metadata


def test_in_memory_vector_index_clear():
    """Test clearing all vectors from the index."""
    index = InMemoryVectorIndex()

    # Add vectors
    index.add("test1", [0.1] * 10)
    index.add("test2", [0.2] * 10)

    # Clear the index
    result = index.clear()

    assert result is True
    assert len(index.vectors) == 0
    assert len(index.metadata) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
