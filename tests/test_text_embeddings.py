# import pytest
# import sys
# import os
# from unittest.mock import patch, MagicMock
# import numpy as np

# # Add the project root to the Python path so we can import agent_memory
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Create a complete mock for SentenceTransformer before importing anything
# sentence_transformer_mock = MagicMock()
# sentence_transformer_mock.get_sentence_embedding_dimension.return_value = 384
# # Return a NumPy array instead of a list to support .tolist() method
# sentence_transformer_mock.encode.return_value = np.array([0.1] * 384)

# # Create the module mocks
# sys.modules['sentence_transformers'] = MagicMock()
# sys.modules['sentence_transformers'].SentenceTransformer = MagicMock(return_value=sentence_transformer_mock)

# # Now we can safely import the TextEmbeddingEngine
# from memory.embeddings.text_embeddings import TextEmbeddingEngine

# def test_text_embedding_engine_init():
#     """Test that TextEmbeddingEngine initializes correctly."""
#     try:
#         engine = TextEmbeddingEngine(model_name="all-MiniLM-L6-v2")
#         assert engine.model is not None
#         assert engine.embedding_dim > 0
#         print(f"Successfully initialized with dimension {engine.embedding_dim}")
#     except Exception as e:
#         pytest.fail(f"TextEmbeddingEngine initialization failed: {e}\n{type(e)}")

# def test_text_embedding_engine_encode():
#     """Test encoding functionality."""
#     try:
#         # Initialize with the mock
#         engine = TextEmbeddingEngine()
        
#         # Test encoding
#         result = engine.encode("test text")
#         assert len(result) == 384
#         engine.model.encode.assert_called_once()
#     except Exception as e:
#         pytest.fail(f"TextEmbeddingEngine encoding failed: {e}")

# def test_text_embedding_object_to_text():
#     """Test the _object_to_text method with various inputs."""
#     engine = TextEmbeddingEngine()
    
#     # Test with string
#     assert "test" in engine._object_to_text("test")
    
#     # Test with dict
#     obj = {"position": {"location": "kitchen", "x": 10.5, "y": 20.3}}
#     text = engine._object_to_text(obj)
#     assert "location is kitchen" in text
#     assert "coordinates" in text
    
#     # Test with inventory
#     obj = {"inventory": ["apple", "knife"]}
#     text = engine._object_to_text(obj)
#     assert "has apple, knife" in text

# def test_text_embedding_context_weights():
#     """Test encoding with context weights."""
#     engine = TextEmbeddingEngine()
    
#     data = {
#         "position": {"location": "kitchen", "x": 10, "y": 20},
#         "inventory": ["apple", "knife"],
#         "health": 100
#     }
    
#     # Test with context weights
#     context_weights = {"position": 1.5, "inventory": 2.0}
#     result = engine.encode(data, context_weights=context_weights)
#     assert len(result) == 384
    
#     # Check that encode was called with a longer text that includes repeated elements
#     args, kwargs = engine.model.encode.call_args
#     combined_text = args[0]
#     assert "location is kitchen" in combined_text
#     assert "has apple" in combined_text
#     assert "has knife" in combined_text

# if __name__ == "__main__":
#     pytest.main(["-xvs", __file__]) 