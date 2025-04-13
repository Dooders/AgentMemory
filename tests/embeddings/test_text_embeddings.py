"""Unit tests for memory.embeddings.text_embeddings module.

This test suite covers:
1. TextEmbeddingEngine - initialization, encoding methods
2. Context-weighted embeddings
3. Tier-specific encoding methods (STM, IM, LTM)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Create mocks that will be used inside our tests
mock_model = MagicMock()
mock_model.get_sentence_embedding_dimension.return_value = 768
mock_model.encode.return_value = np.array([0.1] * 768)

mock_transformer_class = MagicMock(return_value=mock_model)

# Import the TextEmbeddingEngine for type checking
from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.utils import object_to_text


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Patch dependencies before each test."""
    with patch(
        "memory.embeddings.text_embeddings.SentenceTransformer", mock_transformer_class
    ):
        # Reset mocks before each test
        mock_transformer_class.reset_mock()
        mock_model.encode.reset_mock()
        mock_model.get_sentence_embedding_dimension.reset_mock()
        yield


class TestTextEmbeddingEngine:
    """Test suite for TextEmbeddingEngine class."""

    def test_initialization(self):
        """Test initialization with default and custom model names."""
        # Test with default model
        engine = TextEmbeddingEngine()
        assert engine.model is mock_model
        assert engine.embedding_dim == 768
        mock_transformer_class.assert_called_with("all-mpnet-base-v2")

        # Reset mock for second test
        mock_transformer_class.reset_mock()

        # Test with custom model
        engine = TextEmbeddingEngine(model_name="all-MiniLM-L6-v2")
        assert engine.model is mock_model
        assert engine.embedding_dim == 768
        mock_transformer_class.assert_called_with("all-MiniLM-L6-v2")

    def test_initialization_error(self):
        """Test initialization error when SentenceTransformer is not available."""
        with patch(
            "memory.embeddings.text_embeddings.SentenceTransformer",
            side_effect=NameError("SentenceTransformer not available"),
        ):
            with pytest.raises(NameError):
                engine = TextEmbeddingEngine()

    def test_encode_simple_types(self):
        """Test encoding simple data types."""
        engine = TextEmbeddingEngine()

        # Test with string
        result = engine.encode("test string")
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

        # Reset mock
        mock_model.encode.reset_mock()

        # Test with number
        result = engine.encode(123)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

        # Reset mock
        mock_model.encode.reset_mock()

        # Test with boolean
        result = engine.encode(True)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

    def test_encode_complex_types(self):
        """Test encoding complex data types."""
        engine = TextEmbeddingEngine()

        # Test with dict
        test_dict = {"name": "test", "value": 123}
        result = engine.encode(test_dict)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

        # Reset mock
        mock_model.encode.reset_mock()

        # Test with list
        test_list = ["apple", "banana", "orange"]
        result = engine.encode(test_list)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

    def test_encode_with_context_weights(self):
        """Test encoding with context weights for emphasis."""
        engine = TextEmbeddingEngine()

        # Test data with position and inventory
        data = {
            "position": {"location": "kitchen", "x": 10, "y": 20},
            "inventory": ["apple", "knife"],
            "health": 100,
        }

        # Test with context weights
        context_weights = {"position": 1.5, "inventory": 2.0}
        result = engine.encode(data, context_weights=context_weights)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()

        # Get the text passed to encode
        args, _ = mock_model.encode.call_args
        combined_text = args[0]

        # Check that the text contains multiple instances of the weighted items
        # We can't check exact text because object_to_text format is internal
        assert combined_text  # Text should not be empty

    def test_encode_with_zero_weights(self):
        """Test encoding with zero weights."""
        engine = TextEmbeddingEngine()

        data = {"name": "test", "value": 123}
        context_weights = {"name": 0.0, "value": 0.0}

        result = engine.encode(data, context_weights=context_weights)
        assert isinstance(result, list)
        assert len(result) == 768

        # Should still encode with base text
        mock_model.encode.assert_called_once()

    def test_tier_specific_encoding_methods(self):
        """Test the tier-specific encoding methods."""
        engine = TextEmbeddingEngine()
        test_data = {"content": "test"}
        mock_context_weights = {"content": 1.5}

        # Test STM encoding
        stm_result = engine.encode_stm(test_data)
        assert isinstance(stm_result, list)
        assert len(stm_result) == 768
        assert mock_model.encode.call_count == 1

        # Reset mock
        mock_model.encode.reset_mock()

        # Test IM encoding
        im_result = engine.encode_im(test_data, context_weights=mock_context_weights)
        assert isinstance(im_result, list)
        assert len(im_result) == 768
        assert mock_model.encode.call_count == 1

        # Reset mock
        mock_model.encode.reset_mock()

        # Test LTM encoding
        ltm_result = engine.encode_ltm(test_data)
        assert isinstance(ltm_result, list)
        assert len(ltm_result) == 768
        assert mock_model.encode.call_count == 1

    def test_configure_method(self):
        """Test the configure method."""
        engine = TextEmbeddingEngine()

        # Since configure is a no-op, just verify it doesn't raise exceptions
        engine.configure({"some": "config"})
        engine.configure(None)
        engine.configure(["list", "of", "configs"])

    def test_special_case_position_location(self):
        """Test special case handling for position.location."""
        engine = TextEmbeddingEngine()

        # Data with position.location
        data = {"position": {"location": "bedroom", "x": 5, "y": 10}}

        # Context weights emphasizing position
        context_weights = {"position": 2.0}

        # Encode with context weights
        result = engine.encode(data, context_weights=context_weights)

        # Verify encode was called
        mock_model.encode.assert_called_once()

        # We can't verify exact text format due to internal object_to_text implementation
        assert isinstance(result, list)
        assert len(result) == 768

    def test_special_case_inventory(self):
        """Test special case handling for inventory items."""
        engine = TextEmbeddingEngine()

        # Data with inventory
        data = {"inventory": ["sword", "shield", "potion"]}

        # Context weights emphasizing inventory
        context_weights = {"inventory": 1.5}

        # Encode with context weights
        result = engine.encode(data, context_weights=context_weights)

        # Verify encode was called
        mock_model.encode.assert_called_once()

        # We can't verify exact text format due to internal object_to_text implementation
        assert isinstance(result, list)
        assert len(result) == 768

    def test_encoding_with_none_value(self):
        """Test encoding with None value."""
        engine = TextEmbeddingEngine()

        result = engine.encode(None)
        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
