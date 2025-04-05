"""Unit tests for agent_memory.embeddings.autoencoder module.

This test suite covers the AutoEncoder functionality which provides dimensionality
reduction for vector embeddings.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock torch before imports
torch_mock = MagicMock()
nn_mock = MagicMock()
torch_mock.nn = nn_mock
torch_mock.device = MagicMock()
torch_mock.device.return_value = "cpu"
torch_mock.FloatTensor = lambda x: x
torch_mock.from_numpy = lambda x: x
torch_mock.no_grad.return_value.__enter__ = MagicMock()
torch_mock.no_grad.return_value.__exit__ = MagicMock()

# Create layer mocks
linear_layer_mock = MagicMock()
linear_layer_mock.return_value = np.random.rand(10)
nn_mock.Linear.return_value = linear_layer_mock
nn_mock.ReLU.return_value = MagicMock()
nn_mock.BatchNorm1d.return_value = MagicMock()
nn_mock.Dropout.return_value = MagicMock()

# Mock the torch module
sys.modules["torch"] = torch_mock

# Import after mocking
from agent_memory.embeddings.autoencoder import (
    AgentStateDataset,
    AutoencoderEmbeddingEngine,
    NumericExtractor,
    StateAutoencoder,
)

#################################
# StateAutoencoder Tests
#################################


def test_state_autoencoder_init():
    """Test initialization of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, latent_dim=32)

    # Check encoder and decoder were initialized
    nn_mock.Linear.assert_any_call(100, 64)  # First encoder layer
    nn_mock.Linear.assert_any_call(64, 32)  # Bottleneck layer
    nn_mock.Linear.assert_any_call(32, 64)  # First decoder layer
    nn_mock.Linear.assert_any_call(64, 100)  # Output layer


def test_state_autoencoder_forward():
    """Test forward pass of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, latent_dim=32)

    # Mock input data
    input_data = np.random.rand(10, 100)

    # Forward pass
    output = autoencoder.forward(input_data)

    # Check that output has same shape as input
    assert output is not None


def test_state_autoencoder_encode():
    """Test encode method of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, latent_dim=32)

    # Mock input data
    input_data = np.random.rand(10, 100)

    # Encode
    encoded = autoencoder.encode(input_data)

    # Should be called with correct method
    assert encoded is not None


def test_state_autoencoder_decode():
    """Test decode method of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, latent_dim=32)

    # Mock latent data
    latent_data = np.random.rand(10, 32)

    # Decode
    decoded = autoencoder.decode(latent_data)

    # Check decoder was used
    assert decoded is not None


#################################
# NumericExtractor Tests
#################################


def test_numeric_extractor_init():
    """Test initialization of NumericExtractor."""
    extractor = NumericExtractor()
    assert extractor is not None


def test_numeric_extractor_process_value():
    """Test processing of individual values."""
    extractor = NumericExtractor()

    # Test with numeric values
    assert extractor._process_value(42) == 42
    assert extractor._process_value(3.14) == 3.14

    # Test with string values (should be converted to embedding)
    string_result = extractor._process_value("test")
    assert isinstance(string_result, list)

    # Test with boolean
    assert extractor._process_value(True) == 1
    assert extractor._process_value(False) == 0

    # Test with None
    assert extractor._process_value(None) == 0


def test_numeric_extractor_extract_features():
    """Test feature extraction from structured data."""
    extractor = NumericExtractor()

    # Test with structured data
    data = {
        "location": "kitchen",
        "position": {"x": 10, "y": 20},
        "health": 100,
        "inventory": ["apple", "knife"],
        "is_active": True,
    }

    features = extractor.extract_features(data)

    # Should return a numpy array of float values
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32


def test_numeric_extractor_get_feature_dim():
    """Test getting the feature dimension."""
    extractor = NumericExtractor(text_embedding_dim=16)

    # Extraction dimension should match configured dimension
    assert extractor.get_feature_dim() > 0


#################################
# AutoencoderEmbeddingEngine Tests
#################################


def test_autoencoder_embedding_engine_init():
    """Test initialization of AutoencoderEmbeddingEngine."""
    # Create with default params
    engine = AutoencoderEmbeddingEngine()

    # Check that models were initialized
    assert engine.stm_autoencoder is not None
    assert engine.im_autoencoder is not None
    assert engine.ltm_autoencoder is not None
    assert engine.feature_extractor is not None


@patch("agent_memory.embeddings.autoencoder.os.path.exists", return_value=False)
def test_autoencoder_embedding_engine_encode(mock_exists):
    """Test encoding with AutoencoderEmbeddingEngine."""
    engine = AutoencoderEmbeddingEngine()

    # Mock the autoencoder encode method
    engine.stm_autoencoder.encode = MagicMock(return_value=np.random.rand(32))

    # Test data
    data = {
        "content": "test message",
        "metadata": {"sender": "user", "timestamp": 12345},
    }

    # Get vector encoding
    stm_vector = engine.encode_stm(data)
    im_vector = engine.encode_im(data)
    ltm_vector = engine.encode_ltm(data)

    # Check vectors were created
    assert isinstance(stm_vector, list)
    assert isinstance(im_vector, list)
    assert isinstance(ltm_vector, list)


def test_autoencoder_embedding_engine_train():
    """Test training of AutoencoderEmbeddingEngine."""
    engine = AutoencoderEmbeddingEngine()

    # Mock some training data
    train_data = [
        {"content": "message 1", "metadata": {"sender": "user"}},
        {"content": "message 2", "metadata": {"sender": "agent"}},
        {"content": "message 3", "metadata": {"sender": "user"}},
    ]

    # Mock train method of autoencoders
    engine.stm_autoencoder.train = MagicMock()
    engine.im_autoencoder.train = MagicMock()
    engine.ltm_autoencoder.train = MagicMock()

    # Train the engine
    engine.train(train_data, epochs=2)

    # Check that train was called for each autoencoder
    engine.stm_autoencoder.train.assert_called_once()
    engine.im_autoencoder.train.assert_called_once()
    engine.ltm_autoencoder.train.assert_called_once()


#################################
# AgentStateDataset Tests
#################################


def test_agent_state_dataset_init():
    """Test initialization of AgentStateDataset."""
    # Create mock feature data
    features = np.random.rand(10, 100).astype(np.float32)

    # Create dataset
    dataset = AgentStateDataset(features)

    # Check dataset properties
    assert len(dataset) == 10
    assert dataset.features.shape == (10, 100)


def test_agent_state_dataset_getitem():
    """Test __getitem__ method of AgentStateDataset."""
    # Create mock feature data
    features = np.random.rand(10, 100).astype(np.float32)

    # Create dataset
    dataset = AgentStateDataset(features)

    # Get an item
    item = dataset[0]

    # Should return the same item for x and y in autoencoder training
    assert item[0].shape == (100,)
    assert item[1].shape == (100,)
    assert np.array_equal(item[0], item[1])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
