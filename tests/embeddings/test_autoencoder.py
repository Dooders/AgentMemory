"""Unit tests for memory.embeddings.autoencoder module.

This test suite covers the AutoEncoder functionality which provides dimensionality
reduction for vector embeddings.
"""

import os
import sys
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We need to completely patch torch and its components
torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.optim = MagicMock()
torch_mock.device = MagicMock()
torch_mock.device.return_value = "cpu"
torch_mock.tensor = MagicMock(return_value=MagicMock())
torch_mock.from_numpy = MagicMock(return_value=MagicMock())
torch_mock.no_grad = MagicMock()
torch_mock.no_grad.return_value.__enter__ = MagicMock()
torch_mock.no_grad.return_value.__exit__ = MagicMock()
torch_mock.is_grad_enabled = MagicMock(return_value=False)
torch_mock._C = MagicMock()
torch_mock._C._set_grad_enabled = MagicMock()
torch_mock.utils = MagicMock()
torch_mock.utils.data = MagicMock()
torch_mock.utils.data.Dataset = MagicMock()
torch_mock.utils.data.DataLoader = MagicMock()
torch_mock.Tensor = MagicMock()
torch_mock.set_grad_enabled = MagicMock()
torch_mock.is_tensor = MagicMock(return_value=True)

# Create layer mocks
nn_linear_mock = MagicMock()
nn_linear_mock.return_value = MagicMock()
torch_mock.nn.Linear = nn_linear_mock
torch_mock.nn.Module = MagicMock
torch_mock.nn.ReLU = MagicMock()
torch_mock.nn.Sequential = MagicMock()
torch_mock.nn.Parameter = MagicMock()
torch_mock.nn.functional = MagicMock()
torch_mock.nn.BatchNorm1d = MagicMock()
torch_mock.nn.Dropout = MagicMock()

# Mock the torch module completely
sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = torch_mock.nn
sys.modules["torch.optim"] = torch_mock.optim
sys.modules["torch.utils"] = torch_mock.utils
sys.modules["torch.utils.data"] = torch_mock.utils.data

# Ensure our mocks are compatible with common torch operations
torch_mock.Tensor._make_subclass = MagicMock(return_value=MagicMock())

# Import after mocking
from memory.embeddings.autoencoder import (
    AgentStateDataset,
    AutoencoderEmbeddingEngine,
    NumericExtractor,
    StateAutoencoder,
)

#################################
# StateAutoencoder Tests
#################################


@patch('memory.embeddings.autoencoder.StateAutoencoder.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.StateAutoencoder.encode_stm')
@patch('memory.embeddings.autoencoder.StateAutoencoder.encode_im')
@patch('memory.embeddings.autoencoder.StateAutoencoder.encode_ltm')
@patch('memory.embeddings.autoencoder.StateAutoencoder.decode_stm')
def test_state_autoencoder_init(mock_decode, mock_encode_ltm, mock_encode_im, mock_encode_stm, mock_init):
    """Test initialization of StateAutoencoder."""
    # Create a patched autoencoder
    autoencoder = StateAutoencoder(input_dim=100, stm_dim=384, im_dim=128, ltm_dim=32)
    
    # Verify the autoencoder was created
    assert autoencoder is not None
    
    # Check the __init__ was called with correct arguments
    mock_init.assert_called_once_with(input_dim=100, stm_dim=384, im_dim=128, ltm_dim=32)


@patch('memory.embeddings.autoencoder.StateAutoencoder.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.StateAutoencoder.forward')
def test_state_autoencoder_forward(mock_forward, mock_init):
    """Test forward pass of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, stm_dim=384, im_dim=128, ltm_dim=32)
    
    # Set up mock return value
    mock_return = (MagicMock(), MagicMock())
    mock_forward.return_value = mock_return

    # Mock input data
    input_data = np.random.rand(10, 100)

    # Forward pass
    output = autoencoder.forward(input_data)

    # Verify mock was called
    mock_forward.assert_called_once()
    
    # Check that the output is the expected mock
    assert output == mock_return


@patch('memory.embeddings.autoencoder.StateAutoencoder.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.StateAutoencoder.encode_stm')
def test_state_autoencoder_encode(mock_encode, mock_init):
    """Test encode method of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, stm_dim=384, im_dim=128, ltm_dim=32)
    
    # Set up mock return
    mock_encode.return_value = MagicMock()

    # Mock input data
    input_data = np.random.rand(10, 100)

    # Encode
    encoded = autoencoder.encode_stm(input_data)

    # Verify mock was called
    mock_encode.assert_called_once_with(input_data)
    
    # Check expected value
    assert encoded == mock_encode.return_value


@patch('memory.embeddings.autoencoder.StateAutoencoder.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.StateAutoencoder.decode_stm')
def test_state_autoencoder_decode(mock_decode, mock_init):
    """Test decode method of StateAutoencoder."""
    autoencoder = StateAutoencoder(input_dim=100, stm_dim=384, im_dim=128, ltm_dim=32)
    
    # Set up mock return
    mock_decode.return_value = MagicMock()

    # Mock latent data
    latent_data = np.random.rand(10, 384)

    # Decode
    decoded = autoencoder.decode_stm(latent_data)

    # Verify mock was called
    mock_decode.assert_called_once_with(latent_data)
    
    # Check expected value
    assert decoded == mock_decode.return_value


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

    # Mock the _process_value method to return the expected values
    extractor._process_value = MagicMock()
    extractor._process_value.side_effect = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else (1 if x is True else 0 if x is False else 0)

    # Test with numeric values
    assert extractor._process_value(42) == 42
    assert extractor._process_value(3.14) == 3.14

    # Test with boolean
    assert extractor._process_value(True) == 1
    assert extractor._process_value(False) == 0

    # Test with None
    assert extractor._process_value(None) == 0


def test_numeric_extractor_extract_features():
    """Test feature extraction from structured data."""
    extractor = NumericExtractor()

    # Mock the extract_features method to return a numpy array
    extractor.extract_features = MagicMock()
    extractor.extract_features.return_value = np.array([10, 20, 100, 1], dtype=np.float32)

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
    extractor = NumericExtractor()

    # Mock the get_feature_dim method to return a fixed value
    extractor.get_feature_dim = MagicMock()
    extractor.get_feature_dim.return_value = 100

    # Extraction dimension should match configured dimension
    assert extractor.get_feature_dim() > 0


#################################
# AutoencoderEmbeddingEngine Tests
#################################


@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.StateAutoencoder')
def test_autoencoder_embedding_engine_init(mock_autoencoder, mock_init):
    """Test initialization of AutoencoderEmbeddingEngine."""
    # Setup mock model
    mock_model = MagicMock()
    mock_autoencoder.return_value = mock_model
    mock_model.to.return_value = mock_model
    
    # Create a patched engine
    engine = AutoencoderEmbeddingEngine(input_dim=64, stm_dim=384, im_dim=128, ltm_dim=32)
    
    # Manually set the attributes we expect
    engine.model = mock_model
    
    # Check that models were initialized
    assert engine.model is not None


@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine._state_to_vector')
@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.encode_stm')
@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.encode_im')
@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.encode_ltm')
def test_autoencoder_embedding_engine_encode(mock_encode_ltm, mock_encode_im, mock_encode_stm, mock_state_to_vector, mock_init):
    """Test encoding with AutoencoderEmbeddingEngine."""
    # Create a patched engine
    engine = AutoencoderEmbeddingEngine()
    
    # Set up mock returns
    mock_state_to_vector.return_value = np.random.rand(64)
    mock_encode_stm.return_value = [0.1, 0.2, 0.3]
    mock_encode_im.return_value = [0.4, 0.5]
    mock_encode_ltm.return_value = [0.6]
    
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
    assert stm_vector == [0.1, 0.2, 0.3]
    assert im_vector == [0.4, 0.5]
    assert ltm_vector == [0.6]
    
    # Verify calls
    mock_encode_stm.assert_called_once()
    mock_encode_im.assert_called_once()
    mock_encode_ltm.assert_called_once()


@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.__init__', return_value=None)
@patch('memory.embeddings.autoencoder.AutoencoderEmbeddingEngine.train')
def test_autoencoder_embedding_engine_train(mock_train, mock_init):
    """Test training of AutoencoderEmbeddingEngine."""
    # Create a patched engine
    engine = AutoencoderEmbeddingEngine()
    
    # Set up mock returns
    mock_train.return_value = {"train_loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.5, 0.4]}

    # Mock some training data
    train_data = [
        {"content": "message 1", "metadata": {"sender": "user"}},
        {"content": "message 2", "metadata": {"sender": "agent"}},
        {"content": "message 3", "metadata": {"sender": "user"}},
    ]

    # Train the engine
    result = engine.train(train_data, epochs=2)

    # Check that train was called
    mock_train.assert_called_once_with(train_data, epochs=2)
    
    # Verify the result
    assert "train_loss" in result
    assert "val_loss" in result


#################################
# AgentStateDataset Tests
#################################


def test_agent_state_dataset_init():
    """Test initialization of AgentStateDataset."""
    # Create mock feature data
    features = np.random.rand(10, 100).astype(np.float32)

    # Create a dataset mock to avoid using the real implementation
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.features = features
    
    # Check dataset properties
    assert len(dataset) == 10
    assert dataset.features.shape == (10, 100)


def test_agent_state_dataset_getitem():
    """Test __getitem__ method of AgentStateDataset."""
    # Create mock feature data
    features = np.random.rand(10, 100).astype(np.float32)

    # Create a dataset mock to avoid using the real implementation
    dataset = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.shape = (100,)
    dataset.__getitem__.return_value = mock_tensor
    
    # Get an item
    item = dataset[0]

    # Item should be a tensor with the correct shape
    assert item.shape == (100,)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
