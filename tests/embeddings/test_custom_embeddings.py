# """Unit tests for the custom embedding model.

# This test suite covers:
# 1. CustomEmbeddingEngine initialization
# 2. Tokenizer functionality
# 3. Encoding with the custom model
# 4. Basic knowledge distillation training
# """

# import os
# import sys
# import tempfile
# import numpy as np
# import pytest
# import torch

# # Add the project root to the Python path if needed
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from memory.embeddings.custom_embeddings import (
#     CustomEmbeddingEngine, 
#     Tokenizer, 
#     EmbeddingModel,
#     EmbeddingDataset
# )
# from memory.embeddings.utils import object_to_text


# def test_tokenizer_initialization():
#     """Test basic tokenizer initialization."""
#     tokenizer = Tokenizer(vocab_size=1000)
    
#     # Check special tokens
#     assert tokenizer.pad_token == "[PAD]"
#     assert tokenizer.unk_token == "[UNK]"
#     assert tokenizer.word_to_id[tokenizer.pad_token] == 0
#     assert tokenizer.word_to_id[tokenizer.unk_token] == 1


# def test_tokenizer_fit_and_encode():
#     """Test tokenizer vocabulary building and encoding."""
#     tokenizer = Tokenizer(vocab_size=100)
    
#     # Sample texts
#     texts = [
#         "agent is at position x=10 y=20",
#         "agent has apple in inventory",
#         "agent moves to position x=15 y=25",
#         "agent picks up banana"
#     ]
    
#     # Build vocabulary
#     tokenizer.fit(texts)
    
#     # Check vocabulary size (should be less than max plus special tokens)
#     assert len(tokenizer.word_to_id) <= 100
#     assert len(tokenizer.word_to_id) > 2  # More than just special tokens
    
#     # Test encoding
#     encoded = tokenizer.encode("agent has apple")
    
#     # Check output format
#     assert "token_ids" in encoded
#     assert "length" in encoded
#     assert isinstance(encoded["token_ids"], torch.Tensor)
#     assert isinstance(encoded["length"], torch.Tensor)
#     assert encoded["length"].item() == 3  # "agent", "has", "apple"


# def test_embedding_model():
#     """Test the embedding model architecture."""
#     model = EmbeddingModel(
#         vocab_size=100,
#         embedding_dim=64,
#         hidden_dim=32,
#         num_layers=1
#     )
    
#     # Test forward pass
#     batch_size = 2
#     seq_len = 10
#     token_ids = torch.randint(0, 100, (batch_size, seq_len))
#     lengths = torch.tensor([8, 6])  # Different lengths
    
#     # Forward pass
#     embeddings = model(token_ids, lengths)
    
#     # Check output shape
#     assert embeddings.shape == (batch_size, 64)
    
#     # Check normalization (vectors should have unit norm)
#     norms = torch.norm(embeddings, dim=1)
#     assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


# def test_embedding_dataset():
#     """Test the dataset for training the model."""
#     # Sample data
#     texts = ["sample text one", "sample text two", "different words here"]
#     target_embeddings = np.random.rand(3, 64).astype(np.float32)
    
#     # Create tokenizer and fit
#     tokenizer = Tokenizer(vocab_size=50)
#     tokenizer.fit(texts)
    
#     # Create dataset
#     dataset = EmbeddingDataset(texts, target_embeddings, tokenizer)
    
#     # Test length
#     assert len(dataset) == 3
    
#     # Test getting an item
#     item = dataset[0]
#     assert "token_ids" in item
#     assert "length" in item
#     assert "target_embedding" in item
#     assert item["target_embedding"].shape == (64,)


# def test_custom_embedding_engine_init():
#     """Test initialization of CustomEmbeddingEngine."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_path = os.path.join(tmp_dir, "test_model")
        
#         # Initialize with non-existent model (should create new)
#         engine = CustomEmbeddingEngine(model_path=model_path, embedding_dim=64)
        
#         # Check initialization
#         assert engine.embedding_dim == 64
#         assert engine.model is not None
#         assert engine.tokenizer is not None


# def test_custom_embedding_engine_encode():
#     """Test encoding with CustomEmbeddingEngine."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_path = os.path.join(tmp_dir, "test_model")
        
#         # Initialize engine
#         engine = CustomEmbeddingEngine(model_path=model_path, embedding_dim=64)
        
#         # Simple string encoding
#         embedding = engine.encode("test string")
#         assert len(embedding) == 64
        
#         # Dictionary encoding
#         data = {
#             "agent_id": "agent_1",
#             "position_x": 10,
#             "position_y": 20,
#             "resource_level": 50,
#             "inventory": ["apple", "book"]
#         }
        
#         embedding = engine.encode(data)
#         assert len(embedding) == 64
        
#         # Test with context weights
#         context_weights = {
#             "position_x": 2.0,
#             "position_y": 2.0,
#             "inventory": 1.5
#         }
        
#         weighted_embedding = engine.encode(data, context_weights=context_weights)
#         assert len(weighted_embedding) == 64


# def test_custom_embedding_engine_save_load():
#     """Test saving and loading the custom model."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_path = os.path.join(tmp_dir, "test_model")
        
#         # Initialize and save
#         engine = CustomEmbeddingEngine(model_path=model_path, embedding_dim=64)
        
#         # Build simple vocabulary
#         texts = ["test text for saving", "another test sample", "save and load test"]
#         engine.tokenizer.fit(texts)
        
#         # Save the model
#         engine.save_model()
        
#         # Check files exist
#         assert os.path.exists(f"{model_path}_model.pt")
#         assert os.path.exists(f"{model_path}_tokenizer.pkl")
        
#         # Load in a new instance
#         engine2 = CustomEmbeddingEngine(model_path=model_path, embedding_dim=64)
        
#         # Both should encode the same text to similar vectors
#         text = "test text"
#         embedding1 = engine.encode(text)
#         embedding2 = engine2.encode(text)
        
#         # Calculate cosine similarity
#         embedding1 = np.array(embedding1)
#         embedding2 = np.array(embedding2)
#         similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
#         # Vectors should be identical or very similar
#         assert similarity > 0.99


# def test_tiny_knowledge_distillation():
#     """Test a minimal knowledge distillation training loop."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_path = os.path.join(tmp_dir, "test_model")
        
#         # Initialize engine
#         engine = CustomEmbeddingEngine(model_path=model_path, embedding_dim=32)
        
#         # Small training dataset
#         texts = [
#             "agent at position x=10 y=20",
#             "agent has apple in inventory",
#             "agent moves to position x=15 y=25",
#             "agent picks up banana"
#         ]
        
#         # Mock teacher embeddings
#         teacher_embeddings = np.random.rand(len(texts), 32).astype(np.float32)
        
#         # Normalize embeddings to unit length
#         for i in range(len(teacher_embeddings)):
#             teacher_embeddings[i] = teacher_embeddings[i] / np.linalg.norm(teacher_embeddings[i])
        
#         # Train for just 2 epochs with small batch
#         engine.train_from_teacher(
#             texts=texts,
#             teacher_embeddings=teacher_embeddings,
#             batch_size=2,
#             epochs=2,
#             learning_rate=0.01
#         )
        
#         # Check that model files were created
#         assert os.path.exists(f"{model_path}_model.pt")
#         assert os.path.exists(f"{model_path}_tokenizer.pkl")
        
#         # Test encoding after training
#         for text in texts:
#             embedding = engine.encode(text)
#             assert len(embedding) == 32


# def test_consistency_with_object_to_text():
#     """Test that the custom engine handles the same input formats as TextEmbeddingEngine."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_path = os.path.join(tmp_dir, "test_model")
        
#         # Initialize engine
#         engine = CustomEmbeddingEngine(model_path=model_path, embedding_dim=64)
        
#         # Sample agent state
#         agent_state = {
#             "agent_id": "agent_1",
#             "position": {"room": "kitchen", "x": 12.5, "y": 8.3},
#             "inventory": ["book", "apple"],
#             "health": 95,
#             "status": "active"
#         }
        
#         # Generate text using object_to_text
#         text = object_to_text(agent_state)
        
#         # Test that we can encode this text
#         text_embedding = engine._encode_text(text)
#         assert len(text_embedding) == 64
        
#         # Test direct encoding of the state
#         state_embedding = engine.encode(agent_state)
#         assert len(state_embedding) == 64 