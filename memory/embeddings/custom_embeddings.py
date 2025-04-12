"""Custom embedding model to replace sentence-transformers dependency.

This module provides a lightweight embedding model that is trained through knowledge
distillation from sentence-transformers. It maintains the same interface as
TextEmbeddingEngine while eliminating the external dependency.
"""

import logging
import os
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from memory.embeddings.utils import object_to_text

logger = logging.getLogger(__name__)


class EmbeddingModel(nn.Module):
    """Lightweight neural network model for text embeddings.
    
    This model is designed to be trained through knowledge distillation
    from a larger sentence-transformers model.
    """
    
    def __init__(
        self, 
        vocab_size: int = 10000, 
        embedding_dim: int = 384, 
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        """Initialize the embedding model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Output embedding dimension
            hidden_dim: Size of hidden layers
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection layer to final embedding dimension
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding model.
        
        Args:
            token_ids: Tensor of token IDs [batch_size, seq_len]
            lengths: Tensor of sequence lengths [batch_size]
            
        Returns:
            Embedding vectors [batch_size, embedding_dim]
        """
        # Create embeddings
        embedded = self.embedding(token_ids)  # [batch_size, seq_len, hidden_dim]
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        output, (hidden, _) = self.lstm(packed)
        
        # Concatenate the final hidden states from both directions
        # [batch_size, hidden_dim * 2]
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Project to the output embedding dimension
        embedding = self.projection(final_hidden)
        
        # Normalize to unit length
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class Tokenizer:
    """Simple tokenizer for processing text into token IDs."""
    
    def __init__(self, vocab_size: int = 10000):
        """Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum size of vocabulary
        """
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = vocab_size
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        
        # Add special tokens
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        
    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from texts.
        
        Args:
            texts: List of text samples to build vocabulary from
        """
        word_counts = {}
        
        # Count word frequencies
        for text in texts:
            for word in text.lower().split():
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocabulary (limit to vocab_size)
        for word, _ in sorted_words[:self.vocab_size - 2]:  # -2 for special tokens
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
    
    def encode(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with token_ids and length tensors
        """
        # Split text into words
        words = text.lower().split()
        
        # Convert words to token IDs
        token_ids = []
        for word in words[:max_length]:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                token_ids.append(self.word_to_id[self.unk_token])
        
        # Get actual length
        length = len(token_ids)
        
        # Pad sequence
        if len(token_ids) < max_length:
            token_ids += [self.word_to_id[self.pad_token]] * (max_length - len(token_ids))
        
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
    
    def batch_encode(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Encode batch of texts.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with batched token_ids and lengths tensors
        """
        batch_token_ids = []
        batch_lengths = []
        
        for text in texts:
            encoded = self.encode(text, max_length)
            batch_token_ids.append(encoded["token_ids"])
            batch_lengths.append(encoded["length"])
        
        return {
            "token_ids": torch.stack(batch_token_ids),
            "lengths": torch.stack(batch_lengths),
        }
    
    def save(self, path: str) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save the tokenizer
        """
        with open(path, "wb") as f:
            pickle.dump({
                "word_to_id": self.word_to_id,
                "id_to_word": self.id_to_word,
                "vocab_size": self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from file.
        
        Args:
            path: Path to load the tokenizer from
            
        Returns:
            Loaded Tokenizer instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.word_to_id = data["word_to_id"]
        tokenizer.id_to_word = data["id_to_word"]
        
        return tokenizer


class EmbeddingDataset(Dataset):
    """Dataset for training the embedding model."""
    
    def __init__(self, texts: List[str], target_embeddings: np.ndarray, tokenizer: Tokenizer):
        """Initialize the dataset.
        
        Args:
            texts: List of input texts
            target_embeddings: Target embeddings from teacher model
            tokenizer: Tokenizer instance
        """
        self.texts = texts
        self.target_embeddings = target_embeddings
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        target_embedding = self.target_embeddings[idx]
        
        encoded = self.tokenizer.encode(text)
        
        return {
            "token_ids": encoded["token_ids"],
            "length": encoded["length"],
            "target_embedding": torch.tensor(target_embedding, dtype=torch.float32)
        }


class CustomEmbeddingEngine:
    """Custom embedding engine to replace sentence-transformers.
    
    This class provides the same interface as TextEmbeddingEngine but uses
    a lightweight custom model instead of sentence-transformers.
    """
    
    def __init__(
        self, 
        model_path: str = "./models/custom_embedding",
        embedding_dim: int = 384
    ):
        """Initialize the custom embedding engine.
        
        Args:
            model_path: Path to load/save the model
            embedding_dim: Embedding dimension
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load or initialize model and tokenizer
        if os.path.exists(f"{model_path}_model.pt") and os.path.exists(f"{model_path}_tokenizer.pkl"):
            self.load_model()
        else:
            logger.warning(f"Model files not found at {model_path}, using untrained model.")
            self.tokenizer = Tokenizer(vocab_size=10000)
            self.model = EmbeddingModel(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=embedding_dim
            )
            self.model.eval()
    
    def load_model(self) -> None:
        """Load model and tokenizer from files."""
        # Load tokenizer
        self.tokenizer = Tokenizer.load(f"{self.model_path}_tokenizer.pkl")
        
        # Load model
        self.model = EmbeddingModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim
        )
        self.model.load_state_dict(torch.load(f"{self.model_path}_model.pt"))
        self.model.eval()
    
    def save_model(self) -> None:
        """Save model and tokenizer to files."""
        # Save tokenizer
        self.tokenizer.save(f"{self.model_path}_tokenizer.pkl")
        
        # Save model
        torch.save(self.model.state_dict(), f"{self.model_path}_model.pt")
    
    def train_from_teacher(
        self, 
        texts: List[str], 
        teacher_embeddings: np.ndarray, 
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> None:
        """Train model through knowledge distillation from a teacher model.
        
        Args:
            texts: List of text samples
            teacher_embeddings: Embeddings from the teacher model
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Build vocabulary
        logger.info("Building vocabulary from training texts...")
        self.tokenizer.fit(texts)
        
        # Initialize model with updated vocabulary size
        self.model = EmbeddingModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim
        )
        
        # Create dataset and dataloader
        dataset = EmbeddingDataset(texts, teacher_embeddings, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Train the model
        self.model.train()
        logger.info(f"Training model for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                token_ids = batch["token_ids"]
                lengths = batch["lengths"]
                target_embeddings = batch["target_embeddings"]
                
                # Get model embeddings
                embeddings = self.model(token_ids, lengths)
                
                # Compute cosine similarity loss (1 - cosine similarity)
                cos_sim = F.cosine_similarity(embeddings, target_embeddings)
                loss = torch.mean(1.0 - cos_sim)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Save the trained model
        self.save_model()
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors
        """
        token_ids = torch.stack([item["token_ids"] for item in batch])
        lengths = torch.stack([item["length"] for item in batch])
        target_embeddings = torch.stack([item["target_embedding"] for item in batch])
        
        return {
            "token_ids": token_ids,
            "lengths": lengths,
            "target_embeddings": target_embeddings
        }
    
    def encode(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data into an embedding vector with optional context weighting.
        
        This method maintains the same interface as TextEmbeddingEngine.encode.
        
        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights
                for context-aware embedding generation
                
        Returns:
            Embedding vector as a list of floats
        """
        # Check for context-aware weighting
        if context_weights and isinstance(data, dict):
            weighted_text = ""
            # Process standard representation
            standard_text = object_to_text(data)

            # Add weighted components
            for key, weight in context_weights.items():
                if key in data:
                    # Special case for position to extract location
                    if (
                        key == "position"
                        and isinstance(data[key], dict)
                        and "location" in data[key]
                    ):
                        location_text = f"location is {data[key]['location']}"
                        # Repeat text based on weight for emphasis (integer multiplier)
                        repeat_count = max(1, int(weight * 5))
                        weighted_text += f" {location_text}" * repeat_count
                    # Special case for inventory to emphasize items
                    elif key == "inventory" and isinstance(data[key], list):
                        for item in data[key]:
                            item_text = f"has {item}"
                            repeat_count = max(1, int(weight * 3))
                            weighted_text += f" {item_text}" * repeat_count
                    else:
                        # Extract and repeat important components based on weight
                        value_text = object_to_text({key: data[key]})
                        # Repeat text based on weight for emphasis (integer multiplier)
                        repeat_count = max(1, int(weight * 3))
                        weighted_text += f" {value_text}" * repeat_count

            # Combine standard and weighted text with more weight on the emphasized parts
            combined_text = f"{standard_text} {weighted_text} {weighted_text}"
            return self._encode_text(combined_text)

        # Default encoding without weighting
        text = object_to_text(data)
        return self._encode_text(text)
    
    def _encode_text(self, text: str) -> List[float]:
        """Encode text string to embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as a list of floats
        """
        # Tokenize the text
        encoded = self.tokenizer.encode(text)
        
        # Prepare inputs (add batch dimension)
        token_ids = encoded["token_ids"].unsqueeze(0)
        length = encoded["length"].unsqueeze(0)
        
        # Get embedding from model
        with torch.no_grad():
            embedding = self.model(token_ids, length)
        
        # Convert to list
        return embedding.squeeze(0).cpu().numpy().tolist()
    
    def encode_stm(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for STM tier with optional context weighting.
        
        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights
            
        Returns:
            Embedding vector for STM
        """
        return self.encode(data, context_weights)
    
    def encode_im(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for IM tier with optional context weighting.
        
        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights
            
        Returns:
            Embedding vector for IM
        """
        return self.encode(data, context_weights)
    
    def encode_ltm(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for LTM tier with optional context weighting.
        
        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights
            
        Returns:
            Embedding vector for LTM
        """
        return self.encode(data, context_weights)
    
    def configure(self, config: Any) -> None:
        """Update configuration of the embedding engine.
        
        Args:
            config: New configuration parameters
        """
        # Nothing to configure for this engine
        pass 