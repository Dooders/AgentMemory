"""Autoencoder-based embeddings for agent memory states."""

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from agent_memory.config import AutoencoderConfig

logger = logging.getLogger(__name__)


class NumericExtractor:
    """Extract numeric values from agent states."""

    def extract(self, state: Dict[str, Any]) -> List[float]:
        """Extract numeric values from a state dictionary.

        Args:
            state: Dictionary to extract values from

        Returns:
            List of numeric values
        """
        # Extract all numeric values from the state dictionary
        vector = []
        for key, value in self._flatten_dict(state).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                vector.append(float(value))
        return vector

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested dictionaries

        Returns:
            Flattened dictionary
        """
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_dict(v, key))
            else:
                result[key] = v
        return result


class StateAutoencoder(nn.Module):
    """Neural network autoencoder for agent state vectorization and compression.

    The autoencoder consists of:
    1. An encoder that compresses input features to the embedding space
    2. A decoder that reconstructs original features from embeddings
    3. Multiple "bottlenecks" for different compression levels
    """

    def __init__(
        self, input_dim: int, stm_dim: int = 384, im_dim: int = 128, ltm_dim: int = 32
    ):
        """Initialize the multi-resolution autoencoder.

        Args:
            input_dim: Dimension of the flattened input features
            stm_dim: Dimension for Short-Term Memory (STM) embeddings
            im_dim: Dimension for Intermediate Memory (IM) embeddings
            ltm_dim: Dimension for Long-Term Memory (LTM) embeddings
        """
        super(StateAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Multi-resolution bottlenecks
        self.stm_bottleneck = nn.Linear(256, stm_dim)
        self.im_bottleneck = nn.Linear(stm_dim, im_dim)
        self.ltm_bottleneck = nn.Linear(im_dim, ltm_dim)

        # Expansion layers (from LTM to IM to STM)
        self.ltm_to_im = nn.Linear(ltm_dim, im_dim)
        self.im_to_stm = nn.Linear(im_dim, stm_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(stm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def encode_stm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to STM embedding space.

        Args:
            x: Input tensor

        Returns:
            STM embedding
        """
        x = self.encoder(x)
        return self.stm_bottleneck(x)

    def encode_im(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to IM embedding space.

        Args:
            x: Input tensor

        Returns:
            IM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        return self.im_bottleneck(x)

    def encode_ltm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to LTM embedding space.

        Args:
            x: Input tensor

        Returns:
            LTM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        x = self.im_bottleneck(x)
        return self.ltm_bottleneck(x)

    def decode_stm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from STM embedding space.

        Args:
            z: STM embedding

        Returns:
            Reconstructed input
        """
        return self.decoder(z)

    def decode_im(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from IM embedding space.

        Args:
            z: IM embedding

        Returns:
            Reconstructed input
        """
        z = self.im_to_stm(z)
        return self.decoder(z)

    def decode_ltm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from LTM embedding space.

        Args:
            z: LTM embedding

        Returns:
            Reconstructed input
        """
        z = self.ltm_to_im(z)
        z = self.im_to_stm(z)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, level: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor
            level: Compression level (0=STM, 1=IM, 2=LTM)

        Returns:
            Tuple of (reconstructed output, embedding)
        """
        # Encode
        encoded = self.encoder(x)

        if level == 0:  # STM
            embedding = self.stm_bottleneck(encoded)
            decoded = self.decoder(embedding)
        elif level == 1:  # IM
            stm_embedding = self.stm_bottleneck(encoded)
            embedding = self.im_bottleneck(stm_embedding)
            expanded = self.im_to_stm(embedding)
            decoded = self.decoder(expanded)
        else:  # LTM
            stm_embedding = self.stm_bottleneck(encoded)
            im_embedding = self.im_bottleneck(stm_embedding)
            embedding = self.ltm_bottleneck(im_embedding)
            expanded_im = self.ltm_to_im(embedding)
            expanded_stm = self.im_to_stm(expanded_im)
            decoded = self.decoder(expanded_stm)

        return decoded, embedding


class AgentStateDataset(torch.utils.data.Dataset):
    """Dataset for agent states."""

    def __init__(self, states: List[Dict[str, Any]], processor=None):
        """Initialize dataset.

        Args:
            states: List of agent states.
            processor: Processor to use for extracting numeric values from states.
        """
        self.states = states
        self.processor = processor or NumericExtractor()
        self.vectors = self._prepare_vectors()

    def _prepare_vectors(self) -> np.ndarray:
        """Extract numeric values from states.

        Returns:
            Array of vectors.
        """
        # Check if all states have "vector" field and use that directly
        if (
            self.states
            and "vector" in self.states[0]
            and isinstance(self.states[0]["vector"], np.ndarray)
        ):
            vectors = [state["vector"] for state in self.states if "vector" in state]
            if vectors:
                # Ensure all vectors have the same dimensions
                # Use the first vector's dimension as the reference
                first_dim = len(vectors[0])
                normalized_vectors = []

                for vec in vectors:
                    if len(vec) < first_dim:
                        # Pad shorter vectors
                        vec = np.pad(vec, (0, first_dim - len(vec)))
                    elif len(vec) > first_dim:
                        # Truncate longer vectors
                        vec = vec[:first_dim]
                    normalized_vectors.append(vec)

                return np.array(normalized_vectors)

        # Fall back to processor extraction if no vectors found
        vectors = []
        for state in self.states:
            try:
                values = self.processor.extract(state)
                if values:
                    vectors.append(values)
            except Exception as e:
                logging.warning(f"Failed to extract numeric values from state: {e}")
                continue

        if not vectors:
            # Return empty array with correct shape - assuming dimension 1 if we can't determine
            return np.array([]).reshape(0, 1)

        return np.array(vectors)

    def __len__(self) -> int:
        """Get the number of samples.

        Returns:
            Dataset size
        """
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tensor representation of the state
        """
        return torch.tensor(self.vectors[idx], dtype=torch.float32)


class AutoencoderEmbeddingEngine:
    """Engine for generating embeddings using the autoencoder model.

    This class handles the training and inference of the autoencoder model,
    providing methods to encode and decode agent states at different
    compression levels.

    Attributes:
        model: The autoencoder model
        input_dim: Dimension of the input features
        stm_dim: Dimension of STM embeddings
        im_dim: Dimension of IM embeddings
        ltm_dim: Dimension of LTM embeddings
        device: Device to run the model on
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 64,
        stm_dim: int = 384,
        im_dim: int = 128,
        ltm_dim: int = 32,
    ):
        """Initialize the embedding engine.

        Args:
            model_path: Path to saved model (if available)
            input_dim: Dimension of the input features
            stm_dim: Dimension of STM embeddings
            im_dim: Dimension of IM embeddings
            ltm_dim: Dimension of LTM embeddings
        """
        self.input_dim = input_dim
        self.stm_dim = stm_dim
        self.im_dim = im_dim
        self.ltm_dim = ltm_dim

        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = StateAutoencoder(input_dim, stm_dim, im_dim, ltm_dim).to(
            self.device
        )

        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info("Loaded autoencoder model from %s", model_path)
        else:
            logger.warning("No pre-trained model found. Using untrained model.")

    def configure(self, config: AutoencoderConfig) -> None:
        """Update the embedding engine configuration.

        Args:
            config: New configuration object for the autoencoder
        """
        # Update dimension settings if they've changed
        needs_new_model = False
        if self.input_dim != config.input_dim:
            self.input_dim = config.input_dim
            needs_new_model = True

        if self.stm_dim != config.stm_dim:
            self.stm_dim = config.stm_dim
            needs_new_model = True

        if self.im_dim != config.im_dim:
            self.im_dim = config.im_dim
            needs_new_model = True

        if self.ltm_dim != config.ltm_dim:
            self.ltm_dim = config.ltm_dim
            needs_new_model = True

        # If dimensions changed, we need to reinitialize the model
        if needs_new_model:
            logger.info("Reinitializing autoencoder model with new dimensions")
            self.model = StateAutoencoder(
                self.input_dim, self.stm_dim, self.im_dim, self.ltm_dim
            ).to(self.device)

            # Try to load model from the path specified in config
            if config.model_path and os.path.exists(config.model_path):
                try:
                    self.load_model(config.model_path)
                    logger.info(
                        "Loaded model from %s after configuration update",
                        config.model_path,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load model after configuration update: %s", str(e)
                    )

    def encode_stm(
        self, state: Dict[str, Any], context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode state to STM embedding space.

        Args:
            state: Agent state dictionary
            context_weights: Optional dictionary mapping keys to importance weights
                            (not used in autoencoder version, but kept for API compatibility)

        Returns:
            STM embedding as list of floats
        """
        # Note: context_weights is ignored in the autoencoder implementation
        # as the model doesn't support weighted encoding, but included for API compatibility
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_stm(x)
        return embedding.cpu().numpy().tolist()

    def encode_im(
        self, state: Dict[str, Any], context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode state to IM embedding space.

        Args:
            state: Agent state dictionary
            context_weights: Optional dictionary mapping keys to importance weights
                            (not used in autoencoder version, but kept for API compatibility)

        Returns:
            IM embedding as list of floats
        """
        # Note: context_weights is ignored in the autoencoder implementation
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_im(x)
        return embedding.cpu().numpy().tolist()

    def encode_ltm(
        self, state: Dict[str, Any], context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode state to LTM embedding space.

        Args:
            state: Agent state dictionary
            context_weights: Optional dictionary mapping keys to importance weights
                            (not used in autoencoder version, but kept for API compatibility)

        Returns:
            LTM embedding as list of floats
        """
        # Note: context_weights is ignored in the autoencoder implementation
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_ltm(x)
        return embedding.cpu().numpy().tolist()

    def convert_embedding(
        self, embedding: List[float], source_tier: str, target_tier: str
    ) -> List[float]:
        """Convert an embedding from one memory tier to another.

        Args:
            embedding: The source embedding to convert
            source_tier: The source tier ('stm', 'im', or 'ltm')
            target_tier: The target tier ('stm', 'im', or 'ltm')

        Returns:
            The converted embedding

        Raises:
            ValueError: If source or target tier is invalid, or if attempting to convert
                       to a higher-dimension tier (which would lose information)
        """
        valid_tiers = {"stm", "im", "ltm"}
        if source_tier not in valid_tiers or target_tier not in valid_tiers:
            raise ValueError(f"Invalid tier specified. Must be one of: {valid_tiers}")

        # If source and target are the same, return the embedding unchanged
        if source_tier == target_tier:
            return embedding

        # Convert to PyTorch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)

        # Define tier hierarchy and dimensions
        tier_levels = {"stm": 0, "im": 1, "ltm": 2}
        tier_dims = {"stm": self.stm_dim, "im": self.im_dim, "ltm": self.ltm_dim}

        # Verify dimensions
        if len(embedding) != tier_dims[source_tier]:
            logger.warning(
                f"Source embedding dimension {len(embedding)} doesn't match expected {tier_dims[source_tier]} for {source_tier.upper()} tier. Attempting to adjust."
            )
            # Pad or truncate to match expected dimension
            if len(embedding) < tier_dims[source_tier]:
                embedding_tensor = torch.nn.functional.pad(
                    embedding_tensor, (0, tier_dims[source_tier] - len(embedding))
                )
            else:
                embedding_tensor = embedding_tensor[: tier_dims[source_tier]]

        # If converting to a higher dimension tier, we need to use the decoder/encoder path
        # (STM->LTM is going to lower dimension, LTM->STM is going to higher dimension)
        source_level = tier_levels[source_tier]
        target_level = tier_levels[target_tier]

        with torch.no_grad():
            # Converting to lower dimension (e.g., STM to IM, IM to LTM, or STM to LTM)
            if target_level > source_level:
                if source_tier == "stm" and target_tier == "im":
                    result = self.model.im_bottleneck(embedding_tensor)
                elif source_tier == "im" and target_tier == "ltm":
                    result = self.model.ltm_bottleneck(embedding_tensor)
                elif source_tier == "stm" and target_tier == "ltm":
                    # First convert to IM, then to LTM
                    im_embedding = self.model.im_bottleneck(embedding_tensor)
                    result = self.model.ltm_bottleneck(im_embedding)

            # Converting to higher dimension (e.g., LTM to IM, IM to STM, or LTM to STM)
            elif target_level < source_level:
                if source_tier == "im" and target_tier == "stm":
                    result = self.model.im_to_stm(embedding_tensor)
                elif source_tier == "ltm" and target_tier == "im":
                    result = self.model.ltm_to_im(embedding_tensor)
                elif source_tier == "ltm" and target_tier == "stm":
                    # First convert to IM, then to STM
                    im_embedding = self.model.ltm_to_im(embedding_tensor)
                    result = self.model.im_to_stm(im_embedding)
            else:
                # Should never reach here due to the equality check at the beginning
                result = embedding_tensor

        # Convert back to list
        return result.cpu().numpy().tolist()

    def ensure_embedding_dimensions(
        self, embedding: List[float], target_tier: str
    ) -> List[float]:
        """Automatically detect embedding dimensions and convert to the target tier if needed.

        This method identifies the source tier based on the embedding dimension and
        converts it to the requested target tier format.

        Args:
            embedding: The embedding vector to convert
            target_tier: The target memory tier ('stm', 'im', or 'ltm')

        Returns:
            The correctly dimensioned embedding for the target tier

        Raises:
            ValueError: If target tier is invalid or embedding dimensions don't match any known tier
        """
        valid_tiers = {"stm", "im", "ltm"}
        if target_tier not in valid_tiers:
            raise ValueError(f"Invalid target tier. Must be one of: {valid_tiers}")

        # Detect source tier based on dimension
        embedding_dim = len(embedding)
        tier_dims = {"stm": self.stm_dim, "im": self.im_dim, "ltm": self.ltm_dim}

        # Check if embedding dimensions match any tier
        source_tier = None
        tolerance = 1  # Allow for slight dimension differences

        for tier, dim in tier_dims.items():
            if abs(embedding_dim - dim) <= tolerance:
                source_tier = tier
                break

        # If we couldn't identify the source tier, try to determine the closest one
        if source_tier is None:
            # Find the closest matching tier
            diffs = [
                (tier, abs(embedding_dim - dim)) for tier, dim in tier_dims.items()
            ]
            source_tier, _ = min(diffs, key=lambda x: x[1])

            logger.warning(
                f"Embedding dimension {embedding_dim} doesn't exactly match any tier. "
                f"Assuming {source_tier.upper()} tier (dimension {tier_dims[source_tier]})."
            )

        # Convert to target tier if needed
        if source_tier != target_tier:
            return self.convert_embedding(embedding, source_tier, target_tier)
        else:
            # If dimensions don't exactly match, pad or truncate
            if embedding_dim != tier_dims[target_tier]:
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(
                    self.device
                )
                if embedding_dim < tier_dims[target_tier]:
                    embedding_tensor = torch.nn.functional.pad(
                        embedding_tensor, (0, tier_dims[target_tier] - embedding_dim)
                    )
                else:
                    embedding_tensor = embedding_tensor[: tier_dims[target_tier]]
                return embedding_tensor.cpu().numpy().tolist()

            return embedding

    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert a state dictionary to a vector.

        Args:
            state: Agent state dictionary

        Returns:
            Numpy array of state features
        """
        # Special case: if state already has a vector field with a numpy array, use it directly
        if "vector" in state and isinstance(state["vector"], (np.ndarray, list)):
            vector = np.asarray(state["vector"], dtype=np.float32)
        else:
            # Regular case: create dataset and extract vector
            dataset = AgentStateDataset([state])
            vector = dataset.vectors[0]

        # Ensure correct dimension
        if len(vector) < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - len(vector)))
        elif len(vector) > self.input_dim:
            vector = vector[: self.input_dim]

        return vector

    def train(
        self,
        states: List[Dict[str, Any]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder on agent states with validation.

        Args:
            states: List of agent state dictionaries
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            early_stopping_patience: Number of epochs to wait for validation improvement before stopping

        Returns:
            Dictionary of training metrics
        """
        # Create dataset
        all_dataset = AgentStateDataset(states)

        # Split into train and validation sets
        dataset_size = len(all_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size

        # Use random_split for unbiased sampling
        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # For reproducibility
        )

        logger.info(
            f"Training on {train_size} samples, validating on {val_size} samples"
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Set up optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Initialize metrics dictionary
        metrics = {
            "train_loss": [],
            "train_stm_loss": [],
            "train_im_loss": [],
            "train_ltm_loss": [],
            "val_loss": [],
            "val_stm_loss": [],
            "val_im_loss": [],
            "val_ltm_loss": [],
            "val_stm_r2": [],
            "val_im_r2": [],
            "val_ltm_r2": [],
        }

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_running_loss = 0.0
            train_running_stm_loss = 0.0
            train_running_im_loss = 0.0
            train_running_ltm_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward passes at all compression levels
                stm_output, _ = self.model(batch, level=0)
                im_output, _ = self.model(batch, level=1)
                ltm_output, _ = self.model(batch, level=2)

                # Calculate losses
                stm_loss = criterion(stm_output, batch)
                im_loss = criterion(im_output, batch)
                ltm_loss = criterion(ltm_output, batch)

                # Combined loss (weighted by importance)
                loss = stm_loss + 0.5 * im_loss + 0.25 * ltm_loss

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Update training metrics
                train_running_loss += loss.item() * batch.size(0)
                train_running_stm_loss += stm_loss.item() * batch.size(0)
                train_running_im_loss += im_loss.item() * batch.size(0)
                train_running_ltm_loss += ltm_loss.item() * batch.size(0)

            # Calculate average training losses
            avg_train_loss = train_running_loss / len(train_dataset)
            avg_train_stm_loss = train_running_stm_loss / len(train_dataset)
            avg_train_im_loss = train_running_im_loss / len(train_dataset)
            avg_train_ltm_loss = train_running_ltm_loss / len(train_dataset)

            # Record training metrics
            metrics["train_loss"].append(avg_train_loss)
            metrics["train_stm_loss"].append(avg_train_stm_loss)
            metrics["train_im_loss"].append(avg_train_im_loss)
            metrics["train_ltm_loss"].append(avg_train_ltm_loss)

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_running_stm_loss = 0.0
            val_running_im_loss = 0.0
            val_running_ltm_loss = 0.0

            # For R² calculation
            all_val_targets = []
            all_val_stm_preds = []
            all_val_im_preds = []
            all_val_ltm_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)

                    # Forward passes at all compression levels
                    stm_output, _ = self.model(batch, level=0)
                    im_output, _ = self.model(batch, level=1)
                    ltm_output, _ = self.model(batch, level=2)

                    # Store predictions and targets for R² calculation
                    all_val_targets.append(batch.cpu().numpy())
                    all_val_stm_preds.append(stm_output.cpu().numpy())
                    all_val_im_preds.append(im_output.cpu().numpy())
                    all_val_ltm_preds.append(ltm_output.cpu().numpy())

                    # Calculate losses
                    stm_loss = criterion(stm_output, batch)
                    im_loss = criterion(im_output, batch)
                    ltm_loss = criterion(ltm_output, batch)

                    # Combined loss
                    loss = stm_loss + 0.5 * im_loss + 0.25 * ltm_loss

                    # Update validation metrics
                    val_running_loss += loss.item() * batch.size(0)
                    val_running_stm_loss += stm_loss.item() * batch.size(0)
                    val_running_im_loss += im_loss.item() * batch.size(0)
                    val_running_ltm_loss += ltm_loss.item() * batch.size(0)

            # Calculate average validation losses
            avg_val_loss = val_running_loss / len(val_dataset)
            avg_val_stm_loss = val_running_stm_loss / len(val_dataset)
            avg_val_im_loss = val_running_im_loss / len(val_dataset)
            avg_val_ltm_loss = val_running_ltm_loss / len(val_dataset)

            # Record validation metrics
            metrics["val_loss"].append(avg_val_loss)
            metrics["val_stm_loss"].append(avg_val_stm_loss)
            metrics["val_im_loss"].append(avg_val_im_loss)
            metrics["val_ltm_loss"].append(avg_val_ltm_loss)

            # Calculate R² scores for each compression level
            all_val_targets = np.vstack(all_val_targets)
            all_val_stm_preds = np.vstack(all_val_stm_preds)
            all_val_im_preds = np.vstack(all_val_im_preds)
            all_val_ltm_preds = np.vstack(all_val_ltm_preds)

            # Compute R² for each compression level
            stm_r2 = r2_score(
                all_val_targets.reshape(-1), all_val_stm_preds.reshape(-1)
            )
            im_r2 = r2_score(all_val_targets.reshape(-1), all_val_im_preds.reshape(-1))
            ltm_r2 = r2_score(
                all_val_targets.reshape(-1), all_val_ltm_preds.reshape(-1)
            )

            # Add R² scores to metrics
            if epoch == 0:
                metrics["val_stm_r2"] = []
                metrics["val_im_r2"] = []
                metrics["val_ltm_r2"] = []

            metrics["val_stm_r2"].append(stm_r2)
            metrics["val_im_r2"].append(im_r2)
            metrics["val_ltm_r2"].append(ltm_r2)

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"STM R²: {stm_r2:.4f}, "
                    f"IM R²: {im_r2:.4f}, "
                    f"LTM R²: {ltm_r2:.4f}"
                )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Restored best model with validation loss: {best_val_loss:.4f}"
            )

        # Final validation metrics summary
        if len(metrics["val_loss"]) > 0:
            best_epoch = np.argmin(metrics["val_loss"])
            logger.info("=" * 50)
            logger.info("Final Validation Metrics Summary:")
            logger.info(f"Best epoch: {best_epoch + 1}")
            logger.info(f"Validation loss: {metrics['val_loss'][best_epoch]:.6f}")
            logger.info(
                f"STM validation loss: {metrics['val_stm_loss'][best_epoch]:.6f}"
            )
            logger.info(f"IM validation loss: {metrics['val_im_loss'][best_epoch]:.6f}")
            logger.info(
                f"LTM validation loss: {metrics['val_ltm_loss'][best_epoch]:.6f}"
            )
            logger.info(f"STM R²: {metrics['val_stm_r2'][best_epoch]:.6f}")
            logger.info(f"IM R²: {metrics['val_im_r2'][best_epoch]:.6f}")
            logger.info(f"LTM R²: {metrics['val_ltm_r2'][best_epoch]:.6f}")
            logger.info("=" * 50)

        return metrics

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "stm_dim": self.stm_dim,
                "im_dim": self.im_dim,
                "ltm_dim": self.ltm_dim,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.stm_dim = checkpoint["stm_dim"]
        self.im_dim = checkpoint["im_dim"]
        self.ltm_dim = checkpoint["ltm_dim"]

        # Re-initialize model with loaded dimensions
        self.model = StateAutoencoder(
            self.input_dim, self.stm_dim, self.im_dim, self.ltm_dim
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded from %s", path)

    def train_with_kfold(
        self,
        states: List[Dict[str, Any]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        n_folds: int = 5,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """Train the autoencoder using k-fold cross-validation.

        Args:
            states: List of agent state dictionaries
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            n_folds: Number of folds for cross-validation
            early_stopping_patience: Number of epochs to wait for validation improvement before stopping

        Returns:
            Dictionary of training metrics and best fold model
        """
        # Create dataset
        all_dataset = AgentStateDataset(states)
        dataset_size = len(all_dataset)
        logger.info(
            f"Starting {n_folds}-fold cross-validation on {dataset_size} samples"
        )

        # Initialize k-fold cross validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionary to store results from all folds
        all_fold_results = {
            "fold_train_loss": [],
            "fold_val_loss": [],
            "fold_val_stm_r2": [],
            "fold_val_im_r2": [],
            "fold_val_ltm_r2": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        best_fold = -1

        # Perform k-fold cross validation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(all_dataset)):
            logger.info(f"FOLD {fold+1}/{n_folds}")
            logger.info(
                f"Training on {len(train_ids)} samples, validating on {len(val_ids)} samples"
            )

            # Define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            # Prepare data loaders
            train_loader = DataLoader(
                all_dataset, batch_size=batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(
                all_dataset, batch_size=batch_size, sampler=val_sampler
            )

            # Reset model for each fold
            self.model = StateAutoencoder(
                self.input_dim, self.stm_dim, self.im_dim, self.ltm_dim
            ).to(self.device)

            # Set up optimizer and loss
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Training metrics for this fold
            fold_metrics = {
                "train_loss": [],
                "val_loss": [],
                "val_stm_r2": [],
                "val_im_r2": [],
                "val_ltm_r2": [],
            }

            # Early stopping variables
            fold_best_val_loss = float("inf")
            patience_counter = 0
            fold_best_model_state = None

            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_running_loss = 0.0

                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()

                    # Forward passes at all compression levels (focusing on STM for training)
                    stm_output, _ = self.model(batch, level=0)

                    # Calculate loss
                    loss = criterion(stm_output, batch)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Update training metrics
                    train_running_loss += loss.item() * batch.size(0)

                # Calculate average training loss
                avg_train_loss = train_running_loss / len(train_ids)
                fold_metrics["train_loss"].append(avg_train_loss)

                # Validation phase
                self.model.eval()
                val_running_loss = 0.0

                # For R² calculation
                all_val_targets = []
                all_val_stm_preds = []
                all_val_im_preds = []
                all_val_ltm_preds = []

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)

                        # Forward passes at all compression levels
                        stm_output, _ = self.model(batch, level=0)
                        im_output, _ = self.model(batch, level=1)
                        ltm_output, _ = self.model(batch, level=2)

                        # Store predictions and targets for R² calculation
                        all_val_targets.append(batch.cpu().numpy())
                        all_val_stm_preds.append(stm_output.cpu().numpy())
                        all_val_im_preds.append(im_output.cpu().numpy())
                        all_val_ltm_preds.append(ltm_output.cpu().numpy())

                        # Calculate primary loss (STM)
                        loss = criterion(stm_output, batch)

                        # Update validation metrics
                        val_running_loss += loss.item() * batch.size(0)

                # Calculate average validation loss
                avg_val_loss = val_running_loss / len(val_ids)
                fold_metrics["val_loss"].append(avg_val_loss)

                # Calculate R² scores for each compression level
                all_val_targets = np.vstack(all_val_targets)
                all_val_stm_preds = np.vstack(all_val_stm_preds)
                all_val_im_preds = np.vstack(all_val_im_preds)
                all_val_ltm_preds = np.vstack(all_val_ltm_preds)

                # Compute R² for each compression level
                stm_r2 = r2_score(
                    all_val_targets.reshape(-1), all_val_stm_preds.reshape(-1)
                )
                im_r2 = r2_score(
                    all_val_targets.reshape(-1), all_val_im_preds.reshape(-1)
                )
                ltm_r2 = r2_score(
                    all_val_targets.reshape(-1), all_val_ltm_preds.reshape(-1)
                )

                # Add R² scores to metrics
                fold_metrics["val_stm_r2"].append(stm_r2)
                fold_metrics["val_im_r2"].append(im_r2)
                fold_metrics["val_ltm_r2"].append(ltm_r2)

                # Log progress
                if (epoch + 1) % 25 == 0 or epoch == 0:
                    logger.info(
                        f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"STM R²: {stm_r2:.4f}"
                    )

                # Early stopping check
                if avg_val_loss < fold_best_val_loss:
                    fold_best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state for this fold
                    fold_best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        break

            # Restore best model for this fold
            if fold_best_model_state is not None:
                self.model.load_state_dict(fold_best_model_state)

            # Get final metrics for this fold using best model
            final_val_loss = min(fold_metrics["val_loss"])
            best_epoch = np.argmin(fold_metrics["val_loss"])
            final_stm_r2 = fold_metrics["val_stm_r2"][best_epoch]
            final_im_r2 = fold_metrics["val_im_r2"][best_epoch]
            final_ltm_r2 = fold_metrics["val_ltm_r2"][best_epoch]

            logger.info(
                f"Fold {fold+1} results: Val Loss={final_val_loss:.6f}, STM R²={final_stm_r2:.4f}"
            )

            # Add fold results to overall results
            all_fold_results["fold_train_loss"].append(
                fold_metrics["train_loss"][best_epoch]
            )
            all_fold_results["fold_val_loss"].append(final_val_loss)
            all_fold_results["fold_val_stm_r2"].append(final_stm_r2)
            all_fold_results["fold_val_im_r2"].append(final_im_r2)
            all_fold_results["fold_val_ltm_r2"].append(final_ltm_r2)

            # Check if this fold produced a better model
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = fold_best_model_state.copy()
                best_fold = fold

        # Calculate average metrics across all folds
        avg_val_loss = np.mean(all_fold_results["fold_val_loss"])
        avg_stm_r2 = np.mean(all_fold_results["fold_val_stm_r2"])
        avg_im_r2 = np.mean(all_fold_results["fold_val_im_r2"])
        avg_ltm_r2 = np.mean(all_fold_results["fold_val_ltm_r2"])

        logger.info("=" * 50)
        logger.info(f"Cross-Validation Summary ({n_folds} folds):")
        logger.info(f"Average validation loss: {avg_val_loss:.6f}")
        logger.info(f"Average STM R²: {avg_stm_r2:.6f}")
        logger.info(f"Average IM R²: {avg_im_r2:.6f}")
        logger.info(f"Average LTM R²: {avg_ltm_r2:.6f}")
        logger.info(
            f"Best fold: {best_fold+1}, with validation loss: {best_val_loss:.6f}"
        )
        logger.info("=" * 50)

        # Restore the best model across all folds
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from fold {best_fold+1}")

        # Return metrics and fold information
        summary = {
            "best_fold": best_fold + 1,
            "best_val_loss": best_val_loss,
            "avg_val_loss": avg_val_loss,
            "avg_stm_r2": avg_stm_r2,
            "avg_im_r2": avg_im_r2,
            "avg_ltm_r2": avg_ltm_r2,
            "fold_results": all_fold_results,
        }

        return summary
