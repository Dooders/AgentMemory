#!/usr/bin/env python3
"""Experiment with autoencoder embeddings for agent states.

This script connects to a simulation database, loads agent states,
and experiments with autoencoder-based embeddings at different
compression levels.
"""

import argparse
import logging
import os
import sqlite3

# Add the root directory to Python path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_memory.embeddings.autoencoder import (
    AgentStateDataset,
    AutoencoderEmbeddingEngine,
    StateAutoencoder,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def connect_to_db(db_path: str = "simulation.db") -> sqlite3.Connection:
    """Connect to the SQLite database.

    Args:
        db_path: Path to the simulation database

    Returns:
        Database connection
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    return sqlite3.connect(db_path)


def load_agent_states(
    conn: sqlite3.Connection, limit: int = None
) -> List[Dict[str, Any]]:
    """Load agent states from the database.

    Args:
        conn: Database connection
        limit: Maximum number of states to load (None for all)

    Returns:
        List of agent state dictionaries
    """
    query = """
    SELECT 
        id, step_number, agent_id, 
        position_x, position_y, position_z,
        resource_level, current_health, 
        is_defending, total_reward, age
    FROM agent_states
    """

    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.cursor()
    cursor.execute(query)

    # Get column names
    columns = [col[0] for col in cursor.description]

    # Convert to list of dictionaries
    states = []
    for row in cursor.fetchall():
        state = {columns[i]: row[i] for i in range(len(columns))}
        # Convert boolean value
        if "is_defending" in state:
            state["is_defending"] = bool(state["is_defending"])
        states.append(state)

    return states


def prepare_autoencoder(
    states: List[Dict[str, Any]],
    input_dim: int = None,
    stm_dim: int = 32,
    im_dim: int = 16,
    ltm_dim: int = 8,
) -> Tuple[AutoencoderEmbeddingEngine, AgentStateDataset, torch.device]:
    """Prepare the autoencoder for agent states.

    Args:
        states: List of agent state dictionaries
        input_dim: Input dimension (auto-determined if None)
        stm_dim: Dimension for Short-Term Memory embeddings
        im_dim: Dimension for Intermediate Memory embeddings
        ltm_dim: Dimension for Long-Term Memory embeddings

    Returns:
        Tuple of (autoencoder engine, agent state dataset, device)
    """
    # Create dataset
    dataset = AgentStateDataset(states)

    # Determine input dimension if not provided
    if input_dim is None:
        input_dim = dataset.vectors.shape[1]

    logger.info(f"Input dimension: {input_dim}")

    # Create autoencoder
    engine = AutoencoderEmbeddingEngine(
        input_dim=input_dim, stm_dim=stm_dim, im_dim=im_dim, ltm_dim=ltm_dim
    )

    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for training (GPU not available)")

    return engine, dataset, device


def train_and_evaluate(
    engine: AutoencoderEmbeddingEngine,
    dataset: AgentStateDataset,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    """Train the autoencoder and evaluate its performance.

    Args:
        engine: Autoencoder engine
        dataset: Agent state dataset
        device: Device to train on (CPU or GPU)
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        early_stopping_patience: Number of epochs to wait before early stopping

    Returns:
        Dictionary of training metrics, original data, embedded data, and trained model on device
    """
    # Initialize the model
    model = engine.model.to(device)
    
    # Make sure we have the correct input dimension
    input_dim = engine.input_dim
    
    # Convert dataset to states format for engine's train method
    states = []
    for i in range(len(dataset)):
        # Create a dummy state dictionary with a vector field
        # Ensure vector has the right dimension for the model
        vector = dataset[i].detach().cpu().numpy()
        if len(vector) < input_dim:
            vector = np.pad(vector, (0, input_dim - len(vector)))
        elif len(vector) > input_dim:
            vector = vector[:input_dim]
            
        state = {"vector": vector}
        states.append(state)
    
    logger.info(f"Training model with validation split of {validation_split}")
    logger.info(f"Input vector dimension: {input_dim}")
    
    # Use the engine's train method with validation
    train_metrics = engine.train(
        states=states,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience
    )
    
    # Extract all embeddings (using the best model from validation)
    embedded_data = []
    original_data = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get original vector
            vector = dataset[i].unsqueeze(0).to(device)
            original_vector = vector.detach().cpu().numpy().flatten()
            original_data.append(original_vector)

            # Get STM embedding
            stm_embedding = model.encode_stm(vector)
            embedded_data.append(stm_embedding.detach().cpu().numpy().flatten())

    # Save a copy of the model on CPU for the engine
    # We'll return the GPU version for further immediate use
    input_dim = engine.input_dim
    cpu_model = StateAutoencoder(
        input_dim=input_dim,
        stm_dim=model.stm_bottleneck.out_features,
        im_dim=model.im_bottleneck.out_features,
        ltm_dim=model.ltm_bottleneck.out_features,
    ).to("cpu")
    cpu_model.load_state_dict(model.state_dict())
    engine.model = cpu_model

    # Return the model on the device for immediate use by subsequent functions
    return train_metrics, np.array(original_data), np.array(embedded_data), model.to(device)


def evaluate_reconstruction(
    model: StateAutoencoder, dataset: AgentStateDataset, device: torch.device
) -> Dict[str, float]:
    """Evaluate reconstruction quality at different compression levels.

    Args:
        model: Trained autoencoder model
        dataset: Agent state dataset
        device: Device to evaluate on (CPU or GPU)

    Returns:
        Dictionary of reconstruction metrics
    """
    results = {}

    # Test a random subset for faster evaluation
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # STM reconstruction
        stm_errors = []
        for i in indices:
            vector = dataset[i].unsqueeze(0).to(device)
            stm_embedding = model.encode_stm(vector)
            reconstructed = model.decode_stm(stm_embedding)

            # Move to CPU for MSE calculation
            vector_cpu = vector.detach().cpu().numpy().flatten()
            reconstructed_cpu = reconstructed.detach().cpu().numpy().flatten()

            error = mean_squared_error(vector_cpu, reconstructed_cpu)
            stm_errors.append(error)
        results["stm_mse"] = np.mean(stm_errors)

        # IM reconstruction
        im_errors = []
        for i in indices:
            vector = dataset[i].unsqueeze(0).to(device)
            im_embedding = model.encode_im(vector)
            reconstructed = model.decode_im(im_embedding)

            # Move to CPU for MSE calculation
            vector_cpu = vector.detach().cpu().numpy().flatten()
            reconstructed_cpu = reconstructed.detach().cpu().numpy().flatten()

            error = mean_squared_error(vector_cpu, reconstructed_cpu)
            im_errors.append(error)
        results["im_mse"] = np.mean(im_errors)

        # LTM reconstruction
        ltm_errors = []
        for i in indices:
            vector = dataset[i].unsqueeze(0).to(device)
            ltm_embedding = model.encode_ltm(vector)
            reconstructed = model.decode_ltm(ltm_embedding)

            # Move to CPU for MSE calculation
            vector_cpu = vector.detach().cpu().numpy().flatten()
            reconstructed_cpu = reconstructed.detach().cpu().numpy().flatten()

            error = mean_squared_error(vector_cpu, reconstructed_cpu)
            ltm_errors.append(error)
        results["ltm_mse"] = np.mean(ltm_errors)

    # Move model back to CPU
    model = model.to("cpu")

    return results


def visualize_embeddings_modified(
    model: StateAutoencoder,
    full_dataset: AgentStateDataset,
    states: List[Dict[str, Any]],
    device: torch.device,
    output_dir: str = ".",
):
    """Generate and visualize embeddings for the dataset using a provided model.

    This function handles all steps: generating embeddings, applying t-SNE, and creating visualizations.

    Args:
        model: Trained autoencoder model
        full_dataset: Dataset of agent states
        states: Original state dictionaries
        device: Computation device
        output_dir: Directory to save output plots
    """
    logger.info("Generating embeddings for visualization...")

    # Make sure model is on the proper device and in eval mode
    model = model.to(device)
    model.eval()

    # Lists to store original data and embeddings
    original_data = []
    embedded_data = []

    # Use a data loader to batch process
    loader = torch.utils.data.DataLoader(full_dataset, batch_size=64, shuffle=False)

    # Process batches
    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            batch = batch.to(device)

            # Get embeddings
            batch_embeddings = model.encode_stm(batch)

            # Store original data and embeddings
            original_data.extend(batch.cpu().numpy())
            embedded_data.extend(batch_embeddings.cpu().numpy())

    # Convert to numpy arrays
    original_data = np.array(original_data)
    embedded_data = np.array(embedded_data)

    logger.info(f"Generated embeddings of shape {embedded_data.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Apply t-SNE to original data
    logger.info("Applying t-SNE to original data...")
    original_tsne = TSNE(n_components=2, random_state=42)
    original_2d = original_tsne.fit_transform(original_data)

    # Apply t-SNE to embedded data
    logger.info("Applying t-SNE to embedded data...")
    embedded_tsne = TSNE(n_components=2, random_state=42)
    embedded_2d = embedded_tsne.fit_transform(embedded_data)

    # Extract agent attributes for coloring
    if states and "age" in states[0]:
        # Color by age
        agent_ages = np.array([state["age"] for state in states])

        # Original data plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            original_2d[:, 0],
            original_2d[:, 1],
            c=agent_ages,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Agent Age")
        plt.title("t-SNE of Original Agent State Vectors")
        plt.savefig(os.path.join(output_dir, "original_tsne.png"))

        # Embedded data plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedded_2d[:, 0],
            embedded_2d[:, 1],
            c=agent_ages,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Agent Age")
        plt.title("t-SNE of Autoencoder Embeddings")
        plt.savefig(os.path.join(output_dir, "embedded_tsne.png"))
    else:
        # Simple plots without coloring
        plt.figure(figsize=(10, 8))
        plt.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.6)
        plt.title("t-SNE of Original Agent State Vectors")
        plt.savefig(os.path.join(output_dir, "original_tsne.png"))

        plt.figure(figsize=(10, 8))
        plt.scatter(embedded_2d[:, 0], embedded_2d[:, 1], alpha=0.6)
        plt.title("t-SNE of Autoencoder Embeddings")
        plt.savefig(os.path.join(output_dir, "embedded_tsne.png"))

    logger.info(f"Plots saved to {output_dir}")

    # Apply 3D t-SNE if necessary
    try:
        from mpl_toolkits.mplot3d import Axes3D

        # Apply t-SNE to original data with 3 components
        logger.info("Applying 3D t-SNE to original data...")
        original_tsne = TSNE(n_components=3, random_state=42)
        original_3d = original_tsne.fit_transform(original_data)

        # Apply t-SNE to embedded data with 3 components
        logger.info("Applying 3D t-SNE to embedded data...")
        embedded_tsne = TSNE(n_components=3, random_state=42)
        embedded_3d = embedded_tsne.fit_transform(embedded_data)

        # Create 3D plots
        if states and "age" in states[0]:
            # Color by age
            agent_ages = np.array([state["age"] for state in states])

            # Original data 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                original_3d[:, 0],
                original_3d[:, 1],
                original_3d[:, 2],
                c=agent_ages,
                cmap="viridis",
                alpha=0.6,
            )
            fig.colorbar(scatter, label="Agent Age")
            ax.set_title("3D t-SNE of Original Agent State Vectors")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.savefig(os.path.join(output_dir, "original_tsne_3d.png"))

            # Embedded data 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                embedded_3d[:, 0],
                embedded_3d[:, 1],
                embedded_3d[:, 2],
                c=agent_ages,
                cmap="viridis",
                alpha=0.6,
            )
            fig.colorbar(scatter, label="Agent Age")
            ax.set_title("3D t-SNE of Autoencoder Embeddings")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.savefig(os.path.join(output_dir, "embedded_tsne_3d.png"))
        else:
            # Simple 3D plots without coloring
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                original_3d[:, 0], original_3d[:, 1], original_3d[:, 2], alpha=0.6
            )
            ax.set_title("3D t-SNE of Original Agent State Vectors")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.savefig(os.path.join(output_dir, "original_tsne_3d.png"))

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                embedded_3d[:, 0], embedded_3d[:, 1], embedded_3d[:, 2], alpha=0.6
            )
            ax.set_title("3D t-SNE of Autoencoder Embeddings")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.savefig(os.path.join(output_dir, "embedded_tsne_3d.png"))

        logger.info("3D visualizations complete")
    except Exception as e:
        logger.warning(f"Could not create 3D visualizations: {e}")

    # Create interactive plots
    try:
        import plotly.express as px

        # Apply t-SNE to data if not already applied
        if "original_3d" not in locals():
            logger.info("Applying 3D t-SNE for interactive plots...")
            original_tsne = TSNE(n_components=3, random_state=42)
            original_3d = original_tsne.fit_transform(original_data)

            embedded_tsne = TSNE(n_components=3, random_state=42)
            embedded_3d = embedded_tsne.fit_transform(embedded_data)

        if states and "age" in states[0]:
            # Original data interactive plot
            fig = px.scatter_3d(
                x=original_3d[:, 0],
                y=original_3d[:, 1],
                z=original_3d[:, 2],
                color=agent_ages,
                labels={"color": "Agent Age"},
                title="3D t-SNE of Original Agent State Vectors",
                opacity=0.7,
            )
            fig.write_html(
                os.path.join(output_dir, "original_tsne_3d_interactive.html")
            )

            # Embedded data interactive plot
            fig = px.scatter_3d(
                x=embedded_3d[:, 0],
                y=embedded_3d[:, 1],
                z=embedded_3d[:, 2],
                color=agent_ages,
                labels={"color": "Agent Age"},
                title="3D t-SNE of Autoencoder Embeddings",
                opacity=0.7,
            )
            fig.write_html(
                os.path.join(output_dir, "embedded_tsne_3d_interactive.html")
            )
        else:
            # Simple interactive plots without coloring
            fig = px.scatter_3d(
                x=original_3d[:, 0],
                y=original_3d[:, 1],
                z=original_3d[:, 2],
                title="3D t-SNE of Original Agent State Vectors",
                opacity=0.7,
            )
            fig.write_html(
                os.path.join(output_dir, "original_tsne_3d_interactive.html")
            )

            fig = px.scatter_3d(
                x=embedded_3d[:, 0],
                y=embedded_3d[:, 1],
                z=embedded_3d[:, 2],
                title="3D t-SNE of Autoencoder Embeddings",
                opacity=0.7,
            )
            fig.write_html(
                os.path.join(output_dir, "embedded_tsne_3d_interactive.html")
            )

        logger.info("Interactive visualizations complete")
    except Exception as e:
        logger.warning(f"Could not create interactive visualizations: {e}")

    return original_data, embedded_data


def visualize_reconstructions(
    model: StateAutoencoder,
    dataset: AgentStateDataset,
    states: List[Dict[str, Any]],
    device: torch.device,
    output_dir: str,
    num_samples: int = 3,
):
    """Visualize original states alongside their reconstructions at different compression levels.

    Args:
        model: Trained autoencoder model
        dataset: Agent state dataset
        states: List of agent state dictionaries
        device: Device to compute on (CPU or GPU)
        output_dir: Directory to save output plots
        num_samples: Number of sample states to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select random samples
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for idx in sample_indices:
            # Get original vector
            vector = dataset[idx].unsqueeze(0).to(device)

            # Get state info for the plot title
            state = states[idx]
            state_id = f"ID: {state.get('id', 'unknown')}, Step: {state.get('step_number', 'unknown')}"

            # Get reconstructions at different levels
            stm_embedding = model.encode_stm(vector)
            stm_reconstructed = model.decode_stm(stm_embedding)

            im_embedding = model.encode_im(vector)
            im_reconstructed = model.decode_im(im_embedding)

            ltm_embedding = model.encode_ltm(vector)
            ltm_reconstructed = model.decode_ltm(ltm_embedding)

            # Move all tensors to CPU for visualization
            original = vector.detach().cpu().numpy().flatten()
            stm_reconstructed = stm_reconstructed.detach().cpu().numpy().flatten()
            im_reconstructed = im_reconstructed.detach().cpu().numpy().flatten()
            ltm_reconstructed = ltm_reconstructed.detach().cpu().numpy().flatten()

            # Create a plot
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

            # Use feature indices as x values
            feature_indices = np.arange(len(original))

            # Plot original state
            axes[0].bar(feature_indices, original, alpha=0.7)
            axes[0].set_title(f"Original State Vector - {state_id}")
            axes[0].set_ylabel("Feature Value")

            # Plot STM reconstruction
            axes[1].bar(feature_indices, stm_reconstructed, alpha=0.7, color="green")
            axes[1].set_title(
                f"STM Reconstruction (dim={model.stm_bottleneck.out_features})"
            )
            axes[1].set_ylabel("Feature Value")

            # Plot IM reconstruction
            axes[2].bar(feature_indices, im_reconstructed, alpha=0.7, color="orange")
            axes[2].set_title(
                f"IM Reconstruction (dim={model.im_bottleneck.out_features})"
            )
            axes[2].set_ylabel("Feature Value")

            # Plot LTM reconstruction
            axes[3].bar(feature_indices, ltm_reconstructed, alpha=0.7, color="red")
            axes[3].set_title(
                f"LTM Reconstruction (dim={model.ltm_bottleneck.out_features})"
            )
            axes[3].set_ylabel("Feature Value")
            axes[3].set_xlabel("Feature Index")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"reconstruction_sample_{idx}.png"))
            plt.close()

        # Also create a visualization showing embedding vectors at each level
        for idx in sample_indices:
            vector = dataset[idx].unsqueeze(0).to(device)
            state_id = f"ID: {states[idx].get('id', 'unknown')}, Step: {states[idx].get('step_number', 'unknown')}"

            # Get embeddings at different levels
            stm_embedding = model.encode_stm(vector).detach().cpu().numpy().flatten()
            im_embedding = model.encode_im(vector).detach().cpu().numpy().flatten()
            ltm_embedding = model.encode_ltm(vector).detach().cpu().numpy().flatten()

            # Create a plot for embeddings
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))

            # Plot STM embedding
            axes[0].bar(
                range(len(stm_embedding)), stm_embedding, alpha=0.7, color="green"
            )
            axes[0].set_title(f"STM Embedding (dim={len(stm_embedding)}) - {state_id}")
            axes[0].set_ylabel("Value")

            # Plot IM embedding
            axes[1].bar(
                range(len(im_embedding)), im_embedding, alpha=0.7, color="orange"
            )
            axes[1].set_title(f"IM Embedding (dim={len(im_embedding)})")
            axes[1].set_ylabel("Value")

            # Plot LTM embedding
            axes[2].bar(
                range(len(ltm_embedding)), ltm_embedding, alpha=0.7, color="red"
            )
            axes[2].set_title(f"LTM Embedding (dim={len(ltm_embedding)})")
            axes[2].set_ylabel("Value")
            axes[2].set_xlabel("Embedding Dimension")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"embeddings_sample_{idx}.png"))
            plt.close()

    # Move model back to CPU when done
    model = model.to("cpu")

    logger.info(f"Reconstruction visualizations saved to {output_dir}")


def save_model(model: StateAutoencoder, output_dir: str):
    """Save the trained autoencoder model.

    Args:
        model: Trained autoencoder model
        output_dir: Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "agent_state_autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def visualize_encoding_decoding_process(
    model: StateAutoencoder,
    dataset: AgentStateDataset,
    states: List[Dict[str, Any]],
    device: torch.device,
    output_dir: str,
    num_samples: int = 3,
    memory_level: str = "stm",  # 'stm', 'im', or 'ltm'
):
    """Visualize the full encoding-decoding process for agent states.
    
    Shows the original state, encoded vector representation, and reconstructed state.
    
    Args:
        model: Trained autoencoder model
        dataset: Agent state dataset
        states: List of agent state dictionaries
        device: Device to compute on (CPU or GPU)
        output_dir: Directory to save output plots
        num_samples: Number of sample states to visualize
        memory_level: Memory level to use for encoding/decoding ('stm', 'im', or 'ltm')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )
    
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for idx in sample_indices:
            # Get original vector
            vector = dataset[idx].unsqueeze(0).to(device)
            
            # Get state info for the plot title
            state = states[idx]
            state_id = f"ID: {state.get('id', 'unknown')}, Step: {state.get('step_number', 'unknown')}"
            
            # Encode and decode based on memory level
            if memory_level == "stm":
                encoded_vector = model.encode_stm(vector)
                reconstructed = model.decode_stm(encoded_vector)
                memory_name = "Short-Term Memory"
                bottleneck_dim = model.stm_bottleneck.out_features
            elif memory_level == "im":
                encoded_vector = model.encode_im(vector)
                reconstructed = model.decode_im(encoded_vector)
                memory_name = "Intermediate Memory"
                bottleneck_dim = model.im_bottleneck.out_features
            elif memory_level == "ltm":
                encoded_vector = model.encode_ltm(vector)
                reconstructed = model.decode_ltm(encoded_vector)
                memory_name = "Long-Term Memory"
                bottleneck_dim = model.ltm_bottleneck.out_features
            else:
                raise ValueError(f"Invalid memory level: {memory_level}")
            
            # Convert tensors to numpy arrays for visualization
            original = vector.detach().cpu().numpy().flatten()
            encoded = encoded_vector.detach().cpu().numpy().flatten()
            reconstructed = reconstructed.detach().cpu().numpy().flatten()
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex='col')
            
            # Plot original state
            axes[0].bar(range(len(original)), original, alpha=0.7, color='blue')
            axes[0].set_title(f"Original State Vector - {state_id}")
            axes[0].set_ylabel("Feature Value")
            
            # Plot encoded vector (embedding)
            axes[1].bar(range(len(encoded)), encoded, alpha=0.7, color='green')
            axes[1].set_title(f"Encoded Vector ({memory_name}, dim={bottleneck_dim})")
            axes[1].set_ylabel("Embedding Value")
            
            # Plot reconstructed state
            axes[2].bar(range(len(reconstructed)), reconstructed, alpha=0.7, color='orange')
            axes[2].set_title(f"Reconstructed State Vector")
            axes[2].set_ylabel("Feature Value")
            axes[2].set_xlabel("Feature Index")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"encoding_decoding_process_{memory_level}_{idx}.png"))
            plt.close()
            
            # Calculate and display reconstruction error metrics
            mse = mean_squared_error(original, reconstructed)
            logger.info(f"Sample {idx} ({memory_level}) - Reconstruction MSE: {mse:.6f}")
            
            # Save raw data for later analysis if needed
            np.savez(
                os.path.join(output_dir, f"encoding_decoding_data_{memory_level}_{idx}.npz"),
                original=original,
                encoded=encoded,
                reconstructed=reconstructed
            )
    
    # Move model back to CPU when done
    model = model.to("cpu")
    
    logger.info(f"Encoding-decoding process visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment with autoencoder embeddings for agent states"
    )
    parser.add_argument(
        "--db_path", type=str, default="simulation.db", help="Path to simulation.db"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of agent states to load"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Number of random samples to use for training (default: use all loaded samples)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--stm_dim", type=int, default=32, help="Dimension for STM embeddings"
    )
    parser.add_argument(
        "--im_dim", type=int, default=16, help="Dimension for IM embeddings"
    )
    parser.add_argument(
        "--ltm_dim", type=int, default=8, help="Dimension for LTM embeddings"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--num_reconstruction_samples",
        type=int,
        default=3,
        help="Number of sample reconstructions to visualize",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument("--skip_3d", action="store_true", help="Skip 3D visualizations")
    parser.add_argument(
        "--validation_split", type=float, default=0.2, help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=10, help="Number of epochs to wait before early stopping"
    )
    parser.add_argument(
        "--k_fold", type=int, default=0, help="Number of folds for k-fold cross-validation (0 to disable)"
    )

    args = parser.parse_args()

    # Connect to database
    logger.info(f"Connecting to database: {args.db_path}")
    conn = connect_to_db(args.db_path)

    # Load agent states
    logger.info("Loading agent states...")
    states = load_agent_states(conn, args.limit)
    logger.info(f"Loaded {len(states)} agent states")

    if not states:
        logger.error("No agent states found in the database")
        return

    # If train_samples is specified, select a random subset of states for training
    if args.train_samples is not None and args.train_samples < len(states):
        import random

        logger.info(f"Selecting {args.train_samples} random samples for training")
        # Set a seed for reproducibility
        random.seed(42)
        training_states = random.sample(states, args.train_samples)
        logger.info(f"Selected {len(training_states)} random samples for training")
    else:
        training_states = states
        logger.info(f"Using all {len(training_states)} samples for training")

    # Prepare autoencoder
    logger.info("Preparing autoencoder...")
    engine, dataset, device = prepare_autoencoder(
        training_states, stm_dim=args.stm_dim, im_dim=args.im_dim, ltm_dim=args.ltm_dim
    )

    # Override device if CPU is requested
    if args.cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as requested")

    # Train and evaluate with either standard training or k-fold cross-validation
    if args.k_fold > 1:
        logger.info(f"Training autoencoder with {args.k_fold}-fold cross-validation...")
        # Use k-fold cross-validation
        # Make sure we have the correct input dimension
        input_dim = engine.input_dim
        
        # Convert dataset to states format for engine's train method
        states_for_kfold = []
        for i in range(len(dataset)):
            # Create a dummy state dictionary with a vector field
            # Ensure vector has the right dimension for the model
            vector = dataset[i].detach().cpu().numpy()
            if len(vector) < input_dim:
                vector = np.pad(vector, (0, input_dim - len(vector)))
            elif len(vector) > input_dim:
                vector = vector[:input_dim]
            
            state = {"vector": vector}
            states_for_kfold.append(state)
        
        logger.info(f"Input vector dimension: {input_dim}")
        
        cv_results = engine.train_with_kfold(
            states=states_for_kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_folds=args.k_fold,
            early_stopping_patience=args.early_stopping_patience
        )
        
        # Create visualization of fold results
        try:
            # Create bar chart of validation loss by fold
            plt.figure(figsize=(10, 6))
            fold_numbers = [f"Fold {i+1}" for i in range(args.k_fold)]
            plt.bar(fold_numbers, cv_results["fold_results"]["fold_val_loss"])
            plt.axhline(y=cv_results["avg_val_loss"], color='r', linestyle='-', label=f"Average: {cv_results['avg_val_loss']:.4f}")
            plt.title("Validation Loss by Fold")
            plt.ylabel("Validation Loss")
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "kfold_val_loss.png"))
            plt.close()
            
            # Create bar chart of R² scores by fold
            plt.figure(figsize=(10, 6))
            x = np.arange(args.k_fold)
            width = 0.25
            
            plt.bar(x - width, cv_results["fold_results"]["fold_val_stm_r2"], width, label='STM R²')
            plt.bar(x, cv_results["fold_results"]["fold_val_im_r2"], width, label='IM R²')
            plt.bar(x + width, cv_results["fold_results"]["fold_val_ltm_r2"], width, label='LTM R²')
            
            plt.xlabel('Fold')
            plt.ylabel('R² Score')
            plt.title('R² Scores by Fold and Memory Level')
            plt.xticks(x, fold_numbers)
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "kfold_r2_scores.png"))
            plt.close()
            
            logger.info(f"K-fold cross-validation plots saved to {args.output_dir}")
        except Exception as e:
            logger.warning(f"Could not generate k-fold plots: {e}")
        
        # Use model from best fold (already loaded in engine)
        model_on_device = engine.model.to(device)
        
        # We need to extract embeddings for later visualization
        # since we're not using the standard train_and_evaluate function
        embedded_data = []
        original_data = []
        
        model_on_device.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get original vector
                vector = dataset[i].unsqueeze(0).to(device)
                original_vector = vector.detach().cpu().numpy().flatten()
                original_data.append(original_vector)
                
                # Get STM embedding
                stm_embedding = model_on_device.encode_stm(vector)
                embedded_data.append(stm_embedding.detach().cpu().numpy().flatten())
        
        original_data = np.array(original_data)
        embedded_data = np.array(embedded_data)
        
        # Set metrics for compatibility with downstream code
        metrics = {
            "cv_results": cv_results,
            "train_loss": [cv_results["avg_val_loss"]],  # Use average validation loss as placeholder
            "val_loss": [cv_results["best_val_loss"]],
            "val_stm_r2": [cv_results["avg_stm_r2"]],
            "val_im_r2": [cv_results["avg_im_r2"]],
            "val_ltm_r2": [cv_results["avg_ltm_r2"]]
        }
        
        logger.info(f"Cross-validation completed. Best model from fold {cv_results['best_fold']} selected.")
    else:
        # Use standard train and validate
        logger.info("Training autoencoder with validation...")
        metrics, original_data, embedded_data, model_on_device = train_and_evaluate(
            engine, 
            dataset, 
            device, 
            args.epochs, 
            args.batch_size,
            args.validation_split,
            args.early_stopping_patience
        )
    
    # Log training results (for both standard and k-fold)
    if 'train_loss' in metrics and 'val_loss' in metrics:
        logger.info("Final training loss: {:.6f}".format(metrics['train_loss'][-1]))
        logger.info("Final validation loss: {:.6f}".format(metrics['val_loss'][-1]))
        
        # Plot training and validation loss curves
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['train_loss'], label='Training Loss')
            plt.plot(metrics['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, "training_validation_loss.png"))
            plt.close()
            logger.info(f"Loss curves saved to {os.path.join(args.output_dir, 'training_validation_loss.png')}")
        except Exception as e:
            logger.warning(f"Could not generate loss curve plot: {e}")
            
        # Plot R² metrics if available
        if 'val_stm_r2' in metrics and 'val_im_r2' in metrics and 'val_ltm_r2' in metrics:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['val_stm_r2'], label='STM R²')
                plt.plot(metrics['val_im_r2'], label='IM R²') 
                plt.plot(metrics['val_ltm_r2'], label='LTM R²')
                plt.title('Validation R² Scores by Memory Level')
                plt.xlabel('Epoch')
                plt.ylabel('R² Score')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, "validation_r2_scores.png"))
                plt.close()
                logger.info(f"R² score curves saved to {os.path.join(args.output_dir, 'validation_r2_scores.png')}")
                
                # Also create a bar chart of final R² scores
                best_epoch = np.argmin(metrics['val_loss'])
                r2_scores = [
                    metrics['val_stm_r2'][best_epoch], 
                    metrics['val_im_r2'][best_epoch], 
                    metrics['val_ltm_r2'][best_epoch]
                ]
                
                plt.figure(figsize=(8, 6))
                bars = plt.bar(['STM', 'IM', 'LTM'], r2_scores, color=['green', 'orange', 'red'])
                plt.title('R² Scores at Different Compression Levels')
                plt.ylabel('R² Score')
                plt.ylim(0, 1.0)  # R² is typically between 0 and 1
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "r2_comparison.png"))
                plt.close()
                logger.info(f"R² comparison saved to {os.path.join(args.output_dir, 'r2_comparison.png')}")
            except Exception as e:
                logger.warning(f"Could not generate R² plots: {e}")

    # Create a dataset with all states for evaluation and visualization
    full_dataset = AgentStateDataset(states)

    # Log detailed device information
    logger.info(
        f"Model device before explicit move: {next(model_on_device.parameters()).device}"
    )
    model_on_device = model_on_device.to(device)
    logger.info(
        f"Model device after explicit move: {next(model_on_device.parameters()).device}"
    )
    logger.info(f"Target device: {device}")

    # Evaluate reconstruction quality using model on device
    logger.info("Evaluating reconstruction quality...")
    reconstruction_metrics = evaluate_reconstruction(
        model_on_device, full_dataset, device
    )

    logger.info("Reconstruction Mean Squared Errors:")
    logger.info(f"  STM: {reconstruction_metrics['stm_mse']:.6f}")
    logger.info(f"  IM: {reconstruction_metrics['im_mse']:.6f}")
    logger.info(f"  LTM: {reconstruction_metrics['ltm_mse']:.6f}")

    # Use the new integrated visualization function
    original_data, embedded_data = visualize_embeddings_modified(
        model_on_device, full_dataset, states, device, args.output_dir
    )

    # Skip the original visualization calls
    # visualize_embeddings(...)
    # visualize_embeddings_3d(...)

    # Visualize example reconstructions
    logger.info("Visualizing example reconstructions...")
    visualize_reconstructions(
        model_on_device,
        full_dataset,
        states,
        device,
        args.output_dir,
        num_samples=args.num_reconstruction_samples,
    )

    # Visualize the encoding-decoding process for each memory level
    logger.info("Visualizing encoding-decoding process...")
    for memory_level in ["stm", "im", "ltm"]:
        visualize_encoding_decoding_process(
            model_on_device,
            full_dataset,
            states,
            device,
            args.output_dir,
            num_samples=args.num_reconstruction_samples,
            memory_level=memory_level
        )

    # Save model (from engine, which has CPU version)
    save_model(engine.model, args.output_dir)

    # Close database connection
    conn.close()
    logger.info("Experiment completed")


if __name__ == "__main__":
    main()
