# Agent State Autoencoder Embedding Experiment

This script allows you to experiment with autoencoder-based embeddings for agent states from a simulation database.

## Overview

The autoencoder model compresses agent state vectors at three different levels:
- **STM (Short-Term Memory)**: Moderate compression while preserving most details
- **IM (Intermediate Memory)**: Higher compression with some loss of detail
- **LTM (Long-Term Memory)**: Maximum compression, retaining only essential information

The experiment:
1. Loads agent states from a simulation database
2. Trains an autoencoder on the state vectors
3. Evaluates reconstruction quality at each compression level
4. Generates t-SNE visualizations of the embeddings
5. Saves the trained model for future use

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- SQLite3

## Usage

```bash
python agent_state_embedding_experiment.py --db_path path/to/simulation.db
```

### Command Line Arguments

- `--db_path`: Path to the simulation database (required)
- `--limit`: Maximum number of agent states to load (default: all)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--stm_dim`: Dimension for STM embeddings (default: 32)
- `--im_dim`: Dimension for IM embeddings (default: 16)
- `--ltm_dim`: Dimension for LTM embeddings (default: 8)
- `--output_dir`: Directory to save results (default: 'results')

## Output

The script produces:
1. Training metrics and reconstruction quality measurements (console output)
2. T-SNE visualizations of original and embedded data (saved as PNG files)
3. Trained autoencoder model (saved as a PyTorch file)

## Example

```bash
# Train with custom dimensions and 200 epochs
python agent_state_embedding_experiment.py --db_path simulation.db --stm_dim 64 --im_dim 32 --ltm_dim 16 --epochs 200 --output_dir my_results
```

## Understanding the Results

- **Lower MSE (Mean Squared Error)** indicates better reconstruction quality
- Look for clear patterns in the t-SNE visualizations, which suggest the embedding is preserving important structure
- Compare the MSE across different compression levels to understand the trade-off between compression and information loss 