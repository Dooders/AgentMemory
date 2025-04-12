#!/usr/bin/env python
"""
Script to train a custom embedding model using knowledge distillation from sentence-transformers.

This script collects agent state data, processes it through the object_to_text function,
and uses the sentence-transformers model to generate target embeddings for training
our custom lightweight model.
"""

import os
import sqlite3
import logging
import argparse
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Import the TextEmbeddingEngine for generating teacher embeddings
from memory.embeddings.text_embeddings import TextEmbeddingEngine
# Import our custom model
from memory.embeddings.custom_embeddings import CustomEmbeddingEngine
# Import utility functions
from memory.embeddings.utils import object_to_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(db_path: str, limit: int = 5000) -> List[Dict[str, Any]]:
    """
    Load agent state data from the simulation database.
    
    Args:
        db_path: Path to the SQLite database
        limit: Maximum number of states to load
        
    Returns:
        List of agent state dictionaries
    """
    logger.info(f"Loading data from {db_path}...")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get unique agent_ids
    cursor.execute('SELECT DISTINCT agent_id FROM agent_states LIMIT 100')
    agent_ids = [row['agent_id'] for row in cursor.fetchall()]
    
    # Prepare the query with placeholders
    placeholders = ', '.join(['?' for _ in agent_ids])
    query = f"""
    SELECT * FROM agent_states 
    WHERE agent_id IN ({placeholders})
    ORDER BY agent_id, step_number
    LIMIT {limit}
    """
    
    # Execute the query with agent_ids as parameters
    cursor.execute(query, agent_ids)
    
    # Convert to dictionaries
    states = []
    for row in cursor.fetchall():
        state_dict = {k: row[k] for k in row.keys()}
        states.append(state_dict)
    
    conn.close()
    logger.info(f"Loaded {len(states)} states from {len(agent_ids)} agents")
    return states


def convert_states_to_text(states: List[Dict[str, Any]]) -> List[str]:
    """
    Convert agent states to text representations using object_to_text.
    
    Args:
        states: List of agent state dictionaries
        
    Returns:
        List of text representations
    """
    logger.info(f"Converting {len(states)} states to text...")
    texts = []
    
    for state in states:
        # Convert state to text
        text = object_to_text(state)
        texts.append(text)
    
    return texts


def generate_teacher_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate embeddings using the teacher model (sentence-transformers).
    
    Args:
        texts: List of text representations
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        NumPy array of embeddings
    """
    logger.info(f"Generating embeddings with teacher model ({model_name})...")
    
    # Initialize the teacher model
    teacher = TextEmbeddingEngine(model_name=model_name)
    
    # Generate embeddings
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = [teacher.model.encode(text) for text in batch_texts]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def train_custom_model(
    texts: List[str],
    teacher_embeddings: np.ndarray,
    model_path: str,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float
) -> None:
    """
    Train the custom embedding model using knowledge distillation.
    
    Args:
        texts: List of text representations
        teacher_embeddings: Embeddings from the teacher model
        model_path: Path to save the trained model
        embedding_dim: Dimension of the embeddings
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    """
    logger.info(f"Training custom model with embedding dimension {embedding_dim}...")
    
    # Initialize the custom model
    custom_model = CustomEmbeddingEngine(
        model_path=model_path,
        embedding_dim=embedding_dim
    )
    
    # Train the model
    custom_model.train_from_teacher(
        texts=texts,
        teacher_embeddings=teacher_embeddings,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    logger.info(f"Model trained and saved to {model_path}")


def evaluate_model(
    teacher: TextEmbeddingEngine,
    custom_model: CustomEmbeddingEngine,
    test_texts: List[str]
) -> Dict[str, float]:
    """
    Evaluate the custom model against the teacher model.
    
    Args:
        teacher: Teacher model (sentence-transformers)
        custom_model: Custom model to evaluate
        test_texts: List of test texts
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating model on {len(test_texts)} test samples...")
    
    # Generate embeddings with both models
    teacher_embeddings = []
    custom_embeddings = []
    
    for text in tqdm(test_texts):
        teacher_emb = teacher.model.encode(text)
        custom_emb = custom_model._encode_text(text)
        
        teacher_embeddings.append(teacher_emb)
        custom_embeddings.append(custom_emb)
    
    teacher_embeddings = np.array(teacher_embeddings)
    custom_embeddings = np.array(custom_embeddings)
    
    # Calculate cosine similarities
    similarities = []
    for i in range(len(test_texts)):
        teacher_emb = teacher_embeddings[i]
        custom_emb = custom_embeddings[i]
        
        # Normalize vectors
        teacher_norm = teacher_emb / np.linalg.norm(teacher_emb)
        custom_norm = custom_emb / np.linalg.norm(custom_emb)
        
        # Calculate cosine similarity
        similarity = np.dot(teacher_norm, custom_norm)
        similarities.append(similarity)
    
    # Calculate metrics
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    median_similarity = np.median(similarities)
    
    return {
        "average_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "median_similarity": median_similarity
    }


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a custom embedding model using knowledge distillation"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/simulation.db",
        help="Path to the SQLite database"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/custom_embedding",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Dimension of the embeddings"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the sentence-transformers model to use as teacher"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of states to load"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    try:
        # Load data
        states = load_data(args.db_path, limit=args.limit)
        
        # Convert states to text
        texts = convert_states_to_text(states)
        
        # Split data into train and evaluation sets
        split_idx = int(len(texts) * (1 - args.eval_split))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        # Generate teacher embeddings for training
        teacher_embeddings = generate_teacher_embeddings(train_texts, args.teacher_model)
        
        # Train custom model
        start_time = time.time()
        train_custom_model(
            texts=train_texts,
            teacher_embeddings=teacher_embeddings,
            model_path=args.model_path,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        teacher = TextEmbeddingEngine(model_name=args.teacher_model)
        custom_model = CustomEmbeddingEngine(
            model_path=args.model_path,
            embedding_dim=args.embedding_dim
        )
        
        metrics = evaluate_model(teacher, custom_model, eval_texts)
        
        # Log evaluation results
        logger.info("Evaluation results:")
        logger.info(f"  Average similarity: {metrics['average_similarity']:.4f}")
        logger.info(f"  Minimum similarity: {metrics['min_similarity']:.4f}")
        logger.info(f"  Median similarity: {metrics['median_similarity']:.4f}")
        
        # Calculate and log model size
        model_size_mb = os.path.getsize(f"{args.model_path}_model.pt") / (1024 * 1024)
        tokenizer_size_mb = os.path.getsize(f"{args.model_path}_tokenizer.pkl") / (1024 * 1024)
        total_size_mb = model_size_mb + tokenizer_size_mb
        
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        logger.info(f"Tokenizer size: {tokenizer_size_mb:.2f} MB")
        logger.info(f"Total size: {total_size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 