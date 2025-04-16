#!/usr/bin/env python
"""
Script to convert vector store JSON files to FAISS database format.
This allows converting previously generated JSON embeddings to the more efficient FAISS format.
"""

import argparse
import json
import logging
import os
import numpy as np

# Import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    raise ImportError("FAISS is not available. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the FaissVectorIndex from the original script
from scripts.embed_agent_states import FaissVectorIndex

def load_json_vector_store(json_filepath):
    """Load vectors and metadata from a JSON vector store file.
    
    Args:
        json_filepath: Path to the JSON vector store file
        
    Returns:
        Tuple of (vectors, metadata, dimension)
    """
    logger.info(f"Loading vector store from {json_filepath}")
    
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    vectors = data.get("vectors", {})
    metadata = data.get("metadata", {})
    dimension = data.get("dimension", 384)
    model = data.get("model", "unknown")
    
    logger.info(f"Loaded {len(vectors)} vectors with dimension {dimension} from model {model}")
    
    return vectors, metadata, dimension

def convert_json_to_faiss(json_filepath, output_filepath, metric="cosine"):
    """Convert a JSON vector store to FAISS format.
    
    Args:
        json_filepath: Path to the JSON vector store file
        output_filepath: Path to save the FAISS index (without extension)
        metric: Distance metric ('cosine', 'l2')
        
    Returns:
        True if successful
    """
    # Load vectors and metadata from JSON
    vectors_dict, metadata_dict, dimension = load_json_vector_store(json_filepath)
    
    # Create FAISS index
    faiss_index = FaissVectorIndex(dimension=dimension, metric=metric)
    
    # Add vectors to FAISS index
    count = 0
    for id, vector in vectors_dict.items():
        metadata = metadata_dict.get(id, {})
        success = faiss_index.add(id=id, vector=vector, metadata=metadata)
        if success:
            count += 1
    
    logger.info(f"Added {count} vectors to FAISS index")
    
    # Save FAISS index
    success = faiss_index.save(output_filepath)
    if success:
        logger.info(f"Successfully saved FAISS index to {output_filepath}")
    else:
        logger.error(f"Failed to save FAISS index to {output_filepath}")
    
    return success

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert vector store JSON files to FAISS database format"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="memory/embeddings/data/agent_state_embeddings.json",
        help="Path to the input JSON vector store file"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="faiss_index",
        help="Path to save the FAISS index (without extension)"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Distance metric to use (cosine or l2)"
    )
    
    args = parser.parse_args()
    
    # If output_file is not specified, use the input_file path with a different extension
    if not args.output_file:
        base_path = os.path.splitext(args.input_file)[0]
        args.output_file = base_path
    
    # Convert JSON to FAISS
    convert_json_to_faiss(args.input_file, args.output_file, args.metric)

if __name__ == "__main__":
    main() 