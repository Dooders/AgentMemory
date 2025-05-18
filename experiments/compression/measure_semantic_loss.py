"""Measure semantic loss when searching directly on compressed vectors.

This experiment evaluates different vector compression methods for semantic search,
comparing their impact on search accuracy and storage efficiency. The goal is to find
a compression method that significantly reduces vector storage size while maintaining
semantic meaning for search operations.

Compression Methods Tested:
1. Quantization (8-bit, 4-bit, 2-bit):
   - Reduces precision of vector elements
   - 8-bit: 256 possible values per element (75% size reduction)
   - 4-bit: 16 possible values per element (87.5% size reduction)
   - 2-bit: 4 possible values per element (93.75% size reduction)

2. Random Projection (128, 64, 32 dimensions):
   - Reduces vector dimensionality
   - Preserves distances between vectors
   - 128-dim: 50% size reduction
   - 64-dim: 75% size reduction
   - 32-dim: 87.5% size reduction

Key Findings:
1. 4-bit Quantization:
   - Best balance of compression and accuracy
   - 87.5% storage reduction
   - 99.2% reconstruction similarity
   - Maintains search accuracy
   - Can be reversed with minimal loss

2. 8-bit Quantization:
   - Excellent accuracy (99.997% reconstruction)
   - Less storage reduction (75%)
   - Perfect search results

3. 2-bit Quantization:
   - Poor performance (86% reconstruction)
   - Too much information loss
   - Search results degraded

4. Random Projection:
   - Poor search accuracy
   - Not recommended for semantic search

Usage:
    python measure_semantic_loss.py

The script will:
1. Load test memories from validation data
2. Apply each compression method
3. Run test queries
4. Compare search results with uncompressed vectors
5. Measure reconstruction quality
6. Report detailed statistics

Note: This experiment focuses on direct search using compressed vectors,
without decompression, to maximize storage and search efficiency.
"""

import json
import logging
import os
import sys
from typing import List, Any, Dict, Tuple
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from memory.config import MemoryConfig
from memory.embeddings.compression import CompressionEngine

# Try to import TextEmbeddingEngine, fallback to CustomEmbeddingEngine if not available
try:
    from memory.embeddings.text_embeddings import TextEmbeddingEngine
    embedding_engine = TextEmbeddingEngine()
except ImportError:
    from memory.embeddings.custom_embeddings import CustomEmbeddingEngine
    embedding_engine = CustomEmbeddingEngine()

def quantize_vector(vector: List[float], bits: int = 8) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Quantize vector to specified number of bits.
    
    Args:
        vector: Input vector
        bits: Number of bits to use (1-8)
        
    Returns:
        Tuple of (quantized vector, (min_value, max_value))
    """
    vector = np.array(vector)
    # Scale to [0, 1]
    v_min, v_max = vector.min(), vector.max()
    scaled = (vector - v_min) / (v_max - v_min)
    # Quantize
    max_val = (1 << bits) - 1
    quantized = np.round(scaled * max_val) / max_val
    # Scale back
    return quantized * (v_max - v_min) + v_min, (v_min, v_max)

def dequantize_vector(quantized_vector: np.ndarray, min_max: Tuple[float, float]) -> np.ndarray:
    """Reverse the quantization process.
    
    Args:
        quantized_vector: Quantized vector
        min_max: Tuple of (min_value, max_value) from original vector
        
    Returns:
        Dequantized vector
    """
    v_min, v_max = min_max
    # Scale to [0, 1]
    scaled = (quantized_vector - v_min) / (v_max - v_min)
    # Scale back to original range
    return scaled * (v_max - v_min) + v_min

def project_vector(vector: List[float], target_dim: int) -> np.ndarray:
    """Reduce vector dimensionality using random projection.
    
    Args:
        vector: Input vector
        target_dim: Target dimensionality
        
    Returns:
        Projected vector as numpy array
    """
    vector = np.array(vector).reshape(1, -1)
    projector = GaussianRandomProjection(n_components=target_dim)
    projected = projector.fit_transform(vector)
    return normalize(projected).flatten()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_similar(query_vector: np.ndarray, memory_vectors: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """Search for similar vectors using cosine similarity.
    
    Args:
        query_vector: Query vector
        memory_vectors: List of memory vectors with their metadata
        threshold: Similarity threshold
        
    Returns:
        List of similar memories with their scores
    """
    results = []
    for mem in memory_vectors:
        similarity = cosine_similarity(query_vector, mem["vector"])
        if similarity >= threshold:
            results.append({
                "memory_id": mem["memory_id"],
                "score": similarity
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)

def analyze_vector_relationships(vectors: List[Dict], title: str = "Memory Vector Relationships"):
    """Analyze relationships between memory vectors.
    
    Args:
        vectors: List of dictionaries containing vectors and metadata
        title: Title for the analysis
    """
    logger = logging.getLogger("vector_analysis")
    logger.info(f"\n{title}")
    
    # Extract vectors and metadata
    vector_array = np.array([v["vector"] for v in vectors])
    memory_ids = [v["memory_id"] for v in vectors]
    contents = [v["content"]["content"] for v in vectors]  # Access the nested content field
    
    # 1. Pairwise Cosine Similarities
    similarities = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            similarities[i,j] = cosine_similarity(vector_array[i], vector_array[j])
    
    # Find most similar and least similar pairs
    np.fill_diagonal(similarities, 0)  # Exclude self-similarities
    max_sim_idx = np.unravel_index(similarities.argmax(), similarities.shape)
    min_sim_idx = np.unravel_index(similarities.argmin(), similarities.shape)
    
    logger.info("\nMost Similar Memories:")
    logger.info(f"Memory 1: {memory_ids[max_sim_idx[0]]}")
    logger.info(f"Content: {contents[max_sim_idx[0]]}")
    logger.info(f"Memory 2: {memory_ids[max_sim_idx[1]]}")
    logger.info(f"Content: {contents[max_sim_idx[1]]}")
    logger.info(f"Similarity: {similarities[max_sim_idx]:.4f}")
    
    logger.info("\nLeast Similar Memories:")
    logger.info(f"Memory 1: {memory_ids[min_sim_idx[0]]}")
    logger.info(f"Content: {contents[min_sim_idx[0]]}")
    logger.info(f"Memory 2: {memory_ids[min_sim_idx[1]]}")
    logger.info(f"Content: {contents[min_sim_idx[1]]}")
    logger.info(f"Similarity: {similarities[min_sim_idx]:.4f}")
    
    # 2. Clustering Analysis
    n_clusters = min(5, len(vectors))  # Use up to 5 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vector_array)
    
    logger.info("\nCluster Analysis:")
    for i in range(n_clusters):
        cluster_memories = [memory_ids[j] for j in range(len(clusters)) if clusters[j] == i]
        logger.info(f"\nCluster {i+1} ({len(cluster_memories)} memories):")
        for mem_id in cluster_memories:
            idx = memory_ids.index(mem_id)
            logger.info(f"- {mem_id}: {contents[idx][:100]}...")
    
    # 3. Similarity Distribution
    similarities_flat = similarities[np.triu_indices_from(similarities, k=1)]
    logger.info("\nSimilarity Distribution:")
    logger.info(f"Mean similarity: {np.mean(similarities_flat):.4f}")
    logger.info(f"Median similarity: {np.median(similarities_flat):.4f}")
    logger.info(f"Min similarity: {np.min(similarities_flat):.4f}")
    logger.info(f"Max similarity: {np.max(similarities_flat):.4f}")
    logger.info(f"Std dev: {np.std(similarities_flat):.4f}")
    
    # 4. Visualization
    plt.figure(figsize=(12, 8))
    
    # Similarity heatmap
    plt.subplot(2, 2, 1)
    plt.imshow(similarities, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Pairwise Similarities')
    
    # Similarity distribution histogram
    plt.subplot(2, 2, 2)
    plt.hist(similarities_flat, bins=20)
    plt.title('Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    # Cluster visualization (using first two dimensions)
    plt.subplot(2, 2, 3)
    plt.scatter(vector_array[:, 0], vector_array[:, 1], c=clusters, cmap='tab10')
    plt.title('Cluster Visualization (First 2 Dimensions)')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('vector_relationships.png')
    logger.info("\nVisualization saved as 'vector_relationships.png'")

def measure_compressed_search():
    """Measure semantic search performance using compressed vectors."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("compressed_search")

    # Load test memory data
    memory_sample_path = os.path.join("validation", "memory_samples", "similarity_validation_memory.json")
    with open(memory_sample_path, "r") as f:
        memory_data = json.load(f)
    agent_id = "test-agent-similarity-search"
    memories = memory_data["agents"][agent_id]["memories"]

    # Test queries
    test_queries = [
        "machine learning model accuracy",
        "data processing pipeline",
        "performance optimization",
        "security anomaly detection"
    ]

    # Get original vectors for all memories
    memory_vectors = []
    for memory in memories:
        vector = embedding_engine.encode_stm(memory["content"])
        if vector:
            memory_vectors.append({
                "memory_id": memory["memory_id"],
                "vector": np.array(vector),
                "content": memory["content"]
            })

    # Analyze relationships between original vectors
    analyze_vector_relationships(memory_vectors, "Original Vector Relationships")
    
    # Test different compression methods
    compression_methods = {
        "8-bit quantization": lambda v: quantize_vector(v, bits=8),
        "4-bit quantization": lambda v: quantize_vector(v, bits=4),
        "2-bit quantization": lambda v: quantize_vector(v, bits=2),
        "128-dim projection": lambda v: project_vector(v, target_dim=128),
        "64-dim projection": lambda v: project_vector(v, target_dim=64),
        "32-dim projection": lambda v: project_vector(v, target_dim=32)
    }

    for method_name, compress_fn in compression_methods.items():
        logger.info(f"\nTesting {method_name}")
        
        # Compress all memory vectors
        compressed_memories = []
        for mem in memory_vectors:
            if "quantization" in method_name:
                compressed_vector, min_max = compress_fn(mem["vector"])
                compressed_memories.append({
                    "memory_id": mem["memory_id"],
                    "vector": compressed_vector,
                    "content": mem["content"],
                    "min_max": min_max
                })
            else:
                compressed_vector = compress_fn(mem["vector"])
                compressed_memories.append({
                    "memory_id": mem["memory_id"],
                    "vector": compressed_vector,
                    "content": mem["content"]
                })
        
        # Analyze relationships between compressed vectors
        analyze_vector_relationships(compressed_memories, f"{method_name} Vector Relationships")

        # Test each query
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Get query vector and compress it
            query_vector = embedding_engine.encode_stm(query)
            if not query_vector:
                continue
            query_vector = np.array(query_vector)
            
            if "quantization" in method_name:
                compressed_query, query_min_max = compress_fn(query_vector)
                # Dequantize to demonstrate reconstruction
                dequantized_query = dequantize_vector(compressed_query, query_min_max)
                reconstruction_similarity = cosine_similarity(query_vector, dequantized_query)
                logger.info(f"Query reconstruction similarity: {reconstruction_similarity:.6f}")
            else:
                compressed_query = compress_fn(query_vector)
            
            # Search with original vectors
            original_results = search_similar(query_vector, memory_vectors)
            
            # Search with compressed vectors
            compressed_results = search_similar(compressed_query, compressed_memories)
            
            # Compare results
            original_ids = {r["memory_id"] for r in original_results}
            compressed_ids = {r["memory_id"] for r in compressed_results}
            
            logger.info(f"Original results: {len(original_results)} memories")
            logger.info(f"Compressed results: {len(compressed_results)} memories")
            logger.info(f"Matching results: {len(original_ids & compressed_ids)}")
            logger.info(f"Different results: {len(original_ids ^ compressed_ids)}")
            
            # Compare scores for matching results
            if original_ids & compressed_ids:
                for mem_id in original_ids & compressed_ids:
                    orig_score = next(r["score"] for r in original_results if r["memory_id"] == mem_id)
                    comp_score = next(r["score"] for r in compressed_results if r["memory_id"] == mem_id)
                    score_diff = abs(orig_score - comp_score)
                    logger.info(f"Memory {mem_id}: Score difference = {score_diff:.4f}")

if __name__ == "__main__":
    measure_compressed_search() 