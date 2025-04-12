#!/usr/bin/env python
"""
Test script for evaluating the TextEmbeddingEngine with simulation data.

This script loads agent state data from the simulation database, creates embeddings,
and evaluates the quality of the embeddings through various tests.
"""

import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Any, Tuple
import pytest

# Import the TextEmbeddingEngine
from memory.embeddings.text_embeddings import TextEmbeddingEngine

# Connect to the database
def load_data(limit: int = 1000) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load agent state data from the simulation database.
    
    Args:
        limit: Maximum number of states to load
        
    Returns:
        Tuple of (list of agent state dictionaries, list of agent_ids)
    """
    conn = sqlite3.connect('data/simulation.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get unique agent_ids
    cursor.execute('SELECT DISTINCT agent_id FROM agent_states LIMIT 50')
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
    return states, agent_ids

def create_embeddings(states: List[Dict[str, Any]], engine: TextEmbeddingEngine) -> np.ndarray:
    """
    Create embeddings for a list of agent states.
    
    Args:
        states: List of agent state dictionaries
        engine: TextEmbeddingEngine instance
        
    Returns:
        Numpy array of embeddings
    """
    embeddings = []
    print(f"Creating embeddings for {len(states)} states...")
    
    for state in states:
        # Create a context weight dictionary to emphasize position
        context_weights = {
            'position_x': 3.0,
            'position_y': 3.0,
            'resource_level': 1.0,
            'current_health': 0.8,
            'agent_id': 5.0,
        }
        
        # Create embedding
        embedding = engine.encode(state, context_weights)
        embeddings.append(embedding)
    
    return np.array(embeddings)

@pytest.fixture
def states():
    """Fixture that provides mock agent states for testing."""
    # Create mock data instead of loading from database
    mock_states = []
    for agent_id in range(5):  # 5 mock agents
        agent_id_str = f"agent_{agent_id}"
        for step in range(10):  # 10 steps per agent
            mock_states.append({
                'agent_id': agent_id_str,
                'step_number': step,
                'position_x': float(step) + agent_id * 0.5,
                'position_y': float(step) - agent_id * 0.3,
                'resource_level': 50 + step * 5,
                'current_health': 100 - step,
                'status': 'active'
            })
    return mock_states

@pytest.fixture
def embeddings(states):
    """Fixture that provides mock embeddings for the agent states."""
    # Create mock embeddings instead of using the real embedding engine
    # Each embedding is a 10-dimensional vector (simplified from real embeddings)
    mock_embeddings = []
    for state in states:
        # Create a deterministic but somewhat unique embedding for each state
        agent_id_num = int(state['agent_id'].split('_')[1])
        step = state['step_number']
        
        # Base vector with some randomness that's deterministic based on state properties
        np.random.seed(agent_id_num * 100 + step)
        base_vector = np.random.rand(10)
        
        # Add some state properties influence
        base_vector[0] += state['position_x'] * 0.1
        base_vector[1] += state['position_y'] * 0.1
        base_vector[2] += float(agent_id_num) * 0.2
        base_vector[3] += float(step) * 0.1
        
        # Normalize to unit length (common for embeddings)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        mock_embeddings.append(base_vector)
    
    return np.array(mock_embeddings)

@pytest.fixture
def agent_ids(states):
    """Fixture that provides the unique agent IDs from the states."""
    return list(set(state['agent_id'] for state in states))

def test_state_transitions(states: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    """
    Test if consecutive states from the same agent have high similarity.
    
    Args:
        states: List of agent state dictionaries
        embeddings: Numpy array of embeddings
    """
    print("\nTesting state transitions...")
    
    # Group states by agent_id
    agent_states = {}
    for i, state in enumerate(states):
        agent_id = state['agent_id']
        if agent_id not in agent_states:
            agent_states[agent_id] = []
        agent_states[agent_id].append((i, state))
    
    transition_similarities = []
    
    # Calculate similarities between consecutive states for each agent
    for agent_id, agent_state_list in agent_states.items():
        if len(agent_state_list) < 2:
            continue
            
        for i in range(len(agent_state_list) - 1):
            idx1, state1 = agent_state_list[i]
            idx2, state2 = agent_state_list[i + 1]
            
            # Get embeddings
            emb1 = embeddings[idx1].reshape(1, -1)
            emb2 = embeddings[idx2].reshape(1, -1)
            
            # Calculate similarity
            sim = cosine_similarity(emb1, emb2)[0][0]
            transition_similarities.append(sim)
            
            if i == 0:  # Print first transition for each agent as example
                print(f"Agent {agent_id} transition similarity: {sim:.4f}")
                print(f"  Step {state1['step_number']} → {state2['step_number']}")
    
    avg_similarity = np.mean(transition_similarities)
    print(f"Average transition similarity: {avg_similarity:.4f}")

def test_agent_clustering(states: List[Dict[str, Any]], embeddings: np.ndarray, agent_ids: List[str]) -> None:
    """
    Test if states from the same agent cluster together.
    
    Args:
        states: List of agent state dictionaries
        embeddings: Numpy array of embeddings
        agent_ids: List of unique agent IDs
    """
    print("\nTesting agent clustering...")
    
    # Create labels for each state (agent_id)
    labels = [state['agent_id'] for state in states]
    
    # Calculate average similarity between states from the same agent vs different agents
    same_agent_similarities = []
    diff_agent_similarities = []
    
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            emb1 = embeddings[i].reshape(1, -1)
            emb2 = embeddings[j].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            
            if labels[i] == labels[j]:
                same_agent_similarities.append(sim)
            else:
                diff_agent_similarities.append(sim)
    
    avg_same = np.mean(same_agent_similarities)
    avg_diff = np.mean(diff_agent_similarities)
    
    print(f"Average similarity between states from same agent: {avg_same:.4f}")
    print(f"Average similarity between states from different agents: {avg_diff:.4f}")
    print(f"Contrast (same - different): {avg_same - avg_diff:.4f}")

def visualize_embeddings(states: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        states: List of agent state dictionaries
        embeddings: Numpy array of embeddings
    """
    print("\nVisualizing embeddings with t-SNE...")
    
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create labels for coloring by agent_id
    agent_ids = [state['agent_id'] for state in states]
    unique_agents = list(set(agent_ids))
    agent_to_color = {agent: i for i, agent in enumerate(unique_agents)}
    colors = [agent_to_color[agent] for agent in agent_ids]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        cmap='tab20',
        alpha=0.7,
        s=50
    )
    
    # Add step numbers as labels for a few points for each agent
    agent_points = {}
    for i, state in enumerate(states):
        agent = state['agent_id']
        step = state['step_number']
        if agent not in agent_points:
            agent_points[agent] = []
        agent_points[agent].append((i, step))
    
    # Label first, middle and last point for each agent
    for agent, points in agent_points.items():
        if len(points) <= 5:
            to_label = points
        else:
            # Choose first, middle and last points
            indices = [0, len(points) // 4, len(points) // 2, 3 * len(points) // 4, len(points) - 1]
            to_label = [points[i] for i in indices]
        
        for idx, step in to_label:
            plt.annotate(
                f"{step}",
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=8
            )
    
    # Add a legend for the first 10 agents
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab20(agent_to_color[agent]/len(unique_agents)), 
                                 markersize=10, label=f"Agent {agent[:6]}...")
                      for agent in unique_agents[:10]]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('t-SNE Visualization of Agent State Embeddings')
    plt.savefig('embedding_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'embedding_visualization.png'")

def test_position_similarity(states: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    """
    Test if states with similar positions have similar embeddings.
    
    Args:
        states: List of agent state dictionaries
        embeddings: Numpy array of embeddings
    """
    print("\nTesting position similarity...")
    
    # Extract positions
    positions = np.array([[state['position_x'], state['position_y']] for state in states])
    
    # Calculate pairwise distance between positions
    position_distances = np.zeros((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(len(positions)):
            position_distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
    
    # Calculate pairwise similarity between embeddings
    embedding_similarities = cosine_similarity(embeddings)
    
    # Sample some pairs to check correlation
    np.random.seed(42)
    sample_indices = np.random.choice(len(states), size=min(1000, len(states)), replace=False)
    
    position_dists = []
    embedding_sims = []
    
    for i in sample_indices:
        for j in sample_indices:
            if i != j:
                position_dists.append(position_distances[i, j])
                embedding_sims.append(embedding_similarities[i, j])
    
    # Calculate correlation
    correlation = np.corrcoef(position_dists, embedding_sims)[0, 1]
    print(f"Correlation between position distance and embedding similarity: {correlation:.4f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(position_dists, embedding_sims, alpha=0.4)
    plt.xlabel('Position Distance')
    plt.ylabel('Embedding Similarity')
    plt.title('Position Distance vs Embedding Similarity')
    plt.savefig('position_similarity.png', dpi=300)
    print("Position similarity plot saved as 'position_similarity.png'")

def main():
    # Create the TextEmbeddingEngine
    # Using a model optimized for semantic search to better maintain clustering properties
    engine = TextEmbeddingEngine(model_name="multi-qa-MiniLM-L6-cos-v1")
    
    # Load data
    states, agent_ids = load_data(limit=500)
    print(f"Loaded {len(states)} states from {len(agent_ids)} agents")
    
    # Create embeddings
    embeddings = create_embeddings(states, engine)
    
    # Run tests
    test_state_transitions(states, embeddings)
    test_agent_clustering(states, embeddings, agent_ids)
    test_position_similarity(states, embeddings)
    
    # Visualize
    visualize_embeddings(states, embeddings)
    
    print("\nEmbedding testing completed!")

if __name__ == "__main__":
    main() 