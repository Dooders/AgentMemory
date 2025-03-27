"""
Demo 2: Memory Retrieval Capabilities

This demo showcases the different memory retrieval methods including:
- Similarity-based retrieval (using text embeddings)
- Attribute-based retrieval
- Temporal retrieval
- Cross-tier retrieval
"""

import random
from typing import Any, Dict, List

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    log_print,
    pretty_print_memories,
    setup_logging,
)


def run_demo():
    """Run the memory retrieval demo."""
    # Demo name to use for logging
    demo_name = "memory_retrieval"

    # Setup logging
    logger = setup_logging(demo_name)
    log_print(logger, "Starting Memory Retrieval Demo")

    # Initialize with custom config for demo purposes
    memory_system = create_memory_system(
        stm_limit=500,  # Smaller limit to trigger transitions sooner
        stm_ttl=3600,  # Shorter TTL (1 hour)
        im_limit=1000,  # Smaller limit for demo purposes
        im_compression_level=1,
        cleanup_interval=10,  # Check for cleanup more frequently for demo
        description="retrieval demo",
        use_embeddings=True,  # Enable embeddings for similarity search
        embedding_type="text",  # Use text embeddings instead of autoencoder
    )

    # Use test agent
    agent_id = "retrieval_agent"

    # Populate with sample memories across multiple steps
    locations = ["forest", "mountain", "village", "dungeon", "castle"]
    actions = ["attack", "defend", "heal", "move", "interact", "observe"]
    items = ["sword", "shield", "potion", "map", "gem", "key"]

    log_print(logger, "Populating memory system with sample data...")

    # Generate 30 steps of varied memories
    for step in range(1, 31):
        # Create varied states
        state = {
            "position": {
                "x": random.uniform(-100, 100),
                "y": random.uniform(-100, 100),
                "location": random.choice(locations),
            },
            "health": random.randint(50, 100),
            "energy": random.randint(30, 100),
            "inventory": random.sample(items, random.randint(1, 3)),
            "level": step // 5 + 1,
        }

        # Store states with varying priorities
        priority = random.uniform(0.3, 1.0)
        memory_system.store_agent_state(agent_id, state, step, priority)

        # Store some actions (about 60% of steps)
        if random.random() < 0.6:
            action = {
                "type": random.choice(actions),
                "target": f"entity_{random.randint(1, 10)}",
                "success": random.choice([True, False, True]),  # Bias toward success
                "location": state["position"]["location"],
            }
            memory_system.store_agent_action(
                agent_id, action, step, random.uniform(0.4, 0.9)
            )

        # Store some interactions (about 30% of steps)
        if random.random() < 0.3:
            interaction = {
                "type": "encounter",
                "entity": f"npc_{random.randint(1, 5)}",
                "mood": random.choice(["friendly", "hostile", "neutral"]),
                "outcome": random.choice(["trade", "information", "quest", "combat"]),
                "location": state["position"]["location"],
            }
            memory_system.store_agent_interaction(
                agent_id, interaction, step, random.uniform(0.5, 1.0)
            )

        # Trigger maintenance every 10 steps to ensure memories are distributed
        if step % 10 == 0:
            memory_system.force_memory_maintenance(agent_id)

    # Force final memory maintenance to ensure memories are distributed across tiers
    log_print(
        logger, "Forcing memory maintenance to distribute memories across tiers..."
    )
    memory_system.force_memory_maintenance(agent_id)

    # Print memory statistics
    stats = memory_system.get_memory_statistics(agent_id)
    log_print(logger, "\nMemory Statistics:")
    for key, value in stats.items():
        log_print(logger, f"  {key}: {value}")

    # Demo 1: Similarity-based Retrieval
    log_print(
        logger, "\n--- Demo: Similarity-based Retrieval (using Text Embeddings) ---"
    )
    query_state = {
        "position": {
            "location": "forest",
        },
        "health": 75,
        "inventory": ["sword", "potion"],
    }
    log_print(logger, f"Query state: {query_state}")

    similar_memories = memory_system.retrieve_similar_states(agent_id, query_state, k=3)
    pretty_print_memories(similar_memories, "Most Similar States", logger)

    # Demo 2: Attribute-based Retrieval
    log_print(logger, "\n--- Demo: Attribute-based Retrieval ---")
    attr_query = {
        "position.location": "village",
    }
    log_print(logger, f"Attribute query: {attr_query}")

    attribute_memories = memory_system.retrieve_by_attributes(agent_id, attr_query)
    pretty_print_memories(
        attribute_memories[:3],
        "Memories with location='village' (showing first 3)",
        logger,
    )

    # Demo 3: Temporal Retrieval
    log_print(logger, "\n--- Demo: Temporal Retrieval ---")
    start_step = 10
    end_step = 15
    log_print(logger, f"Time range: steps {start_step} to {end_step}")

    temporal_memories = memory_system.retrieve_by_time_range(
        agent_id, start_step, end_step
    )
    pretty_print_memories(
        temporal_memories, f"Memories from steps {start_step}-{end_step}", logger
    )

    # Demo 4: Specific Memory Type Retrieval
    log_print(logger, "\n--- Demo: Specific Memory Type Retrieval ---")
    action_memories = memory_system.retrieve_by_attributes(
        agent_id, {"type": "attack"}, memory_type="action"
    )
    pretty_print_memories(action_memories, "Attack Action Memories", logger)

    # Demo 5: Content-based Search
    log_print(logger, "\n--- Demo: Content-based Search ---")
    content_query = "potion"  # Search for memories involving potions
    log_print(logger, f"Content query: '{content_query}'")

    content_memories = memory_system.search_by_content(agent_id, content_query, k=3)
    pretty_print_memories(
        content_memories, f"Memories matching '{content_query}'", logger
    )

    # Demo 6: Hybrid Search (New)
    log_print(logger, "\n--- Demo: Hybrid Search (Combining Similarity + Attribute Matching) ---")
    hybrid_query = {
        "position": {
            "location": "village",
        },
        "health": 75,
        "inventory": ["potion"],
    }
    log_print(logger, f"Hybrid query: {hybrid_query}")

    hybrid_memories = memory_system.hybrid_retrieve(agent_id, hybrid_query, k=3)
    pretty_print_memories(
        hybrid_memories, "Hybrid Search Results", logger
    )

    # Demo 7: Similarity Search with Context Weighting (New)
    log_print(logger, "\n--- Demo: Similarity Search with Context Weighting ---")
    
    # Find a memory with forest location and sword for our example
    forest_memories = memory_system.retrieve_by_attributes(
        agent_id, {"position.location": "forest"}
    )
    
    # Use this memory to construct our query
    example_memory = None
    for memory in forest_memories:
        if (memory.get("metadata", {}).get("memory_type") == "state" and 
            isinstance(memory.get("content", {}).get("inventory", []), list) and
            "sword" in memory.get("content", {}).get("inventory", [])):
            example_memory = memory
            break
    
    if not example_memory:
        # Fallback if no exact match
        weighted_query = {
            "position": {
                "location": "forest",
            },
            "health": 60,
            "inventory": ["sword"],
        }
    else:
        # Clone a memory to use as our query
        content = example_memory["content"]
        weighted_query = {
            "position": {
                "location": content["position"]["location"],
                "x": content["position"]["x"],
                "y": content["position"]["y"]
            },
            "health": content["health"],
            "inventory": content["inventory"][:1]  # Just take the first item
        }
    
    context_weights = {
        "position": 3.0,  # Very high emphasis on location
        "inventory": 1.5,  # Moderate emphasis on inventory items
    }
    log_print(logger, f"Weighted query: {weighted_query}")
    log_print(logger, f"Context weights: {context_weights}")
    
    # First without weights for comparison
    log_print(logger, "First, standard search without weights:")
    standard_memories = memory_system.retrieve_similar_states(
        agent_id, weighted_query, k=3, threshold=0.25
    )
    pretty_print_memories(
        standard_memories, "Standard Search Results", logger
    )
    
    # Then with weights
    log_print(logger, "Now, with context weighting:")
    weighted_memories = memory_system.retrieve_similar_states(
        agent_id, weighted_query, k=3, threshold=0.25, context_weights=context_weights
    )
    pretty_print_memories(
        weighted_memories, "Context-Weighted Search Results", logger
    )
    
    # Show difference in results
    standard_ids = set(memory["memory_id"] for memory in standard_memories)
    weighted_ids = set(memory["memory_id"] for memory in weighted_memories)
    unique_to_weighted = weighted_ids - standard_ids
    
    if unique_to_weighted:
        log_print(logger, f"Context weighting found {len(unique_to_weighted)} memories that standard search missed")
    else:
        log_print(logger, "Context weighting returned the same results as standard search in this example")

    # Demo 8: Evaluate Search Quality (New)
    log_print(logger, "\n--- Demo: Evaluating Similarity Search Quality ---")
    
    # Define test queries and expected relevant results (memory IDs)
    test_queries = [
        {  # Query 1: Looking for forest encounters
            "position": {"location": "forest"},
            "health": 80,
        },
        {  # Query 2: Low health combat situation
            "health": 55,
            "energy": 40,
            "inventory": ["sword"],
        },
        {  # Query 3: Village interactions
            "position": {"location": "village"},
            "inventory": ["map", "gem"],
        }
    ]
    
    # For demo purposes, we'll create a better function to identify "ground truth" relevant memories
    def identify_relevant_memories(memory_system, agent_id, query):
        """Find relevant memories based on more flexible criteria (our "ground truth")"""
        relevant_ids = []
        
        # Get all memories first
        all_memories = memory_system.retrieve_by_time_range(agent_id, 1, 30)
        
        # Extract location if present in query
        location = None
        if "position" in query and "location" in query["position"]:
            location = query["position"]["location"]
        
        # Extract health range if present in query
        health_range = None
        if "health" in query:
            health_val = query["health"]
            # Define a generous health range (±25%)
            health_range = (health_val * 0.75, health_val * 1.25)
        
        # Extract inventory items if present in query
        inventory_items = []
        if "inventory" in query and isinstance(query["inventory"], list):
            inventory_items = query["inventory"]
        
        # For each memory, check if it matches our flexible criteria
        for memory in all_memories:
            is_relevant = False
            
            # Skip if content is not a dict
            if not isinstance(memory.get("content", {}), dict):
                continue
            
            memory_content = memory["content"]
            memory_type = memory.get("metadata", {}).get("memory_type", "")
            
            # Check for location match based on memory type
            if location:
                # For state memories, check in position.location
                if memory_type == "state" and isinstance(memory_content.get("position", {}), dict):
                    memory_location = memory_content["position"].get("location")
                    if memory_location == location:
                        is_relevant = True
                # For action/interaction memories, check direct location field
                elif memory_type in ["action", "interaction"] and "location" in memory_content:
                    memory_location = memory_content.get("location")
                    if memory_location == location:
                        is_relevant = True
            
            # Check for health range match (usually only in state memories)
            if health_range and "health" in memory_content:
                memory_health = memory_content["health"]
                if health_range[0] <= memory_health <= health_range[1]:
                    is_relevant = True
            
            # Check for inventory item match (usually only in state memories)
            if inventory_items and "inventory" in memory_content and isinstance(memory_content["inventory"], list):
                memory_inventory = memory_content["inventory"]
                # If any item in query inventory matches any item in memory inventory
                if any(item in memory_inventory for item in inventory_items):
                    is_relevant = True
                    
            # For state 2 (low health combat), also match action memories for combat
            if "health" in query and query["health"] < 60 and memory_type == "action":
                action_type = memory_content.get("type")
                if action_type in ["attack", "defend", "combat"]:
                    is_relevant = True
            
            # For state 3 (village interactions), also match interaction memories
            if location == "village" and memory_type == "interaction":
                if "location" in memory_content and memory_content["location"] == "village":
                    is_relevant = True
            
            if is_relevant:
                relevant_ids.append(memory["memory_id"])
        
        # Log ground truth for debugging
        log_print(logger, f"  Found {len(relevant_ids)} relevant memories for the query")
        return relevant_ids
    
    # Get ground truth for each query
    log_print(logger, "Identifying ground truth relevant memories for evaluation...")
    ground_truth = []
    for query in test_queries:
        relevant_ids = identify_relevant_memories(memory_system, agent_id, query)
        ground_truth.append(relevant_ids)
    
    # Now evaluate different retrieval methods
    def evaluate_search(name, search_function, queries, ground_truth):
        """Evaluate a search function against ground truth data"""
        log_print(logger, f"\nEvaluating {name}:")
        
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        
        for i, query in enumerate(queries):
            # Get results using the search function
            results = search_function(query)
            
            # Log similarity scores for debugging
            log_print(logger, f"  Query {i+1} returned {len(results)} results")
            for j, result in enumerate(results[:3]):  # Log first 3 results
                similarity = result.get("similarity_score", result.get("hybrid_score", 0))
                log_print(logger, f"    Result {j+1} similarity score: {similarity:.4f}")
            
            result_ids = [r["memory_id"] for r in results]
            
            # Calculate metrics
            if ground_truth[i]:
                # True positives: results that are in ground truth
                tp = sum(1 for rid in result_ids if rid in ground_truth[i])
                
                # Calculate precision and recall
                precision = tp / len(results) if results else 0
                recall = tp / len(ground_truth[i]) if ground_truth[i] else 0
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics["precision"].append(precision)
                metrics["recall"].append(recall)
                metrics["f1"].append(f1)
                
                log_print(logger, f"  Query {i+1}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
            else:
                log_print(logger, f"  Query {i+1}: No ground truth available")
        
        # Calculate averages
        avg_precision = sum(metrics["precision"]) / len(metrics["precision"]) if metrics["precision"] else 0
        avg_recall = sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0
        avg_f1 = sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0
        
        log_print(logger, f"  Average: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")
        return metrics
    
    # Create search functions that use different methods
    def standard_search(query):
        return memory_system.retrieve_similar_states(agent_id, query, k=5)
    
    def weighted_search(query):
        weights = {"position": 2.0, "health": 1.5, "inventory": 1.5}
        return memory_system.retrieve_similar_states(agent_id, query, k=5, context_weights=weights)
    
    def hybrid_search(query):
        return memory_system.hybrid_retrieve(agent_id, query, k=5)
    
    # Evaluate all methods
    standard_metrics = evaluate_search("Standard Similarity Search", standard_search, test_queries, ground_truth)
    weighted_metrics = evaluate_search("Context-Weighted Search", weighted_search, test_queries, ground_truth)
    hybrid_metrics = evaluate_search("Hybrid Search", hybrid_search, test_queries, ground_truth)
    
    log_print(logger, "\nSearch Method Comparison:")
    methods = ["Standard", "Context-Weighted", "Hybrid"]
    for metric in ["precision", "recall", "f1"]:
        log_print(logger, f"  Average {metric.capitalize()}:")
        for i, method in enumerate(methods):
            method_metrics = [standard_metrics, weighted_metrics, hybrid_metrics][i]
            avg_value = sum(method_metrics[metric]) / len(method_metrics[metric]) if method_metrics[metric] else 0
            log_print(logger, f"    {method}: {avg_value:.2f}")

    log_print(logger, "\nMemory retrieval demo completed!")
    log_print(logger, f"Log file saved at: logs/{demo_name}.log")


if __name__ == "__main__":
    run_demo()
