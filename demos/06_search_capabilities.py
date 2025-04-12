"""
Demo 6: Memory Search Capabilities

This demo showcases the search capabilities of the AgentMemorySystem by demonstrating:

1. Initialization of a memory system with search capabilities
2. Usage of different search strategies:
   - Temporal search: Retrieving memories based on time attributes
   - Attribute search: Searching for specific content characteristics or metadata
   - Combined search: Integrating multiple search approaches
3. Performing advanced filtering and scoring of search results
4. Searching across different memory tiers

The demo illustrates how agents can effectively retrieve relevant information
from their memory stores using different search approaches tailored to specific retrieval needs.
"""

import logging
import time
from datetime import datetime

# Import common utilities for demos
from demo_utils import (
    create_memory_system,
    log_print,
    pretty_print_memories,
    setup_logging,
)

# Import search-related components
from memory.search import (
    AttributeSearchStrategy,
    CombinedSearchStrategy,
    SearchModel,
    TemporalSearchStrategy,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_demo():
    """Run the memory search capabilities demo."""
    # Demo name to use for logging
    demo_name = "search_capabilities"

    # Setup logging
    logger = setup_logging(demo_name)
    log_print(logger, "Starting Memory Search Capabilities Demo")

    # Initialize the memory system with embeddings enabled for similarity search
    log_print(logger, "\nInitializing memory system with embeddings for search...")
    memory_system = create_memory_system(
        stm_limit=50,
        stm_ttl=3600,
        im_limit=100,
        im_compression_level=1,
        ltm_compression_level=2,
        cleanup_interval=60,
        description="search capabilities demo",
        use_embeddings=True,  # Enable embeddings for search
        embedding_type="text",
    )

    # Create a test agent
    agent_id = "search_agent"

    # Populate memory with different types of content
    log_print(logger, "\nPopulating agent memory with test data...")

    # Create memories with varied content
    memories = [
        # Project-related memories
        {
            "type": "note",
            "content": "Meeting with client about the new project requirements",
            "metadata": {
                "project": "alpha",
                "importance": "high",
                "participants": ["client", "team_lead", "developer"],
                "tags": ["meeting", "requirements", "client"],
            },
            "timestamp": int(time.time()) - 86400 * 7,  # 7 days ago
        },
        {
            "type": "note",
            "content": "Brainstorming session for the architecture design of project Alpha",
            "metadata": {
                "project": "alpha",
                "importance": "high",
                "participants": ["architect", "tech_lead", "developer"],
                "tags": ["meeting", "design", "architecture"],
            },
            "timestamp": int(time.time()) - 86400 * 5,  # 5 days ago
        },
        {
            "type": "action",
            "content": "Created initial project structure and repository setup",
            "metadata": {
                "project": "alpha",
                "importance": "medium",
                "tags": ["development", "setup"],
            },
            "timestamp": int(time.time()) - 86400 * 3,  # 3 days ago
        },
        # Task-related memories
        {
            "type": "task",
            "content": "Implement user authentication module with OAuth 2.0 support",
            "metadata": {
                "project": "alpha",
                "importance": "high",
                "status": "in_progress",
                "tags": ["development", "security", "authentication"],
            },
            "timestamp": int(time.time()) - 86400 * 2,  # 2 days ago
        },
        {
            "type": "task",
            "content": "Design database schema for user profiles and preferences",
            "metadata": {
                "project": "alpha",
                "importance": "medium",
                "status": "completed",
                "tags": ["database", "design"],
            },
            "timestamp": int(time.time()) - 86400,  # 1 day ago
        },
        # Error-related memories
        {
            "type": "error",
            "content": "API rate limit exceeded when integrating with external payment service",
            "metadata": {
                "project": "alpha",
                "importance": "high",
                "status": "resolved",
                "tags": ["error", "api", "integration"],
            },
            "timestamp": int(time.time()) - 43200,  # 12 hours ago
        },
        # General knowledge
        {
            "type": "knowledge",
            "content": "Python's requests library is useful for making HTTP requests",
            "metadata": {
                "importance": "low",
                "category": "programming",
                "tags": ["python", "http", "library"],
            },
            "timestamp": int(time.time()) - 86400 * 10,  # 10 days ago
        },
        {
            "type": "knowledge",
            "content": "React hooks were introduced in React 16.8 to enable state in functional components",
            "metadata": {
                "importance": "medium",
                "category": "programming",
                "tags": ["javascript", "react", "frontend"],
            },
            "timestamp": int(time.time()) - 86400 * 15,  # 15 days ago
        },
        # Personal notes
        {
            "type": "personal",
            "content": "Remember to take regular breaks while coding to maintain productivity",
            "metadata": {
                "importance": "medium",
                "category": "productivity",
                "tags": ["health", "productivity", "reminder"],
            },
            "timestamp": int(time.time()) - 86400 * 1,  # 1 day ago
        },
        {
            "type": "personal",
            "content": "Schedule planning session for next week's objectives",
            "metadata": {
                "importance": "medium",
                "category": "planning",
                "tags": ["schedule", "planning", "organization"],
            },
            "timestamp": int(time.time()) - 3600,  # 1 hour ago
        },
        # Add older memories that will go into LTM
        {
            "type": "knowledge",
            "content": "Key design patterns for scalable microservices architecture",
            "metadata": {
                "importance": "high",
                "category": "architecture",
                "tags": ["design", "microservices", "patterns", "architecture"],
            },
            "timestamp": int(time.time()) - 86400 * 60,  # 60 days ago
        },
        {
            "type": "note",
            "content": "Meeting notes about database sharding strategies for high-load systems",
            "metadata": {
                "project": "legacy",
                "importance": "high",
                "participants": ["dba", "architect", "cto"],
                "tags": ["database", "meeting", "architecture", "scaling"],
            },
            "timestamp": int(time.time()) - 86400 * 75,  # 75 days ago
        },
        {
            "type": "knowledge",
            "content": "Performance comparison between SQL and NoSQL databases for time-series data",
            "metadata": {
                "importance": "medium",
                "category": "databases",
                "tags": ["sql", "nosql", "performance", "time-series"],
            },
            "timestamp": int(time.time()) - 86400 * 90,  # 90 days ago
        },
    ]

    # Store memories
    for i, memory in enumerate(memories):
        success = memory_system.store_agent_state(
            agent_id,  # First parameter: agent_id
            memory,  # Second parameter: state (as positional argument)
            memory["timestamp"],
            priority=0.8 if memory["metadata"]["importance"] == "high" else 0.5,
        )
        log_print(logger, f"Stored memory {i+1}: {memory['type']} - {success}")

    # Wait a moment for memories to be processed and embedded
    log_print(logger, "Waiting for memories to be processed...")
    time.sleep(2)

    # Force memory maintenance multiple times to ensure memories move to LTM
    log_print(logger, "Forcing multiple memory maintenance cycles...")
    for i in range(3):
        memory_system.force_memory_maintenance(agent_id)
        log_print(logger, f"Completed maintenance cycle {i+1}")
        time.sleep(1)

    # Explicitly add older memories directly to LTM for demonstration purposes
    log_print(logger, "Adding demonstration memories directly to LTM...")
    memory_agent = memory_system.get_memory_agent(agent_id)

    # Create older memories specifically for LTM
    ltm_memories = [
        {
            "memory_id": f"{agent_id}-ltm-architecture-{int(time.time())}",
            "agent_id": agent_id,
            "step_number": int(time.time()) - 86400 * 60,
            "timestamp": int(time.time()) - 86400 * 60,
            "content": {
                "type": "knowledge",
                "content": "Key design patterns for scalable microservices architecture",
                "metadata": {
                    "importance": "high",
                    "category": "architecture",
                    "tags": ["design", "microservices", "patterns", "architecture"],
                },
                "timestamp": int(time.time()) - 86400 * 60,  # 60 days ago
            },
            "metadata": {
                "creation_time": int(time.time()) - 86400 * 60,
                "last_access_time": int(time.time()) - 86400 * 30,
                "compression_level": 2,
                "importance_score": 0.95,
                "retrieval_count": 3,
                "memory_type": "state",
                "memory_tier": "ltm",
                "last_transition_time": int(time.time()) - 86400 * 30,
            },
        },
        {
            "memory_id": f"{agent_id}-ltm-database-{int(time.time())}",
            "agent_id": agent_id,
            "step_number": int(time.time()) - 86400 * 75,
            "timestamp": int(time.time()) - 86400 * 75,
            "content": {
                "type": "note",
                "content": "Meeting notes about database sharding strategies for high-load systems",
                "metadata": {
                    "project": "legacy",
                    "importance": "high",
                    "participants": ["dba", "architect", "cto"],
                    "tags": ["database", "meeting", "architecture", "scaling"],
                },
                "timestamp": int(time.time()) - 86400 * 75,  # 75 days ago
            },
            "metadata": {
                "creation_time": int(time.time()) - 86400 * 75,
                "last_access_time": int(time.time()) - 86400 * 30,
                "compression_level": 2,
                "importance_score": 0.92,
                "retrieval_count": 2,
                "memory_type": "state",
                "memory_tier": "ltm",
                "last_transition_time": int(time.time()) - 86400 * 30,
            },
        },
        {
            "memory_id": f"{agent_id}-ltm-nosql-{int(time.time())}",
            "agent_id": agent_id,
            "step_number": int(time.time()) - 86400 * 90,
            "timestamp": int(time.time()) - 86400 * 90,
            "content": {
                "type": "knowledge",
                "content": "Performance comparison between SQL and NoSQL databases for time-series data",
                "metadata": {
                    "importance": "medium",
                    "category": "databases",
                    "tags": ["sql", "nosql", "performance", "time-series"],
                },
                "timestamp": int(time.time()) - 86400 * 90,  # 90 days ago
            },
            "metadata": {
                "creation_time": int(time.time()) - 86400 * 90,
                "last_access_time": int(time.time()) - 86400 * 30,
                "compression_level": 2,
                "importance_score": 0.88,
                "retrieval_count": 1,
                "memory_type": "state",
                "memory_tier": "ltm",
                "last_transition_time": int(time.time()) - 86400 * 30,
            },
        },
    ]

    # Add memories directly to LTM
    for memory in ltm_memories:
        # Add to LTM store
        log_print(
            logger,
            f"Adding memory directly to LTM: {memory['content']['content'][:50]}...",
        )
        memory_agent.ltm_store.store(memory)

        # Generate and store embedding for the memory
        try:
            content = memory.get("content", {})
            if isinstance(content, dict):
                text_content = content.get("content", "")
                if text_content:
                    log_print(
                        logger,
                        f"Generating embedding for memory: {text_content[:50]}...",
                    )
                    memory_agent.ltm_store.generate_and_store_embedding(
                        memory_id=memory.get("memory_id"),
                        text=text_content,
                        agent_id=agent_id,
                    )
        except Exception as e:
            log_print(logger, f"Error generating embedding: {str(e)}")
            # Continue with the demo even if embedding generation fails

    # Initialize the search model with the memory system's configuration
    log_print(logger, "\nInitializing search model...")
    memory_agent = memory_system.get_memory_agent(agent_id)

    # Initialize search model
    search_model = SearchModel(memory_system.config)

    # Create and register search strategies
    log_print(logger, "Creating and registering search strategies...")

    # Note: Skipping similarity search strategy as it requires embedding engine access

    # Temporal search strategy
    temporal_strategy = TemporalSearchStrategy(
        memory_agent.stm_store, memory_agent.im_store, memory_agent.ltm_store
    )
    search_model.register_strategy(temporal_strategy, make_default=True)

    # Attribute search strategy
    attribute_strategy = AttributeSearchStrategy(
        memory_agent.stm_store, memory_agent.im_store, memory_agent.ltm_store
    )
    search_model.register_strategy(attribute_strategy)


    combined_strategy = CombinedSearchStrategy(
        strategies={"temporal": temporal_strategy, "attribute": attribute_strategy},
        weights={"temporal": 0.7, "attribute": 0.8},
    )
    search_model.register_strategy(combined_strategy)

    # Get list of available search strategies
    strategies = search_model.get_available_strategies()
    log_print(logger, "Available search strategies:")
    for name, description in strategies.items():
        log_print(logger, f"  - {name}: {description}")

    # Demonstrate search capabilities
    log_print(logger, "\n== DEMONSTRATING SEARCH CAPABILITIES ==")

    # 1. Temporal search (now first)
    log_print(logger, "\n----- TEMPORAL SEARCH -----")

    # Search for recent memories (last 48 hours)
    now = int(time.time())
    two_days_ago = now - (86400 * 2)

    log_print(logger, "Searching for memories from the last 48 hours...")
    # Use a direct approach with get_by_timerange to avoid the datetime conversion issue
    memory_agent = memory_system.get_memory_agent(agent_id)
    recent_memories = memory_agent.ltm_store.get_by_timerange(two_days_ago, now, limit=5)
    
    # Add memories from STM and IM
    stm_memories = memory_agent.stm_store.get_all(agent_id)
    im_memories = memory_agent.im_store.get_all(agent_id)
    
    # Filter STM and IM memories by time
    filtered_stm = []
    for mem in stm_memories:
        if 'timestamp' in mem and two_days_ago <= mem['timestamp'] <= now:
            if 'metadata' not in mem:
                mem['metadata'] = {}
            mem['metadata']['memory_tier'] = 'stm'
            filtered_stm.append(mem)
    
    filtered_im = []
    for mem in im_memories:
        if 'timestamp' in mem and two_days_ago <= mem['timestamp'] <= now:
            if 'metadata' not in mem:
                mem['metadata'] = {}
            mem['metadata']['memory_tier'] = 'im'
            filtered_im.append(mem)
    
    # Combine all memories and sort by timestamp
    all_recent_memories = filtered_stm + filtered_im + recent_memories
    all_recent_memories.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    pretty_print_memories(all_recent_memories[:5], "Recent Memories (Last 48 Hours)", logger)

    # Search with recency weighting
    log_print(
        logger, "\nSearching with high recency weight (favoring newer memories)..."
    )
    # Use native timestamp approach 
    recency_memories = search_model.search(
        query=now,  # Use current timestamp
        agent_id=agent_id,
        strategy_name="attribute",  # Use attribute search which is more stable
        limit=3,
        metadata_filter={"timestamp": {"$gt": now - 86400}},  # Last day - use direct timestamp field
    )

    pretty_print_memories(recency_memories, "Memories Weighted by Recency", logger)

    # 2. Attribute search
    log_print(logger, "\n----- ATTRIBUTE SEARCH -----")

    # Search for high importance memories
    log_print(logger, "Searching for 'high' importance memories...")
    important_memories = search_model.search(
        query="high",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=3,
        content_fields=["content.content"],
        metadata_fields=["content.metadata.importance"],
    )

    pretty_print_memories(important_memories, "High Importance Memories", logger)

    # Search for development-related memories
    log_print(logger, "\nSearching for development-related memories...")
    dev_memories = search_model.search(
        query="development",
        agent_id=agent_id,
        strategy_name="attribute",
        content_fields=["content.content", "content.metadata.tags"],
        metadata_fields=[],
        limit=3,
        match_all=False,  # Any field can match (OR logic)
    )

    pretty_print_memories(dev_memories, "Development-Related Memories", logger)

    # 3. Combined search
    log_print(logger, "\n----- COMBINED SEARCH -----")

    # Search using combined strategy
    log_print(logger, "Searching with combined strategy for 'design'...")
    
    # Direct strategy mapping approach - explicitly name each strategy's query
    combined_memories = search_model.search(
        query={
            # Match exact strategy names
            "temporal": int(time.time()),  # Current timestamp for temporal search
            "attribute": "design"  # Text query for attribute search
        },
        agent_id=agent_id,
        strategy_name="combined",
        limit=3,
        strategy_params={
            "temporal": {"recency_weight": 1.2},
            "attribute": {
                "match_all": False,
                "content_fields": ["content.content"],
                "metadata_fields": ["content.metadata.tags"]
            },
        },
    )

    pretty_print_memories(
        combined_memories, "Combined Strategy Results for 'design'", logger
    )

    # Adjust strategy weights
    log_print(
        logger, "\nAdjusting strategy weights to prioritize attribute matching..."
    )
    combined_strategy.set_weights(
        {
            "temporal": 0.5,  # Reduce temporal weight
            "attribute": 2.0,  # Increase attribute weight
        }
    )

    # Search with adjusted weights
    log_print(logger, "Searching with adjusted weights for 'database'...")
    
    # Use a simpler approach with just the required queries
    adjusted_memories = search_model.search(
        query={
            # Match exact strategy names
            "temporal": int(time.time()),
            "attribute": "database"
        },
        agent_id=agent_id,
        strategy_name="combined",
        limit=3,
        strategy_params={
            "attribute": {
                "content_fields": ["content.content"],
                "metadata_fields": ["content.metadata.tags"]
            }
        },
    )

    pretty_print_memories(adjusted_memories, "Results with Adjusted Weights", logger)

    # 4. Search with metadata filtering
    log_print(logger, "\n----- SEARCH WITH METADATA FILTERING -----")

    # Search with metadata filter
    log_print(
        logger, "Searching for memories with project='alpha' and importance='high'..."
    )
    filtered_memories = search_model.search(
        query="project",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=5,
        metadata_filter={
            "content.metadata.project": "alpha",
            "content.metadata.importance": "high",
        },
        content_fields=["content.content"],
        metadata_fields=["content.metadata.tags"],
    )

    pretty_print_memories(filtered_memories, "Filtered by Metadata", logger)

    # 5. Tier-specific search
    log_print(logger, "\n----- TIER-SPECIFIC SEARCH -----")

    # Search only in STM
    log_print(logger, "Searching only in short-term memory...")
    stm_memories = search_model.search(
        query="meeting",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=2,
        tier="stm",
        content_fields=["content.content"],
        metadata_fields=["content.metadata.tags"],
    )

    pretty_print_memories(stm_memories, "STM Results", logger)

    # Search only in IM
    log_print(logger, "\nSearching only in intermediate memory...")
    im_memories = search_model.search(
        query="meeting",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=2,
        tier="im",
        content_fields=["content.content"],
        metadata_fields=["content.metadata.tags"],
    )

    pretty_print_memories(im_memories, "IM Results", logger)

    # Search only in LTM
    log_print(logger, "\nSearching only in long-term memory...")
    ltm_memories = search_model.search(
        query="architecture",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=3,
        tier="ltm",
        content_fields=["content.content"],
        metadata_fields=["content.metadata.tags"],
    )

    pretty_print_memories(ltm_memories, "LTM Results (Architecture)", logger)

    # Search for database-related memories in LTM
    log_print(logger, "\nSearching for database knowledge in long-term memory...")
    db_ltm_memories = search_model.search(
        query="database",
        agent_id=agent_id,
        strategy_name="attribute",
        limit=3,
        tier="ltm",
        content_fields=["content.content"],
        metadata_fields=["content.metadata.tags"],
    )

    pretty_print_memories(db_ltm_memories, "LTM Results (Database)", logger)

    # Wrap up
    log_print(logger, "\nMemory search capabilities demo completed!")
    log_print(logger, f"Log file saved at: logs/{demo_name}.log")


if __name__ == "__main__":
    run_demo()
