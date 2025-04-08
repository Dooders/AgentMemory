"""
Shared Utilities for AgentMemory System Demos

This module provides common utilities and helper functions used across
the various demonstration scripts in the AgentMemory system.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Union

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.config import (
    AutoencoderConfig,
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.core import AgentMemorySystem

# Path constants
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../memory.db")


def setup_logging(demo_name: str) -> logging.Logger:
    """Set up logging to both console and file.

    Args:
        demo_name: Name of the demo for log file naming

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Use a fixed log filename based on demo name (without timestamp)
    log_file = os.path.join(LOGS_DIR, f"{demo_name}.log")

    # Clear the existing log file if it exists
    with open(log_file, "w") as f:
        # Empty the file by opening it in write mode
        pass

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # File handler for logging to file
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler for logging to console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # Create a filter to exclude embedding-related log messages
    class EmbeddingFilter(logging.Filter):
        def filter(self, record):
            # Skip any log messages containing "embedding" or "vector"
            return not any(
                term in record.getMessage().lower()
                for term in ["embedding", "vector", "encoded"]
            )

    # Add the filter to both handlers
    embedding_filter = EmbeddingFilter()
    file_handler.addFilter(embedding_filter)
    console_handler.addFilter(embedding_filter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_print(logger: logging.Logger, message: str) -> None:
    """Log a message and print it to console."""
    logger.info(message)


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def pretty_print_memories(
    memories: List[Dict[str, Any]],
    title: str = "Memories",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print memories in a readable format.

    Args:
        memories: List of memory objects to display
        title: Title to show above the memories
        logger: Optional logger for writing to log file
    """
    message = f"\n{title} ({len(memories)} results):"
    if logger:
        logger.info(message)
    else:
        print(message)

    if not memories:
        no_memories_msg = "  No memories found."
        if logger:
            logger.info(no_memories_msg)
        else:
            print(no_memories_msg)
        return

    for i, memory in enumerate(memories):
        memory_header = f"  Memory {i+1}:"
        if logger:
            logger.info(memory_header)
        else:
            print(memory_header)

        # Convert complex nested objects to strings for better display
        formatted_memory = {}
        for k, v in memory.items():
            if isinstance(v, dict) or isinstance(v, list):
                formatted_memory[k] = json.dumps(v, indent=2)
            else:
                formatted_memory[k] = v

        for k, v in formatted_memory.items():
            property_msg = f"    {k}: {v}"
            if logger:
                logger.info(property_msg)
            else:
                print(property_msg)

        # Extra line for readability
        if logger:
            logger.info("")
        else:
            print()


def print_memory_details(
    memory_system: AgentMemorySystem, agent_id: str, title: str = "Current Memory State"
) -> None:
    """Print detailed memory information across tiers.

    Args:
        memory_system: The initialized memory system
        agent_id: ID of the agent to get stats for
        title: Title for the stats display
    """
    stats = memory_system.get_memory_statistics(agent_id)

    print(f"\n{title}:")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  STM count: {stats.get('tiers', {}).get('stm', {}).get('count', 0)}")
    print(f"  IM count: {stats.get('tiers', {}).get('im', {}).get('count', 0)}")
    print(f"  LTM count: {stats.get('tiers', {}).get('ltm', {}).get('count', 0)}")

    # Additional statistics if available
    if "avg_compression_ratio" in stats:
        print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2f}x")
    if (
        "tiers" in stats
        and "stm" in stats["tiers"]
        and "size_bytes" in stats["tiers"]["stm"]
    ):
        print(f"  STM size: {stats['tiers']['stm']['size_bytes'] / 1024:.2f} KB")
    if (
        "tiers" in stats
        and "im" in stats["tiers"]
        and "size_bytes" in stats["tiers"]["im"]
    ):
        print(f"  IM size: {stats['tiers']['im']['size_bytes'] / 1024:.2f} KB")
    if (
        "tiers" in stats
        and "ltm" in stats["tiers"]
        and "size_bytes" in stats["tiers"]["ltm"]
    ):
        print(f"  LTM size: {stats['tiers']['ltm']['size_bytes'] / 1024:.2f} KB")


def create_memory_system(
    stm_limit: int = 500,
    stm_ttl: int = 3600,
    im_limit: int = 1000,
    im_compression_level: int = 1,
    ltm_compression_level: int = 2,
    ltm_batch_size: int = 20,
    logging_level: str = "DEBUG",
    cleanup_interval: int = 10,
    enable_hooks: bool = False,
    description: str = "demo",
    use_embeddings: bool = False,  # Set to False by default
    embedding_type: str = "text",  # "text" or "autoencoder"
    use_mock_redis: bool = True,  # Default to using mockredis
) -> AgentMemorySystem:
    """Create and configure a memory system with customizable parameters.

    Args:
        stm_limit: Memory limit for short-term memory
        stm_ttl: Time-to-live for STM items in seconds
        im_limit: Memory limit for intermediate memory
        im_compression_level: Compression level for IM (0-2)
        ltm_compression_level: Compression level for LTM (0-2)
        ltm_batch_size: Batch size for LTM operations
        logging_level: Logging level (INFO, WARNING, ERROR, etc.)
        cleanup_interval: Interval for memory maintenance in seconds
        enable_hooks: Whether to enable memory event hooks
        description: Description of this memory system instance
        use_embeddings: Whether to enable neural embeddings
        embedding_type: Type of embeddings to use ("text" or "autoencoder")
        use_mock_redis: Whether to use MockRedis instead of real Redis

    Returns:
        Configured AgentMemorySystem instance
    """
    print(f"Creating memory system with embeddings={use_embeddings}, type={embedding_type}")
    
    stm_config = RedisSTMConfig(
        host="127.0.0.1",
        port=6379,
        memory_limit=stm_limit,
        ttl=stm_ttl,
        use_mock=use_mock_redis,
    )

    im_config = RedisIMConfig(
        host="127.0.0.1",
        port=6379,
        memory_limit=im_limit,
        compression_level=im_compression_level,
        use_mock=use_mock_redis,
    )

    ltm_config = SQLiteLTMConfig(
        db_path=DB_PATH,
        compression_level=ltm_compression_level,
        batch_size=ltm_batch_size,
    )

    memory_config = MemoryConfig(
        logging_level=logging_level,
        stm_config=stm_config,
        im_config=im_config,
        ltm_config=ltm_config,
        enable_memory_hooks=enable_hooks,
        cleanup_interval=cleanup_interval,
    )

    if use_embeddings:
        if embedding_type == "autoencoder":
            # Configure autoencoder (neural embedding) settings
            autoencoder_config = AutoencoderConfig(
                use_neural_embeddings=True,
                embedding_dim=64,
                hidden_dims=[128, 256, 512],
                activation="relu",
                latent_dim=32,
                learning_rate=0.001,
                weight_decay=1e-5,
                batch_size=32,
                num_epochs=10,
                device="auto",
                random_seed=42,
                logger=None,
            )
            memory_config.autoencoder_config = autoencoder_config
        else:
            # Use text embeddings
            memory_config.autoencoder_config = AutoencoderConfig(
                use_neural_embeddings=True,
                embedding_type="text",
                text_model_name="all-mpnet-base-v2",  # Upgraded from all-MiniLM-L6-v2
                logger=None,
            )

    memory_system = AgentMemorySystem.get_instance(memory_config)
    print(f"Initialized AgentMemorySystem for {description}")
    if use_embeddings:
        print(f"Using {embedding_type} embeddings")

    # Debug: Check database tables
    import sqlite3

    print(f"SQLite database path: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"SQLite tables: {[table[0] for table in tables]}")
        conn.close()
    except Exception as e:
        print(f"Error checking SQLite database: {e}")

    return memory_system


def generate_random_state(agent_id: str, step: int) -> Dict[str, Any]:
    """Generate a random agent state for testing.

    Args:
        agent_id: ID of the agent
        step: Current time step

    Returns:
        Randomly generated state dictionary
    """
    import random

    return {
        "agent_id": agent_id,
        "step": step,
        "position": {
            "x": random.uniform(-100, 100),
            "y": random.uniform(-100, 100),
            "z": random.uniform(-50, 50),
        },
        "health": random.randint(50, 100),
        "energy": random.randint(30, 100),
        "inventory": {
            "items": [
                {"name": f"item_{i}", "count": random.randint(1, 5)}
                for i in range(random.randint(1, 5))
            ]
        },
        "status": random.choice(["exploring", "resting", "fighting", "trading"]),
    }
