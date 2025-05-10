"""
Memory System Utilities

This module provides utilities for creating and configuring AgentMemorySystem instances
with various configurations and settings.
"""

import os
import sqlite3
import subprocess
from os import PathLike
from typing import List, Optional, Union

from memory.config import (
    AutoencoderConfig,
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.core import AgentMemorySystem


def create_memory_system(
    db_path: str,
    memory_samples_dir: Optional[str] = None,
    stm_limit: int = 500,
    stm_ttl: int = 3600,
    im_limit: int = 1000,
    im_compression_level: int = 1,
    ltm_compression_level: int = 2,
    ltm_batch_size: int = 20,
    logging_level: str = "DEBUG",
    cleanup_interval: int = 10,
    enable_hooks: bool = False,
    description: str = "default",
    use_embeddings: bool = False,
    embedding_type: str = "text",
    use_mock_redis: bool = True,
    memory_file: Optional[Union[str, PathLike]] = None,
    clear_db: bool = True,
) -> AgentMemorySystem:
    """Create and configure a memory system with customizable parameters.

    Args:
        db_path: Path to the SQLite database file
        memory_samples_dir: Directory containing memory sample files (optional)
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
        memory_file: Optional path to a memory JSON file to load instead of creating new.
                     Can be a full path, a filename in the memory_samples directory,
                     or one of the preset sample types: "simple", "multi_agent", or "tiered"
        clear_db: Whether to delete the existing LTM database file (default True)

    Returns:
        Configured AgentMemorySystem instance
    """
    # Delete the existing LTM database file if it exists to ensure a clean start
    if clear_db:
        # Get absolute path and log it for debugging
        absolute_db_path = os.path.abspath(db_path)
        logging.info(f"Checking for database file at: {absolute_db_path}")

        if os.path.exists(absolute_db_path):
            logging.info(f"Database file exists. Attempting to delete it.")
            # Close any open connections
            try:
                conn = sqlite3.connect(absolute_db_path)
                conn.close()
                logging.info("Closed any existing database connections")
            except Exception as e:
                logging.warning(f"Note: Could not connect to database: {e}")

            # On Windows, using a different approach for file deletion
            if os.name == "nt":  # Windows
                try:
                    logging.info("Using Windows-specific deletion method")
                    # Force delete using del command
                    subprocess.run(
                        [
                            "powershell",
                            "-Command",
                            f'Remove-Item -Path "{absolute_db_path}" -Force',
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    if not os.path.exists(absolute_db_path):
                        print("Successfully deleted database file using Windows method")
                    else:
                        print(
                            "Warning: File still exists after Windows deletion attempt"
                        )
                except subprocess.CalledProcessError as e:
                    print(f"Windows deletion failed: {e.stderr}")
                    # Fallback to regular method
                    try:
                        os.remove(absolute_db_path)
                        print(
                            "Successfully deleted database file using fallback method"
                        )
                    except Exception as e:
                        print(f"Fallback deletion also failed: {e}")
            else:
                # Regular deletion for non-Windows systems
                try:
                    os.remove(absolute_db_path)
                    print(f"Successfully deleted database file: {absolute_db_path}")
                except Exception as e:
                    print(f"Failed to delete database file: {e}")

            # Verify deletion
            if os.path.exists(absolute_db_path):
                print("Warning: Database file still exists after deletion attempts")
            else:
                print("Confirmed: Database file has been successfully deleted")
        else:
            print(f"No existing database file found at: {absolute_db_path}")

    # If a memory file is provided, try to load from it
    if memory_file:
        # Handle preset sample types
        if memory_samples_dir and memory_file in ["simple", "simple_agent"]:
            memory_file = os.path.join(memory_samples_dir, "simple_agent_memory.json")
        elif memory_samples_dir and memory_file in ["multi_agent", "multi"]:
            memory_file = os.path.join(memory_samples_dir, "multi_agent_memory.json")
        elif memory_samples_dir and memory_file in ["tiered"]:
            memory_file = os.path.join(memory_samples_dir, "tiered_memory.json")
        # If it's just a filename without path and doesn't exist, check in samples dir
        elif (
            memory_samples_dir
            and not os.path.isabs(memory_file)
            and not os.path.exists(memory_file)
        ):
            sample_path = os.path.join(memory_samples_dir, memory_file)
            if os.path.exists(sample_path):
                memory_file = sample_path

        print(f"Loading memory system from file: {memory_file}")
        if os.path.exists(memory_file):
            memory_system = AgentMemorySystem.load_from_json(
                memory_file, use_mock_redis=use_mock_redis
            )
            if memory_system:
                print(f"Successfully loaded memory system from {memory_file}")
                return memory_system
            else:
                print(
                    f"Failed to load memory system from {memory_file}, creating new system"
                )
        else:
            print(f"Memory file not found: {memory_file}, creating new system")

    print(
        f"Creating memory system with embeddings={use_embeddings}, type={embedding_type}"
    )

    stm_config = RedisSTMConfig(
        host="127.0.0.1",
        port=6379,
        memory_limit=stm_limit,
        ttl=stm_ttl,
        use_mock=use_mock_redis,
        test_mode=True,
    )

    im_config = RedisIMConfig(
        host="127.0.0.1",
        port=6379,
        memory_limit=im_limit,
        compression_level=im_compression_level,
        use_mock=use_mock_redis,
        test_mode=True,
    )

    ltm_config = SQLiteLTMConfig(
        db_path=db_path,
        compression_level=ltm_compression_level,
        batch_size=ltm_batch_size,
        test_mode=True,
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
                text_model_name="all-mpnet-base-v2",
                logger=None,
            )

    memory_system = AgentMemorySystem.get_instance(memory_config)
    print(f"Initialized AgentMemorySystem for {description}")
    if use_embeddings:
        print(f"Using {embedding_type} embeddings")

    # Debug: Check database tables
    print(f"SQLite database path: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"SQLite tables: {[table[0] for table in tables]}")
        conn.close()
    except Exception as e:
        print(f"Error checking SQLite database: {e}")

    return memory_system


def get_memory_statistics(memory_system: AgentMemorySystem, agent_id: str) -> dict:
    """Get memory statistics for a specific agent.

    Args:
        memory_system: The initialized memory system
        agent_id: ID of the agent to get stats for

    Returns:
        Dictionary containing memory statistics
    """
    return memory_system.get_memory_statistics(agent_id)


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
