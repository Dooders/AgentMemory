"""
Shared Utilities for AgentMemory System Demos

This module provides common utilities and helper functions used across
the various demonstration scripts in the AgentMemory system.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pprint import pprint

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_memory.core import AgentMemorySystem
from agent_memory.config import (
    MemoryConfig, 
    RedisSTMConfig, 
    RedisIMConfig, 
    SQLiteLTMConfig
)

# Path constants
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../agent_memory.db")

def setup_logging(demo_name: str) -> logging.Logger:
    """Set up logging to both console and file.
    
    Args:
        demo_name: Name of the demo for log file naming
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{demo_name}_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler for logging to file
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler for logging to console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Create a filter to exclude embedding-related log messages
    class EmbeddingFilter(logging.Filter):
        def filter(self, record):
            # Skip any log messages containing "embedding" or "vector"
            return not any(term in record.getMessage().lower() 
                          for term in ["embedding", "vector", "encoded"])
    
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
    os.system('cls' if os.name == 'nt' else 'clear')

def pretty_print_memories(memories: List[Dict[str, Any]], title: str = "Memories") -> None:
    """Print memories in a readable format.
    
    Args:
        memories: List of memory objects to display
        title: Title to show above the memories
    """
    print(f"\n{title} ({len(memories)} results):")
    if not memories:
        print("  No memories found.")
        return
        
    for i, memory in enumerate(memories):
        print(f"  Memory {i+1}:")
        # Convert complex nested objects to strings for better display
        formatted_memory = {}
        for k, v in memory.items():
            if isinstance(v, dict) or isinstance(v, list):
                formatted_memory[k] = json.dumps(v, indent=2)
            else:
                formatted_memory[k] = v
        
        for k, v in formatted_memory.items():
            print(f"    {k}: {v}")
        print()  # Extra line for readability

def print_memory_details(memory_system: AgentMemorySystem, agent_id: str, title: str = "Current Memory State") -> None:
    """Print detailed memory information across tiers.
    
    Args:
        memory_system: The initialized memory system
        agent_id: ID of the agent to get stats for
        title: Title for the stats display
    """
    stats = memory_system.get_memory_statistics(agent_id)
    
    print(f"\n{title}:")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  STM count: {stats.get('stm_count', 0)}")
    print(f"  IM count: {stats.get('im_count', 0)}")
    print(f"  LTM count: {stats.get('ltm_count', 0)}")
    
    # Additional statistics if available
    if 'avg_compression_ratio' in stats:
        print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2f}x")
    if 'stm_size_bytes' in stats:
        print(f"  STM size: {stats['stm_size_bytes'] / 1024:.2f} KB")
    if 'im_size_bytes' in stats:
        print(f"  IM size: {stats['im_size_bytes'] / 1024:.2f} KB")
    if 'ltm_size_bytes' in stats:
        print(f"  LTM size: {stats['ltm_size_bytes'] / 1024:.2f} KB")

def create_memory_system(
    stm_limit: int = 500,
    stm_ttl: int = 3600,
    im_limit: int = 1000,
    im_compression_level: int = 1,
    ltm_compression_level: int = 2,
    ltm_batch_size: int = 20,
    logging_level: str = "INFO",
    cleanup_interval: int = 10,
    enable_hooks: bool = False,
    description: str = "demo"
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
        
    Returns:
        Configured AgentMemorySystem instance
    """
    stm_config = RedisSTMConfig(
        memory_limit=stm_limit,
        ttl=stm_ttl,
    )
    
    im_config = RedisIMConfig(
        memory_limit=im_limit,
        compression_level=im_compression_level,
    )
    
    ltm_config = SQLiteLTMConfig(
        db_path=DB_PATH,
        compression_level=ltm_compression_level,
        batch_size=ltm_batch_size,
    )
    
    config = MemoryConfig(
        logging_level=logging_level,
        stm_config=stm_config,
        im_config=im_config,
        ltm_config=ltm_config,
        enable_memory_hooks=enable_hooks,
        cleanup_interval=cleanup_interval,
    )
    
    memory_system = AgentMemorySystem.get_instance(config)
    print(f"Initialized AgentMemorySystem for {description}")
    
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