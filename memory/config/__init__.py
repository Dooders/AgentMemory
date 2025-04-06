"""Agent Memory Configuration Module

This module provides configuration components for the agent memory system,
allowing customization of memory tiers and behavior. It centralizes configuration
options for different memory stores and processing components.

Key components:

1. MemoryConfig: The main configuration class that integrates settings for all
   memory tiers (STM, IM, LTM) and components like embedding autoencoder.

2. RedisSTMConfig: Configuration for the Short-Term Memory (STM) tier using Redis,
   including connection settings, TTL, and capacity limits.

3. RedisIMConfig: Configuration for the Intermediate Memory (IM) tier using Redis,
   with settings for persistence and organization.

4. SQLiteLTMConfig: Configuration for the Long-Term Memory (LTM) tier using SQLite,
   with database location and indexing options.

5. AutoencoderConfig: Settings for the neural network-based dimensionality reduction
   of memory embeddings, including model architecture and training parameters.

Usage example:
```python
from memory.config import MemoryConfig, RedisSTMConfig, SQLiteLTMConfig

# Create configuration for memory system
memory_config = MemoryConfig(
    stm=RedisSTMConfig(
        host="localhost",
        port=6379,
        ttl=3600,  # 1 hour TTL for short-term memories
    ),
    ltm=SQLiteLTMConfig(
        db_path="./memory.db",
        enable_compression=True
    ),
    autoencoder=AutoencoderConfig(
        latent_dim=128,
        training_epochs=100
    )
)

# Initialize memory system with configuration
from memory import AgentMemorySystem
memory_system = AgentMemorySystem(config=memory_config)
```
"""

# Import and re-export configuration classes
from memory.config.memory_config import (
    AutoencoderConfig,
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)

# For backwards compatibility
STMConfig = RedisSTMConfig
IMConfig = RedisIMConfig
LTMConfig = SQLiteLTMConfig

# Make the Pydantic models available
try:
    from memory.config.models import AutoencoderConfig as PydanticAutoencoderConfig
    from memory.config.models import IMConfig as PydanticIMConfig
    from memory.config.models import LTMConfig as PydanticLTMConfig
    from memory.config.models import MemoryConfigModel
    from memory.config.models import STMConfig as PydanticSTMConfig
except ImportError:
    # Pydantic might not be installed, provide a fallback
    pass

__all__ = [
    "MemoryConfig",
    "RedisSTMConfig",
    "RedisIMConfig",
    "SQLiteLTMConfig",
    "STMConfig",
    "IMConfig",
    "LTMConfig",
    "AutoencoderConfig",
    "MemoryConfigModel",
]
