"""Configuration packages for the agent memory system."""

# Import and re-export configuration classes
from agent_memory.config.memory_config import (
    MemoryConfig,
    RedisSTMConfig,
    RedisIMConfig,
    SQLiteLTMConfig,
    AutoencoderConfig
)

# For backwards compatibility
STMConfig = RedisSTMConfig
IMConfig = RedisIMConfig
LTMConfig = SQLiteLTMConfig

# Make the Pydantic models available
try:
    from agent_memory.config.models import (
        MemoryConfigModel,
        STMConfig as PydanticSTMConfig,
        IMConfig as PydanticIMConfig,
        LTMConfig as PydanticLTMConfig,
        AutoencoderConfig as PydanticAutoencoderConfig,
    )
except ImportError:
    # Pydantic might not be installed, provide a fallback
    pass

__all__ = [
    'MemoryConfig',
    'RedisSTMConfig',
    'RedisIMConfig',
    'SQLiteLTMConfig',
    'STMConfig', 
    'IMConfig',
    'LTMConfig',
    'AutoencoderConfig',
    'MemoryConfigModel',
]
