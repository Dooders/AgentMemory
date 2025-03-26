"""Pydantic models for configuration validation in the agent memory system."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class RedisSTMConfigModel(BaseModel):
    """Configuration model for Redis-based Short-Term Memory."""

    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, description="Redis server port", ge=1, le=65535)
    memory_limit: int = Field(
        default=10000, description="Maximum number of memories in STM", ge=100
    )
    ttl: int = Field(
        default=3600, description="Time-to-live for STM entries in seconds", ge=60
    )

    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator("memory_limit")
    def validate_memory_limit(cls, v):
        if v < 100:
            raise ValueError("Memory limit must be at least 100")
        return v

    @validator("ttl")
    def validate_ttl(cls, v):
        if v < 60:
            raise ValueError("TTL must be at least 60 seconds")
        return v


class RedisIMConfigModel(BaseModel):
    """Configuration model for Redis-based Intermediate Memory."""

    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, description="Redis server port", ge=1, le=65535)
    memory_limit: int = Field(
        default=50000, description="Maximum number of memories in IM", ge=1000
    )
    ttl: int = Field(
        default=86400, description="Time-to-live for IM entries in seconds", ge=3600
    )

    @validator("memory_limit")
    def validate_memory_limit(cls, v):
        if v < 1000:
            raise ValueError("IM memory limit must be at least 1000")
        return v

    @validator("ttl")
    def validate_ttl(cls, v):
        if v < 3600:
            raise ValueError("IM TTL must be at least 3600 seconds (1 hour)")
        return v


class SQLiteLTMConfigModel(BaseModel):
    """Configuration model for SQLite-based Long-Term Memory."""

    db_path: str = Field(default="./ltm.db", description="Path to SQLite database file")

    @validator("db_path")
    def validate_db_path(cls, v):
        if not v:
            raise ValueError("Database path cannot be empty")
        return v


class AutoencoderConfigModel(BaseModel):
    """Configuration model for memory autoencoder."""

    stm_dim: int = Field(default=768, description="Embedding dimension for STM", ge=64)
    im_dim: int = Field(default=384, description="Embedding dimension for IM", ge=32)
    ltm_dim: int = Field(default=128, description="Embedding dimension for LTM", ge=16)

    @validator("stm_dim", "im_dim", "ltm_dim")
    def validate_dimensions(cls, v, values, **kwargs):
        field = kwargs.get("field")

        if field == "im_dim" and "stm_dim" in values and v >= values["stm_dim"]:
            raise ValueError("IM dimension must be smaller than STM dimension")

        if field == "ltm_dim":
            if "im_dim" in values and v >= values["im_dim"]:
                raise ValueError("LTM dimension must be smaller than IM dimension")
            if "stm_dim" in values and v >= values["stm_dim"]:
                raise ValueError("LTM dimension must be smaller than STM dimension")

        return v


class MemoryConfigModel(BaseModel):
    """Configuration model for the memory system."""

    cleanup_interval: int = Field(
        default=100, description="Check for memory maintenance every N insertions", ge=1
    )
    memory_priority_decay: float = Field(
        default=0.95,
        description="Priority decay factor for older memories",
        ge=0.0,
        le=1.0,
    )
    stm_config: Optional[RedisSTMConfigModel] = None
    im_config: Optional[RedisIMConfigModel] = None
    ltm_config: Optional[SQLiteLTMConfigModel] = None
    autoencoder_config: Optional[AutoencoderConfigModel] = None

    @validator("cleanup_interval")
    def validate_cleanup_interval(cls, v):
        if v <= 0:
            raise ValueError("Cleanup interval must be positive")
        return v

    @validator("memory_priority_decay")
    def validate_memory_priority_decay(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Memory priority decay must be between 0.0 and 1.0")
        return v

    @root_validator(skip_on_failure=True)
    def validate_config_consistency(cls, values):
        """Validate consistency between different config sections."""
        stm_config = values.get("stm_config")
        im_config = values.get("im_config")

        # If both STM and IM configs exist, ensure they use the same Redis server
        if stm_config and im_config:
            if stm_config.host != im_config.host or stm_config.port != im_config.port:
                raise ValueError(
                    "STM and IM configurations should use the same Redis server"
                )

        return values

    def to_config_object(self, config):
        """Apply validated model values to a config object.

        Args:
            config: The config object to update

        Returns:
            The updated config object
        """
        # Update top-level parameters
        if hasattr(config, "cleanup_interval"):
            config.cleanup_interval = self.cleanup_interval
        if hasattr(config, "memory_priority_decay"):
            config.memory_priority_decay = self.memory_priority_decay

        # Update STM config
        if self.stm_config and hasattr(config, "stm_config"):
            for key, value in self.stm_config.dict().items():
                if hasattr(config.stm_config, key):
                    setattr(config.stm_config, key, value)

        # Update IM config
        if self.im_config and hasattr(config, "im_config"):
            for key, value in self.im_config.dict().items():
                if hasattr(config.im_config, key):
                    setattr(config.im_config, key, value)

        # Update LTM config
        if self.ltm_config and hasattr(config, "ltm_config"):
            for key, value in self.ltm_config.dict().items():
                if hasattr(config.ltm_config, key):
                    setattr(config.ltm_config, key, value)

        # Update autoencoder config
        if self.autoencoder_config and hasattr(config, "autoencoder_config"):
            for key, value in self.autoencoder_config.dict().items():
                if hasattr(config.autoencoder_config, key):
                    setattr(config.autoencoder_config, key, value)

        return config
