"""Pydantic models for configuration validation in the agent memory system."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

class STMConfig(BaseModel):
    """Configuration for Short-Term Memory (STM)."""
    host: str = "localhost"
    port: int = 6379
    memory_limit: int = 10000
    ttl: int = 3600  # 1 hour
    db: int = 0
    namespace: str = "agent-stm"
    password: Optional[str] = None
    
    @validator('port')
    def port_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Port must be a positive integer")
        return v
    
    @validator('memory_limit')
    def memory_limit_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Memory limit must be a positive integer")
        return v
    
    @validator('ttl')
    def ttl_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("TTL must be a non-negative integer")
        return v

class IMConfig(BaseModel):
    """Configuration for Intermediate Memory (IM)."""
    host: str = "localhost"
    port: int = 6379
    memory_limit: int = 50000
    ttl: int = 86400  # 24 hours
    db: int = 1
    namespace: str = "agent-im"
    password: Optional[str] = None
    
    @validator('port')
    def port_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Port must be a positive integer")
        return v
    
    @validator('memory_limit')
    def memory_limit_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Memory limit must be a positive integer")
        return v
    
    @validator('ttl')
    def ttl_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("TTL must be a non-negative integer")
        return v

class LTMConfig(BaseModel):
    """Configuration for Long-Term Memory (LTM)."""
    db_path: str = "./ltm.db"
    compression_level: int = 1
    batch_size: int = 100
    table_prefix: str = "ltm"
    
    @validator('db_path')
    def db_path_must_be_valid(cls, v):
        if not v:
            raise ValueError("Database path cannot be empty")
        return v

class AutoencoderConfig(BaseModel):
    """Configuration for the embedding autoencoder."""
    input_dim: int = 64
    stm_dim: int = 768
    im_dim: int = 384
    ltm_dim: int = 128
    use_neural_embeddings: bool = False
    model_path: str = "./models/autoencoder.pt"
    
    @validator('stm_dim', 'im_dim', 'ltm_dim', 'input_dim')
    def dimensions_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Dimension must be a positive integer")
        return v

class MemoryConfigModel(BaseModel):
    """Main configuration model for the memory system."""
    cleanup_interval: int = 100
    memory_priority_decay: float = 0.95
    enable_memory_hooks: bool = True
    logging_level: str = "INFO"
    stm_config: STMConfig = Field(default_factory=STMConfig)
    im_config: IMConfig = Field(default_factory=IMConfig)
    ltm_config: LTMConfig = Field(default_factory=LTMConfig)
    autoencoder_config: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    
    @validator('cleanup_interval')
    def cleanup_interval_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Cleanup interval must be a positive integer")
        return v
    
    @validator('memory_priority_decay')
    def memory_priority_decay_must_be_valid(cls, v):
        if v <= 0.0 or v > 1.0:
            raise ValueError("Memory priority decay must be between 0.0 and 1.0 (exclusive)")
        return v

    def to_config_object(self, existing_config=None):
        """Convert the validated model to a configuration object.
        
        Args:
            existing_config: Optional existing configuration object to update
            
        Returns:
            Updated configuration object
        """
        if existing_config is None:
            from agent_memory.config import MemoryConfig
            config = MemoryConfig()
        else:
            config = existing_config
            
        # Update top-level attributes
        config.cleanup_interval = self.cleanup_interval
        config.memory_priority_decay = self.memory_priority_decay
        config.enable_memory_hooks = self.enable_memory_hooks
        config.logging_level = self.logging_level
        
        # Update nested configurations
        for config_key in ['stm_config', 'im_config', 'ltm_config', 'autoencoder_config']:
            if hasattr(config, config_key):
                src_config = getattr(self, config_key)
                dest_config = getattr(config, config_key)
                
                for key, value in src_config.dict().items():
                    setattr(dest_config, key, value)
        
        return config 