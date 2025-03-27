"""Configuration classes for the agent memory system."""


class BaseConfig:
    """Base configuration class that all other config classes inherit from.
    
    This class provides common functionality and properties for all
    configuration classes in the memory system.
    """
    
    def __init__(self, **kwargs):
        """Initialize with keyword arguments.
        
        Args:
            **kwargs: Key-value pairs to set as attributes
        """
        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert the configuration object to a dictionary.
        
        Returns:
            Dict containing the configuration settings
        """
        return {
            key: value 
            for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }


class RedisSTMConfig(BaseConfig):
    """Configuration for Redis-based Short-Term Memory storage."""
    
    def __init__(self, **kwargs):
        # Default configuration values
        self.host = "localhost"
        self.port = 6379
        self.memory_limit = 10000
        self.ttl = 3600  # 1 hour
        self.db = 0      # Redis DB number
        self.namespace = "agent-stm"
        self.password = None
        
        # Handle redis_url for convenience
        if "redis_url" in kwargs:
            redis_url = kwargs.pop("redis_url")
            # Parse Redis URL and set individual properties
            # Format: redis://username:password@host:port/db
            if redis_url.startswith("redis://"):
                from urllib.parse import urlparse
                parsed = urlparse(redis_url)
                if parsed.hostname:
                    self.host = parsed.hostname
                if parsed.port:
                    self.port = parsed.port
                if parsed.password:
                    self.password = parsed.password
                if parsed.path and parsed.path != "/":
                    try:
                        self.db = int(parsed.path.strip("/"))
                    except ValueError:
                        pass
        
        # Initialize base class
        super().__init__(**kwargs)


class RedisIMConfig(BaseConfig):
    """Configuration for Redis-based Intermediate Memory storage."""
    
    def __init__(self, **kwargs):
        # Default configuration values
        self.host = "localhost"
        self.port = 6379
        self.memory_limit = 50000
        self.ttl = 86400  # 24 hours
        self.db = 1       # Redis DB number
        self.namespace = "agent-im"
        self.password = None
        self.embedding_dim = 1536  # Default dimension for embeddings
        
        # Default TTL settings
        self.base_ttl = 86400  # 24 hours baseline TTL
        
        # Handle redis_url for convenience
        if "redis_url" in kwargs:
            redis_url = kwargs.pop("redis_url")
            # Parse Redis URL and set individual properties
            # Format: redis://username:password@host:port/db
            if redis_url.startswith("redis://"):
                from urllib.parse import urlparse
                parsed = urlparse(redis_url)
                if parsed.hostname:
                    self.host = parsed.hostname
                if parsed.port:
                    self.port = parsed.port
                if parsed.password:
                    self.password = parsed.password
                if parsed.path and parsed.path != "/":
                    try:
                        self.db = int(parsed.path.strip("/"))
                    except ValueError:
                        pass
        
        # Initialize base class
        super().__init__(**kwargs)


class SQLiteLTMConfig(BaseConfig):
    """Configuration for SQLite-based Long-Term Memory storage."""
    
    def __init__(self, **kwargs):
        # Default configuration values
        self.db_path = "agent_memory.db"
        self.memory_limit = 1000000
        self.table_prefix = "ltm"
        self.compression_level = 2  # Maximum compression
        
        # Initialize base class
        super().__init__(**kwargs)


class AutoencoderConfig(BaseConfig):
    """Configuration for the memory autoencoder."""
    
    def __init__(self, **kwargs):
        # Default configuration values
        self.enabled = False
        self.model_path = None
        self.input_dim = 1536
        self.compressed_dim = 768
        self.batch_size = 32
        
        # Initialize base class
        super().__init__(**kwargs)


class MemoryConfig(BaseConfig):
    """Configuration for the agent memory system."""

    def __init__(self, **kwargs):
        # Default configuration values
        self.cleanup_interval = 100
        self.memory_priority_decay = 0.95
        self.enable_memory_hooks = True  # Default to enabled
        self.logging_level = "INFO"  # Default logging level

        # Create config objects for each tier with default settings
        self.stm_config = RedisSTMConfig()
        self.im_config = RedisIMConfig()
        self.ltm_config = SQLiteLTMConfig()
        self.autoencoder_config = AutoencoderConfig()
        
        # Initialize base class
        super().__init__(**kwargs)
