"""Configuration classes for the agent memory system."""


class RedisSTMConfig:
    """Configuration for Redis-based Short-Term Memory storage."""

    def __init__(self, **kwargs):
        # Default configuration values
        self.host = "localhost"
        self.port = 6379
        self.memory_limit = 10000
        self.ttl = 3600  # 1 hour
        self.db = 0  # Redis DB number
        self.namespace = "agent-stm"
        self.password = None
        self.use_mock = False

        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RedisIMConfig:
    """Configuration for Redis-based Intermediate Memory storage."""

    def __init__(self, **kwargs):
        # Default configuration values
        self.host = "localhost"
        self.port = 6379
        self.memory_limit = 50000
        self.ttl = 86400  # 24 hours
        self.db = 1  # Redis DB number
        self.namespace = "agent-im"
        self.password = None
        self.use_mock = False

        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class SQLiteLTMConfig:
    """Configuration for SQLite-based Long-Term Memory storage."""

    def __init__(self, **kwargs):
        # Default configuration values
        self.db_path = "./ltm.db"
        self.compression_level = 1
        self.batch_size = 100
        self.table_prefix = "ltm"
        self.use_mock = False

        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class AutoencoderConfig:
    """Configuration for the memory autoencoder."""

    def __init__(self, **kwargs):
        # Default configuration values
        self.input_dim = 64
        self.stm_dim = 768
        self.im_dim = 384
        self.ltm_dim = 128
        self.use_neural_embeddings = False
        self.model_path = "./models/autoencoder.pt"

        # Text embedding options
        self.embedding_type = "autoencoder"  # Options: "autoencoder" or "text"
        self.text_model_name = (
            "all-MiniLM-L6-v2"  # Default to smaller model for faster startup
        )

        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class MemoryConfig:
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

        # Update with any provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
