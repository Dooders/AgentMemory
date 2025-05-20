"""Unit tests for memory config classes."""

import pytest
import os
from memory.config import (
    MemoryConfig,
    RedisSTMConfig,
    RedisIMConfig,
    SQLiteLTMConfig,
    AutoencoderConfig,
)


class TestRedisSTMConfig:
    """Test cases for RedisSTMConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RedisSTMConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.memory_limit == 10000
        assert config.ttl == 3600
        assert config.db == 0
        assert config.namespace == "agent-stm"
        assert config.password is None
        assert config.use_mock is True

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = RedisSTMConfig(
            host="redis.example.com",
            port=6380,
            memory_limit=5000,
            ttl=1800,
            db=1,
            namespace="custom-stm",
            password="secret",
            use_mock=True,
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.memory_limit == 5000
        assert config.ttl == 1800
        assert config.db == 1
        assert config.namespace == "custom-stm"
        assert config.password == "secret"
        assert config.use_mock is True

    def test_partial_override(self):
        """Test that partial overrides work correctly."""
        config = RedisSTMConfig(host="redis.example.com", port=6380)
        assert config.host == "redis.example.com"
        assert config.port == 6380
        # Other values should remain at defaults
        assert config.memory_limit == 10000
        assert config.ttl == 3600


class TestRedisIMConfig:
    """Test cases for RedisIMConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RedisIMConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.memory_limit == 50000
        assert config.ttl == 86400
        assert config.db == 1
        assert config.namespace == "agent-im"
        assert config.password is None
        assert config.use_mock is True

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = RedisIMConfig(
            host="redis.example.com",
            port=6380,
            memory_limit=25000,
            ttl=43200,
            db=2,
            namespace="custom-im",
            password="secret",
            use_mock=True,
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.memory_limit == 25000
        assert config.ttl == 43200
        assert config.db == 2
        assert config.namespace == "custom-im"
        assert config.password == "secret"
        assert config.use_mock is True


class TestSQLiteLTMConfig:
    """Test cases for SQLiteLTMConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SQLiteLTMConfig()
        assert config.db_path == "./ltm.db"
        assert config.compression_level == 1
        assert config.batch_size == 100
        assert config.table_prefix == "ltm"
        assert config.use_mock is True

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = SQLiteLTMConfig(
            db_path="./custom.db",
            compression_level=2,
            batch_size=200,
            table_prefix="custom_ltm",
            use_mock=True,
        )
        assert config.db_path == "./custom.db"
        assert config.compression_level == 2
        assert config.batch_size == 200
        assert config.table_prefix == "custom_ltm"
        assert config.use_mock is True


class TestAutoencoderConfig:
    """Test cases for AutoencoderConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AutoencoderConfig()
        assert config.input_dim == 64
        assert config.stm_dim == 768
        assert config.im_dim == 384
        assert config.ltm_dim == 128
        assert config.use_neural_embeddings is False
        assert config.model_path == "./models/autoencoder.pt"
        assert config.embedding_type == "autoencoder"
        assert config.text_model_name == "all-MiniLM-L6-v2"

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = AutoencoderConfig(
            input_dim=128,
            stm_dim=1024,
            im_dim=512,
            ltm_dim=256,
            use_neural_embeddings=True,
            model_path="./custom/model.pt",
            embedding_type="text",
            text_model_name="all-mpnet-base-v2",
        )
        assert config.input_dim == 128
        assert config.stm_dim == 1024
        assert config.im_dim == 512
        assert config.ltm_dim == 256
        assert config.use_neural_embeddings is True
        assert config.model_path == "./custom/model.pt"
        assert config.embedding_type == "text"
        assert config.text_model_name == "all-mpnet-base-v2"


class TestMemoryConfig:
    """Test cases for MemoryConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = MemoryConfig()
        assert config.cleanup_interval == 100
        assert config.memory_priority_decay == 0.95
        assert config.enable_memory_hooks is True
        assert config.logging_level == "INFO"

        # Test default subconfigs are initialized correctly
        assert isinstance(config.stm_config, RedisSTMConfig)
        assert isinstance(config.im_config, RedisIMConfig)
        assert isinstance(config.ltm_config, SQLiteLTMConfig)
        assert isinstance(config.autoencoder_config, AutoencoderConfig)

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        stm_config = RedisSTMConfig(host="redis1.example.com")
        im_config = RedisIMConfig(host="redis2.example.com")
        ltm_config = SQLiteLTMConfig(db_path="./custom.db")
        autoencoder_config = AutoencoderConfig(stm_dim=1024)
        
        config = MemoryConfig(
            cleanup_interval=200,
            memory_priority_decay=0.9,
            enable_memory_hooks=False,
            logging_level="DEBUG",
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            autoencoder_config=autoencoder_config,
        )
        
        assert config.cleanup_interval == 200
        assert config.memory_priority_decay == 0.9
        assert config.enable_memory_hooks is False
        assert config.logging_level == "DEBUG"
        
        # Test custom subconfigs
        assert config.stm_config is stm_config
        assert config.stm_config.host == "redis1.example.com"
        assert config.im_config is im_config
        assert config.im_config.host == "redis2.example.com"
        assert config.ltm_config is ltm_config
        assert config.ltm_config.db_path == "./custom.db"
        assert config.autoencoder_config is autoencoder_config
        assert config.autoencoder_config.stm_dim == 1024 