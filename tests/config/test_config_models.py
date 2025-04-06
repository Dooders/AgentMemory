"""Unit tests for memory config Pydantic models."""

import pytest
from pydantic import ValidationError

try:
    from memory.config.models import (
        RedisSTMConfigModel,
        RedisIMConfigModel,
        SQLiteLTMConfigModel,
        AutoencoderConfigModel,
        MemoryConfigModel,
    )
    from memory.config import (
        MemoryConfig,
        RedisSTMConfig,
        RedisIMConfig,
        SQLiteLTMConfig,
    )
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


# Skip all tests in this module if Pydantic is not installed
pytestmark = pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")


class TestRedisSTMConfigModel:
    """Test cases for RedisSTMConfigModel."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model = RedisSTMConfigModel()
        assert model.host == "localhost"
        assert model.port == 6379
        assert model.memory_limit == 10000
        assert model.ttl == 3600

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        model = RedisSTMConfigModel(
            host="redis.example.com",
            port=6380,
            memory_limit=5000,
            ttl=1800,
        )
        assert model.host == "redis.example.com"
        assert model.port == 6380
        assert model.memory_limit == 5000
        assert model.ttl == 1800

    def test_port_validation(self):
        """Test port number validation."""
        # Valid port numbers
        RedisSTMConfigModel(port=1)
        RedisSTMConfigModel(port=65535)

        # Invalid port numbers
        with pytest.raises(ValidationError):
            RedisSTMConfigModel(port=0)
        with pytest.raises(ValidationError):
            RedisSTMConfigModel(port=65536)

    def test_memory_limit_validation(self):
        """Test memory limit validation."""
        # Valid memory limits
        RedisSTMConfigModel(memory_limit=100)
        RedisSTMConfigModel(memory_limit=1000000)

        # Invalid memory limits
        with pytest.raises(ValidationError):
            RedisSTMConfigModel(memory_limit=99)

    def test_ttl_validation(self):
        """Test TTL validation."""
        # Valid TTL values
        RedisSTMConfigModel(ttl=60)
        RedisSTMConfigModel(ttl=86400)

        # Invalid TTL values
        with pytest.raises(ValidationError):
            RedisSTMConfigModel(ttl=59)


class TestRedisIMConfigModel:
    """Test cases for RedisIMConfigModel."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model = RedisIMConfigModel()
        assert model.host == "localhost"
        assert model.port == 6379
        assert model.memory_limit == 50000
        assert model.ttl == 86400

    def test_memory_limit_validation(self):
        """Test memory limit validation."""
        # Valid memory limits
        RedisIMConfigModel(memory_limit=1000)
        RedisIMConfigModel(memory_limit=1000000)

        # Invalid memory limits
        with pytest.raises(ValidationError):
            RedisIMConfigModel(memory_limit=999)

    def test_ttl_validation(self):
        """Test TTL validation."""
        # Valid TTL values
        RedisIMConfigModel(ttl=3600)
        RedisIMConfigModel(ttl=604800)

        # Invalid TTL values
        with pytest.raises(ValidationError):
            RedisIMConfigModel(ttl=3599)


class TestSQLiteLTMConfigModel:
    """Test cases for SQLiteLTMConfigModel."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model = SQLiteLTMConfigModel()
        assert model.db_path == "./ltm.db"

    def test_db_path_validation(self):
        """Test database path validation."""
        # Valid paths
        SQLiteLTMConfigModel(db_path="./custom.db")
        SQLiteLTMConfigModel(db_path="/tmp/memory.db")

        # Invalid paths
        with pytest.raises(ValidationError):
            SQLiteLTMConfigModel(db_path="")


class TestAutoencoderConfigModel:
    """Test cases for AutoencoderConfigModel."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model = AutoencoderConfigModel()
        assert model.stm_dim == 768
        assert model.im_dim == 384
        assert model.ltm_dim == 128

    def test_dimension_validation(self):
        """Test dimension validation."""
        # Valid dimensions
        AutoencoderConfigModel(stm_dim=768, im_dim=384, ltm_dim=128)
        
        # Invalid: im_dim >= stm_dim
        with pytest.raises(ValidationError):
            AutoencoderConfigModel(stm_dim=384, im_dim=384, ltm_dim=128)
        
        # Invalid: ltm_dim >= im_dim
        with pytest.raises(ValidationError):
            AutoencoderConfigModel(stm_dim=768, im_dim=384, ltm_dim=384)
        
        # Invalid: ltm_dim >= stm_dim
        with pytest.raises(ValidationError):
            AutoencoderConfigModel(stm_dim=768, im_dim=256, ltm_dim=768)


class TestMemoryConfigModel:
    """Test cases for MemoryConfigModel."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model = MemoryConfigModel()
        assert model.cleanup_interval == 100
        assert model.memory_priority_decay == 0.95
        assert model.stm_config is None
        assert model.im_config is None
        assert model.ltm_config is None
        assert model.autoencoder_config is None

    def test_cleanup_interval_validation(self):
        """Test cleanup interval validation."""
        # Valid intervals
        MemoryConfigModel(cleanup_interval=1)
        MemoryConfigModel(cleanup_interval=1000)
        
        # Invalid intervals
        with pytest.raises(ValidationError):
            MemoryConfigModel(cleanup_interval=0)
        with pytest.raises(ValidationError):
            MemoryConfigModel(cleanup_interval=-1)

    def test_memory_priority_decay_validation(self):
        """Test memory priority decay validation."""
        # Valid decay values
        MemoryConfigModel(memory_priority_decay=0.0)
        MemoryConfigModel(memory_priority_decay=0.5)
        MemoryConfigModel(memory_priority_decay=1.0)
        
        # Invalid decay values
        with pytest.raises(ValidationError):
            MemoryConfigModel(memory_priority_decay=-0.1)
        with pytest.raises(ValidationError):
            MemoryConfigModel(memory_priority_decay=1.1)

    def test_subconfig_validation(self):
        """Test validation of subconfigs."""
        # Valid configs
        model = MemoryConfigModel(
            stm_config=RedisSTMConfigModel(host="redis1.example.com"),
            im_config=RedisIMConfigModel(host="redis1.example.com"),
        )
        assert model.stm_config.host == "redis1.example.com"
        assert model.im_config.host == "redis1.example.com"
        
        # Invalid: different Redis servers for STM and IM
        with pytest.raises(ValidationError):
            MemoryConfigModel(
                stm_config=RedisSTMConfigModel(host="redis1.example.com"),
                im_config=RedisIMConfigModel(host="redis2.example.com"),
            )

    def test_to_config_object(self):
        """Test conversion to plain config object."""
        # Create Pydantic model
        model = MemoryConfigModel(
            cleanup_interval=200,
            memory_priority_decay=0.9,
            stm_config=RedisSTMConfigModel(
                host="redis.example.com",
                port=6380,
                memory_limit=5000,
                ttl=1800,
            ),
            ltm_config=SQLiteLTMConfigModel(db_path="./custom.db"),
        )
        
        # Convert to plain config object
        config = MemoryConfig()
        model.to_config_object(config)
        
        # Check values were transferred
        assert config.cleanup_interval == 200
        assert config.memory_priority_decay == 0.9
        assert config.stm_config.host == "redis.example.com"
        assert config.stm_config.port == 6380
        assert config.stm_config.memory_limit == 5000
        assert config.stm_config.ttl == 1800
        assert config.ltm_config.db_path == "./custom.db" 