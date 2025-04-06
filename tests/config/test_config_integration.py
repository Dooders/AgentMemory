"""Integration tests for the config module components."""

import pytest
import os
import tempfile
import json
from memory.config import (
    MemoryConfig,
    RedisSTMConfig,
    RedisIMConfig,
    SQLiteLTMConfig,
    AutoencoderConfig,
)

try:
    from memory.config.models import (
        MemoryConfigModel,
        RedisSTMConfigModel,
        RedisIMConfigModel,
        SQLiteLTMConfigModel,
    )
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class TestConfigCreation:
    """Test cases for creating and using configs together."""
    
    def test_memory_config_with_custom_subconfigs(self):
        """Test creating a MemoryConfig with custom subconfigs."""
        # Create custom subconfigs
        stm_config = RedisSTMConfig(
            host="redis1.example.com",
            port=6380,
            memory_limit=5000,
        )
        
        im_config = RedisIMConfig(
            host="redis1.example.com",  # Same Redis server as STM
            port=6380,
            memory_limit=20000,
            db=2,  # Different DB number
        )
        
        ltm_config = SQLiteLTMConfig(
            db_path="./custom_ltm.db",
            compression_level=2,
        )
        
        autoencoder_config = AutoencoderConfig(
            stm_dim=1024,
            im_dim=512,
            ltm_dim=256,
        )
        
        # Create master config with custom subconfigs
        config = MemoryConfig(
            cleanup_interval=200,
            memory_priority_decay=0.9,
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            autoencoder_config=autoencoder_config,
        )
        
        # Verify all configs are correctly set
        assert config.cleanup_interval == 200
        assert config.memory_priority_decay == 0.9
        
        # STM config checks
        assert config.stm_config.host == "redis1.example.com"
        assert config.stm_config.port == 6380
        assert config.stm_config.memory_limit == 5000
        
        # IM config checks
        assert config.im_config.host == "redis1.example.com"
        assert config.im_config.port == 6380
        assert config.im_config.memory_limit == 20000
        assert config.im_config.db == 2
        
        # LTM config checks
        assert config.ltm_config.db_path == "./custom_ltm.db"
        assert config.ltm_config.compression_level == 2
        
        # Autoencoder config checks
        assert config.autoencoder_config.stm_dim == 1024
        assert config.autoencoder_config.im_dim == 512
        assert config.autoencoder_config.ltm_dim == 256


class TestConfigIO:
    """Test reading and writing configs from/to files."""
    
    def test_config_serialization(self):
        """Test serializing a config to a dictionary and back."""
        # Create a config with non-default values
        config = MemoryConfig(
            cleanup_interval=300,
            stm_config=RedisSTMConfig(host="redis.example.com"),
            ltm_config=SQLiteLTMConfig(db_path="./test.db"),
        )
        
        # Convert to dictionary (manually as there's no built-in method)
        config_dict = {
            "cleanup_interval": config.cleanup_interval,
            "memory_priority_decay": config.memory_priority_decay,
            "enable_memory_hooks": config.enable_memory_hooks,
            "logging_level": config.logging_level,
            "stm_config": {
                "host": config.stm_config.host,
                "port": config.stm_config.port,
                "memory_limit": config.stm_config.memory_limit,
                "ttl": config.stm_config.ttl,
                "db": config.stm_config.db,
                "namespace": config.stm_config.namespace,
                "password": config.stm_config.password,
                "use_mock": config.stm_config.use_mock,
            },
            "ltm_config": {
                "db_path": config.ltm_config.db_path,
                "compression_level": config.ltm_config.compression_level,
                "batch_size": config.ltm_config.batch_size,
                "table_prefix": config.ltm_config.table_prefix,
                "use_mock": config.ltm_config.use_mock,
            },
        }
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(config_dict, f)
            config_file = f.name
            
        try:
            # Read the file
            with open(config_file, "r") as f:
                loaded_dict = json.load(f)
                
            # Create a new config from the loaded dictionary
            new_config = MemoryConfig(
                cleanup_interval=loaded_dict["cleanup_interval"],
                memory_priority_decay=loaded_dict["memory_priority_decay"],
                enable_memory_hooks=loaded_dict["enable_memory_hooks"],
                logging_level=loaded_dict["logging_level"],
                stm_config=RedisSTMConfig(**loaded_dict["stm_config"]),
                ltm_config=SQLiteLTMConfig(**loaded_dict["ltm_config"]),
            )
            
            # Verify the config was loaded correctly
            assert new_config.cleanup_interval == config.cleanup_interval
            assert new_config.memory_priority_decay == config.memory_priority_decay
            assert new_config.stm_config.host == config.stm_config.host
            assert new_config.ltm_config.db_path == config.ltm_config.db_path
        finally:
            # Clean up
            os.unlink(config_file)


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not installed")
class TestConfigValidation:
    """Test validation between config classes and Pydantic models."""
    
    def test_pydantic_model_to_config(self):
        """Test converting from Pydantic models to regular config objects."""
        # Create a Pydantic model
        model = MemoryConfigModel(
            cleanup_interval=250,
            memory_priority_decay=0.85,
            stm_config=RedisSTMConfigModel(
                host="redis.example.com",
                port=6380,
                memory_limit=5000,
            ),
            ltm_config=SQLiteLTMConfigModel(
                db_path="./validated.db",
            ),
        )
        
        # Convert to config object
        config = MemoryConfig()
        model.to_config_object(config)
        
        # Verify the conversion worked
        assert config.cleanup_interval == 250
        assert config.memory_priority_decay == 0.85
        assert config.stm_config.host == "redis.example.com"
        assert config.stm_config.port == 6380
        assert config.stm_config.memory_limit == 5000
        assert config.ltm_config.db_path == "./validated.db"
    
    def test_config_with_invalid_values(self):
        """Test that invalid configurations are rejected."""
        # These should fail validation
        with pytest.raises(ValueError):
            MemoryConfigModel(
                cleanup_interval=0,  # Invalid: must be positive
            )
        
        with pytest.raises(ValueError):
            MemoryConfigModel(
                memory_priority_decay=2.0,  # Invalid: must be between 0 and 1
            )
        
        with pytest.raises(ValueError):
            RedisSTMConfigModel(
                port=70000,  # Invalid: must be between 1 and 65535
            )
        
        with pytest.raises(ValueError):
            RedisIMConfigModel(
                memory_limit=500,  # Invalid: must be at least 1000
            )
        
        with pytest.raises(ValueError):
            SQLiteLTMConfigModel(
                db_path="",  # Invalid: must not be empty
            ) 