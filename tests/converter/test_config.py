"""
Tests for the configuration system.
"""

import pytest

from converter.config import ConverterConfig
from converter.mapping import MemoryTypeMapper
from converter.tiering import (
    ImportanceAwareTieringStrategy,
    SimpleTieringStrategy,
    StepBasedTieringStrategy,
)


def test_default_config():
    """Test default configuration values."""
    config = ConverterConfig()
    assert config.use_mock_redis is True
    assert config.validate is True
    assert config.error_handling == "skip"
    assert config.batch_size == 100
    assert config.show_progress is True
    assert config.import_mode == "full"
    assert config.selective_agents is None
    assert config.tiering_strategy_type == "simple"
    assert isinstance(config.tiering_strategy, SimpleTieringStrategy)
    assert isinstance(config.memory_type_mapper, MemoryTypeMapper)
    assert config.memory_type_mapping == {
        "AgentStateModel": "state",
        "ActionModel": "action",
        "SocialInteractionModel": "interaction",
    }


def test_custom_config():
    """Test custom configuration values."""
    custom_mapper = MemoryTypeMapper()
    config = ConverterConfig(
        use_mock_redis=False,
        validate=False,
        error_handling="fail",
        batch_size=200,
        show_progress=False,
        import_mode="incremental",
        selective_agents=[1, 2, 3],
        tiering_strategy_type="importance_aware",
        memory_type_mapper=custom_mapper,
    )
    assert config.use_mock_redis is False
    assert config.validate is False
    assert config.error_handling == "fail"
    assert config.batch_size == 200
    assert config.show_progress is False
    assert config.import_mode == "incremental"
    assert config.selective_agents == [1, 2, 3]
    assert config.tiering_strategy_type == "importance_aware"
    assert isinstance(config.tiering_strategy, ImportanceAwareTieringStrategy)
    assert config.memory_type_mapper is custom_mapper


def test_invalid_error_handling():
    """Test invalid error handling mode."""
    with pytest.raises(ValueError, match="Invalid error_handling mode"):
        ConverterConfig(error_handling="invalid")


def test_invalid_import_mode():
    """Test invalid import mode."""
    with pytest.raises(ValueError, match="Invalid import_mode"):
        ConverterConfig(import_mode="invalid")


def test_invalid_batch_size():
    """Test invalid batch size."""
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        ConverterConfig(batch_size=0)


def test_memory_type_mapping_validation():
    """Test memory type mapping validation."""
    # Test missing required model
    with pytest.raises(ValueError, match="Missing required memory type mappings"):
        ConverterConfig(
            memory_type_mapping={"AgentStateModel": "state", "ActionModel": "action"}
        )

    # Test invalid memory type
    with pytest.raises(ValueError, match="Invalid memory types in mapping"):
        ConverterConfig(
            memory_type_mapping={
                "AgentStateModel": "invalid",
                "ActionModel": "action",
                "SocialInteractionModel": "interaction",
            }
        )


def test_tiering_strategy_validation():
    """Test tiering strategy validation."""
    # Test invalid strategy type
    with pytest.raises(ValueError, match="Invalid tiering_strategy_type"):
        ConverterConfig(tiering_strategy_type="invalid")

    # Test custom strategy instance
    custom_strategy = StepBasedTieringStrategy()
    config = ConverterConfig(tiering_strategy=custom_strategy)
    assert config.tiering_strategy is custom_strategy


def test_memory_type_mapper_initialization():
    """Test memory type mapper initialization."""
    # Test default mapper
    config = ConverterConfig()
    assert isinstance(config.memory_type_mapper, MemoryTypeMapper)
    assert config.memory_type_mapper.get_memory_type("AgentStateModel") == "state"

    # Test custom mapper
    custom_mapper = MemoryTypeMapper()
    config = ConverterConfig(memory_type_mapper=custom_mapper)
    assert config.memory_type_mapper is custom_mapper


def test_total_steps():
    """Test total_steps configuration."""
    config = ConverterConfig(total_steps=1000)
    assert config.total_steps == 1000
    
    # Test default value
    config = ConverterConfig()
    assert config.total_steps is None


def test_log_error_handling():
    """Test error handling in log mode."""
    config = ConverterConfig(error_handling="log")
    assert config.error_handling == "log"


def test_default_config_matches_class():
    """Test that DEFAULT_CONFIG matches the default values of ConverterConfig."""
    from converter.config import DEFAULT_CONFIG
    config = ConverterConfig()
    
    assert DEFAULT_CONFIG["use_mock_redis"] == config.use_mock_redis
    assert DEFAULT_CONFIG["validate"] == config.validate
    assert DEFAULT_CONFIG["error_handling"] == config.error_handling
    assert DEFAULT_CONFIG["batch_size"] == config.batch_size
    assert DEFAULT_CONFIG["show_progress"] == config.show_progress
    assert DEFAULT_CONFIG["memory_type_mapping"] == config.memory_type_mapping
    assert DEFAULT_CONFIG["tiering_strategy_type"] == config.tiering_strategy_type
    assert DEFAULT_CONFIG["import_mode"] == config.import_mode
    assert DEFAULT_CONFIG["selective_agents"] == config.selective_agents
