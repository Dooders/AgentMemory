"""
Tests for the configuration system.
"""

import pytest
from converter.config import ConverterConfig
from converter.tiering import StepBasedTieringStrategy, ImportanceAwareTieringStrategy
from converter.mapping import MemoryTypeMapper

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
    assert config.tiering_strategy_type == "step_based"
    assert isinstance(config.tiering_strategy, StepBasedTieringStrategy)
    assert isinstance(config.memory_type_mapper, MemoryTypeMapper)
    assert config.memory_type_mapping == {
        'AgentStateModel': 'state',
        'ActionModel': 'action',
        'SocialInteractionModel': 'interaction'
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
        memory_type_mapper=custom_mapper
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
        ConverterConfig(memory_type_mapping={
            'AgentStateModel': 'state',
            'ActionModel': 'action'
        })
    
    # Test invalid memory type
    with pytest.raises(ValueError, match="Invalid memory types in mapping"):
        ConverterConfig(memory_type_mapping={
            'AgentStateModel': 'invalid',
            'ActionModel': 'action',
            'SocialInteractionModel': 'interaction'
        })

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
    assert config.memory_type_mapper.get_memory_type('AgentStateModel') == 'state'
    
    # Test custom mapper
    custom_mapper = MemoryTypeMapper()
    config = ConverterConfig(memory_type_mapper=custom_mapper)
    assert config.memory_type_mapper is custom_mapper 