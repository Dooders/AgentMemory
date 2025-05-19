"""
Tests for the memory type mapping system.
"""

import pytest
from converter.mapping import MemoryTypeMapping, MemoryTypeMapper

def test_default_memory_type_mapping():
    """Test default memory type mapping configuration."""
    mapping = MemoryTypeMapping({
        'AgentStateModel': 'state',
        'ActionModel': 'action',
        'SocialInteractionModel': 'interaction'
    })
    
    assert mapping.model_to_type['AgentStateModel'] == 'state'
    assert mapping.model_to_type['ActionModel'] == 'action'
    assert mapping.model_to_type['SocialInteractionModel'] == 'interaction'
    
    assert 'AgentStateModel' in mapping.required_models
    assert 'ActionModel' in mapping.required_models
    assert 'SocialInteractionModel' in mapping.required_models
    
    assert 'state' in mapping.valid_types
    assert 'action' in mapping.valid_types
    assert 'interaction' in mapping.valid_types

def test_custom_memory_type_mapping():
    """Test custom memory type mapping configuration."""
    custom_models = {'CustomModel'}
    custom_types = {'custom_type'}
    
    mapping = MemoryTypeMapping(
        model_to_type={'CustomModel': 'custom_type'},
        required_models=custom_models,
        valid_types=custom_types
    )
    
    assert mapping.model_to_type['CustomModel'] == 'custom_type'
    assert mapping.required_models == custom_models
    assert mapping.valid_types == custom_types

def test_missing_required_model():
    """Test validation of missing required models."""
    with pytest.raises(ValueError, match="Missing required memory type mappings"):
        MemoryTypeMapping({
            'AgentStateModel': 'state',
            'ActionModel': 'action'
            # Missing SocialInteractionModel
        })

def test_invalid_memory_type():
    """Test validation of invalid memory types."""
    with pytest.raises(ValueError, match="Invalid memory types in mapping"):
        MemoryTypeMapping({
            'AgentStateModel': 'invalid_type',
            'ActionModel': 'action',
            'SocialInteractionModel': 'interaction'
        })

def test_memory_type_mapper():
    """Test memory type mapper functionality."""
    mapper = MemoryTypeMapper()
    
    # Test getting memory type from model
    assert mapper.get_memory_type('AgentStateModel') == 'state'
    assert mapper.get_memory_type('ActionModel') == 'action'
    assert mapper.get_memory_type('SocialInteractionModel') == 'interaction'
    
    # Test getting model from memory type
    assert mapper.get_model_name('state') == 'AgentStateModel'
    assert mapper.get_model_name('action') == 'ActionModel'
    assert mapper.get_model_name('interaction') == 'SocialInteractionModel'
    
    # Test invalid model name
    with pytest.raises(ValueError, match="No memory type mapping for model"):
        mapper.get_memory_type('InvalidModel')
        
    # Test invalid memory type
    with pytest.raises(ValueError, match="No model mapping for memory type"):
        mapper.get_model_name('invalid_type')

def test_custom_memory_type_mapper():
    """Test memory type mapper with custom mapping."""
    custom_mapping = {
        'CustomStateModel': 'state',
        'CustomActionModel': 'action',
        'CustomInteractionModel': 'interaction'
    }
    
    custom_required_models = {'CustomStateModel', 'CustomActionModel', 'CustomInteractionModel'}
    custom_valid_types = {'state', 'action', 'interaction'}
    
    mapper = MemoryTypeMapper(
        mapping=custom_mapping,
        required_models=custom_required_models,
        valid_types=custom_valid_types
    )
    
    assert mapper.get_memory_type('CustomStateModel') == 'state'
    assert mapper.get_memory_type('CustomActionModel') == 'action'
    assert mapper.get_memory_type('CustomInteractionModel') == 'interaction'

def test_memory_data_validation():
    """Test memory data validation."""
    mapper = MemoryTypeMapper()
    
    # Test valid state data
    valid_state = {
        'agent_id': 1,
        'step_number': 1,
        'state_data': {'key': 'value'}
    }
    assert mapper.validate_memory_data('state', valid_state) is True
    
    # Test valid action data
    valid_action = {
        'agent_id': 1,
        'step_number': 1,
        'action_type': 'move'
    }
    assert mapper.validate_memory_data('action', valid_action) is True
    
    # Test valid interaction data
    valid_interaction = {
        'agent_id': 1,
        'step_number': 1,
        'interaction_type': 'talk'
    }
    assert mapper.validate_memory_data('interaction', valid_interaction) is True
    
    # Test invalid data
    invalid_data = {'agent_id': 1}
    assert mapper.validate_memory_data('state', invalid_data) is False
    assert mapper.validate_memory_data('action', invalid_data) is False
    assert mapper.validate_memory_data('interaction', invalid_data) is False 