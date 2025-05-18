"""
Tests for the memory import system.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from converter.memory_import import MemoryImporter, MemoryMetadata
from converter.config import ConverterConfig
from converter.db import DatabaseManager
from converter.tiering import StepBasedTieringStrategy
from converter.mapping import MemoryTypeMapper

@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock(spec=DatabaseManager)
    
    # Mock memory models
    class MockStateModel:
        def __init__(self, id, agent_id, step_number, type, status, properties, settings):
            self.id = id
            self.agent_id = agent_id
            self.step_number = step_number
            self.current_step = 100
            self.total_steps = 1000
            self.type = type
            self.status = status
            self.properties = properties
            self.settings = settings
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            
    class MockActionModel:
        def __init__(self, id, agent_id, step_number, action_type, status, properties):
            self.id = id
            self.agent_id = agent_id
            self.step_number = step_number
            self.current_step = 100
            self.total_steps = 1000
            self.action_type = action_type
            self.status = status
            self.properties = properties
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            
    class MockInteractionModel:
        def __init__(self, id, agent_id, step_number, interaction_type, status, properties):
            self.id = id
            self.agent_id = agent_id
            self.step_number = step_number
            self.current_step = 100
            self.total_steps = 1000
            self.interaction_type = interaction_type
            self.status = status
            self.properties = properties
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            
    manager.AgentStateModel = MockStateModel
    manager.ActionModel = MockActionModel
    manager.SocialInteractionModel = MockInteractionModel
    return manager

@pytest.fixture
def config():
    """Create a test configuration."""
    return ConverterConfig(
        validate=True,
        error_handling="fail",
        batch_size=2,
        import_mode="full"
    )

@pytest.fixture
def tiering_strategy():
    """Create a test tiering strategy."""
    return StepBasedTieringStrategy()

@pytest.fixture
def memory_type_mapper():
    """Create a test memory type mapper."""
    return MemoryTypeMapper()

def test_import_memories(mock_db_manager, config, tiering_strategy, memory_type_mapper):
    """Test successful memory import."""
    # Create test memories
    memories = [
        mock_db_manager.AgentStateModel(
            id=1,
            agent_id=1,
            step_number=50,
            type="state",
            status="active",
            properties={"key": "value"},
            settings={"setting": "value"}
        ),
        mock_db_manager.ActionModel(
            id=2,
            agent_id=1,
            step_number=60,
            action_type="move",
            status="completed",
            properties={"direction": "north"}
        )
    ]
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.side_effect = [
        [memories[0]],  # First batch (state)
        [],  # Empty batch to end
        [memories[1]],  # First batch (action)
        []  # Empty batch to end
    ]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import memories
    importer = MemoryImporter(
        mock_db_manager,
        config,
        tiering_strategy,
        memory_type_mapper
    )
    imported_memories = importer.import_memories(1)
    
    # Verify results
    assert len(imported_memories) == 2
    assert isinstance(imported_memories[0], MemoryMetadata)
    assert imported_memories[0].memory_id == 1
    assert imported_memories[0].memory_type == "state"
    assert imported_memories[1].memory_id == 2
    assert imported_memories[1].memory_type == "action"

def test_import_memories_with_validation_error(
    mock_db_manager,
    config,
    tiering_strategy,
    memory_type_mapper
):
    """Test memory import with validation error."""
    # Create invalid memory
    memory = mock_db_manager.AgentStateModel(
        id=1,
        agent_id=None,  # Invalid missing agent_id
        step_number=50,
        type="state",
        status="active",
        properties={},
        settings={}
    )
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.return_value = [memory]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import memories
    importer = MemoryImporter(
        mock_db_manager,
        config,
        tiering_strategy,
        memory_type_mapper
    )
    with pytest.raises(ValueError, match="AgentStateModel must have an agent ID"):
        importer.import_memories(1)

def test_import_memories_with_error_handling(
    mock_db_manager,
    tiering_strategy,
    memory_type_mapper
):
    """Test memory import with different error handling modes."""
    # Create test memory
    memory = mock_db_manager.AgentStateModel(
        id=1,
        agent_id=1,
        step_number=50,
        type="state",
        status="active",
        properties={},
        settings={}
    )
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.return_value = [memory]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Test skip mode
    config_skip = ConverterConfig(error_handling="skip")
    importer = MemoryImporter(
        mock_db_manager,
        config_skip,
        tiering_strategy,
        memory_type_mapper
    )
    imported_memories = importer.import_memories(1)
    assert len(imported_memories) == 1
    
    # Test log mode
    config_log = ConverterConfig(error_handling="log")
    importer = MemoryImporter(
        mock_db_manager,
        config_log,
        tiering_strategy,
        memory_type_mapper
    )
    with patch('converter.memory_import.logger') as mock_logger:
        imported_memories = importer.import_memories(1)
        assert len(imported_memories) == 1
        mock_logger.error.assert_not_called()

def test_memory_metadata_extraction(
    mock_db_manager,
    config,
    tiering_strategy,
    memory_type_mapper
):
    """Test extraction of memory metadata."""
    # Create test memory with various fields
    memory = mock_db_manager.AgentStateModel(
        id=1,
        agent_id=1,
        step_number=50,
        type="state",
        status="active",
        properties={"key": "value"},
        settings={"setting": "value"}
    )
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.return_value = [memory]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import memories
    importer = MemoryImporter(
        mock_db_manager,
        config,
        tiering_strategy,
        memory_type_mapper
    )
    imported_memories = importer.import_memories(1)
    
    # Verify metadata
    assert len(imported_memories) == 1
    metadata = imported_memories[0].metadata
    assert metadata['type'] == "state"
    assert metadata['status'] == "active"
    assert metadata['properties'] == {"key": "value"}
    assert metadata['settings'] == {"setting": "value"} 