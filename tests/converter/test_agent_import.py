"""
Tests for the agent import system.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from converter.agent_import import AgentImporter, AgentMetadata
from converter.config import ConverterConfig
from converter.db import DatabaseManager

@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock(spec=DatabaseManager)
    
    # Mock agent model
    class MockAgentModel:
        def __init__(self, id, name, type, status, properties, settings):
            self.id = id
            self.name = name
            self.type = type
            self.status = status
            self.properties = properties
            self.settings = settings
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            
    manager.AgentModel = MockAgentModel
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

def test_import_agents(mock_db_manager, config):
    """Test successful agent import."""
    # Create test agents
    agents = [
        mock_db_manager.AgentModel(
            id=1,
            name="Agent1",
            type="basic",
            status="active",
            properties={"key": "value"},
            settings={"setting": "value"}
        ),
        mock_db_manager.AgentModel(
            id=2,
            name="Agent2",
            type="advanced",
            status="inactive",
            properties={},
            settings={}
        )
    ]
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.side_effect = [
        [agents[0], agents[1]],  # First batch
        []  # Empty batch to end
    ]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import agents
    importer = AgentImporter(mock_db_manager, config)
    imported_agents = importer.import_agents()
    
    # Verify results
    assert len(imported_agents) == 2
    assert isinstance(imported_agents[0], AgentMetadata)
    assert imported_agents[0].agent_id == 1
    assert imported_agents[0].name == "Agent1"
    assert imported_agents[1].agent_id == 2
    assert imported_agents[1].name == "Agent2"

def test_import_agents_with_validation_error(mock_db_manager, config):
    """Test agent import with validation error."""
    # Create invalid agent
    agent = mock_db_manager.AgentModel(
        id=1,
        name="",  # Invalid empty name
        type="basic",
        status="active",
        properties={},
        settings={}
    )
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.return_value = [agent]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import agents
    importer = AgentImporter(mock_db_manager, config)
    with pytest.raises(ValueError, match="Agent must have a name"):
        importer.import_agents()

def test_import_agents_with_error_handling(mock_db_manager):
    """Test agent import with different error handling modes."""
    # Create test agent
    agent = mock_db_manager.AgentModel(
        id=1,
        name="Agent1",
        type="basic",
        status="active",
        properties={},
        settings={}
    )
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.return_value = [agent]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Test skip mode
    config_skip = ConverterConfig(error_handling="skip")
    importer = AgentImporter(mock_db_manager, config_skip)
    imported_agents = importer.import_agents()
    assert len(imported_agents) == 1
    
    # Test log mode
    config_log = ConverterConfig(error_handling="log")
    importer = AgentImporter(mock_db_manager, config_log)
    with patch('converter.agent_import.logger') as mock_logger:
        imported_agents = importer.import_agents()
        assert len(imported_agents) == 1
        mock_logger.error.assert_not_called()

def test_selective_agent_import(mock_db_manager, config):
    """Test importing specific agents."""
    config.selective_agents = [1, 3]
    
    # Create test agents
    agents = [
        mock_db_manager.AgentModel(
            id=1,
            name="Agent1",
            type="basic",
            status="active",
            properties={},
            settings={}
        ),
        mock_db_manager.AgentModel(
            id=2,
            name="Agent2",
            type="basic",
            status="active",
            properties={},
            settings={}
        ),
        mock_db_manager.AgentModel(
            id=3,
            name="Agent3",
            type="basic",
            status="active",
            properties={},
            settings={}
        )
    ]
    
    # Mock session and query
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.offset.return_value.limit.return_value.all.side_effect = [
        [agents[0], agents[2]],  # First batch (only agents 1 and 3)
        []  # Empty batch to end
    ]
    mock_session.query.return_value = mock_query
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    
    # Import agents
    importer = AgentImporter(mock_db_manager, config)
    imported_agents = importer.import_agents()
    
    # Verify results
    assert len(imported_agents) == 2
    assert imported_agents[0].agent_id == 1
    assert imported_agents[1].agent_id == 3 