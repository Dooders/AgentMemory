"""
Tests for edge cases and additional coverage of the converter module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from converter.config import ConverterConfig
from converter.converter import from_agent_farm
from memory.core import AgentMemorySystem


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        "use_mock_redis": True,
        "batch_size": 200,
        "validate": True,
        "error_handling": "fail",
    }


def test_from_agent_farm_with_empty_database(tmp_path, config):
    """Test handling of empty database."""
    db_path = tmp_path / "empty.db"
    mock_db_manager = MagicMock()
    mock_db_manager.get_total_steps.return_value = 0
    mock_db_manager.get_agent_count.return_value = 0
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager):
        memory_system = from_agent_farm(str(db_path), config)
        assert len(memory_system.agents) == 0


def test_from_agent_farm_with_large_dataset(tmp_path, config):
    """Test handling of large dataset."""
    db_path = tmp_path / "large.db"
    mock_db_manager = MagicMock()
    mock_db_manager.get_total_steps.return_value = 1000000
    mock_db_manager.get_agent_count.return_value = 1000
    mock_db_manager.validate_database.return_value = True
    
    # Create many mock agents and memories
    mock_agents = [MagicMock(agent_id=i) for i in range(1000)]
    mock_memories = [MagicMock(agent_id=i, memory_id=j) for i in range(1000) for j in range(100)]
    
    mock_agent_importer = MagicMock()
    mock_agent_importer.import_agents.return_value = mock_agents
    
    mock_memory_importer = MagicMock()
    mock_memory_importer.import_memories.side_effect = [
        [m for m in mock_memories if m.agent_id == i] for i in range(1000)
    ]
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer), \
         patch("converter.converter.MemoryImporter", return_value=mock_memory_importer), \
         patch("memory.core.AgentMemorySystem") as mock_memory_system:
        memory_system = from_agent_farm(str(db_path), config)
        assert mock_memory_importer.import_memories.call_count == 1000


def test_from_agent_farm_with_invalid_memory_types(tmp_path, config):
    """Test handling of invalid memory types."""
    db_path = tmp_path / "test.db"
    mock_db_manager = MagicMock()
    mock_db_manager.validate_database.return_value = True
    
    mock_agent_importer = MagicMock()
    mock_agent_importer.import_agents.return_value = [MagicMock(agent_id=1)]
    
    mock_memory_importer = MagicMock()
    invalid_memory = MagicMock()
    invalid_memory.memory_type = "invalid_type"
    mock_memory_importer.import_memories.return_value = [invalid_memory]
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer), \
         patch("converter.converter.MemoryImporter", return_value=mock_memory_importer):
        with pytest.raises(ValueError, match="Invalid memory type"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_with_duplicate_agent_ids(tmp_path, config):
    """Test handling of duplicate agent IDs."""
    db_path = tmp_path / "test.db"
    mock_db_manager = MagicMock()
    mock_db_manager.validate_database.return_value = True
    
    mock_agent_importer = MagicMock()
    duplicate_agents = [MagicMock(agent_id=1), MagicMock(agent_id=1)]
    mock_agent_importer.import_agents.return_value = duplicate_agents
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer):
        with pytest.raises(ValueError, match="Duplicate agent ID"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_with_corrupted_database(tmp_path, config):
    """Test handling of corrupted database."""
    db_path = tmp_path / "corrupted.db"
    mock_db_manager = MagicMock()
    mock_db_manager.initialize.side_effect = SQLAlchemyError("Database corrupted")
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager):
        with pytest.raises(SQLAlchemyError, match="Database corrupted"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_with_memory_system_error(tmp_path, config):
    """Test handling of memory system errors."""
    db_path = tmp_path / "test.db"
    mock_db_manager = MagicMock()
    mock_db_manager.validate_database.return_value = True
    
    mock_agent_importer = MagicMock()
    mock_agent_importer.import_agents.return_value = [MagicMock(agent_id=1)]
    
    mock_memory_importer = MagicMock()
    mock_memory_importer.import_memories.return_value = [MagicMock(agent_id=1, memory_id=1)]
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer), \
         patch("converter.converter.MemoryImporter", return_value=mock_memory_importer), \
         patch("memory.core.AgentMemorySystem.get_instance", side_effect=Exception("Memory system error")):
        with pytest.raises(Exception, match="Memory system error"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_with_custom_memory_config(tmp_path, config):
    """Test with custom memory configuration."""
    db_path = tmp_path / "test.db"
    config["memory_config"] = {
        "use_mock_redis": False,
        "logging_level": "DEBUG",
        "stm_config": {
            "memory_limit": 5000,
            "ttl": 43200,
            "namespace": "custom-stm"
        }
    }
    
    mock_db_manager = MagicMock()
    mock_db_manager.validate_database.return_value = True
    
    mock_agent_importer = MagicMock()
    mock_agent_importer.import_agents.return_value = [MagicMock(agent_id=1)]
    
    mock_memory_importer = MagicMock()
    mock_memory_importer.import_memories.return_value = [MagicMock(agent_id=1, memory_id=1)]
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer), \
         patch("converter.converter.MemoryImporter", return_value=mock_memory_importer), \
         patch("memory.core.AgentMemorySystem") as mock_memory_system:
        from_agent_farm(str(db_path), config)
        # Verify memory system was configured with custom settings
        mock_memory_system.get_instance.assert_called_once()
        call_args = mock_memory_system.get_instance.call_args[0][0]
        assert call_args.use_mock_redis is False
        assert call_args.logging_level == "DEBUG"
        assert call_args.stm_config.memory_limit == 5000


def test_from_agent_farm_resource_cleanup(tmp_path, config):
    """Test proper resource cleanup after import."""
    db_path = tmp_path / "test.db"
    mock_db_manager = MagicMock()
    mock_db_manager.validate_database.return_value = True
    
    mock_agent_importer = MagicMock()
    mock_agent_importer.import_agents.return_value = [MagicMock(agent_id=1)]
    
    mock_memory_importer = MagicMock()
    mock_memory_importer.import_memories.return_value = [MagicMock(agent_id=1, memory_id=1)]
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager), \
         patch("converter.converter.AgentImporter", return_value=mock_agent_importer), \
         patch("converter.converter.MemoryImporter", return_value=mock_memory_importer), \
         patch("memory.core.AgentMemorySystem"):
        from_agent_farm(str(db_path), config)
        # Verify database connection was closed
        mock_db_manager.close.assert_called_once()


def test_from_agent_farm_cleanup_on_error(tmp_path, config):
    """Test resource cleanup when errors occur."""
    db_path = tmp_path / "test.db"
    mock_db_manager = MagicMock()
    mock_db_manager.initialize.side_effect = Exception("Test error")
    
    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager):
        with pytest.raises(Exception):
            from_agent_farm(str(db_path), config)
        # Verify cleanup still occurred
        mock_db_manager.close.assert_called_once() 