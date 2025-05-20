"""
Tests for the main converter functionality.
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
        "use_mock_redis": True,  # Use mock Redis for testing
        "batch_size": 200,
        "validate": True,
        "error_handling": "fail",
    }


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock()
    manager.get_total_steps.return_value = 1000
    manager.get_agent_count.return_value = 2
    manager.validate_database.return_value = True
    return manager


@pytest.fixture
def mock_agent_importer():
    """Create a mock agent importer."""
    importer = MagicMock()
    importer.import_agents.return_value = [MagicMock(agent_id=1), MagicMock(agent_id=2)]
    return importer


@pytest.fixture
def mock_memory_importer():
    """Create a mock memory importer."""
    importer = MagicMock()
    importer.import_memories.side_effect = [
        [MagicMock(agent_id=1, memory_id=1)],
        [MagicMock(agent_id=2, memory_id=2)],
    ]
    return importer


def test_from_agent_farm_successful_import(
    tmp_path, config, mock_db_manager, mock_agent_importer, mock_memory_importer
):
    """Test successful import of agents and memories."""
    db_path = tmp_path / "test.db"

    # Create mock memory stores for agent 1
    mock_stm_store1 = MagicMock()
    mock_stm_store1.count.return_value = 1  # One memory in STM
    mock_im_store1 = MagicMock()
    mock_im_store1.count.return_value = 0
    mock_ltm_store1 = MagicMock()
    mock_ltm_store1.count.return_value = 0

    # Create mock memory stores for agent 2
    mock_stm_store2 = MagicMock()
    mock_stm_store2.count.return_value = 1  # One memory in STM
    mock_im_store2 = MagicMock()
    mock_im_store2.count.return_value = 0
    mock_ltm_store2 = MagicMock()
    mock_ltm_store2.count.return_value = 0

    # Create two distinct mock agents with memory stores
    mock_agent1 = MagicMock()
    mock_agent1.agent_id = 1
    mock_agent1.stm_store = mock_stm_store1
    mock_agent1.im_store = mock_im_store1
    mock_agent1.ltm_store = mock_ltm_store1

    mock_agent2 = MagicMock()
    mock_agent2.agent_id = 2
    mock_agent2.stm_store = mock_stm_store2
    mock_agent2.im_store = mock_im_store2
    mock_agent2.ltm_store = mock_ltm_store2

    # Configure mock agent importer to return 2 agents
    mock_agent_importer.import_agents.return_value = [
        MagicMock(agent_id=1),
        MagicMock(agent_id=2),
    ]

    # Create mock memories with all required attributes
    mock_memory1 = MagicMock()
    mock_memory1.agent_id = 1
    mock_memory1.memory_id = 1
    mock_memory1.memory_type = "stm"
    mock_memory1.step_number = 1
    mock_memory1.metadata = {}
    mock_memory1.tier = "stm"
    mock_memory1.created_at = "2023-01-01"
    mock_memory1.updated_at = "2023-01-01"

    mock_memory2 = MagicMock()
    mock_memory2.agent_id = 2
    mock_memory2.memory_id = 2
    mock_memory2.memory_type = "stm"
    mock_memory2.step_number = 1
    mock_memory2.metadata = {}
    mock_memory2.tier = "stm"
    mock_memory2.created_at = "2023-01-01"
    mock_memory2.updated_at = "2023-01-01"

    # Configure mock memory importer to return one memory per agent
    mock_memory_importer.import_memories.side_effect = [
        [mock_memory1],  # One memory for agent 1
        [mock_memory2],  # One memory for agent 2
    ]

    with patch(
        "converter.converter.DatabaseManager", return_value=mock_db_manager
    ), patch(
        "converter.converter.AgentImporter", return_value=mock_agent_importer
    ), patch(
        "converter.converter.MemoryImporter", return_value=mock_memory_importer
    ), patch(
        "memory.core.AgentMemorySystem"
    ) as mock_memory_system:

        # Configure mock memory system with two distinct agents
        mock_memory_system.return_value.agents = {1: mock_agent1, 2: mock_agent2}

        memory_system = from_agent_farm(str(db_path), config)

        assert isinstance(memory_system, AgentMemorySystem)
        assert len(memory_system.agents) == 2
        mock_db_manager.initialize.assert_called_once()
        mock_db_manager.validate_database.assert_called_once()
        mock_agent_importer.import_agents.assert_called_once()
        assert mock_memory_importer.import_memories.call_count == 2


def test_from_agent_farm_database_error(tmp_path, config):
    """Test handling of database errors."""
    db_path = tmp_path / "test.db"
    os.makedirs(db_path)  # Create directory instead of file to cause error

    with pytest.raises(SQLAlchemyError):
        from_agent_farm(str(db_path), config)


def test_from_agent_farm_validation_fail(tmp_path, config, mock_db_manager):
    """Test database validation failure handling."""
    db_path = tmp_path / "test.db"
    mock_db_manager.validate_database.return_value = False

    with patch("converter.converter.DatabaseManager", return_value=mock_db_manager):
        with pytest.raises(ValueError, match="Database validation failed"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_validation_skip(tmp_path, config, mock_db_manager):
    """Test database validation skip handling."""
    db_path = tmp_path / "test.db"
    mock_db_manager.validate_database.return_value = False
    config["error_handling"] = "skip"

    with patch(
        "converter.converter.DatabaseManager", return_value=mock_db_manager
    ), patch("converter.converter.AgentImporter"), patch(
        "converter.converter.MemoryImporter"
    ):

        memory_system = from_agent_farm(str(db_path), config)
        assert isinstance(memory_system, AgentMemorySystem)


def test_from_agent_farm_memory_import_error(
    tmp_path, config, mock_db_manager, mock_agent_importer
):
    """Test handling of memory import errors."""
    db_path = tmp_path / "test.db"
    mock_memory_importer = MagicMock()
    mock_memory_importer.import_memories.side_effect = ValueError("Import failed")

    with patch(
        "converter.converter.DatabaseManager", return_value=mock_db_manager
    ), patch(
        "converter.converter.AgentImporter", return_value=mock_agent_importer
    ), patch(
        "converter.converter.MemoryImporter", return_value=mock_memory_importer
    ):

        with pytest.raises(ValueError, match="Failed to import memories for agent"):
            from_agent_farm(str(db_path), config)


def test_from_agent_farm_import_verification(
    tmp_path, config, mock_db_manager, mock_agent_importer, mock_memory_importer
):
    """Test import verification."""
    db_path = tmp_path / "test.db"

    # Create mock memory stores for agent 1
    mock_stm_store1 = MagicMock()
    mock_stm_store1.count.return_value = 0  # No memories to avoid memory count issues
    mock_im_store1 = MagicMock()
    mock_im_store1.count.return_value = 0
    mock_ltm_store1 = MagicMock()
    mock_ltm_store1.count.return_value = 0

    # Create mock agent with memory stores
    mock_agent1 = MagicMock()
    mock_agent1.agent_id = 1
    mock_agent1.stm_store = mock_stm_store1
    mock_agent1.im_store = mock_im_store1
    mock_agent1.ltm_store = mock_ltm_store1

    # Configure mock agent importer to return 2 agents
    mock_agent_importer.import_agents.return_value = [
        MagicMock(agent_id=1),
        MagicMock(agent_id=2),
    ]

    # Configure mock memory importer to return no memories
    # to avoid memory count mismatch
    mock_memory_importer.import_memories.side_effect = [
        [],  # No memories for agent 1
        [],  # No memories for agent 2
    ]

    with patch(
        "converter.converter.DatabaseManager", return_value=mock_db_manager
    ), patch(
        "converter.converter.AgentImporter", return_value=mock_agent_importer
    ), patch(
        "converter.converter.MemoryImporter", return_value=mock_memory_importer
    ), patch(
        "memory.core.AgentMemorySystem"
    ) as mock_memory_system:

        # Mock memory system to simulate verification failure
        # Only one agent in the system when we expect two
        mock_memory_system.return_value.agents = {1: mock_agent1}

        # Test should fail with either agent count or memory count mismatch
        with pytest.raises(ValueError) as exc_info:
            from_agent_farm(str(db_path), config)

        # Verify the error message contains either agent count or memory count mismatch
        error_msg = str(exc_info.value)
        assert any(
            msg in error_msg
            for msg in [
                "Import verification failed: agent count mismatch",
                "Import verification failed: memory count mismatch",
            ]
        )
