"""
Unit tests for the memory import module.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from converter.config import ConverterConfig
from converter.db import DatabaseManager
from converter.mapping import MemoryTypeMapper
from converter.memory_import import MemoryImporter, MemoryMetadata
from converter.tiering import TieringStrategy


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = Mock(spec=DatabaseManager)
    manager.AgentModel = Mock()
    manager.SimulationStepModel = Mock()
    manager.ActionModel = Mock()
    manager.SocialInteractionModel = Mock()
    manager.AgentStateModel = Mock()
    return manager


@pytest.fixture
def mock_config():
    """Create a mock converter config."""
    config = Mock(spec=ConverterConfig)
    config.validate = True
    config.error_handling = "fail"
    config.batch_size = 100
    config.total_steps = 1000
    config.import_mode = "full"
    return config


@pytest.fixture
def mock_tiering_strategy():
    """Create a mock tiering strategy."""
    strategy = Mock(spec=TieringStrategy)
    strategy.determine_tier.return_value = "long_term"
    return strategy


@pytest.fixture
def mock_memory_type_mapper():
    """Create a mock memory type mapper."""
    mapper = Mock(spec=MemoryTypeMapper)
    mapper.required_models = [
        "ActionModel",
        "SocialInteractionModel",
        "AgentStateModel",
    ]
    mapper.get_memory_type.return_value = "test_memory_type"
    return mapper


@pytest.fixture
def memory_importer(
    mock_db_manager, mock_config, mock_tiering_strategy, mock_memory_type_mapper
):
    """Create a MemoryImporter instance with mocked dependencies."""
    return MemoryImporter(
        mock_db_manager, mock_config, mock_tiering_strategy, mock_memory_type_mapper
    )


class TestMemoryMetadata:
    """Test cases for the MemoryMetadata dataclass."""

    def test_memory_metadata_creation(self):
        """Test creating a MemoryMetadata instance."""
        metadata = MemoryMetadata(
            memory_id=1,
            agent_id="123",
            memory_type="test_type",
            step_number=5,
            tier="long_term",
            metadata={"test": "data"},
            created_at="2024-01-01",
            updated_at="2024-01-01",
        )

        assert metadata.memory_id == 1
        assert metadata.agent_id == "123"
        assert metadata.memory_type == "test_type"
        assert metadata.step_number == 5
        assert metadata.tier == "long_term"
        assert metadata.metadata == {"test": "data"}
        assert metadata.created_at == "2024-01-01"
        assert metadata.updated_at == "2024-01-01"


class TestMemoryImporter:
    """Test cases for the MemoryImporter class."""

    def test_import_memories_agent_not_found(self, memory_importer, mock_db_manager):
        """Test importing memories for non-existent agent."""
        # Create a mock session that works as a context manager
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Set up the session context manager
        mock_db_manager.session.return_value = mock_session
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None

        memories = memory_importer.import_memories(agent_id=999)
        assert len(memories) == 0

    def test_import_memory_type_validation_failure(self, memory_importer, mock_config):
        """Test memory validation failure."""
        mock_config.validate = True
        mock_config.error_handling = "fail"

        invalid_memory = Mock()
        invalid_memory.agent_id = None
        invalid_memory.step_number = 1

        with pytest.raises(ValueError):
            memory_importer._import_memory(invalid_memory, "test_type", "ActionModel")

    def test_import_memory_success(self, memory_importer, mock_tiering_strategy):
        """Test successful memory import."""
        memory = Mock()
        memory.agent_id = 123
        memory.step_number = 5
        memory.position_x = 10.0
        memory.position_y = 20.0
        memory.timestamp = datetime.now()

        mock_tiering_strategy.determine_tier.return_value = "long_term"

        metadata = memory_importer._import_memory(memory, "test_type", "ActionModel")

        assert metadata is not None
        assert metadata.agent_id == "123"
        assert metadata.step_number == 5
        assert metadata.tier == "long_term"
        assert "position" in metadata.metadata

    def test_batch_query(self, memory_importer):
        """Test batch query processing."""
        query = Mock()
        query.offset.return_value.limit.return_value.all.side_effect = [
            [Mock() for _ in range(3)],
            [Mock() for _ in range(2)],
            [],
        ]

        batches = list(memory_importer._batch_query(query))
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2

    def test_handle_import_error_fail_mode(self, memory_importer, mock_config):
        """Test error handling in fail mode."""
        mock_config.error_handling = "fail"
        error = ValueError("Test error")

        with pytest.raises(ValueError):
            memory_importer._handle_import_error(error, 123, "ActionModel")

    def test_handle_import_error_log_mode(self, memory_importer, mock_config):
        """Test error handling in log mode."""
        mock_config.error_handling = "log"
        error = ValueError("Test error")

        # Should not raise an exception
        memory_importer._handle_import_error(error, 123, "ActionModel")

    def test_extract_memory_metadata(self, memory_importer):
        """Test metadata extraction from memory."""
        memory = Mock()
        memory.type = "test_type"
        memory.status = "active"
        memory.properties = {"prop1": "value1"}
        memory.settings = {"setting1": "value1"}
        memory.action_type = "test_action"

        metadata = memory_importer._extract_memory_metadata(memory)

        assert metadata["type"] == "test_type"
        assert metadata["status"] == "active"
        assert metadata["properties"] == {"prop1": "value1"}
        assert metadata["settings"] == {"setting1": "value1"}
        assert metadata["action_type"] == "test_action"

    @patch("converter.memory_import.generate_memory_id")
    def test_import_memory_with_generated_id(self, mock_generate_id, memory_importer):
        """Test memory import with generated memory ID."""
        mock_generate_id.return_value = 999
        memory = Mock()
        memory.agent_id = 123
        memory.step_number = 5
        memory.position_x = 10.0
        memory.position_y = 20.0
        memory.timestamp = datetime.now()

        metadata = memory_importer._import_memory(memory, "test_type", "ActionModel")

        assert metadata.memory_id == 999
        mock_generate_id.assert_called_once_with("test_type", 123, 5)
