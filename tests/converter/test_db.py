"""
Tests for the database connection manager.
"""

import os
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from converter.db import DatabaseManager
from converter.config import ConverterConfig

@pytest.fixture
def config():
    """Create a test configuration."""
    return ConverterConfig(
        use_mock_redis=True,
        validate=True,
        error_handling='fail',
        batch_size=100
    )

@pytest.fixture
def db_manager(tmp_path, config):
    """Create a test database manager with a temporary database."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    return manager

def test_database_initialization(db_manager):
    """Test database initialization."""
    db_manager.initialize()
    assert db_manager._engine is not None
    assert db_manager._Session is not None

def test_session_context_manager(db_manager):
    """Test session context manager."""
    db_manager.initialize()
    with db_manager.session() as session:
        assert session is not None
        # Test that session is working
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1

def test_validate_database_with_error_handling_skip(config, tmp_path):
    """Test database validation with error handling set to skip."""
    config.error_handling = 'skip'
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    manager.initialize()
    assert not manager.validate_database()

def test_validate_database_with_error_handling_fail(config, tmp_path):
    """Test database validation with error handling set to fail."""
    config.error_handling = 'fail'
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    manager.initialize()
    with pytest.raises(ValueError):
        manager.validate_database()

def test_get_total_steps_empty_db(db_manager):
    """Test getting total steps from empty database."""
    db_manager.initialize()
    assert db_manager.get_total_steps() == 0

def test_get_agent_count_empty_db(db_manager):
    """Test getting agent count from empty database."""
    db_manager.initialize()
    assert db_manager.get_agent_count() == 0

def test_close_connection(db_manager):
    """Test closing database connection."""
    db_manager.initialize()
    db_manager.close()
    assert db_manager._engine is None 