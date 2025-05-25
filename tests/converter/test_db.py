"""
Tests for the database connection manager.
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from converter.config import ConverterConfig
from converter.db import DatabaseManager
from data.models import AgentModel, AgentStateModel


@pytest.fixture
def config():
    """Create a test configuration."""
    return ConverterConfig(
        use_mock_redis=True, validate=True, error_handling="fail", batch_size=100
    )


@pytest.fixture
def db_manager(tmp_path, config):
    """Create a test database manager with a temporary database."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    return manager


@pytest.fixture
def initialized_db(db_manager):
    """Create and initialize a database manager."""
    db_manager.initialize()
    return db_manager


def test_database_initialization(db_manager):
    """Test database initialization."""
    db_manager.initialize()
    assert db_manager._engine is not None
    assert db_manager._Session is not None

    # Test engine configuration
    assert db_manager._engine.pool.size() == 5
    assert db_manager._engine.pool._max_overflow == 10
    assert db_manager._engine.pool._timeout == 30
    assert db_manager._engine.pool._recycle == 1800


def test_session_context_manager(db_manager):
    """Test session context manager."""
    db_manager.initialize()
    with db_manager.session() as session:
        assert session is not None
        # Test that session is working
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_session_rollback_on_error(db_manager):
    """Test that session rolls back on error."""
    db_manager.initialize()
    with pytest.raises(SQLAlchemyError):
        with db_manager.session() as session:
            # Force an error
            session.execute(text("SELECT * FROM nonexistent_table"))
            session.commit()


def test_validate_database_with_error_handling_skip(config, tmp_path):
    """Test database validation with error handling set to skip."""
    config.error_handling = "skip"
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    manager.initialize()
    assert not manager.validate_database()


def test_validate_database_with_error_handling_fail(config, tmp_path):
    """Test database validation with error handling set to fail."""
    config.error_handling = "fail"
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(str(db_path), config)
    manager.initialize()
    with pytest.raises(ValueError):
        manager.validate_database()


def test_validate_database_with_valid_schema(initialized_db):
    """Test database validation with a valid schema."""
    # Create required tables
    with initialized_db.session() as session:
        session.execute(
            text(
                """
            CREATE TABLE agents (
                agent_id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
            )
        )
        session.execute(
            text(
                """
            CREATE TABLE agent_states (
                state_id INTEGER PRIMARY KEY,
                agent_id INTEGER
            )
        """
            )
        )
        session.execute(
            text(
                """
            CREATE TABLE agent_actions (
                action_id INTEGER PRIMARY KEY,
                agent_id INTEGER
            )
        """
            )
        )
        session.execute(
            text(
                """
            CREATE TABLE social_interactions (
                interaction_id INTEGER PRIMARY KEY,
                agent_id INTEGER
            )
        """
            )
        )
        session.execute(
            text(
                """
            CREATE TABLE simulations (
                simulation_id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
            )
        )
        session.commit()

    assert initialized_db.validate_database()


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


def test_in_memory_database():
    """Test in-memory database initialization."""
    config = ConverterConfig(
        use_mock_redis=True, validate=True, error_handling="fail", batch_size=100
    )
    manager = DatabaseManager("sqlite:///:memory:", config)
    manager.initialize()
    assert manager._engine is not None
    assert manager._Session is not None


def test_database_reinitialization(db_manager):
    """Test that reinitializing the database doesn't cause issues."""
    db_manager.initialize()
    first_engine = db_manager._engine
    db_manager.initialize()
    assert db_manager._engine is not None
    assert db_manager._engine is not first_engine  # Should create new engine


def test_session_after_close(db_manager):
    """Test that session creation fails after closing the connection."""
    db_manager.initialize()
    db_manager.close()
    with pytest.raises(RuntimeError, match="Database connection is closed"):
        with db_manager.session():
            pass
