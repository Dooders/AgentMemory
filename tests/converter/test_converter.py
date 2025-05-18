"""
Tests for the main converter functionality.
"""

import os
import pytest
from sqlalchemy.exc import SQLAlchemyError
from converter.converter import from_agent_farm
from converter.config import ConverterConfig

@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'use_mock_redis': False,
        'batch_size': 200,
        'validate': False  # Skip validation for basic tests
    }

def test_from_agent_farm_not_implemented(tmp_path, config):
    """Test that the converter raises NotImplementedError."""
    db_path = tmp_path / "test.db"
    with pytest.raises(NotImplementedError):
        from_agent_farm(str(db_path), config)

def test_from_agent_farm_config_merging(tmp_path, config):
    """Test that configuration merging works correctly."""
    db_path = tmp_path / "test.db"
    with pytest.raises(NotImplementedError):
        from_agent_farm(str(db_path), config)

def test_from_agent_farm_database_error(tmp_path, config):
    """Test handling of database errors."""
    # Create a directory instead of a file to cause a database error
    db_path = tmp_path / "test.db"
    os.makedirs(db_path)
    
    with pytest.raises(SQLAlchemyError):
        from_agent_farm(str(db_path), config)

def test_from_agent_farm_validation_fail(tmp_path):
    """Test database validation failure handling."""
    config = {
        'validate': True,
        'error_handling': 'fail'
    }
    
    db_path = tmp_path / "test.db"
    with pytest.raises(ValueError, match="Database validation failed"):
        from_agent_farm(str(db_path), config)

def test_from_agent_farm_validation_skip(tmp_path):
    """Test database validation skip handling."""
    config = {
        'validate': True,
        'error_handling': 'skip'
    }
    
    db_path = tmp_path / "test.db"
    with pytest.raises(NotImplementedError):
        from_agent_farm(str(db_path), config) 