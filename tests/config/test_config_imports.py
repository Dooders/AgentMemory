"""Tests for config module imports and structure."""

import pytest


class TestConfigImports:
    """Test that the config module imports correctly."""
    
    def test_import_config_module(self):
        """Test that the main config module can be imported."""
        import memory.config
        assert memory.config is not None
    
    def test_import_classes(self):
        """Test importing primary classes from config module."""
        from memory.config import (
            MemoryConfig,
            RedisSTMConfig,
            RedisIMConfig,
            SQLiteLTMConfig,
            AutoencoderConfig,
        )
        
        # Test classes exist and can be instantiated
        assert MemoryConfig() is not None
        assert RedisSTMConfig() is not None
        assert RedisIMConfig() is not None
        assert SQLiteLTMConfig() is not None
        assert AutoencoderConfig() is not None
    
    def test_imported_aliases(self):
        """Test that backward compatibility aliases are working."""
        from memory.config import STMConfig, IMConfig, LTMConfig
        from memory.config import RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig
        
        # Check that aliases point to the correct classes
        assert STMConfig is RedisSTMConfig
        assert IMConfig is RedisIMConfig
        assert LTMConfig is SQLiteLTMConfig
    

class TestPydanticImports:
    """Test Pydantic model imports."""
    
    def test_pydantic_imports(self):
        """Test importing Pydantic models from config module."""
        try:
            from memory.config import MemoryConfigModel
            has_pydantic = True
        except ImportError:
            has_pydantic = False
        
        if has_pydantic:
            # If Pydantic is available, test the model
            assert MemoryConfigModel() is not None
        else:
            # Skip this test if Pydantic isn't installed
            pytest.skip("Pydantic not installed, skipping model import test")
    
    def test_pydantic_fallback(self):
        """Test the fallback when Pydantic is not available."""
        # This won't raise an exception even if Pydantic is not installed
        import memory.config
        # The module should be imported successfully regardless of Pydantic 