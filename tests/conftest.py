"""Pytest configuration for agent memory tests.

This module contains shared fixtures and setup for all tests in the agent memory system.
"""

import logging
import pytest
import sys
import os
from unittest import mock

from memory.config.memory_config import RedisSTMConfig, RedisIMConfig


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test requiring external services"
    )


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging during tests to keep the test output clean."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def patch_redis_config_for_mocking():
    """Patch Redis configuration classes to use MockRedis by default.
    
    Returns:
        tuple: Original init methods for STM and IM to be used for restoration
    """
    # Save original __init__ methods
    original_stm_init = RedisSTMConfig.__init__
    original_im_init = RedisIMConfig.__init__
    
    # Create patched init methods that set use_mock=True by default
    def patched_stm_init(self, **kwargs):
        if 'use_mock' not in kwargs:
            kwargs['use_mock'] = True
        original_stm_init(self, **kwargs)
    
    def patched_im_init(self, **kwargs):
        if 'use_mock' not in kwargs:
            kwargs['use_mock'] = True
        original_im_init(self, **kwargs)
    
    # Apply the patches
    RedisSTMConfig.__init__ = patched_stm_init
    RedisIMConfig.__init__ = patched_im_init
    
    return original_stm_init, original_im_init


def restore_redis_config(original_methods):
    """Restore Redis configuration classes to their original state."""
    RedisSTMConfig.__init__ = original_methods[0]
    RedisIMConfig.__init__ = original_methods[1]


@pytest.fixture
def with_mock_redis():
    """Fixture to ensure tests use MockRedis.
    
    This fixture should be explicitly included in tests that need MockRedis.
    The configuration classes will be patched at the start of the test and
    restored after the test completes.
    """
    original_methods = patch_redis_config_for_mocking()
    yield
    restore_redis_config(original_methods)


# Apply mock Redis by default for all tests except configuration tests
@pytest.fixture(autouse=True)  # Remove session scope to allow per-test checking
def default_mock_redis(request):
    """Use MockRedis by default for all tests except config tests.
    
    This fixture runs for all tests but excludes configuration tests
    which need to verify the original default values.
    """
    # Skip mocking for config tests
    if "test_config_classes" in request.module.__name__:
        # Don't patch anything for config tests
        yield
    else:
        # Patch for all other tests
        original_methods = patch_redis_config_for_mocking()
        yield
        restore_redis_config(original_methods)


@pytest.fixture
def memory_entry():
    """Create a sample memory entry for testing."""
    return {
        "memory_id": "test-memory-1",
        "content": "This is a test memory",
        "timestamp": 1234567890.0,
        "metadata": {
            "compression_level": 1,
            "importance_score": 0.8,
            "retrieval_count": 0,
            "source": "test"
        }
    } 