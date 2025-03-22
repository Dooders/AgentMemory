"""Pytest configuration for agent memory tests.

This module contains shared fixtures and setup for all tests in the agent memory system.
"""

import logging
import pytest


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