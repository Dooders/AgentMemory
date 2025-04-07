"""
Test script to verify mock Redis is working correctly.
"""

import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.storage.redis_factory import RedisFactory

def test_mock_redis():
    """Test if mock Redis works properly."""
    print("Creating mock Redis client...")
    mock_client = RedisFactory.create_client(
        client_name="test-mock",
        use_mock=True,
        host="localhost",
        port=6379
    )
    
    # Test basic operations
    print("Testing basic operations...")
    mock_client.set("test_key", "test_value")
    value = mock_client.get("test_key")
    print(f"Retrieved value: {value}")
    
    # Test if this is actually a mock client
    print(f"Client type: {type(mock_client).__name__}")
    print(f"Underlying client type: {type(mock_client._client).__name__}")
    
    print("Mock Redis test completed successfully!")

if __name__ == "__main__":
    test_mock_redis() 