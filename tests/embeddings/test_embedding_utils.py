"""Unit tests for agent_memory.embeddings utility functions.

This test suite covers:
1. Text embedding utility functions
2. Edge cases for embedding operations
3. Integration of embedding utilities with other components
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
from memory.embeddings.utils import (
    cosine_similarity,
    flatten_dict,
    object_to_text,
    filter_dict_keys
)

#################################
# Utility Function Tests
#################################

def test_cosine_similarity_edge_cases():
    """Test cosine similarity with edge cases."""
    # Test with zero vectors
    zero_vec = np.zeros(5)
    unit_vec = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    
    assert cosine_similarity(zero_vec, unit_vec) == 0.0
    assert cosine_similarity(unit_vec, zero_vec) == 0.0
    assert cosine_similarity(zero_vec, zero_vec) == 0.0
    
    # Test with very small values
    small_vec1 = np.array([1e-10, 2e-10, 3e-10])
    small_vec2 = np.array([3e-10, 2e-10, 1e-10])
    similarity = cosine_similarity(small_vec1, small_vec2)
    assert 0.0 <= similarity <= 1.0
    
    # Test with identical vectors of different magnitudes
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([2.0, 4.0, 6.0])  # Same direction, different magnitude
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 1e-10  # Should be ~1.0

def test_flatten_dict_complex_cases():
    """Test flatten_dict with more complex cases."""
    # Test with nested empty dictionaries
    nested_empty = {"a": {}, "b": {"c": {}}}
    flattened = flatten_dict(nested_empty)
    assert flattened == {"a": {}, "b.c": {}}
    
    # Test with lists and other non-dict values
    complex_dict = {
        "a": [1, 2, 3],
        "b": {"c": [4, 5], "d": {"e": "text"}}
    }
    flattened = flatten_dict(complex_dict)
    assert flattened["a"] == [1, 2, 3]
    assert flattened["b.c"] == [4, 5]
    assert flattened["b.d.e"] == "text"
    
    # Test with numeric keys
    numeric_keys = {
        1: "one",
        2: {3: "three", 4: {5: "five"}}
    }
    flattened = flatten_dict(numeric_keys)
    assert flattened[1] == "one"
    assert flattened["2.3"] == "three"
    assert flattened["2.4.5"] == "five"

def test_object_to_text_custom_objects():
    """Test object_to_text with custom and complex objects."""
    # Test with None
    assert object_to_text(None) == ""
    
    # Test with custom class
    class TestClass:
        def __str__(self):
            return "TestClass instance"
    
    test_obj = TestClass()
    assert "TestClass instance" in object_to_text(test_obj)
    
    # Test with nested custom objects
    nested_obj = {
        "name": "test",
        "object": test_obj,
        "items": [1, test_obj, "text"]
    }
    text = object_to_text(nested_obj)
    assert "name: test" in text
    assert "TestClass instance" in text
    
    # Test with empty containers
    assert "empty" in object_to_text({})
    assert "empty" in object_to_text([])
    
    # Test with special keys
    special_dict = {
        "position": {"x": 10, "y": 20},
        "inventory": [],
        "status": None
    }
    text = object_to_text(special_dict)
    assert "coordinates" in text
    assert "empty inventory" in text

def test_filter_dict_keys_edge_cases():
    """Test filter_dict_keys with edge cases."""
    # Test with empty dict
    assert filter_dict_keys({}, {"a", "b"}) == {}
    
    # Test with empty exclude set
    test_dict = {"a": 1, "b": 2}
    assert filter_dict_keys(test_dict, set()) == test_dict
    
    # Test with exclude keys not in dict
    assert filter_dict_keys(test_dict, {"c", "d"}) == test_dict
    
    # Test excluding all keys
    assert filter_dict_keys(test_dict, {"a", "b"}) == {}
    
    # Test with nested dictionaries (should only filter top-level)
    nested_dict = {
        "a": 1,
        "b": {"a": 2, "c": 3},
        "c": 4
    }
    result = filter_dict_keys(nested_dict, {"a"})
    assert "a" not in result
    assert "b" in result
    assert "a" in result["b"]  # Nested "a" should not be filtered

#################################
# Integration Tests
#################################

def test_object_to_text_for_embedding():
    """Test how object_to_text formats objects for embedding."""
    # Test complex nested structure with specific format needs
    complex_obj = {
        "user": {
            "name": "Alice",
            "location": {"city": "New York", "coordinates": [40.7128, -74.0060]}
        },
        "inventory": ["laptop", "phone", "keys"],
        "status": "active",
        "history": [
            {"action": "login", "timestamp": "2023-01-01T12:00:00"},
            {"action": "view_page", "timestamp": "2023-01-01T12:05:00"}
        ]
    }
    
    text = object_to_text(complex_obj)
    
    # Check expected special formatting
    assert "has laptop" in text
    assert "has phone" in text
    assert "has keys" in text
    assert "coordinates" in text
    assert "New York" in text
    
    # Check that important information is preserved
    assert "Alice" in text
    assert "active" in text
    assert "login" in text
    assert "view_page" in text

def test_text_processing_for_embeddings():
    """Test text processing specific to embedding generation."""
    # Simple cases
    assert "test" == object_to_text("test")
    assert "123" == object_to_text(123)
    assert "True" == object_to_text(True)
    
    # Lists of different types
    assert "items: 1, 2, 3" == object_to_text([1, 2, 3])
    assert "items: apple, banana, orange" == object_to_text(["apple", "banana", "orange"])
    
    # Dictionary with inventory
    inventory_dict = {"inventory": ["sword", "shield", "potion"]}
    assert "has sword, shield, potion" in object_to_text(inventory_dict)
    
    # Dictionary with position
    position_dict = {"position": {"room": "kitchen", "x": 5, "y": 10}}
    position_text = object_to_text(position_dict)
    assert "room is kitchen" in position_text
    assert "coordinates" in position_text
    assert "5" in position_text
    assert "10" in position_text

def test_object_to_text_realistic_agent_state():
    """Test object_to_text with realistic agent state information."""
    agent_state = {
        "id": "agent_1",
        "position": {
            "room": "living_room",
            "x": 12.5,
            "y": 8.3,
            "facing": "north"
        },
        "inventory": ["book", "glasses", "remote"],
        "status": {
            "health": 95,
            "energy": 80,
            "mood": "content"
        },
        "goals": [
            {"description": "Find the keys", "priority": "high"},
            {"description": "Water the plants", "priority": "medium"}
        ],
        "knowledge": {
            "living_room": {
                "objects": ["sofa", "tv", "bookshelf"],
                "exits": ["kitchen", "hallway"]
            }
        }
    }
    
    text = object_to_text(agent_state)
    
    # Check that important information is preserved in a queryable format
    assert "living_room" in text
    assert "north" in text
    assert "has book" in text
    assert "has glasses" in text
    assert "health" in text and "95" in text
    assert "Find the keys" in text
    assert "sofa" in text and "tv" in text
    assert "kitchen" in text and "hallway" in text


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 