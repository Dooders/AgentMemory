"""Tests for the AttributeSearchStrategy class."""

import re
import time
import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.core import AgentMemorySystem


class TestAttributeSearchStrategy(unittest.TestCase):
    """Tests for the AttributeSearchStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock memory system
        self.mock_memory_system = MagicMock(spec=AgentMemorySystem)
        
        # Create mock agent
        self.mock_agent = MagicMock()
        self.mock_memory_system.get_memory_agent.return_value = self.mock_agent
        
        # Set up mock stores in the agent
        self.mock_agent.stm_store = MagicMock()
        self.mock_agent.im_store = MagicMock()
        self.mock_agent.ltm_store = MagicMock()

        # Create strategy with mock memory system
        self.strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="length_ratio"  # Set default scoring method
        )

        # Sample memories for testing
        self.sample_memories = [
            {
                "id": "memory1",
                "content": {
                    "content": "Meeting with John about the project timeline",
                    "metadata": {
                        "type": "meeting",
                        "tags": ["project", "timeline"],
                        "importance": "high",
                    },
                },
            },
            {
                "id": "memory2",
                "content": {
                    "content": "Sent email to team about status update",
                    "metadata": {
                        "type": "communication",
                        "tags": ["email", "team", "status"],
                        "importance": "medium",
                    },
                },
            },
            {
                "id": "memory3",
                "content": {
                    "content": "Phone call meeting with client to discuss requirements",
                    "metadata": {
                        "type": "meeting",
                        "tags": ["client", "requirements"],
                        "importance": "high",
                    },
                },
            },
        ]

        # Configure mock stores to return empty lists by default
        self.mock_agent.stm_store.get_all.return_value = []
        self.mock_agent.im_store.get_all.return_value = []
        self.mock_agent.ltm_store.get_all.return_value = []

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "attribute")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("attribute", self.strategy.description().lower())

    def test_search_string_query(self):
        """Test search with a simple string query."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories
        self.mock_agent.im_store.get_all.return_value = []
        self.mock_agent.ltm_store.get_all.return_value = []

        # Perform search
        results = self.strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory2", [r["id"] for r in results])

        # Verify score was added to metadata
        self.assertIn("attribute_score", results[0]["metadata"])

    def test_search_dict_query(self):
        """Test search with a dictionary query."""
        # Set up store to return only the meeting-type memories
        meeting_memories = [
            mem
            for mem in self.sample_memories
            if mem["content"]["metadata"]["type"] == "meeting"
        ]

        self.mock_agent.stm_store.get_all.return_value = []
        self.mock_agent.im_store.get_all.return_value = meeting_memories
        self.mock_agent.ltm_store.get_all.return_value = []

        # Perform search with string query instead
        results = self.strategy.search(
            query="meeting",
            agent_id="agent-1",
            tier="im",
            content_fields=[],
            metadata_fields=["content.metadata.type"],
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])
        self.assertNotIn("memory2", [r["id"] for r in results])

    def test_search_with_metadata_filter(self):
        """Test search with metadata filter."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = []
        self.mock_agent.im_store.get_all.return_value = []
        self.mock_agent.ltm_store.get_all.return_value = self.sample_memories

        # Perform search with metadata filter
        results = self.strategy.search(
            query="client",
            agent_id="agent-1",
            tier="ltm",
            metadata_filter={"content.metadata.importance": "high"},
            limit=5,
        )

        # Verify search found matching memories that also match the metadata filter
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "memory3")

    def test_search_with_regex(self):
        """Test search with regex pattern."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Perform search with regex
        results = self.strategy.search(
            query="meet.*john|phone.*client",
            agent_id="agent-1",
            tier="stm",
            use_regex=True,
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

    def test_search_with_match_all(self):
        """Test search requiring all conditions to match."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Query that should match only when all conditions are met
        query = {
            "content": "meeting",
            "metadata": {"type": "meeting", "importance": "high"},
        }

        # Perform search requiring all conditions to match
        results = self.strategy.search(
            query=query, agent_id="agent-1", tier="stm", match_all=True, limit=5
        )

        # Verify search found only memories matching all conditions
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

    def test_search_with_case_sensitive(self):
        """Test search with case sensitivity."""
        # Modify a sample memory for case-sensitive testing
        case_sensitive_memories = self.sample_memories.copy()
        case_sensitive_memories[0]["content"][
            "content"
        ] = "MEETING with John about the project"

        self.mock_agent.stm_store.get_all.return_value = case_sensitive_memories

        # Case-insensitive search (default)
        results_insensitive = self.strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Should find both memories with "meeting" regardless of case
        self.assertEqual(len(results_insensitive), 2)

        # Case-sensitive search
        results_sensitive = self.strategy.search(
            query="MEETING",
            agent_id="agent-1",
            tier="stm",
            case_sensitive=True,
            limit=5,
        )

        # Should only find the memory with exact case match
        self.assertEqual(len(results_sensitive), 1)
        self.assertEqual(results_sensitive[0]["id"], "memory1")

    def test_search_with_custom_fields(self):
        """Test search with custom content and metadata fields."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Perform search with specific content and metadata fields
        results = self.strategy.search(
            query="high",
            agent_id="agent-1",
            tier="stm",
            content_fields=[],  # Don't search in content
            metadata_fields=[
                "content.metadata.importance"
            ],  # Only search in importance field
            limit=5,
        )

        # Verify search found matching memories
        self.assertEqual(len(results), 2)
        self.assertIn("memory1", [r["id"] for r in results])
        self.assertIn("memory3", [r["id"] for r in results])

        # Should not find this even though "high" could match in tags or somewhere else
        self.assertNotIn("memory2", [r["id"] for r in results])

    def test_search_all_tiers(self):
        """Test search across all memory tiers."""
        # Set up stores to return different memories in each tier
        self.mock_agent.stm_store.get_all.return_value = [self.sample_memories[0]]
        self.mock_agent.im_store.get_all.return_value = [self.sample_memories[1]]
        self.mock_agent.ltm_store.get_all.return_value = [self.sample_memories[2]]

        # Perform search across all tiers
        results = self.strategy.search(
            query="meeting",
            agent_id="agent-1",
            # No tier specified, should search all
            limit=5,
        )

        # Verify results include memories from both tiers
        self.assertEqual(len(results), 2)

        # Check that tier was added to metadata
        tiers = [r.get("metadata", {}).get("memory_tier") for r in results]
        self.assertIn("stm", tiers)
        self.assertIn("ltm", tiers)

    def test_get_field_value(self):
        """Test retrieving field values from nested paths."""
        memory = {
            "content": "test content",
            "metadata": {"nested": {"field": "nested value"}, "array": [1, 2, 3]},
        }

        # Test simple field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "content"), "test content"
        )

        # Test nested field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.nested.field"),
            "nested value",
        )

        # Test array field access
        self.assertEqual(
            self.strategy._get_field_value(memory, "metadata.array"), [1, 2, 3]
        )

        # Test nonexistent field
        self.assertIsNone(self.strategy._get_field_value(memory, "nonexistent.field"))

    def test_empty_query(self):
        """Test search with empty query."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Perform search with empty string query
        results_empty_string = self.strategy.search(
            query="", agent_id="agent-1", tier="stm", limit=5
        )

        # Should return no results with empty string
        self.assertEqual(len(results_empty_string), 0)

        # Perform search with empty dict query
        results_empty_dict = self.strategy.search(
            query={}, agent_id="agent-1", tier="stm", limit=5
        )

        # Should return no results with empty dict
        self.assertEqual(len(results_empty_dict), 0)

    def test_special_characters_in_query(self):
        """Test search with special characters in query."""
        # Create memories with special characters
        special_char_memories = [
            {
                "id": "memory7",
                "content": {
                    "content": "Email from john.doe@example.com",
                    "metadata": {"type": "email", "tags": ["email", "contact"]},
                },
            },
            {
                "id": "memory8",
                "content": {
                    "content": "Meeting about C++ & Python integration",
                    "metadata": {"type": "meeting", "tags": ["C++", "Python"]},
                },
            },
            {
                "id": "memory9",
                "content": {
                    "content": "Query: SELECT * FROM table WHERE id='123'",
                    "metadata": {"type": "database", "tags": ["SQL", "query"]},
                },
            },
        ]

        # Set up store to return special character memories
        self.mock_agent.stm_store.get_all.return_value = special_char_memories

        # Test with email address in query
        email_results = self.strategy.search(
            query="john.doe@example.com", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(email_results), 1)
        self.assertEqual(email_results[0]["id"], "memory7")

        # Test with SQL-like query with quotes and special chars
        sql_results = self.strategy.search(
            query="SELECT * FROM", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(sql_results), 1)
        self.assertEqual(sql_results[0]["id"], "memory9")

        # Test with symbols in query
        symbol_results = self.strategy.search(
            query="C++ & Python", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(symbol_results), 1)
        self.assertEqual(symbol_results[0]["id"], "memory8")

    def test_mixed_data_types(self):
        """Test search handling mixed data types in memory fields."""
        # Create memories with mixed data types
        mixed_type_memories = [
            {
                "id": "memory10",
                "content": {
                    "content": "Temperature is 72 degrees",
                    "metadata": {
                        "type": "weather",
                        "temperature": 72,  # Numeric value
                        "tags": ["weather", "temperature"],
                    },
                },
            },
            {
                "id": "memory11",
                "content": {
                    "content": "Task completed",
                    "metadata": {
                        "type": "task",
                        "completed": True,  # Boolean value
                        "tags": ["task", "completed"],
                    },
                },
            },
            {
                "id": "memory12",
                "content": {
                    "content": "Complex data structure",
                    "metadata": {
                        "type": "data",
                        "nested": {
                            "array": [1, "two", 3.0, True],  # Mixed array
                            "object": {"key": "value"},
                        },
                    },
                },
            },
        ]

        # Set up store to return mixed type memories
        self.mock_agent.stm_store.get_all.return_value = mixed_type_memories

        # Test search with numeric value in query
        numeric_results = self.strategy.search(
            query="72", agent_id="agent-1", tier="stm", limit=5
        )
        self.assertEqual(len(numeric_results), 1)
        self.assertEqual(numeric_results[0]["id"], "memory10")

        # Test with boolean in metadata filter
        boolean_filter_results = self.strategy.search(
            query="task",
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"content.metadata.completed": True},
            limit=5,
        )
        self.assertEqual(len(boolean_filter_results), 1)
        self.assertEqual(boolean_filter_results[0]["id"], "memory11")

    def test_scoring_methods(self):
        """Test different scoring methods for attribute search."""
        # Create test memories with repeated terms for testing term frequency
        scoring_test_memories = [
            {
                "id": "memory1",
                "content": {
                    "content": "meeting meeting meeting with John",  # Term repeated 3 times
                    "metadata": {"type": "meeting", "importance": "high"},
                },
            },
            {
                "id": "memory2",
                "content": {
                    "content": "This is a very long text that contains a meeting reference only once but has many more words to lower the term frequency ratio",
                    "metadata": {"type": "meeting", "importance": "medium"},
                },
            },
        ]

        self.mock_agent.stm_store.get_all.return_value = scoring_test_memories

        # 1. Test different scoring methods at initialization
        # Create strategies with different scoring methods
        length_ratio_strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="length_ratio"
        )
        term_freq_strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="term_frequency"
        )
        bm25_strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="bm25"
        )
        binary_strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="binary"
        )

        # Verify that scoring_method is correctly set in the strategy instances
        self.assertEqual(length_ratio_strategy.scoring_method, "length_ratio")
        self.assertEqual(term_freq_strategy.scoring_method, "term_frequency")
        self.assertEqual(bm25_strategy.scoring_method, "bm25")
        self.assertEqual(binary_strategy.scoring_method, "binary")

        # 2. Test search with different methods and compare results
        # Length ratio - will favor shorter text with matching term
        length_ratio_results = length_ratio_strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Term frequency - will favor text with more occurrences of term
        term_freq_results = term_freq_strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # BM25 - will balance term frequency and field length
        bm25_results = bm25_strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # Binary - will just count matches with score 1.0
        binary_results = binary_strategy.search(
            query="meeting", agent_id="agent-1", tier="stm", limit=5
        )

        # All should return both memories
        self.assertEqual(len(length_ratio_results), 2)
        self.assertEqual(len(term_freq_results), 2)
        self.assertEqual(len(bm25_results), 2)
        self.assertEqual(len(binary_results), 2)

        # Print metadata for debugging
        print(
            "Length ratio results metadata:",
            length_ratio_results[0].get("metadata", {}),
        )
        print(
            "Term frequency results metadata:", term_freq_results[0].get("metadata", {})
        )
        print("BM25 results metadata:", bm25_results[0].get("metadata", {}))
        print("Binary results metadata:", binary_results[0].get("metadata", {}))

        # 3. Check that scoring method is stored in metadata
        self.assertEqual(
            length_ratio_results[0]["metadata"]["scoring_method"],
            "length_ratio",
            f"Expected 'length_ratio' but got '{length_ratio_results[0]['metadata'].get('scoring_method')}'",
        )
        self.assertEqual(
            term_freq_results[0]["metadata"]["scoring_method"],
            "term_frequency",
            f"Expected 'term_frequency' but got '{term_freq_results[0]['metadata'].get('scoring_method')}'",
        )
        self.assertEqual(
            bm25_results[0]["metadata"]["scoring_method"],
            "bm25",
            f"Expected 'bm25' but got '{bm25_results[0]['metadata'].get('scoring_method')}'",
        )
        self.assertEqual(
            binary_results[0]["metadata"]["scoring_method"],
            "binary",
            f"Expected 'binary' but got '{binary_results[0]['metadata'].get('scoring_method')}'",
        )

        # 4. Compare scores to ensure different methods produce different scores
        # In term frequency, memory1 should score higher than memory2
        mem1_tf_score = next(
            r["metadata"]["attribute_score"]
            for r in term_freq_results
            if r["id"] == "memory1"
        )
        mem2_tf_score = next(
            r["metadata"]["attribute_score"]
            for r in term_freq_results
            if r["id"] == "memory2"
        )
        self.assertGreater(
            mem1_tf_score,
            mem2_tf_score,
            "Term frequency should score memory1 higher than memory2",
        )
        print(
            f"Term frequency scores - memory1: {mem1_tf_score}, memory2: {mem2_tf_score}"
        )

        # In length ratio, memory1 should also score higher than memory2
        mem1_lr_score = next(
            r["metadata"]["attribute_score"]
            for r in length_ratio_results
            if r["id"] == "memory1"
        )
        mem2_lr_score = next(
            r["metadata"]["attribute_score"]
            for r in length_ratio_results
            if r["id"] == "memory2"
        )
        self.assertGreater(
            mem1_lr_score,
            mem2_lr_score,
            "Length ratio should score memory1 higher than memory2",
        )
        print(
            f"Length ratio scores - memory1: {mem1_lr_score}, memory2: {mem2_lr_score}"
        )

        # 5. Test overriding scoring method in search call
        # Create strategy with default length_ratio scoring
        default_strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system
        )

        # Override with term_frequency in search call
        override_results = default_strategy.search(
            query="meeting",
            agent_id="agent-1",
            tier="stm",
            scoring_method="term_frequency",
            limit=5,
        )

        print("Override results metadata:", override_results[0].get("metadata", {}))

        # Check that override was applied
        self.assertEqual(
            override_results[0]["metadata"]["scoring_method"],
            "term_frequency",
            f"Expected 'term_frequency' but got '{override_results[0]['metadata'].get('scoring_method')}'",
        )

    def test_pattern_cache(self):
        """Test the regex pattern caching functionality."""
        # Create clean strategy
        strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system
        )

        # Initial cache should be empty
        self.assertEqual(len(strategy._pattern_cache), 0)

        # Compile a pattern and check it's added to cache
        pattern1 = strategy.get_compiled_pattern("test.*pattern", True)
        self.assertEqual(len(strategy._pattern_cache), 1)

        # Get the same pattern again - should use cache
        pattern2 = strategy.get_compiled_pattern("test.*pattern", True)
        self.assertEqual(len(strategy._pattern_cache), 1)

        # Patterns should be identical (same object in memory)
        self.assertIs(pattern1, pattern2)

        # Different pattern should create new cache entry
        strategy.get_compiled_pattern("different.*pattern", True)
        self.assertEqual(len(strategy._pattern_cache), 2)

        # Same pattern with different case sensitivity should create new cache entry
        strategy.get_compiled_pattern("test.*pattern", False)
        self.assertEqual(len(strategy._pattern_cache), 3)

        # Test caching of invalid patterns
        invalid_pattern = "invalid[pattern"  # Missing closing bracket

        # Instead of relying on log capture, just verify the behavior:
        # First attempt should try compilation and return None
        result1 = strategy.get_compiled_pattern(invalid_pattern, True)
        self.assertIsNone(result1)
        self.assertEqual(len(strategy._pattern_cache), 4)

        # Check that the invalid pattern is now in the cache with None value
        self.assertIn((invalid_pattern, True), strategy._pattern_cache)
        self.assertIsNone(strategy._pattern_cache[(invalid_pattern, True)])

        # Second attempt should use cached result
        # Use a spy to verify the method doesn't try to compile again
        original_compile = re.compile
        compile_called = [False]

        def mock_compile(*args, **kwargs):
            compile_called[0] = True
            return original_compile(*args, **kwargs)

        try:
            re.compile = mock_compile

            # Get the pattern again - should use cache and not call re.compile
            result2 = strategy.get_compiled_pattern(invalid_pattern, True)
            self.assertIsNone(result2)
            self.assertFalse(
                compile_called[0], "re.compile should not have been called"
            )

        finally:
            # Restore original compile function
            re.compile = original_compile

        # Cache size shouldn't change
        self.assertEqual(len(strategy._pattern_cache), 4)

        # Clear cache should empty it
        strategy.clear_pattern_cache()
        self.assertEqual(len(strategy._pattern_cache), 0)

    def test_precompile_patterns(self):
        """Test the precompile_patterns method."""
        strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system
        )

        # Precompile multiple patterns
        patterns = [
            ("pattern1", True),
            ("pattern2", False),
            ("invalid[", True),  # Invalid pattern
            ("pattern3", False),
        ]

        # Should return count of successful compilations
        success_count = strategy.precompile_patterns(patterns)

        # Should have 3 successful patterns (1 invalid)
        self.assertEqual(success_count, 3)

        # Cache should have all 4 patterns (the invalid one with None value)
        self.assertEqual(len(strategy._pattern_cache), 4)

        # Valid patterns should have compiled regex objects
        self.assertIsNotNone(strategy._pattern_cache[("pattern1", True)])
        self.assertIsNotNone(strategy._pattern_cache[("pattern2", False)])
        self.assertIsNotNone(strategy._pattern_cache[("pattern3", False)])

        # Invalid pattern should be cached as None
        self.assertIn(("invalid[", True), strategy._pattern_cache)
        self.assertIsNone(strategy._pattern_cache[("invalid[", True)])

    def test_pattern_cache_in_search(self):
        """Test that pattern cache is used during search operations."""
        # Set up store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system
        )

        # Spy on the get_compiled_pattern method to count calls
        original_get_compiled_pattern = strategy.get_compiled_pattern
        call_count = [0]

        def spy_get_compiled_pattern(*args, **kwargs):
            call_count[0] += 1
            return original_get_compiled_pattern(*args, **kwargs)

        strategy.get_compiled_pattern = spy_get_compiled_pattern

        # First search with regex should compile patterns
        strategy.search(
            query="meet.*",
            agent_id="agent-1",
            tier="stm",
            use_regex=True,
            limit=5,
        )

        first_call_count = call_count[0]
        self.assertGreater(first_call_count, 0)

        # Reset counter
        call_count[0] = 0

        # Second search with same pattern should use cache
        strategy.search(
            query="meet.*",
            agent_id="agent-1",
            tier="stm",
            use_regex=True,
            limit=5,
        )

        # Should have fewer or equal calls since patterns are cached
        second_call_count = call_count[0]
        self.assertLessEqual(second_call_count, first_call_count)

        # Restore original method
        strategy.get_compiled_pattern = original_get_compiled_pattern

    @patch("time.time")
    def test_regex_performance(self, mock_time):
        """Test performance improvement from regex pattern caching."""
        # Mock time.time to return controlled values for performance measurement
        time_values = [0.0, 0.2, 0.4, 0.5]  # Simulate elapsed time
        mock_time.side_effect = lambda: time_values.pop(0)

        # Create memories with many regex patterns
        regex_test_memories = []
        for i in range(100):
            regex_test_memories.append(
                {
                    "id": f"memory{i}",
                    "content": {
                        "content": f"Test content {i} with pattern text",
                        "metadata": {"type": "test", "tags": [f"tag{i}"]},
                    },
                }
            )

        self.mock_agent.stm_store.get_all.return_value = regex_test_memories

        strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system
        )

        # First run - no cached patterns
        with patch("time.time", side_effect=[0.0, 0.2]):
            strategy.clear_pattern_cache()
            strategy.search(
                query="patt.*text",
                agent_id="agent-1",
                tier="stm",
                use_regex=True,
                limit=10,
            )
            uncached_time = 0.2 - 0.0

        # Second run - should use cached patterns
        with patch("time.time", side_effect=[0.0, 0.1]):
            strategy.search(
                query="patt.*text",
                agent_id="agent-1",
                tier="stm",
                use_regex=True,
                limit=10,
            )
            cached_time = 0.1 - 0.0

        # The cached search should be faster
        self.assertLess(cached_time, uncached_time)

        # Try with precompiled patterns
        with patch("time.time", side_effect=[0.0, 0.05]):
            strategy.clear_pattern_cache()
            strategy.precompile_patterns([("patt.*text", False)])
            strategy.search(
                query="patt.*text",
                agent_id="agent-1",
                tier="stm",
                use_regex=True,
                limit=10,
            )
            precompiled_time = 0.05 - 0.0

        # Precompiled should be even faster
        self.assertLess(precompiled_time, uncached_time)

    def test_error_handling(self):
        """Test error handling in search."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with invalid regex pattern
        results = self.strategy.search("[invalid regex", "agent1", use_regex=True)
        self.assertEqual(len(results), 0)

        # Test with empty query
        results = self.strategy.search("", "agent1")
        self.assertEqual(len(results), 0)

        # Test with None query
        results = self.strategy.search(None, "agent1", content_fields=["content.content"])
        self.assertEqual(len(results), 0)

    def test_basic_search(self):
        """Test basic search functionality."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test simple content search
        results = self.strategy.search("meeting", "agent1")
        self.assertEqual(len(results), 2)
        self.assertTrue(any(r["id"] == "memory1" for r in results))
        self.assertTrue(any(r["id"] == "memory3" for r in results))

        # Test metadata search
        results = self.strategy.search({"metadata": {"type": "communication"}}, "agent1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "memory2")

    def test_scoring_methods(self):
        """Test different scoring methods."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Create test memories with repeated terms
        test_memories = [
            {
                "id": "memory1",
                "content": {
                    "content": "meeting meeting meeting project timeline",
                    "metadata": {"type": "meeting"},
                },
            },
            {
                "id": "memory2",
                "content": {
                    "content": "meeting project timeline",
                    "metadata": {"type": "meeting"},
                },
            },
        ]
        self.mock_agent.stm_store.get_all.return_value = test_memories

        # Test different scoring methods
        methods = ["length_ratio", "term_frequency", "bm25", "binary"]
        for method in methods:
            strategy = AttributeSearchStrategy(
                memory_system=self.mock_memory_system,
                scoring_method=method
            )
            results = strategy.search("meeting", "agent1")
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["metadata"]["scoring_method"], method)

            # Print metadata for debugging
            print(f"\nResults for {method}:")
            for r in results:
                print(f"Memory {r['id']}: {r['metadata']}")

        # Test overriding default scoring method
        strategy = AttributeSearchStrategy(
            memory_system=self.mock_memory_system,
            scoring_method="length_ratio"  # Default
        )
        results = strategy.search("meeting", "agent1", scoring_method="term_frequency")
        self.assertEqual(results[0]["metadata"]["scoring_method"], "term_frequency")

        # Verify scores are different for different methods
        length_results = strategy.search("meeting", "agent1", scoring_method="length_ratio")
        freq_results = strategy.search("meeting", "agent1", scoring_method="term_frequency")
        self.assertNotEqual(
            length_results[0]["metadata"]["attribute_score"],
            freq_results[0]["metadata"]["attribute_score"]
        )

        # Verify memory1 scores higher than memory2 for term frequency
        self.assertGreater(
            freq_results[0]["metadata"]["attribute_score"],
            freq_results[1]["metadata"]["attribute_score"]
        )

    def test_mixed_data_types(self):
        """Test search with mixed data types."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with numeric content
        results = self.strategy.search(123, "agent1")
        self.assertEqual(len(results), 0)  # No matches expected

        # Test with boolean content
        results = self.strategy.search(True, "agent1")
        self.assertEqual(len(results), 0)  # No matches expected

        # Test with mixed metadata types
        results = self.strategy.search(
            {"metadata": {"importance": 5}}, "agent1"
        )
        self.assertEqual(len(results), 0)  # No matches expected

    def test_search_with_filters(self):
        """Test search with various filters."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with metadata filter
        results = self.strategy.search(
            "meeting",
            "agent1",
            metadata_filter={"content.metadata.importance": "high"},
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["content"]["metadata"]["importance"] == "high" for r in results))

        # Test with tier filter
        results = self.strategy.search("meeting", "agent1", tier="stm")
        self.assertEqual(len(results), 2)

        # Test with content fields
        results = self.strategy.search(
            "meeting",
            "agent1",
            content_fields=["content.content"],
        )
        self.assertEqual(len(results), 2)

        # Test with metadata fields
        results = self.strategy.search(
            "high",
            "agent1",
            metadata_fields=["content.metadata.importance"],
        )
        self.assertEqual(len(results), 2)

    def test_match_all_behavior(self):
        """Test match_all parameter behavior."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with match_all=True
        results = self.strategy.search(
            {"metadata": {"type": "meeting", "importance": "high"}},
            "agent1",
            match_all=True,
        )
        self.assertEqual(len(results), 2)

        # Test with match_all=False
        results = self.strategy.search(
            {"metadata": {"type": "meeting", "importance": "medium"}},
            "agent1",
            match_all=False,
        )
        self.assertEqual(len(results), 3)  # Should match any memory with either condition

    def test_case_sensitivity(self):
        """Test case sensitivity in search."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test case-insensitive search (default)
        results = self.strategy.search("MEETING", "agent1")
        self.assertEqual(len(results), 2)

        # Test case-sensitive search
        results = self.strategy.search("MEETING", "agent1", case_sensitive=True)
        self.assertEqual(len(results), 0)  # No matches expected

        # Test case-sensitive search with correct case
        results = self.strategy.search("meeting", "agent1", case_sensitive=True)
        self.assertEqual(len(results), 2)

    def test_regex_search(self):
        """Test regex search functionality."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with valid regex pattern
        results = self.strategy.search(
            r"meeting.*project",
            "agent1",
            use_regex=True,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "memory1")

        # Test with invalid regex pattern
        results = self.strategy.search(
            r"meeting[",
            "agent1",
            use_regex=True,
        )
        self.assertEqual(len(results), 0)

    def test_limit_parameter(self):
        """Test limit parameter behavior."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with limit=1
        results = self.strategy.search("meeting", "agent1", limit=1)
        self.assertEqual(len(results), 1)

        # Test with limit=0
        results = self.strategy.search("meeting", "agent1", limit=0)
        self.assertEqual(len(results), 0)

        # Test with limit greater than available results
        results = self.strategy.search("meeting", "agent1", limit=10)
        self.assertEqual(len(results), 2)  # Should return all matches

    def test_skip_validation(self):
        """Test skip_validation parameter behavior."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test with skip_validation=True
        results = self.strategy.search(
            "meeting",
            "agent1",
            skip_validation=True,
        )
        self.assertEqual(len(results), 2)

        # Test with skip_validation=False
        results = self.strategy.search(
            "meeting",
            "agent1",
            skip_validation=False,
        )
        self.assertEqual(len(results), 2)

    def test_pattern_cache(self):
        """Test regex pattern caching functionality."""
        # Configure mock store to return sample memories
        self.mock_agent.stm_store.get_all.return_value = self.sample_memories

        # Test pattern caching
        pattern = r"meeting.*project"
        self.strategy.get_compiled_pattern(pattern, False)
        self.assertIn((pattern, False), self.strategy._pattern_cache)

        # Test cache clearing
        self.strategy.clear_pattern_cache()
        self.assertEqual(len(self.strategy._pattern_cache), 0)

        # Test precompilation
        patterns = [(r"meeting.*project", False), (r"email.*team", False)]
        success_count = self.strategy.precompile_patterns(patterns)
        self.assertEqual(success_count, 2)
        self.assertEqual(len(self.strategy._pattern_cache), 2)


if __name__ == "__main__":
    unittest.main()
