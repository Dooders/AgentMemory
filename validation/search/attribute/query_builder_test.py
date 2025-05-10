"""Test for the AttributeQueryBuilder."""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from memory.search.strategies.query_builder import AttributeQueryBuilder


class AttributeQueryBuilderTest(unittest.TestCase):
    """Test case for the AttributeQueryBuilder."""

    def test_simple_content_query(self):
        """Test simple content query."""
        builder = AttributeQueryBuilder()
        query, kwargs = builder.content("meeting").build()
        
        self.assertEqual(query, "meeting")
        self.assertEqual(kwargs["limit"], 10)
        self.assertEqual(kwargs["match_all"], False)
        self.assertEqual(kwargs["case_sensitive"], False)
        self.assertEqual(kwargs["use_regex"], False)
        self.assertNotIn("content_fields", kwargs)
        self.assertNotIn("metadata_fields", kwargs)

    def test_content_with_type(self):
        """Test content query with type metadata."""
        builder = AttributeQueryBuilder()
        query, kwargs = builder.content("meeting").type("meeting").build()
        
        self.assertIsInstance(query, dict)
        self.assertEqual(query["content"], "meeting")
        self.assertIsInstance(query["metadata"], dict)
        self.assertEqual(query["metadata"]["type"], "meeting")
        self.assertEqual(kwargs["limit"], 10)

    def test_complex_query(self):
        """Test a complex query with multiple parameters."""
        builder = AttributeQueryBuilder()
        query, kwargs = (builder
            .content("security")
            .type("note")
            .importance("high")
            .tag("development")
            .match_all(True)
            .case_sensitive(True)
            .use_regex(True)
            .in_content_fields("content.content")
            .in_metadata_fields("content.metadata.tags")
            .limit(5)
            .in_tier("stm")
            .score_by("bm25")
            .filter_metadata("content.metadata.source", "email")
            .build())
        
        # Check query structure
        self.assertIsInstance(query, dict)
        self.assertEqual(query["content"], "security")
        self.assertIsInstance(query["metadata"], dict)
        self.assertEqual(query["metadata"]["type"], "note")
        self.assertEqual(query["metadata"]["importance"], "high")
        self.assertEqual(query["metadata"]["tags"], "development")
        
        # Check kwargs
        self.assertEqual(kwargs["limit"], 5)
        self.assertTrue(kwargs["match_all"])
        self.assertTrue(kwargs["case_sensitive"])
        self.assertTrue(kwargs["use_regex"])
        self.assertEqual(kwargs["content_fields"], ["content.content"])
        self.assertEqual(kwargs["metadata_fields"], ["content.metadata.tags"])
        self.assertEqual(kwargs["tier"], "stm")
        self.assertEqual(kwargs["scoring_method"], "bm25")
        self.assertEqual(kwargs["metadata_filter"], {"content.metadata.source": "email"})

    def test_metadata_only_query(self):
        """Test query with only metadata fields."""
        builder = AttributeQueryBuilder()
        query, kwargs = builder.type("task").importance("high").build()
        
        self.assertIsInstance(query, dict)
        self.assertNotIn("content", query)
        self.assertIsInstance(query["metadata"], dict)
        self.assertEqual(query["metadata"]["type"], "task")
        self.assertEqual(query["metadata"]["importance"], "high")

    def test_filter_metadata(self):
        """Test metadata filtering."""
        builder = AttributeQueryBuilder()
        query, kwargs = (builder
            .content("project")
            .filter_metadata("content.metadata.source", "email")
            .build())
        
        self.assertEqual(query, "project")
        self.assertEqual(kwargs["metadata_filter"], {"content.metadata.source": "email"})

    def test_chaining_methods(self):
        """Test that all methods return self for chaining."""
        builder = AttributeQueryBuilder()
        
        # This test will fail if any method doesn't return self
        result = (builder
            .content("test")
            .metadata("field", "value")
            .type("test")
            .importance("high")
            .tag("tag")
            .match_all()
            .case_sensitive()
            .use_regex()
            .in_content_fields("field")
            .in_metadata_fields("field")
            .limit(5)
            .filter_metadata("field", "value")
            .in_tier("stm")
            .score_by("bm25"))
        
        self.assertIs(result, builder)


if __name__ == "__main__":
    unittest.main() 