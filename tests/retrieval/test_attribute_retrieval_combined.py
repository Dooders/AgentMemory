"""Tests for combined usage patterns of attribute-based memory retrieval."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import random

from agent_memory.retrieval.attribute import AttributeRetrieval
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore


class TestAttributeRetrievalCombined:
    """Test suite for combined patterns and performance aspects."""

    @pytest.fixture
    def mock_stores(self):
        """Create all mock stores in one fixture.
        
        Uses autospec=False to allow any method to be mocked.
        """
        return {
            "stm": Mock(name="mock_stm_store", autospec=False),
            "im": Mock(name="mock_im_store", autospec=False),
            "ltm": Mock(name="mock_ltm_store", autospec=False),
        }

    @pytest.fixture
    def retriever(self, mock_stores):
        """Create an AttributeRetrieval instance with mocked stores."""
        return AttributeRetrieval(mock_stores["stm"], mock_stores["im"], mock_stores["ltm"])

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            {
                "memory_id": "mem1",
                "memory_type": "state",
                "importance": 0.8,
                "metadata": {"location": "kitchen", "mood": "happy"},
                "contents": {"text": "I am in the kitchen feeling happy"}
            },
            {
                "memory_id": "mem2",
                "memory_type": "action",
                "importance": 0.6,
                "metadata": {"location": "kitchen", "action": "cooking"},
                "contents": {"text": "I am cooking pasta"}
            },
            {
                "memory_id": "mem3",
                "memory_type": "observation",
                "importance": 0.9,
                "metadata": {"location": "living_room", "subject": "TV"},
                "contents": {"text": "The TV is on"}
            },
            {
                "memory_id": "mem4",
                "memory_type": "interaction",
                "importance": 0.7,
                "metadata": {"location": "kitchen", "person": "Bob"},
                "contents": {"text": "Talking to Bob while cooking"}
            },
            {
                "memory_id": "mem5",
                "memory_type": "reflection",
                "importance": 0.95,
                "metadata": {"topic": "day", "mood": "satisfied"},
                "contents": {"text": "It was a good day"}
            }
        ]

    @pytest.fixture
    def large_sample_memories(self):
        """Create a larger set of sample memories for testing performance and limits."""
        memories = []
        locations = ["kitchen", "living_room", "bedroom"]
        moods = ["happy", "sad"]
        memory_types = ["state", "action", "observation", "interaction"]
        
        for i in range(100):
            location = locations[i % 3]
            mood = moods[i % 2]
            memory_type = memory_types[i % 4]
            person = f"person{i % 5}"
            
            # Create some pattern in importance scores
            importance = 0.5
            if i % 10 == 0:
                importance = 0.9  # Every 10th memory is highly important
            
            # Create text content (make some None to test edge cases)
            text = f"This is memory number {i} in {location}"
            if i % 3 == 2:
                text = None
                
            # Add tags to some memories
            tags = []
            if i % 5 == 0:
                tags.append("common_tag")  # Every 5th memory has common_tag
            if i % 7 == 0:
                tags.append("special_tag")  # Every 7th memory has special_tag
            
            memory = {
                "memory_id": f"mem{i}",
                "memory_type": memory_type,
                "step_number": i,
                "importance": importance,
                "metadata": {"location": location, "mood": mood},
                "contents": {
                    "location": {"name": location},
                    "mood": mood,
                    "person": person,
                    "text": text
                },
                "tags": tags
            }
            memories.append(memory)
        
        return memories

    def test_chained_retrieval_by_type_then_metadata(self, retriever, mock_stores, sample_memories):
        """Test retrieving memories by type and then filtering by metadata."""
        # Setup
        mock_stores["stm"].get_by_type.return_value = [m for m in sample_memories if m["memory_type"] == "state"]
        mock_stores["stm"].get_all.return_value = mock_stores["stm"].get_by_type.return_value
        
        # Step 1: Get all state memories
        state_memories = retriever.retrieve_by_memory_type("state", tier="stm")
        
        # Step 2: From state memories, filter by metadata
        kitchen_state_memories = retriever.retrieve_by_metadata(
            metadata_filters={"location": "kitchen"},
            tier="stm"
        )
        
        # Verify results
        assert len(state_memories) == 1
        assert len(kitchen_state_memories) == 1
        assert kitchen_state_memories[0]["memory_id"] == "mem1"
        assert kitchen_state_memories[0]["metadata"]["location"] == "kitchen"
        assert kitchen_state_memories[0]["memory_type"] == "state"

    def test_combined_filtering_by_importance_and_content(self, retriever, mock_stores, sample_memories):
        """Test retrieving memories by importance and then filtering by content."""
        # Setup - get all memories with importance >= 0.8
        important_memories = [m for m in sample_memories if m["importance"] >= 0.8]
        mock_stores["stm"].get_by_importance.return_value = important_memories
        mock_stores["stm"].get_all.return_value = important_memories
        
        # Step 1: Get important memories
        high_importance_memories = retriever.retrieve_by_importance(
            min_importance=0.8,
            tier="stm"
        )
        
        # Step 2: Filter for kitchen-related content
        kitchen_important_memories = []
        for memory in high_importance_memories:
            content_text = memory["contents"].get("text", "")
            if "kitchen" in content_text:
                kitchen_important_memories.append(memory)
        
        # Verify results
        assert len(high_importance_memories) == 3  # mem1, mem3, mem5
        assert len(kitchen_important_memories) == 1  # Only mem1
        assert kitchen_important_memories[0]["memory_id"] == "mem1"

    def test_cross_tier_search(self, retriever, mock_stores, large_sample_memories):
        """Test searching across multiple memory tiers and combining results."""
        # Setup - distribute memories across tiers
        stm_memories = large_sample_memories[:30]
        im_memories = large_sample_memories[30:60]
        ltm_memories = large_sample_memories[60:]

        # Set up mock responses for tag-based searches in each tier
        mock_stores["stm"].get_by_tag.return_value = [m for m in stm_memories if "common_tag" in m.get("tags", [])][:10]
        mock_stores["im"].get_by_tag.return_value = [m for m in im_memories if "common_tag" in m.get("tags", [])][:10]
        mock_stores["ltm"].get_by_tag.return_value = [m for m in ltm_memories if "common_tag" in m.get("tags", [])][:10]

        # Perform the searches across all tiers
        stm_results = retriever.retrieve_by_tag("common_tag", limit=10, tier="stm")
        im_results = retriever.retrieve_by_tag("common_tag", limit=10, tier="im")
        ltm_results = retriever.retrieve_by_tag("common_tag", limit=10, tier="ltm")

        # Combine results (simulating what an application might do)
        all_results = stm_results + im_results + ltm_results

        # Verify cross-tier search
        assert len(all_results) > 0
        assert len(stm_results) <= 10
        assert len(im_results) <= 10
        assert len(ltm_results) <= 10

    def test_compound_query_with_metadata_and_pattern(self, retriever, mock_stores, large_sample_memories):
        """Test using compound queries with both metadata and pattern matching."""
        # Setup
        mock_stores["stm"].get_all.return_value = large_sample_memories[:30]
        
        # Define compound conditions
        conditions = [
            {
                "type": "metadata",
                "key": "location",
                "value": "kitchen"
            },
            {
                "type": "pattern",
                "path": "contents.text",
                "pattern": "memory number"
            }
        ]
        
        # Apply compound query with AND operator
        results_and = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="AND",
            limit=10,
            tier="stm"
        )
        
        # Apply compound query with OR operator
        results_or = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="OR",
            limit=10,
            tier="stm"
        )
        
        # Verify results
        # AND should give only memories that match both conditions
        assert all(m["metadata"]["location"] == "kitchen" for m in results_and)
        assert all(
            m["contents"].get("text") and "memory number" in m["contents"]["text"]
            for m in results_and
        )
        
        # OR should include memories that match either condition
        assert any(m["metadata"]["location"] == "kitchen" for m in results_or)
        assert any(
            m["contents"].get("text") and "memory number" in m["contents"]["text"]
            for m in results_or
        )
        
        # OR should return more results than AND
        assert len(results_or) >= len(results_and)

    def test_performance_large_dataset(self, retriever, mock_stores, large_sample_memories):
        """Test performance with a large dataset."""
        # Setup
        mock_stores["stm"].get_all.return_value = large_sample_memories
        
        # Measure time for a metadata query
        start_time = time.time()
        metadata_results = retriever.retrieve_by_metadata(
            metadata_filters={"location": "kitchen"},
            limit=10,
            tier="stm"
        )
        metadata_time = time.time() - start_time
        
        # Measure time for a content pattern query
        start_time = time.time()
        pattern_results = retriever.retrieve_by_content_pattern(
            path="contents.text",
            pattern="memory number",
            limit=10,
            tier="stm"
        )
        pattern_time = time.time() - start_time
        
        # Measure time for a compound query
        conditions = [
            {"type": "metadata", "key": "location", "value": "kitchen"},
            {"type": "pattern", "path": "contents.text", "pattern": "memory"}
        ]
        
        start_time = time.time()
        compound_results = retriever.retrieve_by_compound_query(
            conditions=conditions,
            operator="AND",
            limit=10,
            tier="stm"
        )
        compound_time = time.time() - start_time
        
        # Verify results came back with reasonable size
        assert len(metadata_results) <= 10
        assert len(pattern_results) <= 10
        assert len(compound_results) <= 10
        
        # Performance expectations will depend on implementation details

    def test_limit_application_in_chain(self, retriever, mock_stores, large_sample_memories):
        """Test how limits affect chained retrievals."""
        # First get all state memories (25% of total)
        state_memories = [m for m in large_sample_memories if m["memory_type"] == "state"][:5]

        # Add some important memories
        high_importance_memories = []
        for memory in state_memories:
            if memory["memory_id"] == "mem0":
                memory["importance"] = 0.9
                high_importance_memories.append(memory)

        mock_stores["stm"].get_by_type.return_value = state_memories

        # First retrieval with small limit
        first_results = retriever.retrieve_by_memory_type("state", limit=5, tier="stm")

        # Now get important memories from those results
        mock_stores["stm"].get_all.return_value = first_results
        mock_stores["stm"].get_by_importance.return_value = high_importance_memories

        # Second retrieval - should only search within first_results
        second_results = retriever.retrieve_by_importance(min_importance=0.7, limit=10, tier="stm")

        # Verify limit applications
        assert len(first_results) <= 5
        # Verify second_results contains the high importance memories
        assert len(second_results) == len(high_importance_memories)
        assert all(memory["memory_id"] == "mem0" for memory in second_results)

    def test_custom_aggregation_workflow(self, retriever, mock_stores, large_sample_memories):
        """Test a complex workflow combining multiple retrieval methods."""
        # Setup the initial data in the stores
        mock_stores["stm"].get_all.return_value = large_sample_memories[:30]
        mock_stores["im"].get_all.return_value = large_sample_memories[30:60]
        mock_stores["ltm"].get_all.return_value = large_sample_memories[60:]

        # Step 1: Get important memories from each tier
        def is_important(memory):
            return memory.get("importance", 0) >= 0.8

        stm_important = retriever.retrieve_by_custom_filter(is_important, limit=5, tier="stm")
        im_important = retriever.retrieve_by_custom_filter(is_important, limit=5, tier="im")
        ltm_important = retriever.retrieve_by_custom_filter(is_important, limit=5, tier="ltm")

        # Mock getting these results for next step
        mock_stores["stm"].get_by_tag.return_value = [m for m in stm_important if "common_tag" in m.get("tags", [])][:3]
        mock_stores["im"].get_by_tag.return_value = [m for m in im_important if "common_tag" in m.get("tags", [])][:3]
        mock_stores["ltm"].get_by_tag.return_value = [m for m in ltm_important if "common_tag" in m.get("tags", [])][:3]

        # Step 2: From important memories, filter to those with common_tag
        stm_tagged = retriever.retrieve_by_tag("common_tag", limit=3, tier="stm")
        im_tagged = retriever.retrieve_by_tag("common_tag", limit=3, tier="im")
        ltm_tagged = retriever.retrieve_by_tag("common_tag", limit=3, tier="ltm")

        # Combine all results
        all_results = stm_tagged + im_tagged + ltm_tagged

        # Verify the workflow produced expected results
        assert all(m.get("importance", 0) >= 0.8 for m in all_results)
        assert all("common_tag" in m.get("tags", []) for m in all_results)

        # Verify limits were respected
        assert len(stm_tagged) <= 3
        assert len(im_tagged) <= 3
        assert len(ltm_tagged) <= 3 