"""Tests for the NarrativeSequenceStrategy class."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from memory.search.strategies.sequence import NarrativeSequenceStrategy


class TestNarrativeSequenceStrategy(unittest.TestCase):
    """Tests for the NarrativeSequenceStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create mock memory system
        self.mock_memory_system = MagicMock()
        self.mock_memory_system.stm_store = self.mock_stm_store
        self.mock_memory_system.im_store = self.mock_im_store
        self.mock_memory_system.ltm_store = self.mock_ltm_store

        # Create strategy with mock memory system
        self.strategy = NarrativeSequenceStrategy(self.mock_memory_system)

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "sequence")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("sequence", self.strategy.description().lower())

    def test_search_with_reference_memory_id(self):
        """Test search with a reference memory ID."""
        # Set up test memories with sequential timestamps
        now = datetime.now()
        memories = [
            {
                "id": "mem1",
                "content": "First memory",
                "metadata": {"timestamp": (now - timedelta(minutes=4)).isoformat()},
            },
            {
                "id": "mem2",
                "content": "Second memory",
                "metadata": {"timestamp": (now - timedelta(minutes=3)).isoformat()},
            },
            {
                "id": "mem3",  # Reference memory
                "content": "Reference memory",
                "metadata": {"timestamp": (now - timedelta(minutes=2)).isoformat()},
            },
            {
                "id": "mem4",
                "content": "Fourth memory",
                "metadata": {"timestamp": (now - timedelta(minutes=1)).isoformat()},
            },
            {
                "id": "mem5",
                "content": "Fifth memory",
                "metadata": {"timestamp": now.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.get_all.return_value = memories

        # Configure mock to return reference memory
        reference_memory = memories[2]  # mem3
        self.mock_stm_store.get.return_value = reference_memory

        # Perform search with reference memory ID and sequence size
        results = self.strategy.search(
            query={
                "reference_id": "mem3",
                "sequence_size": 5,  # 2 before + 1 reference + 2 after
            },
            agent_id="agent-1",
            tier="stm",
            limit=10,
        )

        # Should return 5 memories in sequence
        self.assertEqual(len(results), 5)

        # Verify the sequence is in chronological order
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem2")
        self.assertEqual(results[2]["id"], "mem3")  # Reference memory
        self.assertEqual(results[3]["id"], "mem4")
        self.assertEqual(results[4]["id"], "mem5")

        # Verify reference memory is marked in metadata
        self.assertTrue(results[2]["metadata"].get("is_reference_memory", False))

    def test_search_with_before_after_counts(self):
        """Test search with specific before and after counts."""
        # Set up test memories with sequential timestamps
        now = datetime.now()
        memories = [
            {
                "id": "mem1",
                "content": "First memory",
                "metadata": {"timestamp": (now - timedelta(minutes=4)).isoformat()},
            },
            {
                "id": "mem2",
                "content": "Second memory",
                "metadata": {"timestamp": (now - timedelta(minutes=3)).isoformat()},
            },
            {
                "id": "mem3",  # Reference memory
                "content": "Reference memory",
                "metadata": {"timestamp": (now - timedelta(minutes=2)).isoformat()},
            },
            {
                "id": "mem4",
                "content": "Fourth memory",
                "metadata": {"timestamp": (now - timedelta(minutes=1)).isoformat()},
            },
            {
                "id": "mem5",
                "content": "Fifth memory",
                "metadata": {"timestamp": now.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.get_all.return_value = memories

        # Configure mock to return reference memory
        reference_memory = memories[2]  # mem3
        self.mock_im_store.get.return_value = reference_memory

        # Perform search with reference memory ID and asymmetric sequence (1 before, 3 after)
        results = self.strategy.search(
            query={"reference_id": "mem3", "before_count": 1, "after_count": 2},
            agent_id="agent-1",
            tier="im",
            limit=10,
        )

        # Should return 4 memories in sequence (1 before + 1 reference + 2 after)
        self.assertEqual(len(results), 4)

        # Verify the sequence is in chronological order
        self.assertEqual(results[0]["id"], "mem2")  # 1 before
        self.assertEqual(results[1]["id"], "mem3")  # Reference
        self.assertEqual(results[2]["id"], "mem4")  # 1 after
        self.assertEqual(results[3]["id"], "mem5")  # 2 after

    def test_search_with_time_window(self):
        """Test search with time window around reference memory."""
        # Set up test memories with sequential timestamps
        now = datetime.now()
        reference_time = now - timedelta(minutes=10)

        memories = [
            {
                "id": "mem1",
                "content": "Far before reference",
                "metadata": {
                    "timestamp": (reference_time - timedelta(minutes=30)).isoformat()
                },
            },
            {
                "id": "mem2",
                "content": "Just before reference",
                "metadata": {
                    "timestamp": (reference_time - timedelta(minutes=4)).isoformat()
                },
            },
            {
                "id": "mem3",  # Reference memory
                "content": "Reference memory",
                "metadata": {"timestamp": reference_time.isoformat()},
            },
            {
                "id": "mem4",
                "content": "Just after reference",
                "metadata": {
                    "timestamp": (reference_time + timedelta(minutes=5)).isoformat()
                },
            },
            {
                "id": "mem5",
                "content": "Far after reference",
                "metadata": {
                    "timestamp": (reference_time + timedelta(minutes=25)).isoformat()
                },
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.get_all.return_value = memories

        # Configure mock to return reference memory
        reference_memory = memories[2]  # mem3
        self.mock_ltm_store.get.return_value = reference_memory

        # Perform search with reference memory ID and time window
        results = self.strategy.search(
            query={
                "reference_id": "mem3",
                "time_window_minutes": 10,  # 10 minutes before and after
                "timestamp_field": "metadata.timestamp"  # Specify the correct timestamp field path
            },
            agent_id="agent-1",
            tier="ltm",
            limit=10,
        )

        # Should return 3 memories within time window
        self.assertEqual(len(results), 3)

        # Verify the sequence contains only memories within time window
        self.assertEqual(
            results[0]["id"], "mem2"
        )  # Just before reference (within window)
        self.assertEqual(results[1]["id"], "mem3")  # Reference memory
        self.assertEqual(
            results[2]["id"], "mem4"
        )  # Just after reference (within window)

        # mem1 and mem5 should not be included (outside time window)
        result_ids = [r["id"] for r in results]
        self.assertNotIn("mem1", result_ids)
        self.assertNotIn("mem5", result_ids)

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        # Set up test memories with sequential timestamps and metadata
        now = datetime.now()
        memories = [
            {
                "id": "mem1",
                "content": "First memory",
                "metadata": {
                    "timestamp": (now - timedelta(minutes=4)).isoformat(),
                    "type": "important",
                },
            },
            {
                "id": "mem2",
                "content": "Second memory",
                "metadata": {
                    "timestamp": (now - timedelta(minutes=3)).isoformat(),
                    "type": "regular",
                },
            },
            {
                "id": "mem3",  # Reference memory
                "content": "Reference memory",
                "metadata": {
                    "timestamp": (now - timedelta(minutes=2)).isoformat(),
                    "type": "important",
                },
            },
            {
                "id": "mem4",
                "content": "Fourth memory",
                "metadata": {
                    "timestamp": (now - timedelta(minutes=1)).isoformat(),
                    "type": "regular",
                },
            },
            {
                "id": "mem5",
                "content": "Fifth memory",
                "metadata": {"timestamp": now.isoformat(), "type": "important"},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.get_all.return_value = memories

        # Configure mock to return reference memory
        reference_memory = memories[2]  # mem3
        self.mock_stm_store.get.return_value = reference_memory

        # Perform search with reference memory ID and metadata filter
        results = self.strategy.search(
            query={"reference_id": "mem3", "sequence_size": 5},
            agent_id="agent-1",
            tier="stm",
            metadata_filter={"type": "important"},
            limit=10,
        )

        # Should return only important memories in sequence
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem3")  # Reference memory
        self.assertEqual(results[2]["id"], "mem5")

    def test_custom_timestamp_field(self):
        """Test search with a custom timestamp field."""
        # Set up test memories with different timestamp field
        now = datetime.now()
        memories = [
            {
                "id": "mem1",
                "content": "First memory",
                "metadata": {"created_at": (now - timedelta(minutes=2)).isoformat()},
            },
            {
                "id": "mem2",  # Reference memory
                "content": "Reference memory",
                "metadata": {"created_at": (now - timedelta(minutes=1)).isoformat()},
            },
            {
                "id": "mem3",
                "content": "Third memory",
                "metadata": {"created_at": now.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.get_all.return_value = memories

        # Configure mock to return reference memory
        reference_memory = memories[1]  # mem2
        self.mock_im_store.get.return_value = reference_memory

        # Perform search with custom timestamp field
        results = self.strategy.search(
            query={
                "reference_id": "mem2",
                "sequence_size": 3,
                "timestamp_field": "metadata.created_at",
            },
            agent_id="agent-1",
            tier="im",
            limit=10,
        )

        # Verify sequence is retrieved using custom timestamp field
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem2")  # Reference memory
        self.assertEqual(results[2]["id"], "mem3")

    def test_reference_at_boundary(self):
        """Test search when reference memory is at the boundary of available memories."""
        # Set up test memories where reference is first or last
        now = datetime.now()
        memories = [
            {
                "id": "mem1",  # Reference at start
                "content": "First memory",
                "metadata": {"timestamp": (now - timedelta(minutes=2)).isoformat()},
            },
            {
                "id": "mem2",
                "content": "Second memory",
                "metadata": {"timestamp": (now - timedelta(minutes=1)).isoformat()},
            },
            {
                "id": "mem3",  # Reference at end
                "content": "Third memory",
                "metadata": {"timestamp": now.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.get_all.return_value = memories

        # Test with reference at start
        self.mock_stm_store.get.return_value = memories[0]  # mem1

        # Should return only memories after reference (including reference)
        results = self.strategy.search(
            query={
                "reference_id": "mem1",
                "sequence_size": 5,  # Request more than available
            },
            agent_id="agent-1",
            tier="stm",
            limit=10,
        )

        self.assertEqual(len(results), 3)  # All 3 memories
        self.assertEqual(results[0]["id"], "mem1")  # Reference memory
        self.assertEqual(results[1]["id"], "mem2")
        self.assertEqual(results[2]["id"], "mem3")

        # Test with reference at end
        self.mock_stm_store.get.return_value = memories[2]  # mem3

        # Should return only memories before reference (including reference)
        results = self.strategy.search(
            query={
                "reference_id": "mem3",
                "sequence_size": 5,  # Request more than available
            },
            agent_id="agent-1",
            tier="stm",
            limit=10,
        )

        self.assertEqual(len(results), 3)  # All 3 memories
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem2")
        self.assertEqual(results[2]["id"], "mem3")  # Reference memory

    def test_invalid_reference_id(self):
        """Test handling of invalid reference memory ID."""
        # Configure mock to return None (memory not found)
        self.mock_stm_store.get.return_value = None

        # Test with invalid reference ID
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"reference_id": "invalid_id", "sequence_size": 3},
                agent_id="agent-1",
                tier="stm",
            )

    def test_invalid_query_parameters(self):
        """Test handling of invalid query parameters."""
        # Configure mock to return a valid reference memory
        reference_memory = {
            "id": "mem1",
            "content": "Reference memory",
            "metadata": {"timestamp": datetime.now().isoformat()},
        }
        self.mock_im_store.get.return_value = reference_memory

        # Test with missing reference_id
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"sequence_size": 3},  # Missing reference_id
                agent_id="agent-1",
                tier="im",
            )

        # Test with negative sequence size
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"reference_id": "mem1", "sequence_size": -1},  # Negative size
                agent_id="agent-1",
                tier="im",
            )

        # Test with negative time window
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={
                    "reference_id": "mem1",
                    "time_window_minutes": -5,  # Negative window
                },
                agent_id="agent-1",
                tier="im",
            )

        # Test with conflicting parameters
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={
                    "reference_id": "mem1",
                    "sequence_size": 3,
                    "before_count": 1,  # Conflict with sequence_size
                    "after_count": 1,
                },
                agent_id="agent-1",
                tier="im",
            )


if __name__ == "__main__":
    unittest.main()
