"""Tests for the TimeWindowStrategy class."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from memory.search.strategies.window import TimeWindowStrategy


class TestTimeWindowStrategy(unittest.TestCase):
    """Tests for the TimeWindowStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create strategy with mock dependencies
        self.strategy = TimeWindowStrategy(
            self.mock_stm_store, self.mock_im_store, self.mock_ltm_store
        )

    def test_name_and_description(self):
        """Test name and description methods."""
        self.assertEqual(self.strategy.name(), "window")
        self.assertTrue(isinstance(self.strategy.description(), str))
        self.assertIn("window", self.strategy.description().lower())

    def test_search_with_time_range(self):
        """Test search with specific time range."""
        # Set up test memories with different timestamps
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)
        three_hours_ago = now - timedelta(hours=3)

        memories = [
            {
                "id": "mem1",
                "content": "Recent memory",
                "metadata": {"timestamp": now.isoformat()},
            },
            {
                "id": "mem2",
                "content": "Memory from 1 hour ago",
                "metadata": {"timestamp": one_hour_ago.isoformat()},
            },
            {
                "id": "mem3",
                "content": "Memory from 2 hours ago",
                "metadata": {"timestamp": two_hours_ago.isoformat()},
            },
            {
                "id": "mem4",
                "content": "Memory from 3 hours ago",
                "metadata": {"timestamp": three_hours_ago.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Search for memories within the last 2 hours
        results = self.strategy.search(
            query={
                "start_time": two_hours_ago.isoformat(),
                "end_time": now.isoformat(),
            },
            agent_id="agent-1",
            tier="stm",
            limit=5,
        )

        # Should return 3 memories from the last 2 hours
        self.assertEqual(len(results), 3)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertIn("mem3", result_ids)
        self.assertNotIn("mem4", result_ids)

        # Verify results are ordered by timestamp (most recent first)
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[1]["id"], "mem2")
        self.assertEqual(results[2]["id"], "mem3")

    def test_search_with_last_n_minutes(self):
        """Test search with the last N minutes parameter."""
        # Set up test memories with different timestamps
        now = datetime.now()
        ten_min_ago = now - timedelta(minutes=10)
        thirty_min_ago = now - timedelta(minutes=30)
        sixty_min_ago = now - timedelta(minutes=60)

        memories = [
            {
                "id": "mem1",
                "content": "Very recent memory",
                "metadata": {"timestamp": now.isoformat()},
            },
            {
                "id": "mem2",
                "content": "Memory from 10 minutes ago",
                "metadata": {"timestamp": ten_min_ago.isoformat()},
            },
            {
                "id": "mem3",
                "content": "Memory from 30 minutes ago",
                "metadata": {"timestamp": thirty_min_ago.isoformat()},
            },
            {
                "id": "mem4",
                "content": "Memory from 60 minutes ago",
                "metadata": {"timestamp": sixty_min_ago.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Search for memories from the last 20 minutes
        results = self.strategy.search(
            query={"last_minutes": 20}, agent_id="agent-1", tier="im", limit=5
        )

        # Should return 2 memories from the last 20 minutes
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertNotIn("mem3", result_ids)
        self.assertNotIn("mem4", result_ids)

    def test_search_with_last_n_hours(self):
        """Test search with the last N hours parameter."""
        # Set up test memories with different timestamps
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        three_hours_ago = now - timedelta(hours=3)
        five_hours_ago = now - timedelta(hours=5)

        memories = [
            {
                "id": "mem1",
                "content": "Recent memory",
                "metadata": {"timestamp": now.isoformat()},
            },
            {
                "id": "mem2",
                "content": "Memory from 1 hour ago",
                "metadata": {"timestamp": one_hour_ago.isoformat()},
            },
            {
                "id": "mem3",
                "content": "Memory from 3 hours ago",
                "metadata": {"timestamp": three_hours_ago.isoformat()},
            },
            {
                "id": "mem4",
                "content": "Memory from 5 hours ago",
                "metadata": {"timestamp": five_hours_ago.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Search for memories from the last 4 hours
        results = self.strategy.search(
            query={"last_hours": 4}, agent_id="agent-1", tier="ltm", limit=5
        )

        # Should return 3 memories from the last 4 hours
        self.assertEqual(len(results), 3)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertIn("mem3", result_ids)
        self.assertNotIn("mem4", result_ids)

    def test_search_with_last_n_days(self):
        """Test search with the last N days parameter."""
        # Set up test memories with different timestamps
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        five_days_ago = now - timedelta(days=5)

        memories = [
            {
                "id": "mem1",
                "content": "Today's memory",
                "metadata": {"timestamp": now.isoformat()},
            },
            {
                "id": "mem2",
                "content": "Yesterday's memory",
                "metadata": {"timestamp": one_day_ago.isoformat()},
            },
            {
                "id": "mem3",
                "content": "Memory from 2 days ago",
                "metadata": {"timestamp": two_days_ago.isoformat()},
            },
            {
                "id": "mem4",
                "content": "Memory from 5 days ago",
                "metadata": {"timestamp": five_days_ago.isoformat()},
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Search for memories from the last 3 days
        results = self.strategy.search(
            query={"last_days": 3}, agent_id="agent-1", tier="stm", limit=5
        )

        # Should return 3 memories from the last 3 days
        self.assertEqual(len(results), 3)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertIn("mem3", result_ids)
        self.assertNotIn("mem4", result_ids)

    def test_search_with_metadata_filter(self):
        """Test search with time window and metadata filter combined."""
        # Set up test memories with different timestamps and metadata
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)

        memories = [
            {
                "id": "mem1",
                "content": "Important recent memory",
                "metadata": {"timestamp": now.isoformat(), "type": "important"},
            },
            {
                "id": "mem2",
                "content": "Regular recent memory",
                "metadata": {"timestamp": one_hour_ago.isoformat(), "type": "regular"},
            },
            {
                "id": "mem3",
                "content": "Important older memory",
                "metadata": {
                    "timestamp": two_hours_ago.isoformat(),
                    "type": "important",
                },
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_im_store.list.return_value = memories

        # Search for important memories from the last 2 hours
        results = self.strategy.search(
            query={"last_hours": 2},
            agent_id="agent-1",
            tier="im",
            metadata_filter={"type": "important"},
            limit=5,
        )

        # Should only return mem1 (important and within last 2 hours)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

    def test_custom_timestamp_field(self):
        """Test search with a custom timestamp field."""
        # Set up test memories with different custom timestamp fields
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)

        memories = [
            {
                "id": "mem1",
                "content": "Recent memory",
                "metadata": {
                    "created_at": now.isoformat(),
                },
            },
            {
                "id": "mem2",
                "content": "Older memory",
                "metadata": {
                    "created_at": two_hours_ago.isoformat(),
                },
            },
        ]

        # Configure mock to return all memories for listing
        self.mock_ltm_store.list.return_value = memories

        # Search with custom timestamp field
        results = self.strategy.search(
            query={"last_hours": 1, "timestamp_field": "metadata.created_at"},
            agent_id="agent-1",
            tier="ltm",
            limit=5,
        )

        # Should only return mem1 (within last hour)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")

    def test_missing_timestamp(self):
        """Test handling of memories with missing timestamp field."""
        # Set up test memories, some missing timestamps
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        memories = [
            {
                "id": "mem1",
                "content": "With timestamp",
                "metadata": {"timestamp": now.isoformat()},
            },
            {
                "id": "mem2",
                "content": "Also with timestamp",
                "metadata": {"timestamp": one_hour_ago.isoformat()},
            },
            {"id": "mem3", "content": "Missing timestamp", "metadata": {}},
        ]

        # Configure mock to return all memories for listing
        self.mock_stm_store.list.return_value = memories

        # Search for memories from the last 2 hours
        results = self.strategy.search(
            query={"last_hours": 2}, agent_id="agent-1", tier="stm", limit=5
        )

        # Should only return memories with valid timestamps in range
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("mem1", result_ids)
        self.assertIn("mem2", result_ids)
        self.assertNotIn("mem3", result_ids)

    def test_invalid_time_format(self):
        """Test handling of invalid time format."""
        # Configure mock to return memories
        self.mock_stm_store.list.return_value = []

        # Test with invalid time format
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={
                    "start_time": "not a valid timestamp",
                    "end_time": datetime.now().isoformat(),
                },
                agent_id="agent-1",
                tier="stm",
            )

    def test_invalid_query_parameters(self):
        """Test handling of invalid or conflicting query parameters."""
        # Configure mock to return memories
        self.mock_im_store.list.return_value = []

        # Test with conflicting time parameters
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"last_minutes": 30, "last_hours": 2},
                agent_id="agent-1",
                tier="im",
            )

        # Test with both time range and last_x parameters
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={
                    "start_time": "2023-01-01T00:00:00",
                    "end_time": "2023-01-02T00:00:00",
                    "last_days": 1,
                },
                agent_id="agent-1",
                tier="im",
            )

        # Test with negative time values
        with self.assertRaises(ValueError):
            self.strategy.search(
                query={"last_minutes": -10}, agent_id="agent-1", tier="im"
            )


if __name__ == "__main__":
    unittest.main()
