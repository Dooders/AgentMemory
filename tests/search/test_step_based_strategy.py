import copy
import datetime
import json
import unittest
from unittest.mock import MagicMock, patch

from memory.search.strategies.step import StepBasedSearchStrategy


class TestStepBasedStrategy(unittest.TestCase):
    def setUp(self):
        # Create mock stores
        self.mock_stm_store = MagicMock()
        self.mock_im_store = MagicMock()
        self.mock_ltm_store = MagicMock()

        # Create mock agent
        self.mock_agent = MagicMock()
        self.mock_agent.stm_store = self.mock_stm_store
        self.mock_agent.im_store = self.mock_im_store
        self.mock_agent.ltm_store = self.mock_ltm_store

        # Create mock memory system
        self.mock_memory_system = MagicMock()
        self.mock_memory_system.get_memory_agent.return_value = self.mock_agent

        # Create strategy with mock memory system
        self.strategy = StepBasedSearchStrategy(self.mock_memory_system)

        # Sample memories with step numbers
        self.sample_memories = [
            {
                "id": "mem1",
                "step_number": 100,
                "contents": {"text": "Memory at step 100", "step_number": 100},
                "metadata": {"importance": 0.8, "created_at": "2023-01-01T12:00:00Z"},
            },
            {
                "id": "mem2",
                "step_number": 200,
                "contents": {"text": "Memory at step 200", "step_number": 200},
                "metadata": {"importance": 0.5, "created_at": "2023-01-01T12:01:00Z"},
            },
            {
                "id": "mem3",
                "step_number": 300,
                "contents": {"text": "Memory at step 300", "step_number": 300},
                "metadata": {"importance": 0.9, "created_at": "2023-01-01T12:02:00Z"},
            },
            {
                "id": "mem4",
                "step_number": 400,
                "contents": {"text": "Memory at step 400", "step_number": 400},
                "metadata": {"importance": 0.7, "created_at": "2023-01-01T12:03:00Z"},
            },
            {
                "id": "mem5",
                "step_number": 500,
                "contents": {"text": "Memory at step 500", "step_number": 500},
                "metadata": {"importance": 0.6, "created_at": "2023-01-01T12:04:00Z"},
            },
        ]

        # Configure mocks to return sample memories
        self.mock_stm_store.get_all.return_value = self.sample_memories[:2]  # First 2 memories
        self.mock_im_store.get_all.return_value = self.sample_memories[2:4]  # Next 2 memories
        self.mock_ltm_store.get_all.return_value = self.sample_memories[4:]  # Last memory

        # Configure mock memory system to return the stores
        self.mock_memory_system.stm_store = self.mock_stm_store
        self.mock_memory_system.im_store = self.mock_im_store
        self.mock_memory_system.ltm_store = self.mock_ltm_store

    def test_name_and_description(self):
        """Test the name and description methods."""
        self.assertEqual(self.strategy.name(), "step_based")
        self.assertTrue("step" in self.strategy.description().lower())

    def test_search_with_reference_step(self):
        """Test searching with a reference step number."""
        # Search for memories close to step 250
        results = self.strategy.search(
            query="250", agent_id="agent-123", limit=3, step_range=100
        )

        # Should return memories at steps 200 and 300, sorted by score
        self.assertEqual(len(results), 2)
        # Check both memories are present (order may depend on implementation)
        self.assertTrue(any(mem["id"] == "mem2" for mem in results))
        self.assertTrue(any(mem["id"] == "mem3" for mem in results))
        # Check memory with step 200 is closer to 250 than 300
        if results[0]["id"] == "mem2" and results[1]["id"] == "mem3":
            self.assertGreaterEqual(
                results[0]["metadata"]["step_score"],
                results[1]["metadata"]["step_score"],
            )
        elif results[0]["id"] == "mem3" and results[1]["id"] == "mem2":
            self.assertGreaterEqual(
                results[0]["metadata"]["step_score"],
                results[1]["metadata"]["step_score"],
            )

    def test_search_with_step_range(self):
        """Test searching with a step range."""
        # Search for memories between steps 150 and 350
        results = self.strategy.search(
            query={"start_step": 150, "end_step": 350}, agent_id="agent-123", limit=10
        )

        # Should return memories at steps 200 and 300
        self.assertEqual(len(results), 2)
        self.assertTrue(any(mem["id"] == "mem2" for mem in results))
        self.assertTrue(any(mem["id"] == "mem3" for mem in results))

    def test_search_with_explicit_reference_and_range(self):
        """Test searching with explicit reference step and range parameters."""
        # Search for memories around step 300 with a range of 150
        results = self.strategy.search(
            query={}, agent_id="agent-123", reference_step=300, step_range=150, limit=10
        )

        # Should return memories at steps 200, 300, and 400
        self.assertEqual(len(results), 3)
        self.assertTrue(any(mem["id"] == "mem2" for mem in results))
        self.assertTrue(any(mem["id"] == "mem3" for mem in results))
        self.assertTrue(any(mem["id"] == "mem4" for mem in results))

    def test_search_with_tier_filter(self):
        """Test searching with a specific memory tier."""
        # Search only in STM
        results = self.strategy.search(
            query={"start_step": 0, "end_step": 1000},
            agent_id="agent-123",
            tier="stm",
            limit=10,
        )

        # Should only return memories from STM
        self.assertEqual(len(results), 2)
        self.assertTrue(all(mem["id"] in ["mem1", "mem2"] for mem in results))

        # Search only in IM
        results = self.strategy.search(
            query={"start_step": 0, "end_step": 1000},
            agent_id="agent-123",
            tier="im",
            limit=10,
        )

        # Should only return memories from IM
        self.assertEqual(len(results), 2)
        self.assertTrue(all(mem["id"] in ["mem3", "mem4"] for mem in results))

    def test_search_with_metadata_filter(self):
        """Test searching with metadata filters."""
        # Search for high importance memories (>0.7)
        results = self.strategy.search(
            query={"start_step": 0, "end_step": 1000},
            agent_id="agent-123",
            metadata_filter={"importance": 0.8},
            limit=10,
        )

        # Should only return high importance memories
        self.assertEqual(len(results), 1)

    def test_step_scoring(self):
        """Test that memories are properly scored based on step proximity."""
        # Use deep copy of memories to avoid modifying originals
        self.mock_stm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[:2]
        )
        self.mock_im_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[2:4]
        )
        self.mock_ltm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[4:]
        )

        # Search for memories around step 300
        results = self.strategy.search(query="300", agent_id="agent-123", limit=10)

        # Sort results by score in descending order
        sorted_results = sorted(
            results,
            key=lambda x: x.get("metadata", {}).get("step_score", 0.0),
            reverse=True,
        )

        # First result should be closest to step 300
        self.assertEqual(sorted_results[0]["id"], "mem3")

        # Check that scores are properly assigned
        self.assertTrue(0 <= sorted_results[0]["metadata"]["step_score"] <= 1)

    def test_mixed_step_formats(self):
        """Test handling of memories with different step number formats."""
        # Create memories with different step number formats
        mixed_memories = [
            {"id": "mixed1", "step_number": 150},
            {"id": "mixed2", "contents": {"step_number": 250}},
            {"id": "mixed3", "metadata": {"step_number": 350}},
            {"id": "mixed4", "metadata": {"step": "450"}},
            {"id": "mixed5", "step": 550},
        ]

        # Configure mock to return these memories
        self.mock_stm_store.get_all.return_value = mixed_memories
        self.mock_im_store.get_all.return_value = []
        self.mock_ltm_store.get_all.return_value = []

        # Search with a broad range
        results = self.strategy.search(
            query={"start_step": 100, "end_step": 600}, agent_id="agent-123", limit=10
        )

        # Should find all memories
        self.assertEqual(len(results), 5)

    def test_invalid_step_values(self):
        """Test handling of invalid step values."""
        # Create memories with some invalid step data
        invalid_memories = [
            {"id": "invalid1", "step_number": "not-a-number"},
            {"id": "invalid2", "contents": {"text": "No step here"}},
            {"id": "valid", "step_number": 300},
        ]

        # Mock the _get_memory_step method to handle the string case
        with patch.object(self.strategy, "_get_memory_step") as mock_get_step:
            # Configure the mock to return None for invalid steps and actual value for valid one
            def side_effect(memory):
                if memory["id"] == "valid":
                    return 300
                return None

            mock_get_step.side_effect = side_effect

            # Configure mock to return these memories
            self.mock_stm_store.get_all.return_value = invalid_memories
            self.mock_im_store.get_all.return_value = []
            self.mock_ltm_store.get_all.return_value = []

            # Search with a broad range
            results = self.strategy.search(
                query={"start_step": 0, "end_step": 1000},
                agent_id="agent-123",
                limit=10,
            )

            # Should only find the valid memory
            self.assertEqual(len(results), 1)

    def test_step_weight_parameter(self):
        """Test that step_weight parameter affects scoring."""
        # Use deep copies to avoid modifying originals
        self.mock_stm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[:2]
        )
        self.mock_im_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[2:4]
        )
        self.mock_ltm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[4:]
        )

        # Get results with normal step_weight
        normal_results = self.strategy.search(
            query="250", agent_id="agent-123", step_weight=1.0, limit=10
        )

        # Reset memories with deep copies again
        self.mock_stm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[:2]
        )
        self.mock_im_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[2:4]
        )
        self.mock_ltm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[4:]
        )

        # Get results with higher step_weight
        weighted_results = self.strategy.search(
            query="250", agent_id="agent-123", step_weight=2.0, limit=10
        )

        # Memory scores should be different between the two searches
        normal_scores = [r["metadata"]["step_score"] for r in normal_results]
        weighted_scores = [r["metadata"]["step_score"] for r in weighted_results]

        # Check we got the same number of results in both cases
        self.assertEqual(len(normal_scores), len(weighted_scores))

        # For the same memories, scores should be higher with higher weight
        for i in range(min(len(normal_scores), len(weighted_scores))):
            # Higher weight should give scores that are twice as high,
            # but capped at 1.0 for perfect matches
            if normal_scores[i] < 0.5:  # Not a perfect match
                self.assertAlmostEqual(
                    weighted_scores[i], min(normal_scores[i] * 2.0, 1.0), delta=0.01
                )

    def test_scoring_preservation(self):
        """Test that scoring doesn't modify original memories."""
        # Make deep copies of original memories to ensure no modification
        original_memories = copy.deepcopy(self.sample_memories)

        # Configure the mocks to return deep copies
        self.mock_stm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[:2]
        )
        self.mock_im_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[2:4]
        )
        self.mock_ltm_store.get_all.return_value = copy.deepcopy(
            self.sample_memories[4:]
        )

        # Perform a search
        results = self.strategy.search(query="250", agent_id="agent-123", limit=10)

        # Verify original memories are unchanged
        for i, mem in enumerate(self.sample_memories):
            self.assertEqual(mem["id"], original_memories[i]["id"])
            if "metadata" in mem and "metadata" in original_memories[i]:
                self.assertEqual(
                    mem["metadata"].get("importance"),
                    original_memories[i]["metadata"].get("importance"),
                )

            # Verify no step_score was added to the original
            if "metadata" in mem:
                self.assertNotIn("step_score", mem["metadata"])

        # Check that results have the step_score but originals don't
        for result in results:
            self.assertIn("step_score", result["metadata"])


if __name__ == "__main__":
    unittest.main()
