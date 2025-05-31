import pytest
import numpy as np
from unittest.mock import MagicMock

from agents.random_agent import RandomAgent, MemoryRandomAgent
from memory.api.models import MazeObservation, MazeActionSpace

@pytest.fixture
def sample_observation():
    return MazeObservation(
        position=(1, 1),
        target=(2, 2),
        nearby_obstacles=[(0, 1), (1, 0)],
        steps=5,
    )


def test_random_agent_act_returns_valid_action(sample_observation):
    agent = RandomAgent(agent_id="test", action_space=4)
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_random_agent_demo_path(sample_observation):
    agent = RandomAgent(agent_id="test", action_space=4)
    agent.set_demo_path([2, 3, 1])
    assert agent.act(sample_observation) == 2
    assert agent.act(sample_observation) == 3
    assert agent.act(sample_observation) == 1
    # After demo path, should revert to random
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_memory_random_agent_act_returns_valid_action(sample_observation, monkeypatch):
    agent = MemoryRandomAgent(agent_id="test", action_space=4)
    # Patch memory.retrieve_similar_states to return empty
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = []
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_memory_random_agent_demo_path(sample_observation):
    agent = MemoryRandomAgent(agent_id="test", action_space=4)
    agent.set_demo_path([1, 0])
    assert agent.act(sample_observation) == 1
    assert agent.act(sample_observation) == 0


def test_memory_random_agent_memory_action(sample_observation):
    agent = MemoryRandomAgent(agent_id="test", action_space=4)
    # Patch memory.retrieve_similar_states to return a memory with action 2
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = [
        {"content": {"action": 2, "reward": 1}}
    ]
    # Patch np.random.random to always return 0.5 (> 0.2)
    np_random_backup = np.random.random
    np.random.random = lambda: 0.5
    action = agent.act(sample_observation)
    np.random.random = np_random_backup
    assert action == 2 