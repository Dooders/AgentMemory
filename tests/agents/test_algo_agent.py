import pytest
import numpy as np
from unittest.mock import MagicMock

from agents.algo_agent import AlgoAgent, MemoryAlgoAgent
from memory.api.models import MazeObservation

@pytest.fixture
def sample_observation():
    return MazeObservation(
        position=(1, 1),
        target=(2, 2),
        nearby_obstacles=[(0, 1), (1, 0)],
        steps=5,
    )


def test_algo_agent_bfs_path(sample_observation):
    agent = AlgoAgent(agent_id="test", action_space=4, search_algo="bfs")
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_algo_agent_dfs_path(sample_observation):
    agent = AlgoAgent(agent_id="test", action_space=4, search_algo="dfs")
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_algo_agent_demo_path(sample_observation):
    agent = AlgoAgent(agent_id="test", action_space=4)
    agent.set_demo_path([1, 2])
    assert agent.act(sample_observation) == 1
    assert agent.act(sample_observation) == 2
    # After demo path, should revert to planning
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_memory_algo_agent_act_returns_valid_action(sample_observation):
    agent = MemoryAlgoAgent(agent_id="test", action_space=4)
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = []
    action = agent.act(sample_observation)
    assert 0 <= action < 4


def test_memory_algo_agent_demo_path(sample_observation):
    agent = MemoryAlgoAgent(agent_id="test", action_space=4)
    agent.set_demo_path([3, 0])
    assert agent.act(sample_observation) == 3
    assert agent.act(sample_observation) == 0


def test_memory_algo_agent_memory_action(sample_observation):
    agent = MemoryAlgoAgent(agent_id="test", action_space=4)
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = [
        {"content": {"action": 1, "reward": 1}}
    ]
    np_random_backup = np.random.random
    np.random.random = lambda: 0.5
    action = agent.act(sample_observation)
    np.random.random = np_random_backup
    assert action == 1 