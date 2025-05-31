import pytest
import numpy as np
from unittest.mock import MagicMock

from agents.q_agent import QAgent, MemoryQAgent
from memory.api.models import MazeObservation

@pytest.fixture
def sample_observation():
    return MazeObservation(
        position=(0, 0),
        target=(1, 1),
        nearby_obstacles=[(0, 1)],
        steps=1,
    )

@pytest.fixture
def next_observation():
    return MazeObservation(
        position=(0, 1),
        target=(1, 1),
        nearby_obstacles=[(1, 1)],
        steps=2,
    )

def test_q_agent_epsilon_greedy_action(sample_observation):
    agent = QAgent(agent_id="test", action_space=4)
    # Force random action
    np_random_backup = np.random.random
    np.random.random = lambda: 0.05
    action = agent.act(sample_observation, epsilon=1.0)
    np.random.random = np_random_backup
    assert 0 <= action < 4

def test_q_agent_demo_path(sample_observation):
    agent = QAgent(agent_id="test", action_space=4)
    agent.set_demo_path([2, 1])
    assert agent.act(sample_observation) == 2
    assert agent.act(sample_observation) == 1

def test_q_agent_q_value_update(sample_observation, next_observation):
    agent = QAgent(agent_id="test", action_space=4)
    action = 1
    reward = 1.0
    done = False
    agent.update_q_value(sample_observation, action, reward, next_observation, done)
    state_key = agent._get_state_key(sample_observation)
    assert agent.q_table[state_key][action] != 0

def test_memory_q_agent_act_returns_valid_action(sample_observation):
    agent = MemoryQAgent(agent_id="test", action_space=4)
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = []
    action = agent.act(sample_observation)
    assert 0 <= action < 4

def test_memory_q_agent_demo_path(sample_observation):
    agent = MemoryQAgent(agent_id="test", action_space=4)
    agent.set_demo_path([3, 0])
    assert agent.act(sample_observation) == 3
    assert agent.act(sample_observation) == 0

def test_memory_q_agent_memory_action(sample_observation):
    agent = MemoryQAgent(agent_id="test", action_space=4)
    agent.memory = MagicMock()
    agent.memory.retrieve_similar_states.return_value = [
        {"content": {"action": 2, "reward": 1}}
    ]
    np_random_backup = np.random.random
    np.random.random = lambda: 0.5
    action = agent.act(sample_observation)
    np.random.random = np_random_backup
    assert action == 2 