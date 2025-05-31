# Agents Module

This module provides a variety of agent classes for use in reinforcement learning and maze navigation environments. Agents can be used as-is or extended for custom behaviors. Many agents have both standard and memory-augmented variants that leverage episodic and semantic memory for improved performance.

## Agent Types

### 1. `Agent` (Abstract Base Class)
Defines the interface for all agents. To implement a custom agent, inherit from this class and implement the required methods.

**API:**
```python
class Agent(ABC):
    def __init__(self, agent_id: str, action_space, **kwargs): ...
    @abstractmethod
    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int: ...
    @abstractmethod
    def set_demo_path(self, path: list[int]) -> None: ...
```

### 2. `RandomAgent`
Selects actions randomly from the action space. Useful as a baseline.

### 3. `MemoryRandomAgent`
A random agent that also stores and retrieves state/action information from a memory system, biasing action selection toward previously successful actions.

### 4. `AlgoAgent`
A planning agent that uses search algorithms (BFS/DFS or custom) to plan a path to the target. Good for deterministic environments.

### 5. `MemoryAlgoAgent`
A planning agent with memory augmentation. Retrieves similar states from memory to bias planning and action selection.

### 6. `QAgent`
Implements tabular Q-learning. Maintains a Q-table for state-action values and uses an epsilon-greedy policy.

### 7. `MemoryQAgent`
A Q-learning agent with memory augmentation. Stores and retrieves states, actions, and interactions from memory to bias exploration and exploitation.

### 8. `DeepQAgent`
Implements Deep Q-Learning using PyTorch. Uses a neural network to approximate Q-values and experience replay for training.

### 9. `MemoryDeepQAgent`
A deep Q-learning agent with memory augmentation. Stores and retrieves states and interactions from memory to bias action selection and learning.

---

## Usage

> **Note:** Only the abstract `Agent` is exposed in `agents/__init__.py`. To use concrete agents, import them directly from their respective files:

```python
from agents.random_agent import RandomAgent, MemoryRandomAgent
from agents.algo_agent import AlgoAgent, MemoryAlgoAgent
from agents.q_agent import QAgent, MemoryQAgent
from agents.deep_q_agent import DeepQAgent, MemoryDeepQAgent
```

## Example

```python
from agents.q_agent import QAgent
from memory.api.models import MazeObservation

agent = QAgent(agent_id="A1", action_space=4)
obs = MazeObservation(position=(0,0), target=(3,3), steps=0, nearby_obstacles=[])
action = agent.act(obs)
```

## Extending Agents
To create your own agent, inherit from `Agent` and implement the `act` and `set_demo_path` methods.

## Memory-Augmented Agents
Memory-augmented agents use a `MemorySpace` object to store and retrieve states, actions, and interactions. This enables:
- Retrieval of similar past states for biasing action selection
- Storing successful actions/interactions for future use
- Episodic and semantic memory integration

## Requirements
- `memory` module (for memory-augmented agents)
- `numpy`, `torch` (for DeepQAgent)

---

## File Overview
- `base.py`: Abstract base class
- `random_agent.py`: RandomAgent, MemoryRandomAgent
- `algo_agent.py`: AlgoAgent, MemoryAlgoAgent
- `q_agent.py`: QAgent, MemoryQAgent
- `deep_q_agent.py`: DeepQAgent, MemoryDeepQAgent

---

For more details, see the docstrings in each agent class. 