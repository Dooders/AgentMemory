import numpy as np

from agents import Agent
from memory import (
    MemoryConfig,
    MemorySpace,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.api.models import MazeActionSpace, MazeObservation
from memory.utils.util import convert_numpy_to_python


class QAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        **kwargs,
    ) -> None:
        """
        Initialize a q-learning agent for reinforcement learning.

        Args:
            agent_id (str): Unique identifier for the agent.
            action_space (int or MazeActionSpace): Number of possible actions or MazeActionSpace object.
            learning_rate (float): Q-learning learning rate.
            discount_factor (float): Q-learning discount factor.
            **kwargs: Additional arguments.
        """
        self.agent_id = agent_id
        if isinstance(action_space, MazeActionSpace):
            self.action_space = action_space.n
            self.action_space_model = action_space
        else:
            self.action_space = action_space
            self.action_space_model = MazeActionSpace(n=action_space)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action values #! fuzzier search than exact match
        self.current_observation = None
        self.demo_path = None  # For scripted demo actions
        self.demo_step = 0
        self.step_number = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_state_key(self, observation: MazeObservation) -> str:
        """
        Generate a unique key for a given observation/state.

        Args:
            observation (MazeObservation): The environment observation.

        Returns:
            str: A string key representing the state.
        """
        return f"{observation.position}|{observation.target}|{observation.steps}"

    def select_action(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        """
        Select an action using an epsilon-greedy policy or a demonstration path.

        Args:
            observation (MazeObservation): The current environment observation.
            epsilon (float): Probability of choosing a random action (exploration).

        Returns:
            int: The selected action index.
        """
        self.current_observation = observation
        state_key = self._get_state_key(observation)

        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)

        # If we have a demo path, follow it first to ensure we explore the correct path
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return action

        # Epsilon-greedy policy
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state_key])

    def update_q_value(
        self,
        observation: MazeObservation,
        action: int,
        reward: float,
        next_observation: MazeObservation,
        done: bool,
    ) -> None:
        """
        Update the Q-value for a state-action pair using the Q-learning rule.

        Args:
            observation (MazeObservation): The current state observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_observation (MazeObservation): The next state observation.
            done (bool): Whether the episode has ended.
        """
        state_key = self._get_state_key(observation)
        next_state_key = self._get_state_key(next_observation)

        # Initialize next state if not seen before
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)

        # Q-learning update
        current_q = self.q_table[state_key][action]

        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        """
        Choose and return an action for the given observation.

        Args:
            observation (MazeObservation): The current environment observation.
            epsilon (float): Probability of choosing a random action (exploration).

        Returns:
            int: The selected action index.
        """
        self.step_number += 1
        # Convert NumPy types to Python types
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        self.current_observation = obs
        action = self.select_action(self.current_observation, epsilon)
        return int(action)

    def set_demo_path(self, path: list[int]) -> None:
        """
        Set a predetermined path of actions for demonstration or scripted exploration.

        Args:
            path (list[int]): List of action indices to follow.
        """
        self.demo_path = path
        self.demo_step = 0


class MemoryQAgent(QAgent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            action_space=action_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            **kwargs,
        )
        memory_config = MemoryConfig(
            stm_config=RedisSTMConfig(
                ttl=120,
                memory_limit=500,
                use_mock=True,
            ),
            im_config=RedisIMConfig(
                ttl=240,
                memory_limit=1000,
                compression_level=0,
                use_mock=True,
            ),
            ltm_config=SQLiteLTMConfig(
                compression_level=0,
                batch_size=20,
                db_path="memory_demo.db",
            ),
            cleanup_interval=1000,
            enable_memory_hooks=False,
            use_embedding_engine=True,
            text_model_name="all-MiniLM-L6-v2",
        )
        self.memory = MemorySpace(agent_id, memory_config)
        self.visited_states = set()
        self.position_memory_cache = {}

    def select_action(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        self.current_observation = observation
        state_key = self._get_state_key(observation)
        position_key = str(observation.position)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return action
        try:
            if state_key not in self.visited_states:
                enhanced_state = {
                    "position": observation.position,
                    "target": observation.target,
                    "steps": observation.steps,
                    "nearby_obstacles": getattr(observation, "nearby_obstacles", None),
                    "manhattan_distance": abs(
                        observation.position[0] - observation.target[0]
                    )
                    + abs(observation.position[1] - observation.target[1]),
                    "state_key": state_key,
                    "position_key": position_key,
                }
                self.memory.store_state(
                    state_data=convert_numpy_to_python(enhanced_state),
                    step_number=self.step_number,
                    priority=0.7,
                )
                self.visited_states.add(state_key)
            query_state = {
                "position": observation.position,
                "target": observation.target,
                "steps": observation.steps,
                "manhattan_distance": abs(
                    observation.position[0] - observation.target[0]
                )
                + abs(observation.position[1] - observation.target[1]),
            }
            similar_states = self.memory.retrieve_similar_states(
                query_state=query_state,
                k=10,
                memory_type="state",
            )
            if len(similar_states) == 0:
                if position_key in self.position_memory_cache:
                    similar_states = self.position_memory_cache[position_key]
            for s in similar_states:
                mem_position = None
                if "position" in s.get("content", {}):
                    mem_position = str(s["content"]["position"])
                elif "next_state" in s.get("content", {}):
                    mem_position = str(s["content"]["next_state"])
                if mem_position:
                    if mem_position not in self.position_memory_cache:
                        self.position_memory_cache[mem_position] = []
                    if s not in self.position_memory_cache[mem_position]:
                        self.position_memory_cache[mem_position].append(s)
            if similar_states and np.random.random() > 0.2:
                actions_from_memory = []
                for s in similar_states:
                    if "action" in s.get("content", {}):
                        reward = s["content"].get("reward", -1)
                        weight = 1
                        if reward > -2:
                            weight = 3
                        if reward > 0:
                            weight = 5
                        for _ in range(weight):
                            actions_from_memory.append(s["content"]["action"])
                if actions_from_memory:
                    chosen_action = max(
                        set(actions_from_memory), key=actions_from_memory.count
                    )
                    return chosen_action
        except Exception:
            pass
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        self.step_number += 1
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        self.current_observation = obs
        action = self.select_action(self.current_observation, epsilon)
        try:
            position_key = str(observation.position)
            action_data = {
                "action": int(action),
                "position": self.current_observation.position,
                "state_key": self._get_state_key(self.current_observation),
                "steps": self.current_observation.steps,
                "position_key": position_key,
            }
            self.memory.store_action(
                action_data=action_data,
                step_number=self.step_number,
                priority=0.6,
            )
            if position_key not in self.position_memory_cache:
                self.position_memory_cache[position_key] = []
            memory_entry = {"content": action_data, "step_number": self.step_number}
            self.position_memory_cache[position_key].append(memory_entry)
        except Exception:
            pass
        return int(action)

    def update_q_value(
        self,
        observation: MazeObservation,
        action: int,
        reward: float,
        next_observation: MazeObservation,
        done: bool,
    ) -> None:
        super().update_q_value(observation, action, reward, next_observation, done)
        try:
            position_key = str(observation.position)
            next_position_key = str(next_observation.position)
            interaction_data = {
                "action": int(action),
                "reward": float(reward),
                "next_state": convert_numpy_to_python(next_observation.position),
                "done": done,
                "state_key": self._get_state_key(observation),
                "next_state_key": self._get_state_key(next_observation),
                "steps": observation.steps,
                "manhattan_distance": abs(
                    observation.position[0] - observation.target[0]
                )
                + abs(observation.position[1] - observation.target[1]),
                "position_key": position_key,
                "next_position_key": next_position_key,
            }
            priority = abs(float(reward)) / 100
            if done and reward > 0:
                priority = 1.0
            self.memory.store_interaction(
                interaction_data=interaction_data,
                step_number=self.step_number,
                priority=priority,
            )
            for pos_key in [position_key, next_position_key]:
                if pos_key not in self.position_memory_cache:
                    self.position_memory_cache[pos_key] = []
                memory_entry = {
                    "content": interaction_data,
                    "step_number": self.step_number,
                }
                self.position_memory_cache[pos_key].append(memory_entry)
            if done and reward > 0:
                for _ in range(10):
                    self.position_memory_cache[position_key].append(memory_entry)
        except Exception:
            pass
