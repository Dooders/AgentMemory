"""
Agent classes for reinforcement learning and memory-augmented agents.

This module defines several agent classes for use in environments such as mazes or gridworlds:

- SimpleAgent: A basic Q-learning agent with epsilon-greedy action selection.
- MemoryAgent: An agent that augments Q-learning with episodic memory, using a MemorySpace for storing and retrieving experiences to inform decisions.
- RandomAgent: An agent that selects actions randomly, for baseline comparison.

Agents can be used with or without memory, and support demonstration paths for scripted exploration. The MemoryAgent leverages a memory system for enhanced learning and recall of past experiences.
"""

import numpy as np

from memory import (
    MemoryConfig,
    MemorySpace,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.utils.util import convert_numpy_to_python


# Base agent class without hooks
class SimpleAgent:
    def __init__(
        self,
        agent_id: str,
        action_space: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        **kwargs,
    ) -> None:
        """
        Initialize a SimpleAgent for reinforcement learning.

        Args:
            agent_id (str): Unique identifier for the agent.
            action_space (int): Number of possible actions.
            learning_rate (float): Q-learning learning rate.
            discount_factor (float): Q-learning discount factor.
            **kwargs: Additional arguments (unused).
        """
        self.agent_id = agent_id
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action values
        self.current_observation = None
        self.demo_path = None  # For scripted demo actions
        self.demo_step = 0
        self.step_number = 0

    def _get_state_key(self, observation: dict) -> str:
        """
        Generate a unique key for a given observation/state.

        Args:
            observation (dict): The environment observation.

        Returns:
            str: A string key representing the state.
        """
        return (
            f"{observation['position']}|{observation['target']}|{observation['steps']}"
        )

    def select_action(self, observation: dict, epsilon: float = 0.1) -> int:
        """
        Select an action using an epsilon-greedy policy or a demonstration path.

        Args:
            observation (dict): The current environment observation.
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
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        """
        Update the Q-value for a state-action pair using the Q-learning rule.

        Args:
            observation (dict): The current state observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_observation (dict): The next state observation.
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

    def act(self, observation: dict, epsilon: float = 0.1) -> int:
        """
        Choose and return an action for the given observation.

        Args:
            observation (dict): The current environment observation.
            epsilon (float): Probability of choosing a random action (exploration).

        Returns:
            int: The selected action index.
        """
        self.step_number += 1
        # Convert NumPy types to Python types
        self.current_observation = convert_numpy_to_python(observation)
        action = self.select_action(self.current_observation, epsilon)
        return int(action)  # Return as integer instead of ActionResult

    def set_demo_path(self, path: list[int]) -> None:
        """
        Set a predetermined path of actions for demonstration or scripted exploration.

        Args:
            path (list[int]): List of action indices to follow.
        """
        self.demo_path = path
        self.demo_step = 0


# Memory-enhanced agent using MemorySpace directly
class MemoryAgent(SimpleAgent):
    def __init__(
        self,
        agent_id: str,
        action_space: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        **kwargs,
    ) -> None:
        """
        Initialize a MemoryAgent that augments Q-learning with episodic memory.

        Args:
            agent_id (str): Unique identifier for the agent.
            action_space (int): Number of possible actions.
            learning_rate (float): Q-learning learning rate.
            discount_factor (float): Q-learning discount factor.
            **kwargs: Additional arguments (unused).
        """
        super().__init__(
            agent_id=agent_id,
            action_space=action_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            **kwargs,
        )

        memory_config = MemoryConfig(
            stm_config=RedisSTMConfig(
                ttl=120,  # Increase TTL to keep more memories active
                memory_limit=500,  # Increase memory limit
                use_mock=True,  # Use mock Redis for easy setup
            ),
            im_config=RedisIMConfig(
                ttl=240,  # Longer TTL for IM
                memory_limit=1000,  # Larger memory limit
                compression_level=0,  # No compression for IM
                use_mock=True,  # Use mock Redis for easy setup
            ),
            ltm_config=SQLiteLTMConfig(
                compression_level=0,  # No compression for LTM
                batch_size=20,  # Larger batch size
                db_path="memory_demo.db",  # Use a real file for SQLite
            ),
            cleanup_interval=1000,  # Reduce cleanup frequency
            enable_memory_hooks=False,  # Disable memory hooks since we're using direct API calls
            use_embedding_engine=True,  # Enable embedding engine for similarity search
            text_model_name="all-MiniLM-L6-v2",  # Use a default text embedding model
        )
        # Store the memory system and get the memory space for this agent
        self.memory_space = MemorySpace(agent_id, memory_config)

        # Keep track of visited states to avoid redundant storage
        self.visited_states = set()
        # Add memory cache for direct position lookups
        self.position_memory_cache = {}  # Mapping from positions to memories

    def select_action(self, observation: dict, epsilon: float = 0.1) -> int:
        """
        Select an action using memory-augmented Q-learning and experience recall.

        Args:
            observation (dict): The current environment observation.
            epsilon (float): Probability of choosing a random action (exploration).

        Returns:
            int: The selected action index.
        """
        self.current_observation = observation
        state_key = self._get_state_key(observation)
        position_key = str(observation["position"])  # Use position as direct lookup key

        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)

        # If we have a demo path, follow it first to ensure we explore the correct path
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return action

        # Try to retrieve similar experiences from memory
        try:
            # Store current state if not already visited
            if state_key not in self.visited_states:
                # Enhanced state representation
                enhanced_state = {
                    "position": observation["position"],
                    "target": observation["target"],
                    "steps": observation["steps"],
                    "nearby_obstacles": observation["nearby_obstacles"],
                    "manhattan_distance": abs(
                        observation["position"][0] - observation["target"][0]
                    )
                    + abs(observation["position"][1] - observation["target"][1]),
                    "state_key": state_key,
                    "position_key": position_key,  # Add position key for direct lookup
                }
                self.memory_space.store_state(
                    state_data=convert_numpy_to_python(enhanced_state),
                    step_number=self.step_number,
                    priority=0.7,  # Medium priority for state
                )
                self.visited_states.add(state_key)

            # Create a query with the enhanced state features
            query_state = {
                "position": observation["position"],
                "target": observation["target"],
                "steps": observation["steps"],
                "manhattan_distance": abs(
                    observation["position"][0] - observation["target"][0]
                )
                + abs(observation["position"][1] - observation["target"][1]),
            }

            # Use search strategy directly
            similar_states = self.memory_space.retrieve_similar_states(
                query_state=query_state,
                k=10,  # Increase from 5 to 10 to find more candidates
                memory_type="state",
            )

            # Direct position-based lookup as fallback
            if len(similar_states) == 0:
                # Try direct lookup from our position memory cache
                if position_key in self.position_memory_cache:
                    direct_memories = self.position_memory_cache[position_key]
                    similar_states = direct_memories

            for s in similar_states:
                # Update our position memory cache with this memory for future direct lookups
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

            # Strong bias toward using memory (higher than epsilon)
            if similar_states and np.random.random() > 0.2:
                # Use any experience with significant reward
                actions_from_memory = []
                for s in similar_states:
                    # Consider any action with a reward, not just positive ones
                    if "action" in s.get("content", {}):
                        # Weight action by reward to prefer better outcomes
                        # Add the action multiple times based on reward magnitude
                        reward = s["content"].get("reward", -1)
                        # Consider any reward better than average
                        # Add actions with better rewards more times
                        weight = 1
                        if reward > -2:  # Better than the typical step penalty
                            weight = 3
                        if reward > 0:  # Positive rewards get even more weight
                            weight = 5

                        for _ in range(weight):
                            actions_from_memory.append(s["content"]["action"])

                if actions_from_memory:
                    # Most common action from similar states, weighted by reward
                    chosen_action = max(
                        set(actions_from_memory), key=actions_from_memory.count
                    )
                    return chosen_action
        except Exception as e:
            # Fallback to regular selection on any error
            pass

        # Epsilon-greedy policy as fallback
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_space)
            return action
        else:
            action = np.argmax(self.q_table[state_key])
            return action

    def act(self, observation: dict, epsilon: float = 0.1) -> int:
        """
        Choose and return an action for the given observation, storing the action in memory.

        Args:
            observation (dict): The current environment observation.
            epsilon (float): Probability of choosing a random action (exploration).

        Returns:
            int: The selected action index.
        """
        self.step_number += 1
        # Convert NumPy types to Python types
        self.current_observation = convert_numpy_to_python(observation)
        action = self.select_action(self.current_observation, epsilon)

        # Store the action using memory space
        try:
            # Include more context in the action data
            position_key = str(observation["position"])
            action_data = {
                "action": int(action),
                "position": self.current_observation["position"],
                "state_key": self._get_state_key(self.current_observation),
                "steps": self.current_observation["steps"],
                "position_key": position_key,
            }
            self.memory_space.store_action(
                action_data=action_data,
                step_number=self.step_number,
                priority=0.6,  # Medium priority
            )

            # Add to position cache
            if position_key not in self.position_memory_cache:
                self.position_memory_cache[position_key] = []

            # Create a memory-like structure for our cache
            memory_entry = {"content": action_data, "step_number": self.step_number}

            self.position_memory_cache[position_key].append(memory_entry)

        except Exception as e:
            pass

        # Return action as integer
        return int(action)

    def update_q_value(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        """
        Update the Q-value and store the reward and outcome in memory.

        Args:
            observation (dict): The current state observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_observation (dict): The next state observation.
            done (bool): Whether the episode has ended.
        """
        # First, call the parent method to update Q-values
        super().update_q_value(observation, action, reward, next_observation, done)

        # Then store the reward and outcome using memory space
        try:
            # Enhance interaction data with more context
            position_key = str(observation["position"])
            next_position_key = str(next_observation["position"])

            interaction_data = {
                "action": int(action),
                "reward": float(reward),
                "next_state": convert_numpy_to_python(next_observation["position"]),
                "done": done,
                "state_key": self._get_state_key(observation),
                "next_state_key": self._get_state_key(next_observation),
                "steps": observation["steps"],
                "manhattan_distance": abs(
                    observation["position"][0] - observation["target"][0]
                )
                + abs(observation["position"][1] - observation["target"][1]),
                "position_key": position_key,
                "next_position_key": next_position_key,
            }

            # Increase priority for successful interactions
            priority = abs(float(reward)) / 100  # Base priority on reward magnitude
            if done and reward > 0:  # Successful completion
                priority = 1.0  # Maximum priority

            self.memory_space.store_interaction(
                interaction_data=interaction_data,
                step_number=self.step_number,
                priority=priority,
            )

            # Add to position cache - very important for successful experiences!
            # This ensures we can directly lookup both the current and next positions
            for pos_key in [position_key, next_position_key]:
                if pos_key not in self.position_memory_cache:
                    self.position_memory_cache[pos_key] = []

                # Create a memory-like structure for our cache
                memory_entry = {
                    "content": interaction_data,
                    "step_number": self.step_number,
                }

                self.position_memory_cache[pos_key].append(memory_entry)

            # If this was a successful completion, store it prominently
            if done and reward > 0:
                # Make extra copies in the cache to increase influence
                for _ in range(10):  # Store 10 copies to really emphasize this path
                    self.position_memory_cache[position_key].append(memory_entry)

        except Exception as e:
            pass


# Random agent that chooses actions randomly
class RandomAgent(SimpleAgent):
    def select_action(self, observation, epsilon=0.1):
        self.current_observation = observation
        return np.random.randint(self.action_space)
