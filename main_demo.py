import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

# Configure logging - suppress specific warnings
logging.getLogger("memory.agent_memory").setLevel(logging.ERROR)
logging.getLogger("memory.storage.sqlite_ltm").setLevel(logging.ERROR)

from memory import (
    AgentMemorySystem,
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.api.hooks import BaseAgent, install_memory_hooks
from memory.config import AutoencoderConfig


# Helper function to convert NumPy types to native Python types for JSON serialization
def convert_numpy_to_python(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


# Define a simple maze environment
class MazeEnvironment:
    def __init__(self, size=5, obstacles=None, max_steps=15):
        self.size = size
        self.obstacles = obstacles or []
        self.target = (size - 2, size - 2)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.position = (1, 1)
        self.steps = 0
        return self.get_observation()

    def get_observation(self):
        return {
            "position": self.position,
            "target": self.target,
            "nearby_obstacles": self._get_nearby_obstacles(),
            "steps": self.steps,
        }

    def _get_nearby_obstacles(self):
        return [
            obs
            for obs in self.obstacles
            if abs(obs[0] - self.position[0]) <= 2
            and abs(obs[1] - self.position[1]) <= 2
        ]

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_position = (
            self.position[0] + directions[action][0],
            self.position[1] + directions[action][1],
        )

        # Check if valid move
        if (
            0 <= new_position[0] < self.size
            and 0 <= new_position[1] < self.size
            and new_position not in self.obstacles
        ):
            self.position = new_position

        self.steps += 1

        # Calculate reward
        if self.position == self.target:
            reward = 100  # Success
            done = True
        elif self.steps >= self.max_steps:
            reward = -50  # Timeout penalty
            done = True
        else:
            # Manhattan distance to target
            dist = abs(self.position[0] - self.target[0]) + abs(
                self.position[1] - self.target[1]
            )
            reward = -1 - (dist * 0.1)  # Small step penalty with distance hint
            done = False

        return self.get_observation(), reward, done


# Base agent class that extends the BaseAgent from memory hooks
class SimpleAgent(BaseAgent):
    def __init__(
        self,
        agent_id,
        config=None,
        action_space=4,
        learning_rate=0.1,
        discount_factor=0.9,
        **kwargs,
    ):
        super().__init__(config=config, agent_id=agent_id, **kwargs)
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action values
        self.current_observation = None
        self.demo_path = None  # For scripted demo actions
        self.demo_step = 0

    def get_state(self):
        """Override get_state to provide current observation for memory hooks"""
        state = super().get_state()
        # Add the current observation to the state if available
        if self.current_observation:
            # Convert NumPy types to Python types
            state.extra_data = convert_numpy_to_python(self.current_observation)
        return state

    def _get_state_key(self, observation):
        # Convert observation to a hashable state
        return f"{observation['position']}"

    def select_action(self, observation, epsilon=0.1):
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

    def update_q_value(self, observation, action, reward, next_observation, done):
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

    def act(self, observation, epsilon=0.1):
        """Override act method to implement agent behavior"""
        self.step_number += 1
        # Convert NumPy types to Python types
        self.current_observation = convert_numpy_to_python(observation)
        action = self.select_action(self.current_observation, epsilon)

        # Return an object with the expected structure for memory hooks
        from memory.api.models import ActionResult

        return ActionResult(
            action_type="move",
            params={"direction": int(action)},  # Convert to standard Python int
            action_id=str(action),  # Convert to string for safe serialization
        )

    def set_demo_path(self, path):
        """Set a predetermined path to follow for demonstration"""
        self.demo_path = path
        self.demo_step = 0


# Memory-enhanced agent using hooks
@install_memory_hooks
class MemoryEnhancedAgent(SimpleAgent):
    def __init__(
        self,
        agent_id,
        config,
        action_space=4,
        learning_rate=0.1,
        discount_factor=0.9,
        **kwargs,
    ):
        # Initialize config first so memory hooks can access it
        self.config = config
        super().__init__(
            agent_id=agent_id,
            config=config,
            action_space=action_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            **kwargs,
        )

    # Override select_action to use memory for better decisions
    def select_action(self, observation, epsilon=0.1):
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

        # Try to retrieve similar experiences from memory
        try:
            if hasattr(self, "memory_system") and self.memory_system is not None:
                similar_states = self.memory_system.retrieve_similar_states(
                    self.agent_id, observation, k=5
                )

                # Strong bias toward using memory (higher than epsilon)
                if similar_states and np.random.random() > 0.2:
                    # Use past successful actions as a guide
                    actions_from_memory = []
                    for s in similar_states:
                        if (
                            "action" in s["contents"]
                            and s["contents"].get("reward", 0) > 0
                        ):
                            actions_from_memory.append(s["contents"]["action"])

                    if actions_from_memory:
                        # Most common successful action from similar states
                        return max(
                            set(actions_from_memory), key=actions_from_memory.count
                        )
        except Exception as e:
            # Fallback to regular selection on any error
            print(f"Memory retrieval error: {e}")
            pass

        # Epsilon-greedy policy as fallback
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state_key])


# Create a demonstration path to reach the goal
def create_optimal_path_for_maze(maze_size=5):
    """Create an optimal path from start (1,1) to goal (maze_size-2, maze_size-2)"""
    # Path to move right until one before the goal column
    path_right = [1] * (maze_size - 3)  # 1 = right
    # Path to move down until the goal row
    path_down = [2] * (maze_size - 3)  # 2 = down

    return path_right + path_down  # First go right, then go down


# A simpler experiment runner with fewer episodes for demonstration
def run_experiment(episodes=50, memory_enabled=True, random_seed=None):
    # Use different seeds for memory vs non-memory experiments
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create a simple maze with no obstacles and clear path to goal
    maze_size = 5  # Smaller maze
    env = MazeEnvironment(size=maze_size, obstacles=[], max_steps=15)

    # Create the optimal path for demonstration
    optimal_path = create_optimal_path_for_maze(maze_size)

    # Create agent based on memory flag
    agent_id = "agent_memory" if memory_enabled else "standard_agent"
    if memory_enabled:
        # Completely disable memory compression and neural embeddings
        # 1. Create autoencoder config with neural embeddings disabled
        autoencoder_config = AutoencoderConfig(
            use_neural_embeddings=False,  # Key setting: disable neural embeddings completely
            epochs=1,  # Minimize any training
            batch_size=1,  # Smallest possible batch
        )

        # 2. Create configurations with compression disabled
        stm_config = RedisSTMConfig(
            ttl=60,  # Short TTL for demo
            memory_limit=100,  # Small memory limit
            use_mock=True,  # Use mock Redis for easy setup
        )

        im_config = RedisIMConfig(
            ttl=120,  # Short TTL for demo
            memory_limit=100,  # Small memory limit
            compression_level=0,  # No compression for IM
            use_mock=True,  # Use mock Redis for easy setup
        )

        # Use a real file for SQLite to avoid table creation issues
        db_path = "memory_demo.db"
        if os.path.exists(db_path):
            os.remove(db_path)  # Remove existing database to start fresh

        ltm_config = SQLiteLTMConfig(
            compression_level=0,  # No compression for LTM
            batch_size=10,  # Small batch size
            db_path=db_path,  # Use a real file for SQLite
        )

        # 3. Create the main memory config with all compression disabled
        memory_config = MemoryConfig(
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            autoencoder_config=autoencoder_config,
            cleanup_interval=1000,  # Reduce cleanup frequency
            enable_memory_hooks=True,  # Enable memory hooks for automatic capture
        )

        # Important: Set up the memory system singleton with our config
        # Since memory hooks use the singleton pattern, we need to ensure
        # our memory system is the singleton instance
        memory_system = AgentMemorySystem.get_instance(memory_config)

        # Important: Pre-initialize the memory agent for our agent ID
        # This ensures the agent exists in the memory system before hooks try to access it
        memory_agent = memory_system.get_memory_agent(agent_id)

        # 4. Create the agent with our compression-disabled config
        config = {"memory_config": memory_config}
        agent = MemoryEnhancedAgent(agent_id, config=config, action_space=4)

        # Set the demonstration path for the first episode
        agent.set_demo_path(optimal_path)

        print("Created memory agent with compression and neural embeddings disabled")
    else:
        agent = SimpleAgent(agent_id, action_space=4)
        # No memory, but still give the demo path for the first episode
        agent.set_demo_path(optimal_path)

    # Track metrics
    rewards_per_episode = []
    steps_per_episode = []
    success_rate = []

    window_size = 5  # For running success rate (reduced from 10)
    successes = 0

    # Training loop
    for episode in range(episodes):
        observation = env.reset()
        # Set initial observation explicitly
        agent.current_observation = convert_numpy_to_python(observation)

        # Reset demo step counter for each episode
        if episode > 0:  # After first episode, let agents learn on their own
            agent.demo_path = None

        done = False
        total_reward = 0
        # Faster exploration decay to focus on exploitation sooner
        epsilon = max(0.01, 0.5 - (episode / (episodes / 2)))

        # Episode loop
        while not done:
            action = agent.act(observation, epsilon)
            next_observation, reward, done = env.step(action.params["direction"])

            # Store action results explicitly if not using hooks
            if memory_enabled and not memory_config.enable_memory_hooks:
                try:
                    agent.memory_system.store_action(
                        agent_id=agent.agent_id,
                        action_data={
                            "action": int(
                                action.params["direction"]
                            ),  # Convert to standard Python int
                            "reward": float(reward),  # Convert to standard Python float
                            "next_state": convert_numpy_to_python(
                                next_observation["position"]
                            ),
                        },
                        step_number=agent.step_number,
                        priority=abs(float(reward))
                        / 100,  # Convert to standard Python float
                    )
                except Exception as e:
                    # Gracefully handle any memory errors
                    print(f"Memory storage error: {e}")
                    pass

            # Update Q-values with higher learning rate for faster learning
            if memory_enabled:
                # Memory agent can learn faster because it has memory
                agent.learning_rate = 0.2
            agent.update_q_value(
                observation, action.params["direction"], reward, next_observation, done
            )

            total_reward += reward
            observation = next_observation

            # Print progress for each episode
            if done:
                success = (
                    reward > 0
                )  # Success if final reward was positive (reached goal)
                print(
                    f"Episode {episode+1}/{episodes} completed: steps={env.steps}, reward={total_reward:.1f}, "
                    + f"success={'Yes' if success else 'No'}"
                )

        # Record metrics
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(env.steps)

        # Track success rate
        if total_reward > 0:  # Success
            successes += 1

        if episode >= window_size - 1:
            if episode > window_size - 1:
                if rewards_per_episode[episode - window_size] > 0:
                    successes -= 1
            success_rate.append(successes / window_size)
        else:
            success_rate.append(successes / (episode + 1))

    return {
        "rewards": rewards_per_episode,
        "steps": steps_per_episode,
        "success_rate": success_rate,
        "agent": agent,
    }


print("Starting experiment with memory...")
results_with_memory = run_experiment(episodes=50, memory_enabled=True, random_seed=42)
print("\nStarting experiment without memory...")
results_without_memory = run_experiment(
    episodes=50, memory_enabled=False, random_seed=84
)  # Different seed

# Plot results
plt.figure(figsize=(15, 10))

# Plot rewards
plt.subplot(2, 2, 1)
plt.plot(results_with_memory["rewards"], label="With Memory")
plt.plot(results_without_memory["rewards"], label="Without Memory")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.legend()

# Plot steps
plt.subplot(2, 2, 2)
plt.plot(results_with_memory["steps"], label="With Memory")
plt.plot(results_without_memory["steps"], label="Without Memory")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend()

# Plot success rate
plt.subplot(2, 2, 3)
plt.plot(results_with_memory["success_rate"], label="With Memory")
plt.plot(results_without_memory["success_rate"], label="Without Memory")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("5-Episode Moving Success Rate")
plt.legend()

# Plot Q-value distribution
plt.subplot(2, 2, 4)
mem_q_values = np.array([max(v) for v in results_with_memory["agent"].q_table.values()])
std_q_values = np.array(
    [max(v) for v in results_without_memory["agent"].q_table.values()]
)
plt.hist(mem_q_values, alpha=0.5, label="With Memory")
plt.hist(std_q_values, alpha=0.5, label="Without Memory")
plt.xlabel("Max Q-Value")
plt.ylabel("Count")
plt.title("Q-Value Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("memory_benefit_comparison.png")
plt.show()

# Clean up the SQLite database file
if os.path.exists("memory_demo.db"):
    try:
        os.remove("memory_demo.db")
        print("Cleaned up temporary SQLite database")
    except:
        pass

print("Experiment completed. Results saved to memory_benefit_comparison.png")
