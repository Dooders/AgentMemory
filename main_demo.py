import os

import matplotlib.pyplot as plt
import numpy as np

from agents import MemoryAgent, SimpleAgent
from maze import MazeEnvironment
from memory.config import MemoryConfig, RedisIMConfig, RedisSTMConfig, SQLiteLTMConfig
from memory.core import AgentMemorySystem
from memory.utils.util import convert_numpy_to_python


# Create a demonstration path to reach the goal
def create_optimal_path_for_maze(maze_size=5):
    """Create an optimal path from start (1,1) to goal (maze_size-2, maze_size-2)"""
    # Path to move right until one before the goal column
    path_right = [1] * (maze_size - 3)  # 1 = right
    # Path to move down until the goal row
    path_down = [2] * (maze_size - 3)  # 2 = down

    return path_right + path_down  # First go right, then go down


# A simpler experiment runner with fewer episodes for demonstration
def run_experiment(episodes=100, memory_enabled=True, random_seed=None):
    # Use different seeds for memory vs non-memory experiments
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create a maze with obstacles
    maze_size = 20  # Smaller maze
    obstacles = [
        (3, 3),
        (3, 4),
        (3, 5),  # Horizontal wall
        (7, 7),
        (8, 7),
        (9, 7),  # Vertical wall
        (12, 12),
        (12, 13),
        (13, 12),  # L-shaped wall
        (15, 15),
        (16, 16),
        (17, 17),  # Diagonal wall
        (5, 10),
        (10, 5),
        (15, 10),  # Scattered obstacles
    ]
    env = MazeEnvironment(size=maze_size, obstacles=obstacles, max_steps=500)

    # Create the optimal path for demonstration
    #! Why do I need this?
    optimal_path = create_optimal_path_for_maze(maze_size)

    # Create agent based on memory flag
    agent_id = "agent_memory" if memory_enabled else "standard_agent"
    if memory_enabled:
        # Create configurations with compression disabled
        stm_config = RedisSTMConfig(
            ttl=120,  # Increase TTL to keep more memories active
            memory_limit=500,  # Increase memory limit
            use_mock=True,  # Use mock Redis for easy setup
        )

        im_config = RedisIMConfig(
            ttl=240,  # Longer TTL for IM
            memory_limit=1000,  # Larger memory limit
            compression_level=0,  # No compression for IM
            use_mock=True,  # Use mock Redis for easy setup
        )

        # Use a real file for SQLite to avoid table creation issues
        db_path = "memory_demo.db"
        if os.path.exists(db_path):
            os.remove(db_path)  # Remove existing database to start fresh

        ltm_config = SQLiteLTMConfig(
            compression_level=0,  # No compression for LTM
            batch_size=20,  # Larger batch size
            db_path=db_path,  # Use a real file for SQLite
        )

        # Create the main memory config with all compression disabled and text embedding engine
        memory_config = MemoryConfig(
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            cleanup_interval=1000,  # Reduce cleanup frequency
            enable_memory_hooks=False,  # Disable memory hooks since we're using direct API calls
            use_embedding_engine=True,  # Enable embedding engine for similarity search
            text_model_name="all-MiniLM-L6-v2",  # Use a default text embedding model
        )

        # Create the memory system
        #! Agent will have memory space, dont need the system
        memory_system = AgentMemorySystem.get_instance(memory_config)

        # Create the agent with memory system
        agent = MemoryAgent(agent_id, memory_system, action_space=4)

        # Set the demonstration path for the first episode
        agent.set_demo_path(optimal_path)

        print("Created memory agent with text embedding engine (no autoencoder)")
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
        # Modified exploration strategy - decay epsilon more slowly
        # to favor following memory early on
        epsilon = max(0.05, 0.5 - (episode / episodes))

        # Episode loop
        while not done:
            action = agent.act(observation, epsilon)
            next_observation, reward, done = env.step(action)

            # Update Q-values with higher learning rate for faster learning
            if memory_enabled:
                # Memory agent can learn faster because it has memory
                agent.learning_rate = 0.2
            agent.update_q_value(observation, action, reward, next_observation, done)

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


# Modify the main execution to include a debug run for examination
if __name__ == "__main__":
    # Run the regular experiment
    print("Starting experiment with memory...")
    results_with_memory = run_experiment(
        episodes=50, memory_enabled=True, random_seed=42
    )
    print("\nStarting experiment without memory...")
    results_without_memory = run_experiment(
        episodes=50, memory_enabled=False, random_seed=84
    )

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
    mem_q_values = np.array(
        [max(v) for v in results_with_memory["agent"].q_table.values()]
    )
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
