import numpy as np
import matplotlib.pyplot as plt
from memory import AgentMemorySystem, MemoryConfig, RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig
from memory.config import AutoencoderConfig
from memory.api.hooks import install_memory_hooks

# Define a simple maze environment
class MazeEnvironment:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.obstacles = obstacles or []
        self.target = (size-2, size-2)
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
            "steps": self.steps
        }
    
    def _get_nearby_obstacles(self):
        return [obs for obs in self.obstacles 
                if abs(obs[0] - self.position[0]) <= 2 and 
                   abs(obs[1] - self.position[1]) <= 2]
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_position = (self.position[0] + directions[action][0], 
                        self.position[1] + directions[action][1])
        
        # Check if valid move
        if (0 <= new_position[0] < self.size and 
            0 <= new_position[1] < self.size and
            new_position not in self.obstacles):
            self.position = new_position
        
        self.steps += 1
        
        # Calculate reward
        if self.position == self.target:
            reward = 100  # Success
            done = True
        elif self.steps >= self.size * 3:
            reward = -50  # Timeout penalty
            done = True
        else:
            # Distance-based reward to encourage progress
            prev_dist = abs(self.position[0] - self.target[0]) + abs(self.position[1] - self.target[1])
            reward = -1  # Small step penalty to encourage efficiency
            done = False
            
        return self.get_observation(), reward, done

# Base agent class
class BaseAgent:
    def __init__(self, agent_id, action_space=4, learning_rate=0.1, discount_factor=0.9):
        self.agent_id = agent_id
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action values
        self.step_number = 0
        self.current_observation = None  # Initialize this to avoid errors
        
    def get_state(self):
        return self.current_observation
        
    def _get_state_key(self, observation):
        # Convert observation to a hashable state
        return f"{observation['position']}"
    
    def select_action(self, observation, epsilon=0.1):
        self.current_observation = observation  # Store for memory hooks
        state_key = self._get_state_key(observation)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
            
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
            
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
    def act(self, observation, epsilon=0.1):
        self.step_number += 1
        action = self.select_action(observation, epsilon)
        return action

# Memory-enhanced agent using hooks
@install_memory_hooks
class MemoryAgent(BaseAgent):
    def __init__(self, agent_id, action_space=4, learning_rate=0.1, discount_factor=0.9):
        # Initialize with a config dict to satisfy memory hooks
        self.config = {"agent_id": agent_id, "memory_config": MemoryConfig()}
        # Initialize current_observation before super().__init__ to ensure it's available for hooks
        self.current_observation = {"position": (1, 1), "steps": 0}  # Default initial observation
        super().__init__(agent_id, action_space, learning_rate, discount_factor)
        # Memory hooks automatically capture agent states and actions

    # Use past similar states to make better decisions
    def select_action(self, observation, epsilon=0.1):
        self.current_observation = observation
        state_key = self._get_state_key(observation)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        
        # Retrieve similar states from memory
        try:
            similar_states = self.memory_system.retrieve_similar_states(
                self.agent_id, 
                observation,
                k=5
            )
            
            if similar_states and np.random.random() > epsilon:
                # Use past successful actions as a guide
                actions_from_memory = []
                for s in similar_states:
                    if 'next_action' in s['contents'] and s['contents'].get('reward', 0) > 0:
                        actions_from_memory.append(s['contents']['next_action'])
                
                if actions_from_memory:
                    # Most common successful action from similar states
                    return max(set(actions_from_memory), key=actions_from_memory.count)
        except Exception as e:
            # Fallback to regular selection on any error
            print(f"Memory retrieval error: {e}")
            pass
            
        # Epsilon-greedy policy as fallback
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state_key])

# A simpler experiment runner with fewer episodes for demonstration
def run_experiment(episodes=20, memory_enabled=True):
    # Create maze with random obstacles
    np.random.seed(42)  # For reproducibility
    size = 10
    obstacles = [(np.random.randint(1, size-1), np.random.randint(1, size-1)) 
                 for _ in range(size)]
    # Remove obstacles at start and goal
    if (1, 1) in obstacles:
        obstacles.remove((1, 1))
    if (size-2, size-2) in obstacles:
        obstacles.remove((size-2, size-2))
    
    env = MazeEnvironment(size=size, obstacles=obstacles)
    
    # Create agent based on memory flag
    agent_id = "memory_agent" if memory_enabled else "standard_agent"
    if memory_enabled:
        # Completely disable memory compression and neural embeddings
        # 1. Create autoencoder config with neural embeddings disabled
        autoencoder_config = AutoencoderConfig(
            use_neural_embeddings=False,  # Key setting: disable neural embeddings completely
            epochs=1,                     # Minimize any training
            batch_size=1                  # Smallest possible batch
        )
        
        # 2. Create configurations with compression disabled
        stm_config = RedisSTMConfig(
            ttl=60,                       # Short TTL for demo
            memory_limit=100              # Small memory limit
        )
        
        im_config = RedisIMConfig(
            ttl=120,                      # Short TTL for demo
            memory_limit=100,             # Small memory limit
            compression_level=0           # No compression for IM
        )
        
        ltm_config = SQLiteLTMConfig(
            compression_level=0,          # No compression for LTM
            batch_size=10                 # Small batch size
        )
        
        # 3. Create the main memory config with all compression disabled
        memory_config = MemoryConfig(
            stm_config=stm_config,
            im_config=im_config,
            ltm_config=ltm_config,
            autoencoder_config=autoencoder_config,
            cleanup_interval=1000         # Reduce cleanup frequency
        )
        
        # 4. Create the agent with our compression-disabled config
        agent = MemoryAgent(agent_id, action_space=4)
        agent.config["memory_config"] = memory_config
        
        print("Created memory agent with compression and neural embeddings disabled")
    else:
        agent = BaseAgent(agent_id, action_space=4)
    
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
        agent.current_observation = observation
        
        done = False
        total_reward = 0
        epsilon = max(0.05, 1.0 - (episode / 10))  # Decay exploration faster
        
        # Episode loop
        while not done:
            action = agent.act(observation, epsilon)
            next_observation, reward, done = env.step(action)
            
            # Store action result in agent's memory
            if memory_enabled:
                try:
                    agent.memory_system.store_agent_action(
                        agent_id=agent.agent_id,
                        action_data={
                            "action": action,
                            "reward": reward,
                            "next_state": next_observation["position"]
                        },
                        step_number=agent.step_number,
                        priority=abs(reward)/100  # Higher priority for significant rewards
                    )
                except Exception as e:
                    # Gracefully handle any memory errors
                    print(f"Memory storage error: {e}")
                    pass
            
            # Update Q-values
            agent.update_q_value(observation, action, reward, next_observation, done)
            
            total_reward += reward
            observation = next_observation
            
            # Print progress for each episode
            if done:
                print(f"Episode {episode+1}/{episodes} completed: steps={env.steps}, reward={total_reward}")
        
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
        'rewards': rewards_per_episode,
        'steps': steps_per_episode,
        'success_rate': success_rate,
        'agent': agent
    }

print("Starting experiment with memory...")
results_with_memory = run_experiment(episodes=20, memory_enabled=True)
print("\nStarting experiment without memory...")
results_without_memory = run_experiment(episodes=20, memory_enabled=False)

# Plot results
plt.figure(figsize=(15, 10))

# Plot rewards
plt.subplot(2, 2, 1)
plt.plot(results_with_memory['rewards'], label='With Memory')
plt.plot(results_without_memory['rewards'], label='Without Memory')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.legend()

# Plot steps
plt.subplot(2, 2, 2)
plt.plot(results_with_memory['steps'], label='With Memory')
plt.plot(results_without_memory['steps'], label='Without Memory')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()

# Plot success rate
plt.subplot(2, 2, 3)
plt.plot(results_with_memory['success_rate'], label='With Memory')
plt.plot(results_without_memory['success_rate'], label='Without Memory')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('5-Episode Moving Success Rate')
plt.legend()

# Plot Q-value distribution
plt.subplot(2, 2, 4)
mem_q_values = np.array([max(v) for v in results_with_memory['agent'].q_table.values()])
std_q_values = np.array([max(v) for v in results_without_memory['agent'].q_table.values()])
plt.hist(mem_q_values, alpha=0.5, label='With Memory')
plt.hist(std_q_values, alpha=0.5, label='Without Memory')
plt.xlabel('Max Q-Value')
plt.ylabel('Count')
plt.title('Q-Value Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('memory_benefit_comparison.png')
plt.show()

print("Experiment completed. Results saved to memory_benefit_comparison.png")
