import argparse
import os.path
import random
import time
import numpy as np
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Import the memory system
from memory.config import MemoryConfig, RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig, AutoencoderConfig
from memory.core import AgentMemorySystem

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MODEL_PATH = "cartpole_model.pth"
AGENT_ID = "cartpole_agent"

# Customize memory config
def get_memory_config():
    # Create memory system configuration with mock Redis (no real Redis needed)
    stm_config = RedisSTMConfig(
        use_mock=True,
        ttl=3600,  # 1 hour TTL
        memory_limit=5000,  # Increased memory limit
        namespace="cartpole:stm"  # Custom namespace
    )
    
    im_config = RedisIMConfig(
        use_mock=True,
        ttl=86400,  # 1 day TTL
        memory_limit=20000,  # Increased memory limit
        namespace="cartpole:im",  # Custom namespace
        compression_level=1  # Low compression for speed
    )
    
    ltm_config = SQLiteLTMConfig(
        db_path="cartpole_memory.db",
        compression_level=1,  # Low compression for speed
        table_prefix="cartpole_memory"  # Custom table prefix
    )
    
    # Configure autoencoder for state dimensions in CartPole (4 dimensions)
    autoencoder_config = AutoencoderConfig(
        input_dim=4,  # CartPole has 4 state dimensions
        stm_dim=16,   # Reduced dimensions for embeddings
        im_dim=8,
        ltm_dim=4,
        use_neural_embeddings=False  # No neural embeddings for this simple example
    )
    
    # Create the full config
    return MemoryConfig(
        stm_config=stm_config,
        im_config=im_config,
        ltm_config=ltm_config,
        autoencoder_config=autoencoder_config,
        cleanup_interval=20,  # More frequent cleanup
        memory_priority_decay=0.8,  # Faster priority decay for transitions
        logging_level="INFO"
    )

# DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.fc(x)

# Memory adapter to bridge between AgentMemorySystem and DQN requirements
class MemoryAdapter:
    def __init__(self, max_size=MEMORY_SIZE):
        self.memory_config = get_memory_config()
        # More frequent cleanup for better tier transitions
        self.memory_config.cleanup_interval = 10  # Reduced from 50
        self.memory_system = AgentMemorySystem.get_instance(self.memory_config)
        self.agent_id = AGENT_ID
        self.max_size = max_size
        self.current_step = 0
        
        # Cache for recent experiences to improve sampling performance
        self.recent_cache = []
        self.cache_size = min(2000, max_size // 2)  # Keep about half in fast cache
        
        # Statistics for debug
        self.priority_sum = 0
        self.transitions_count = 0
        
    def append(self, state, action, reward, next_state, done):
        # Create memory entry
        memory_data = {
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": int(action),
            "reward": float(reward),
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            "done": bool(done),
        }
        
        # Calculate priority based on reward and episode completion
        # Simplified priority calculation for better performance
        # Give higher priority to terminal states and states with non-zero rewards
        # For CartPole, the goal is to keep the pole upright, so falls are important learning opportunities
        priority = 0.5  # Base priority
        if done:
            priority = 0.9  # Terminal states are important
        elif abs(reward) > 0.1:
            priority = 0.7  # Significant reward states
            
        self.priority_sum += priority
        self.transitions_count += 1
        
        # Add to local cache for fast retrieval
        self.recent_cache.append((memory_data, priority))
        if len(self.recent_cache) > self.cache_size:
            self.recent_cache.pop(0)  # Remove oldest item when cache is full
        
        # Store the experience in memory system
        self.memory_system.store_agent_state(self.agent_id, memory_data, self.current_step, priority)
        
        # Debug: Print the memory structure for first few entries
        if self.current_step < 2:
            # Retrieve the memory we just stored to see its structure
            recent_memory = self.memory_system.retrieve_by_time_range(
                self.agent_id, 
                start_step=self.current_step,
                end_step=self.current_step
            )
            if recent_memory:
                print(f"DEBUG - Stored memory structure at step {self.current_step}:")
                print(f"Memory keys: {recent_memory[0].keys() if recent_memory else 'No memory'}")
                if "data" in recent_memory[0]:
                    print(f"Data keys: {recent_memory[0]['data'].keys() if 'data' in recent_memory[0] else 'No data'}")
        
        # Force memory maintenance more aggressively
        if self.current_step % 100 == 0:
            self.memory_system.force_memory_maintenance(self.agent_id)
            
        self.current_step += 1
    
    def sample(self, batch_size):
        # First try to use the cache for faster sampling
        if len(self.recent_cache) >= batch_size:
            # Sample from cache with 80% probability for speed
            if random.random() < 0.8 or self.current_step < 200:
                # Extract a random batch from cache
                sampled_items = random.sample(self.recent_cache, batch_size)
                # Unpack cached items
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                
                for memory_item, _ in sampled_items:
                    states.append(memory_item["state"])
                    actions.append(memory_item["action"])
                    rewards.append(memory_item["reward"])
                    next_states.append(memory_item["next_state"])
                    dones.append(memory_item["done"])
                
                return states, actions, rewards, next_states, dones
        
        # If we're here, either cache is insufficient or we're doing a deeper retrieval
        # Get memories from the memory system, focusing on recent ones for better performance
        recent_step_count = min(2000, self.current_step)  # Look at most recent 2000 steps
        step_threshold = max(0, self.current_step - recent_step_count)
        
        # Get memories prioritizing the higher tiers for better performance
        try:
            # First try STM for fastest access
            stm_memories = self.memory_system.retrieve_by_time_range(
                self.agent_id, 
                start_step=step_threshold, 
                end_step=self.current_step,
                memory_type="stm"  # Specify STM tier
            )
            
            # If we don't have enough, try IM
            if len(stm_memories) < batch_size:
                im_memories = self.memory_system.retrieve_by_time_range(
                    self.agent_id, 
                    start_step=step_threshold, 
                    end_step=self.current_step,
                    memory_type="im"  # Specify IM tier
                )
                memories = stm_memories + im_memories
            else:
                memories = stm_memories
            
            # If still not enough, try all tiers
            if len(memories) < batch_size:
                all_memories = self.memory_system.retrieve_by_time_range(
                    self.agent_id, 
                    start_step=step_threshold, 
                    end_step=self.current_step
                )
                memories = all_memories
        except Exception as e:
            # Fallback to all memories if tier-based retrieval isn't working
            print(f"Memory tier retrieval failed: {e}, falling back to general retrieval")
            memories = self.memory_system.retrieve_by_time_range(
                self.agent_id, 
                start_step=step_threshold, 
                end_step=self.current_step
            )
        
        # If we don't have enough memories, return None
        if len(memories) < batch_size:
            return None
        
        # Sample from these memories
        sampled_memories = random.sample(memories, batch_size)
        
        # Extract the SARSD tuples
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for memory in sampled_memories:
            # Try different possible data structures
            if "data" in memory:
                data = memory["data"]
            else:
                # Memory might already be the data dictionary itself
                data = memory
                
            # Extract fields with better error handling
            if data:
                try:
                    state_value = data.get("state", [0, 0, 0, 0])
                    action_value = data.get("action", 0)
                    reward_value = data.get("reward", 0.0)
                    next_state_value = data.get("next_state", [0, 0, 0, 0])
                    done_value = data.get("done", False)
                    
                    # Ensure state is a list with correct dimensions
                    if state_value is None:
                        state_value = [0, 0, 0, 0]
                    if next_state_value is None:
                        next_state_value = [0, 0, 0, 0]
                    
                    states.append(state_value)
                    actions.append(action_value)
                    rewards.append(reward_value)
                    next_states.append(next_state_value)
                    dones.append(done_value)
                except Exception as e:
                    if self.current_step <= 60:
                        print(f"Error extracting memory data: {e}")
                        print(f"Memory data: {data}")
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        # Use cache length for fast size check
        return len(self.recent_cache)
    
    def get_stats(self):
        """Get statistics about the memory adapter"""
        avg_priority = self.priority_sum / max(1, self.transitions_count)
        
        return {
            "cache_size": len(self.recent_cache),
            "current_step": self.current_step,
            "avg_priority": avg_priority
        }

# Environment setup
env = None
model = None
target_model = None
optimizer = None
criterion = None
memory = None

def select_action(state, epsilon, model, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return q_values.argmax().item()

def train_step(model, target_model, optimizer, criterion, memory):
    batch = memory.sample(BATCH_SIZE)
    if batch is None:
        return  # Not enough samples yet
        
    states, actions, rewards, next_states, dones = batch
    
    # Check for None values or invalid data
    if not states or None in states or None in next_states:
        return  # Skip this batch
    
    try:
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute Q values and targets more efficiently
        with torch.no_grad():
            next_q_values = target_model(next_states).max(1, keepdim=True)[0]
            target = rewards + (GAMMA * next_q_values * (1 - dones))
        
        # Current Q values
        current_q = model(states).gather(1, actions)
        
        # Compute loss
        loss = criterion(current_q, target)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        return loss.item()
    except Exception as e:
        # Handle any unexpected errors during training
        print(f"Error in training step: {e}")
        return None

# Training and visualization function definitions
def train():
    # Environment setup
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Initialize memory with AgentMemorySystem adapter
    memory = MemoryAdapter(max_size=MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    # Collect initial experiences before training
    print("Collecting initial experiences...")
    initial_experiences = 500  # Increased from 100 to better populate memory tiers
    state, _ = env.reset()
    
    for step in range(initial_experiences):
        action = env.action_space.sample()  # Use random actions for exploration
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append(state, action, reward, next_state, done or truncated)
        
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
        
        if step % 100 == 0:
            print(f"Collecting experience: {step}/{initial_experiences}")
    
    # Force memory maintenance to establish memory tiers
    print("Forcing memory maintenance for initial experiences...")
    memory_system = AgentMemorySystem.get_instance()
    memory_system.force_memory_maintenance(AGENT_ID)
    
    print(f"Collected {initial_experiences} experiences, starting training...")

    # Print memory stats after initial experience collection
    memory_stats = memory.get_stats()
    print(f"Memory adapter stats: {memory_stats}")

    # Training loop
    best_reward = 0
    no_improvement_count = 0
    rewards_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0

        for t in range(500):
            # More efficient action selection
            action = select_action(state, epsilon, model, env)
            next_state, reward, done, truncated, _ = env.step(action)
            memory.append(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            episode_steps += 1

            # Train more efficiently - every step in early episodes, less frequently later
            train_freq = 1 if episode < 20 else (5 if episode < 50 else 10)
            if t % train_freq == 0:
                train_step(model, target_model, optimizer, criterion, memory)

            if done or truncated:
                break

        # Decay epsilon with a faster schedule for CartPole
        epsilon = max(EPSILON_END, epsilon * (EPSILON_DECAY if episode < 100 else 0.99))

        # Update target network more frequently in early training
        target_update_freq = 2 if episode < 20 else (5 if episode < 50 else 10)
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Keep track of reward history
        rewards_history.append(total_reward)
        if len(rewards_history) > 10:
            rewards_history.pop(0)
        
        # Track best performance and detect convergence
        avg_reward = sum(rewards_history) / len(rewards_history)
        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Print progress with more detail
        print(f"Episode {episode}, Steps: {episode_steps}, Reward: {total_reward}, Avg(10): {avg_reward:.1f}, Epsilon: {epsilon:.3f}")

        # Perform memory maintenance occasionally
        if episode % 10 == 0:
            memory_system = AgentMemorySystem.get_instance()
            stats = memory_system.get_memory_statistics(AGENT_ID)
            print(f"Memory tiers: STM={stats.get('stm_count', 0)}, IM={stats.get('im_count', 0)}, LTM={stats.get('ltm_count', 0)}")
            memory_system.force_memory_maintenance(AGENT_ID)
        
        # Early stopping if performance plateaus
        if no_improvement_count >= 30 and episode > 100 and avg_reward > 400:
            print(f"Early stopping at episode {episode} - no improvement for 30 episodes with good performance")
            break

    env.close()

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, state_size, action_size

def visualize_model(num_episodes=5, model=None, state_size=None, action_size=None):
    """
    Visualize the performance of the trained model
    """
    print("\nVisualizing trained model performance...")

    if model is None:
        # Environment setup just to get dimensions
        temp_env = gym.make("CartPole-v1")
        state_size = temp_env.observation_space.shape[0]
        action_size = temp_env.action_space.n
        temp_env.close()

        # Load the model
        model = DQN(state_size, action_size)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    for episode in range(num_episodes):
        # Render mode needs to be set to 'human' for visualization
        test_env = gym.make("CartPole-v1", render_mode="human")
        state, _ = test_env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Visualization Episode {episode+1}, Total Reward: {total_reward}")
        test_env.close()

def analyze_memory():
    """
    Analyze what's stored in the memory system
    """
    memory_system = AgentMemorySystem.get_instance()
    
    # Get memory statistics
    stats = memory_system.get_memory_statistics(AGENT_ID)
    print("\nMemory System Statistics:")
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"STM entries: {stats.get('stm_count', 0)}")
    print(f"IM entries: {stats.get('im_count', 0)}")
    print(f"LTM entries: {stats.get('ltm_count', 0)}")
    
    # Analyze reward distribution
    memories = memory_system.retrieve_by_time_range(
        AGENT_ID, 
        start_step=0,
        end_step=999999
    )
    
    if memories:
        rewards = [memory.get('data', {}).get('reward', 0) for memory in memories]
        print(f"\nReward analysis:")
        print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
        print(f"Max reward: {max(rewards):.2f}")
        print(f"Min reward: {min(rewards):.2f}")
    
    # Get high-priority memories
    high_priority_memories = [mem for mem in memories if mem.get('priority', 0) > 0.7]
    print(f"\nNumber of high-priority memories: {len(high_priority_memories)}")
    
    return stats

def reset_memory_system():
    """Reset the memory system for a clean start"""
    print("Resetting memory system...")
    try:
        # Try to delete the SQLite database file
        if os.path.exists("cartpole_memory.db"):
            os.remove("cartpole_memory.db")
            print("Removed existing memory database file")
        
        # Reset the singleton instance of the memory system
        AgentMemorySystem._instance = None
        
        # Create a new instance with fresh config
        memory_config = get_memory_config()
        memory_system = AgentMemorySystem.get_instance(memory_config)
        print("Memory system reset complete")
        
        return memory_system
    except Exception as e:
        print(f"Error resetting memory system: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole DQN with Memory System")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the trained model"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze the memory system"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the memory system before training"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to visualize"
    )

    args = parser.parse_args()

    # Default behavior: if no args are provided, do both train and visualize
    if not args.train and not args.visualize and not args.analyze:
        args.train = True
        args.visualize = True
        args.analyze = True
    
    # Always reset unless specified not to
    if args.reset or args.train:
        reset_memory_system()

    if args.train:
        model, state_size, action_size = train()
        if args.visualize:
            visualize_model(args.episodes, model, state_size, action_size)
    elif args.visualize:
        visualize_model(args.episodes)
    
    if args.analyze:
        analyze_memory() 