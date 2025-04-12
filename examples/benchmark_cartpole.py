import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
import pandas as pd

# Import both implementations
import cartpole
import cartpole_memory

def run_benchmark(num_episodes: int = 100, num_runs: int = 5, use_memory: bool = False) -> Dict[str, Any]:
    """
    Run benchmark for a specific implementation
    
    Args:
        num_episodes: Number of episodes to train for
        num_runs: Number of runs to average over
        use_memory: Whether to use the memory-based implementation
    
    Returns:
        Dictionary of benchmark results
    """
    rewards_per_run = []
    steps_per_run = []
    times_per_run = []
    losses_per_run = []
    
    for run in range(num_runs):
        print(f"Starting run {run+1}/{num_runs} {'with' if use_memory else 'without'} memory system")
        
        # Setup for this run
        rewards = []
        steps = []
        losses = []
        start_time = time.time()
        
        if use_memory:
            # Reset memory system for this run for a clean start
            cartpole_memory.reset_memory_system()
            
            # Reset memory system for this run
            memory_impl = cartpole_memory.MemoryAdapter(max_size=cartpole_memory.MEMORY_SIZE)
            
            # Create environment and model
            env = cartpole_memory.gym.make("CartPole-v1")
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            model = cartpole_memory.DQN(state_size, action_size)
            target_model = cartpole_memory.DQN(state_size, action_size)
            target_model.load_state_dict(model.state_dict())
            optimizer = cartpole_memory.optim.Adam(model.parameters(), lr=cartpole_memory.LR)
            criterion = cartpole_memory.nn.MSELoss()
            
            # Pre-fill memory with random experiences
            state, _ = env.reset()
            print("Collecting initial experiences...")
            for i in range(500):  # Increased from 100 to match memory implementation
                action = env.action_space.sample()
                next_state, reward, done, truncated, _ = env.step(action)
                memory_impl.append(state, action, reward, next_state, done or truncated)
                if done or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state
                if i % 100 == 0:
                    print(f"  Collected {i}/500 experiences")
            
            # Force memory maintenance before training
            memory_system = cartpole_memory.AgentMemorySystem.get_instance()
            memory_system.force_memory_maintenance(cartpole_memory.AGENT_ID)
            
            # Training loop
            epsilon = cartpole_memory.EPSILON_START
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                
                for t in range(500):  # Max steps per episode
                    action = cartpole_memory.select_action(state, epsilon, model, env)
                    next_state, reward, done, truncated, _ = env.step(action)
                    memory_impl.append(state, action, reward, next_state, done or truncated)
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    
                    # Train with optimized frequency
                    train_freq = 1 if episode < 20 else (5 if episode < 50 else 10)
                    if t % train_freq == 0:
                        loss = cartpole_memory.train_step(model, target_model, optimizer, criterion, memory_impl)
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    if done or truncated:
                        break
                
                # Decay epsilon with optimized schedule
                epsilon = max(cartpole_memory.EPSILON_END, epsilon * (cartpole_memory.EPSILON_DECAY if episode < 100 else 0.99))
                
                # Update target network with optimized frequency
                target_update_freq = 2 if episode < 20 else (5 if episode < 50 else 10)
                if episode % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                if episode_losses:
                    losses.append(sum(episode_losses) / len(episode_losses))
                else:
                    losses.append(0)
                
                # Perform memory maintenance occasionally
                if episode % 10 == 0:
                    memory_system = cartpole_memory.AgentMemorySystem.get_instance()
                    memory_system.force_memory_maintenance(cartpole_memory.AGENT_ID)
                
                # Verbose output every 10 episodes
                if episode % 10 == 0:
                    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
                    print(f"Episode {episode}, Reward: {episode_reward}, Steps: {episode_steps}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")
            
            env.close()
        
        else:
            # Original implementation
            # Setup deque memory
            memory = cartpole.deque(maxlen=cartpole.MEMORY_SIZE)
            
            # Create environment and model
            env = cartpole.gym.make("CartPole-v1")
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            model = cartpole.DQN(state_size, action_size)
            target_model = cartpole.DQN(state_size, action_size)
            target_model.load_state_dict(model.state_dict())
            optimizer = cartpole.optim.Adam(model.parameters(), lr=cartpole.LR)
            criterion = cartpole.nn.MSELoss()
            
            # Pre-fill memory with random experiences (for fairness)
            state, _ = env.reset()
            print("Collecting initial experiences...")
            for i in range(500):  # Match memory implementation
                action = env.action_space.sample()
                next_state, reward, done, truncated, _ = env.step(action)
                memory.append((state, action, reward, next_state, done or truncated))
                if done or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state
                if i % 100 == 0:
                    print(f"  Collected {i}/500 experiences")
            
            # Training loop
            epsilon = cartpole.EPSILON_START
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                
                for t in range(500):  # Max steps per episode
                    action = cartpole.select_action(state, epsilon, model, env)
                    next_state, reward, done, truncated, _ = env.step(action)
                    memory.append((state, action, reward, next_state, done or truncated))
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    
                    # Train using same schedule as memory implementation
                    train_freq = 1 if episode < 20 else (5 if episode < 50 else 10)
                    if t % train_freq == 0 and len(memory) >= cartpole.BATCH_SIZE:
                        batch = cartpole.random.sample(memory, cartpole.BATCH_SIZE)
                        states, actions, reward_batch, next_states, dones = zip(*batch)
                        
                        states = cartpole.torch.FloatTensor(states)
                        actions = cartpole.torch.LongTensor(actions).unsqueeze(1)
                        reward_batch = cartpole.torch.FloatTensor(reward_batch).unsqueeze(1)
                        next_states = cartpole.torch.FloatTensor(next_states)
                        dones = cartpole.torch.FloatTensor(dones).unsqueeze(1)
                        
                        q_values = model(states).gather(1, actions)
                        next_q_values = target_model(next_states).max(1, keepdim=True)[0].detach()
                        target = reward_batch + (cartpole.GAMMA * next_q_values * (1 - dones))
                        
                        loss = criterion(q_values, target)
                        optimizer.zero_grad()
                        loss.backward()
                        # Add same gradient clipping as memory implementation
                        cartpole.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        episode_losses.append(loss.item())
                    
                    if done or truncated:
                        break
                
                # Use same epsilon decay schedule
                epsilon = max(cartpole.EPSILON_END, epsilon * (cartpole.EPSILON_DECAY if episode < 100 else 0.99))
                
                # Use same target network update schedule
                target_update_freq = 2 if episode < 20 else (5 if episode < 50 else 10)
                if episode % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                if episode_losses:
                    losses.append(sum(episode_losses) / len(episode_losses))
                else:
                    losses.append(0)
                
                # Verbose output every 10 episodes
                if episode % 10 == 0:
                    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
                    print(f"Episode {episode}, Reward: {episode_reward}, Steps: {episode_steps}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")
            
            env.close()
        
        run_time = time.time() - start_time
        rewards_per_run.append(rewards)
        steps_per_run.append(steps)
        times_per_run.append(run_time)
        losses_per_run.append(losses)
        
        print(f"Run {run+1} completed in {run_time:.2f} seconds")
    
    # Calculate statistics across runs
    rewards_mean = np.mean(rewards_per_run, axis=0)
    rewards_std = np.std(rewards_per_run, axis=0)
    steps_mean = np.mean(steps_per_run, axis=0)
    steps_std = np.std(steps_per_run, axis=0)
    losses_mean = np.mean(losses_per_run, axis=0)
    losses_std = np.std(losses_per_run, axis=0)
    time_mean = np.mean(times_per_run)
    time_std = np.std(times_per_run)
    
    return {
        "rewards_mean": rewards_mean,
        "rewards_std": rewards_std,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "losses_mean": losses_mean,
        "losses_std": losses_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "all_rewards": rewards_per_run,
        "all_steps": steps_per_run,
        "all_times": times_per_run,
        "all_losses": losses_per_run
    }

def analyze_memory_system():
    """
    Analyze the memory system after benchmarking
    """
    print("\nAnalyzing memory system contents...")
    memory_system = cartpole_memory.AgentMemorySystem.get_instance()
    agent_id = cartpole_memory.AGENT_ID
    
    # Get memory statistics
    stats = memory_system.get_memory_statistics(agent_id)
    print("\nMemory System Statistics:")
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"STM entries: {stats.get('stm_count', 0)}")
    print(f"IM entries: {stats.get('im_count', 0)}")
    print(f"LTM entries: {stats.get('ltm_count', 0)}")
    
    # Retrieve all memories
    memories = memory_system.retrieve_by_time_range(
        agent_id, 
        start_step=0,
        end_step=999999
    )
    
    if memories:
        # Analyze memory priorities
        priorities = [mem.get('priority', 0) for mem in memories]
        plt.figure(figsize=(10, 6))
        plt.hist(priorities, bins=20, alpha=0.7)
        plt.title('Memory Priority Distribution')
        plt.xlabel('Priority')
        plt.ylabel('Count')
        plt.savefig("benchmark_results/memory_priorities.png")
        
        # Analyze rewards in memories
        rewards = []
        for memory in memories:
            data = memory.get('data', {})
            if data and 'reward' in data:
                rewards.append(data['reward'])
        
        if rewards:
            plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=20, alpha=0.7)
            plt.title('Reward Distribution in Memories')
            plt.xlabel('Reward')
            plt.ylabel('Count')
            plt.savefig("benchmark_results/memory_rewards.png")
            
            print(f"\nReward Statistics in Memory:")
            print(f"Average reward: {np.mean(rewards):.2f}")
            print(f"Std dev of rewards: {np.std(rewards):.2f}")
            print(f"Min reward: {min(rewards):.2f}")
            print(f"Max reward: {max(rewards):.2f}")
    
    return stats

def plot_results(results_standard: Dict[str, Any], results_memory: Dict[str, Any], num_episodes: int):
    """
    Plot comparison results
    
    Args:
        results_standard: Results from standard implementation
        results_memory: Results from memory implementation
        num_episodes: Number of episodes used in training
    """
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    
    episodes = range(num_episodes)
    plt.plot(episodes, results_standard["rewards_mean"], label="Standard", color="blue")
    plt.fill_between(
        episodes,
        results_standard["rewards_mean"] - results_standard["rewards_std"],
        results_standard["rewards_mean"] + results_standard["rewards_std"],
        alpha=0.2,
        color="blue"
    )
    
    plt.plot(episodes, results_memory["rewards_mean"], label="With Memory System", color="red")
    plt.fill_between(
        episodes,
        results_memory["rewards_mean"] - results_memory["rewards_std"],
        results_memory["rewards_mean"] + results_memory["rewards_std"],
        alpha=0.2,
        color="red"
    )
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("benchmark_results/rewards_comparison.png")
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    
    plt.plot(episodes, results_standard["steps_mean"], label="Standard", color="blue")
    plt.fill_between(
        episodes,
        results_standard["steps_mean"] - results_standard["steps_std"],
        results_standard["steps_mean"] + results_standard["steps_std"],
        alpha=0.2,
        color="blue"
    )
    
    plt.plot(episodes, results_memory["steps_mean"], label="With Memory System", color="red")
    plt.fill_between(
        episodes,
        results_memory["steps_mean"] - results_memory["steps_std"],
        results_memory["steps_mean"] + results_memory["steps_std"],
        alpha=0.2,
        color="red"
    )
    
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("benchmark_results/steps_comparison.png")
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    
    plt.plot(episodes, results_standard["losses_mean"], label="Standard", color="blue")
    plt.fill_between(
        episodes,
        results_standard["losses_mean"] - results_standard["losses_std"],
        results_standard["losses_mean"] + results_standard["losses_std"],
        alpha=0.2,
        color="blue"
    )
    
    plt.plot(episodes, results_memory["losses_mean"], label="With Memory System", color="red")
    plt.fill_between(
        episodes,
        results_memory["losses_mean"] - results_memory["losses_std"],
        results_memory["losses_mean"] + results_memory["losses_std"],
        alpha=0.2,
        color="red"
    )
    
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss per Episode")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("benchmark_results/loss_comparison.png")
    
    # Plot learning curves (smoothed rewards)
    plt.figure(figsize=(12, 6))
    
    # Apply smoothing
    window_size = max(1, num_episodes // 20)  # 5% of episodes
    
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    smoothed_standard = smooth(results_standard["rewards_mean"], window_size)
    smoothed_memory = smooth(results_memory["rewards_mean"], window_size)
    
    # Plot smoothed curves
    valid_episodes = range(window_size-1, num_episodes)
    plt.plot(valid_episodes, smoothed_standard, label="Standard (Smoothed)", color="blue")
    plt.plot(valid_episodes, smoothed_memory, label="With Memory System (Smoothed)", color="red")
    
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title(f"Learning Curves (Window Size: {window_size})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("benchmark_results/learning_curves.png")
    
    # Plot efficiency comparison (rewards per time)
    plt.figure(figsize=(12, 6))
    
    # Calculate time per episode (assuming linear distribution)
    standard_time_per_episode = results_standard['time_mean'] / num_episodes
    memory_time_per_episode = results_memory['time_mean'] / num_episodes
    
    # Scale rewards by time spent
    standard_scaled = results_standard["rewards_mean"] / standard_time_per_episode
    memory_scaled = results_memory["rewards_mean"] / memory_time_per_episode
    
    plt.plot(episodes, standard_scaled, label="Standard", color="blue")
    plt.plot(episodes, memory_scaled, label="With Memory System", color="red")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward / Time (s)")
    plt.title("Time Efficiency Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("benchmark_results/efficiency_comparison.png")
    
    # Create statistical summary table
    summary = {
        "Metric": [
            "Avg Training Time (s)", 
            "Final Avg Reward", 
            "Final Avg Steps",
            "Avg Reward (Last 10)",
            "Sample Efficiency (AUC)",
            "Final Loss",
            "Time Efficiency (Reward/s)",
            "Learning Speed (Ep to 200)"
        ],
        "Standard": [
            f"{results_standard['time_mean']:.2f} ± {results_standard['time_std']:.2f}",
            f"{results_standard['rewards_mean'][-1]:.2f} ± {results_standard['rewards_std'][-1]:.2f}",
            f"{results_standard['steps_mean'][-1]:.2f} ± {results_standard['steps_std'][-1]:.2f}",
            f"{np.mean(results_standard['rewards_mean'][-10:]):.2f}",
            f"{np.trapz(results_standard['rewards_mean']):.2f}",
            f"{results_standard['losses_mean'][-1]:.4f}",
            f"{results_standard['rewards_mean'][-1] / standard_time_per_episode:.2f}",
            f"{np.argmax(results_standard['rewards_mean'] >= 200) if any(results_standard['rewards_mean'] >= 200) else 'N/A'}"
        ],
        "Memory System": [
            f"{results_memory['time_mean']:.2f} ± {results_memory['time_std']:.2f}",
            f"{results_memory['rewards_mean'][-1]:.2f} ± {results_memory['rewards_std'][-1]:.2f}",
            f"{results_memory['steps_mean'][-1]:.2f} ± {results_memory['steps_std'][-1]:.2f}",
            f"{np.mean(results_memory['rewards_mean'][-10:]):.2f}",
            f"{np.trapz(results_memory['rewards_mean']):.2f}",
            f"{results_memory['losses_mean'][-1]:.4f}",
            f"{results_memory['rewards_mean'][-1] / memory_time_per_episode:.2f}",
            f"{np.argmax(results_memory['rewards_mean'] >= 200) if any(results_memory['rewards_mean'] >= 200) else 'N/A'}"
        ]
    }
    
    df = pd.DataFrame(summary)
    df.to_csv("benchmark_results/summary_stats.csv", index=False)
    
    # Print summary
    print("\nPerformance Summary:")
    print(df.to_string(index=False))
    
    # Calculate relative performance
    standard_auc = np.trapz(results_standard['rewards_mean'])
    memory_auc = np.trapz(results_memory['rewards_mean'])
    relative_improvement = (memory_auc - standard_auc) / standard_auc * 100
    
    standard_time = results_standard['time_mean']
    memory_time = results_memory['time_mean']
    time_overhead = (memory_time - standard_time) / standard_time * 100
    
    standard_final = np.mean(results_standard['rewards_mean'][-10:])
    memory_final = np.mean(results_memory['rewards_mean'][-10:])
    performance_diff = (memory_final - standard_final) / standard_final * 100
    
    print("\nRelative Performance:")
    print(f"Final performance difference: {performance_diff:.2f}%")
    print(f"Sample efficiency difference: {relative_improvement:.2f}%")
    print(f"Computational overhead: {time_overhead:.2f}%")
    
    # Save raw data for further analysis
    with open("benchmark_results/raw_data.txt", "w") as f:
        f.write("Standard Implementation:\n")
        f.write(f"Rewards: {results_standard['all_rewards']}\n")
        f.write(f"Steps: {results_standard['all_steps']}\n")
        f.write(f"Times: {results_standard['all_times']}\n")
        f.write(f"Losses: {results_standard['all_losses']}\n\n")
        
        f.write("Memory System Implementation:\n")
        f.write(f"Rewards: {results_memory['all_rewards']}\n")
        f.write(f"Steps: {results_memory['all_steps']}\n")
        f.write(f"Times: {results_memory['all_times']}\n")
        f.write(f"Losses: {results_memory['all_losses']}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark CartPole implementations")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train for")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average over")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze memory without running benchmarks")
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_memory_system()
        return
    
    # Run benchmarks
    print(f"Running benchmarks with {args.episodes} episodes and {args.runs} runs for each implementation")
    
    print("\n=== Standard Implementation ===")
    results_standard = run_benchmark(args.episodes, args.runs, use_memory=False)
    
    print("\n=== Memory System Implementation ===")
    results_memory = run_benchmark(args.episodes, args.runs, use_memory=True)
    
    # Plot and save results
    plot_results(results_standard, results_memory, args.episodes)
    
    # Analyze memory system
    analyze_memory_system()
    
    print("\nBenchmarking complete! Results saved to 'benchmark_results' directory.")

if __name__ == "__main__":
    main() 