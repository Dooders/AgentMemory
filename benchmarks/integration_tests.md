# Integration Tests Benchmarks

These benchmarks evaluate how well the Agent Memory System integrates with existing agent architectures and measure the overhead of integration.

## API Overhead

### Benchmark Description
Measure the performance impact of integrating the memory system via its API interface.

### Methodology
1. Create baseline agent operations without memory integration
2. Implement the same operations with memory API integration
3. Measure performance differences in throughput, latency, and resource usage
4. Analyze the overhead introduced by the memory API

### Code Example
```python
from agent_memory import AgentMemorySystem, MemoryConfig
from agent_memory.api.memory_api import MemoryAPI
import time
import numpy as np
import random

def benchmark_api_overhead(num_operations=1000):
    # Create a memory system
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    memory_api = MemoryAPI(memory_system)
    
    # Prepare test data
    test_states = [generate_test_state() for _ in range(num_operations)]
    
    # Benchmark 1: Direct operations without using API
    direct_times = []
    start_time = time.time()
    
    for i, state in enumerate(test_states):
        # Store state directly
        op_start = time.time()
        memory_system.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"direct_state_{i}",
            state_data=state
        )
        direct_times.append(time.time() - op_start)
    
    direct_duration = time.time() - start_time
    
    # Benchmark 2: Operations using API
    api_times = []
    start_time = time.time()
    
    for i, state in enumerate(test_states):
        # Store state via API
        op_start = time.time()
        memory_api.store_agent_state(
            agent_id="benchmark_agent",
            state_id=f"api_state_{i}",
            state_data=state
        )
        api_times.append(time.time() - op_start)
    
    api_duration = time.time() - start_time
    
    # Calculate metrics
    direct_throughput = num_operations / direct_duration
    api_throughput = num_operations / api_duration
    
    throughput_overhead = (direct_throughput - api_throughput) / direct_throughput * 100
    latency_overhead = (np.mean(api_times) - np.mean(direct_times)) / np.mean(direct_times) * 100
    
    return {
        "direct": {
            "total_duration_seconds": direct_duration,
            "avg_operation_time_ms": np.mean(direct_times) * 1000,
            "p95_operation_time_ms": np.percentile(direct_times, 95) * 1000,
            "throughput_ops_per_second": direct_throughput
        },
        "api": {
            "total_duration_seconds": api_duration,
            "avg_operation_time_ms": np.mean(api_times) * 1000,
            "p95_operation_time_ms": np.percentile(api_times, 95) * 1000,
            "throughput_ops_per_second": api_throughput
        },
        "overhead": {
            "throughput_overhead_percent": throughput_overhead,
            "latency_overhead_percent": latency_overhead,
            "absolute_latency_diff_ms": (np.mean(api_times) - np.mean(direct_times)) * 1000
        }
    }
```

### Expected Metrics
- API operation latency (ms)
- Throughput overhead (%)
- Latency overhead (%)
- Memory overhead introduced by API layer

## Hook Implementation

### Benchmark Description
Evaluate the performance and resource impact of using memory hooks for agent integration.

### Methodology
1. Implement a test agent with and without memory hooks
2. Run standardized agent workloads
3. Measure the performance impact of memory hooks
4. Analyze how hooks affect agent operation

### Code Example
```python
from agent_memory import MemoryConfig
from agent_memory.api.hooks import install_memory_hooks, BaseAgent

class TestAgentWithoutHooks:
    def __init__(self):
        self.states = []
        self.actions = []
    
    def observe(self, observation):
        self.states.append(observation)
        return observation
    
    def act(self, observation):
        action = self.decide_action(observation)
        self.actions.append(action)
        return action
    
    def decide_action(self, observation):
        # Simplified action decision
        return {"move": random.choice(["north", "south", "east", "west"])}

@install_memory_hooks
class TestAgentWithHooks(BaseAgent):
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
    
    def observe(self, observation):
        result = super().observe(observation)
        self.states.append(observation)
        return result
    
    def act(self, observation):
        action = self.decide_action(observation)
        result = super().act(observation, action)
        self.actions.append(action)
        return result
    
    def decide_action(self, observation):
        # Simplified action decision
        return {"move": random.choice(["north", "south", "east", "west"])}

def benchmark_hook_implementation(num_steps=1000):
    # Setup memory system
    memory_config = MemoryConfig()
    
    # Create agents
    agent_without_hooks = TestAgentWithoutHooks()
    agent_with_hooks = TestAgentWithHooks(agent_id="test_agent", memory_config=memory_config)
    
    # Generate test observations
    test_observations = [generate_test_observation() for _ in range(num_steps)]
    
    # Run agent without hooks
    without_hooks_times = {
        "observe": [],
        "act": []
    }
    
    start_time = time.time()
    
    for observation in test_observations:
        # Observe
        observe_start = time.time()
        agent_without_hooks.observe(observation)
        without_hooks_times["observe"].append(time.time() - observe_start)
        
        # Act
        act_start = time.time()
        agent_without_hooks.act(observation)
        without_hooks_times["act"].append(time.time() - act_start)
    
    without_hooks_duration = time.time() - start_time
    
    # Run agent with hooks
    with_hooks_times = {
        "observe": [],
        "act": []
    }
    
    start_time = time.time()
    
    for observation in test_observations:
        # Observe
        observe_start = time.time()
        agent_with_hooks.observe(observation)
        with_hooks_times["observe"].append(time.time() - observe_start)
        
        # Act
        act_start = time.time()
        agent_with_hooks.act(observation)
        with_hooks_times["act"].append(time.time() - act_start)
    
    with_hooks_duration = time.time() - start_time
    
    # Calculate metrics
    without_hooks_throughput = num_steps / without_hooks_duration
    with_hooks_throughput = num_steps / with_hooks_duration
    
    throughput_overhead = (without_hooks_throughput - with_hooks_throughput) / without_hooks_throughput * 100
    
    observe_latency_overhead = (
        np.mean(with_hooks_times["observe"]) - np.mean(without_hooks_times["observe"])
    ) / np.mean(without_hooks_times["observe"]) * 100
    
    act_latency_overhead = (
        np.mean(with_hooks_times["act"]) - np.mean(without_hooks_times["act"])
    ) / np.mean(without_hooks_times["act"]) * 100
    
    return {
        "without_hooks": {
            "total_duration_seconds": without_hooks_duration,
            "throughput_ops_per_second": without_hooks_throughput,
            "avg_observe_time_ms": np.mean(without_hooks_times["observe"]) * 1000,
            "avg_act_time_ms": np.mean(without_hooks_times["act"]) * 1000,
            "p95_observe_time_ms": np.percentile(without_hooks_times["observe"], 95) * 1000,
            "p95_act_time_ms": np.percentile(without_hooks_times["act"], 95) * 1000
        },
        "with_hooks": {
            "total_duration_seconds": with_hooks_duration,
            "throughput_ops_per_second": with_hooks_throughput,
            "avg_observe_time_ms": np.mean(with_hooks_times["observe"]) * 1000,
            "avg_act_time_ms": np.mean(with_hooks_times["act"]) * 1000,
            "p95_observe_time_ms": np.percentile(with_hooks_times["observe"], 95) * 1000,
            "p95_act_time_ms": np.percentile(with_hooks_times["act"], 95) * 1000
        },
        "overhead": {
            "throughput_overhead_percent": throughput_overhead,
            "observe_latency_overhead_percent": observe_latency_overhead,
            "act_latency_overhead_percent": act_latency_overhead
        }
    }
```

### Expected Metrics
- Hook installation overhead (%)
- Per-operation latency impact (ms)
- Memory usage increase with hooks
- CPU utilization increase with hooks

## Real-world Agent Integration

### Benchmark Description
Evaluate integration with popular agent frameworks and measure the performance impact.

### Methodology
1. Integrate with standard agent frameworks (e.g., RLlib, Stable Baselines)
2. Run standardized agent training and evaluation scenarios
3. Measure performance impact of memory integration
4. Compare memory access patterns across frameworks

### Code Example
```python
import gym
import time
import numpy as np
from agent_memory import AgentMemorySystem, MemoryConfig
from agent_memory.api.hooks import integrate_with_rllib  # Hypothetical integration

def benchmark_rllib_integration(env_name="CartPole-v1", training_iterations=10):
    # Setup memory system
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    # Import RLlib (in actual implementation)
    # import ray
    # from ray import tune
    # from ray.rllib.agents import ppo
    
    # Define test environment
    env = gym.make(env_name)
    
    # Setup standard agent without memory integration
    agent_config = {
        "env": env_name,
        "num_workers": 1,
        "train_batch_size": 1000
    }
    
    # This would be actual RLlib agent in real implementation
    # standard_agent = ppo.PPOTrainer(config=agent_config)
    standard_agent = MockRLlibAgent(env, agent_config)
    
    # Train standard agent
    standard_times = []
    standard_rewards = []
    
    for i in range(training_iterations):
        start_time = time.time()
        result = standard_agent.train()
        standard_times.append(time.time() - start_time)
        standard_rewards.append(result["episode_reward_mean"])
    
    # Setup agent with memory integration
    memory_agent_config = agent_config.copy()
    
    # This would be actual integration in real implementation
    # memory_agent = integrate_with_rllib(ppo.PPOTrainer, memory_system, config=memory_agent_config)
    memory_agent = MockRLlibAgentWithMemory(env, memory_agent_config, memory_system)
    
    # Train memory-integrated agent
    memory_times = []
    memory_rewards = []
    
    for i in range(training_iterations):
        start_time = time.time()
        result = memory_agent.train()
        memory_times.append(time.time() - start_time)
        memory_rewards.append(result["episode_reward_mean"])
    
    # Evaluate memory access patterns
    memory_access_counts = memory_system.get_access_statistics()
    
    # Calculate metrics
    avg_standard_time = np.mean(standard_times)
    avg_memory_time = np.mean(memory_times)
    time_overhead = (avg_memory_time - avg_standard_time) / avg_standard_time * 100
    
    return {
        "standard_agent": {
            "avg_iteration_time_seconds": avg_standard_time,
            "total_training_time_seconds": sum(standard_times),
            "reward_progress": standard_rewards
        },
        "memory_agent": {
            "avg_iteration_time_seconds": avg_memory_time,
            "total_training_time_seconds": sum(memory_times),
            "reward_progress": memory_rewards
        },
        "overhead": {
            "time_overhead_percent": time_overhead,
            "memory_accesses_per_iteration": memory_access_counts["total"] / training_iterations,
            "stm_accesses": memory_access_counts["stm"],
            "im_accesses": memory_access_counts["im"],
            "ltm_accesses": memory_access_counts["ltm"]
        },
        "memory_access_patterns": {
            "read_write_ratio": memory_access_counts["reads"] / memory_access_counts["writes"],
            "access_by_operation_type": memory_access_counts["by_operation"]
        }
    }

# Mock classes for example purposes
class MockRLlibAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.iteration = 0
    
    def train(self):
        # Simulate training
        time.sleep(0.1)  # Simulate computation
        self.iteration += 1
        return {
            "episode_reward_mean": 100 + self.iteration * 10,
            "episodes_total": self.iteration * 10
        }

class MockRLlibAgentWithMemory(MockRLlibAgent):
    def __init__(self, env, config, memory_system):
        super().__init__(env, config)
        self.memory_system = memory_system
    
    def train(self):
        # Simulate training with memory access
        result = super().train()
        
        # Simulate memory operations during training
        for i in range(10):
            state = {"observation": np.random.random(4), "iteration": self.iteration}
            self.memory_system.store_agent_state(
                agent_id="rllib_agent",
                state_id=f"state_{self.iteration}_{i}",
                state_data=state
            )
        
        return result
```

### Expected Metrics
- Training iteration time overhead (%)
- Memory access patterns during training
- Impact on agent performance (reward)
- Tier utilization during agent operation 