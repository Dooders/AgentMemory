import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from agents import Agent
from memory.api.models import MazeActionSpace, MazeObservation
from memory.utils.util import convert_numpy_to_python
from memory.config.memory_config import MemoryConfig, RedisIMConfig, RedisSTMConfig, SQLiteLTMConfig
from memory.space import MemorySpace


def observation_to_tensor(obs: MazeObservation) -> torch.Tensor:
    # Flatten observation: position (2), target (2), steps (1), obstacles (up to 8 nearby obstacles, each 2)
    pos = list(obs.position)
    tgt = list(obs.target)
    steps = [obs.steps]
    # Pad or truncate nearby_obstacles to 8
    obstacles = list(obs.nearby_obstacles)[:8]
    flat_obs = pos + tgt + steps
    for ob in obstacles:
        flat_obs.extend(list(ob))
    # Pad if fewer than 8 obstacles
    while len(flat_obs) < 2 + 2 + 1 + 8 * 2:
        flat_obs.append(0)
    return torch.tensor(flat_obs, dtype=torch.float32)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepQAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        batch_size: int = 32,
        memory_size: int = 10000,
        target_update: int = 100,
        device: str = None,
        **kwargs,
    ):
        if isinstance(action_space, MazeActionSpace):
            self.action_space = action_space.n
            self.action_space_model = action_space
        else:
            self.action_space = action_space
            self.action_space_model = MazeActionSpace(n=action_space)
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update = target_update
        self.step_number = 0
        self.demo_path = None
        self.demo_step = 0
        self.current_observation = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 2 + 2 + 1 + 8 * 2  # pos(2), tgt(2), steps(1), obstacles(8x2)
        self.policy_net = DQN(self.input_dim, self.action_space).to(self.device)
        self.target_net = DQN(self.input_dim, self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        self.step_number += 1
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        self.current_observation = obs
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        obs_tensor = observation_to_tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        return int(torch.argmax(q_values).item())

    def set_demo_path(self, path: list[int]) -> None:
        self.demo_path = path
        self.demo_step = 0

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        obs_tensor = torch.stack([observation_to_tensor(o) for o in obs_batch]).to(self.device)
        action_tensor = torch.tensor(action_batch, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_tensor = torch.stack([observation_to_tensor(o) for o in next_obs_batch]).to(self.device)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        # Q(s,a)
        q_values = self.policy_net(obs_tensor).gather(1, action_tensor)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_tensor).max(1, keepdim=True)[0]
            target = reward_tensor + self.discount_factor * next_q_values * (1 - done_tensor)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Periodically update target network
        if self.step_number % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def observe(self, observation, action, reward, next_observation, done):
        # Store experience and train
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        next_obs = convert_numpy_to_python(next_observation)
        if not isinstance(next_obs, MazeObservation):
            next_obs = MazeObservation(**next_obs)
        self.remember(obs, action, reward, next_obs, done)
        self.update()


class MemoryDeepQAgent(DeepQAgent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        batch_size: int = 32,
        memory_size: int = 10000,
        target_update: int = 100,
        device: str = None,
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            action_space=action_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            memory_size=memory_size,
            target_update=target_update,
            device=device,
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
        state_key = f"{observation.position}|{observation.target}|{observation.steps}"
        position_key = str(observation.position)
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        try:
            if state_key not in self.visited_states:
                enhanced_state = {
                    "position": observation.position,
                    "target": observation.target,
                    "steps": observation.steps,
                    "nearby_obstacles": getattr(observation, "nearby_obstacles", None),
                    "manhattan_distance": abs(observation.position[0] - observation.target[0]) + abs(observation.position[1] - observation.target[1]),
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
                "manhattan_distance": abs(observation.position[0] - observation.target[0]) + abs(observation.position[1] - observation.target[1]),
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
                    chosen_action = max(set(actions_from_memory), key=actions_from_memory.count)
                    return int(chosen_action)
        except Exception:
            pass
        # Fallback to DQN policy or random
        if np.random.random() < epsilon:
            return np.random.randint(self.action_space)
        obs_tensor = observation_to_tensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        return int(torch.argmax(q_values).item())

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
                "state_key": f"{self.current_observation.position}|{self.current_observation.target}|{self.current_observation.steps}",
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

    def observe(self, observation, action, reward, next_observation, done):
        # Store experience and train
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        next_obs = convert_numpy_to_python(next_observation)
        if not isinstance(next_obs, MazeObservation):
            next_obs = MazeObservation(**next_obs)
        self.remember(obs, action, reward, next_obs, done)
        try:
            position_key = str(observation.position)
            next_position_key = str(next_observation.position)
            interaction_data = {
                "action": int(action),
                "reward": float(reward),
                "next_state": convert_numpy_to_python(next_observation.position),
                "done": done,
                "state_key": f"{observation.position}|{observation.target}|{observation.steps}",
                "next_state_key": f"{next_observation.position}|{next_observation.target}|{next_observation.steps}",
                "steps": observation.steps,
                "manhattan_distance": abs(observation.position[0] - observation.target[0]) + abs(observation.position[1] - observation.target[1]),
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
        self.update()
