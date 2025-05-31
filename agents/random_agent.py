import numpy as np

from agents import Agent
from memory.api.models import MazeActionSpace, MazeObservation
from memory.config.memory_config import (
    MemoryConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from memory.space import MemorySpace
from memory.utils.util import convert_numpy_to_python


class RandomAgent(Agent):
    def __init__(
        self, agent_id: str, action_space: int | MazeActionSpace = 4, **kwargs
    ):
        if isinstance(action_space, MazeActionSpace):
            self.action_space = action_space.n
            self.action_space_model = action_space
        else:
            self.action_space = action_space
            self.action_space_model = MazeActionSpace(n=action_space)
        self.agent_id = agent_id
        self.demo_path = None
        self.demo_step = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        return int(np.random.randint(self.action_space))

    def set_demo_path(self, path: list[int]) -> None:
        self.demo_path = path
        self.demo_step = 0


class MemoryRandomAgent(RandomAgent):
    def __init__(
        self, agent_id: str, action_space: int | MazeActionSpace = 4, **kwargs
    ):
        super().__init__(agent_id, action_space, **kwargs)
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
        self.position_memory_cache = {}
        self.step_number = 0

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        self.step_number += 1
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        position_key = str(observation.position)
        state_key = f"{observation.position}|{observation.target}|{observation.steps}"
        # Try to retrieve similar states from memory
        try:
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
            if len(similar_states) == 0 and position_key in self.position_memory_cache:
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
                    return int(chosen_action)
        except Exception:
            pass
        # Otherwise, act randomly
        return int(np.random.randint(self.action_space))

    def set_demo_path(self, path: list[int]) -> None:
        self.demo_path = path
        self.demo_step = 0
