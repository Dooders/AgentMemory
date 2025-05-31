from collections import deque

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


class AlgoAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        search_algo: str = "bfs",
        **kwargs,
    ):
        if isinstance(action_space, MazeActionSpace):
            self.action_space = action_space.n
            self.action_space_model = action_space
        else:
            self.action_space = action_space
            self.action_space_model = MazeActionSpace(n=action_space)
        self.agent_id = agent_id
        self.search_algo = search_algo
        self.demo_path = None
        self.demo_step = 0
        self.last_plan = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        # Plan path if needed
        if not self.last_plan or obs.position != self.last_plan[0]:
            self.last_plan = self._plan_path(obs)
        if len(self.last_plan) < 2:
            # No path or already at target
            return np.random.randint(self.action_space)
        # Determine action to move from current to next position
        current = self.last_plan[0]
        next_pos = self.last_plan[1]
        action = self._get_action_from_positions(current, next_pos)
        # Advance plan
        self.last_plan = self.last_plan[1:]
        return int(action)

    def set_demo_path(self, path: list[int]) -> None:
        self.demo_path = path
        self.demo_step = 0

    def _plan_path(self, obs: MazeObservation):
        if callable(self.search_algo):
            return self.search_algo(obs)
        if self.search_algo == "bfs":
            return self._bfs(obs)
        elif self.search_algo == "dfs":
            return self._dfs(obs)
        else:
            # Default to BFS
            return self._bfs(obs)

    def _bfs(self, obs: MazeObservation):
        start = obs.position
        target = obs.target
        size = max(max(start), max(target)) + 2  # crude estimate
        obstacles = set(obs.nearby_obstacles)
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        while queue:
            pos, path = queue.popleft()
            if pos == target:
                return path
            if pos in visited:
                continue
            visited.add(pos)
            for action, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                new_pos = (pos[0] + dr, pos[1] + dc)
                if new_pos in visited or new_pos in obstacles:
                    continue
                if (
                    new_pos[0] < 0
                    or new_pos[1] < 0
                    or new_pos[0] >= size
                    or new_pos[1] >= size
                ):
                    continue
                queue.append((new_pos, path + [new_pos]))
        return [start]  # No path found

    def _dfs(self, obs: MazeObservation):
        start = obs.position
        target = obs.target
        size = max(max(start), max(target)) + 2
        obstacles = set(obs.nearby_obstacles)
        stack = [(start, [start])]
        visited = set()
        while stack:
            pos, path = stack.pop()
            if pos == target:
                return path
            if pos in visited:
                continue
            visited.add(pos)
            for action, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                new_pos = (pos[0] + dr, pos[1] + dc)
                if new_pos in visited or new_pos in obstacles:
                    continue
                if (
                    new_pos[0] < 0
                    or new_pos[1] < 0
                    or new_pos[0] >= size
                    or new_pos[1] >= size
                ):
                    continue
                stack.append((new_pos, path + [new_pos]))
        return [start]

    def _get_action_from_positions(self, current, next_pos):
        # Returns action index to move from current to next_pos
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]
        if dr == -1 and dc == 0:
            return 0  # up
        elif dr == 0 and dc == 1:
            return 1  # right
        elif dr == 1 and dc == 0:
            return 2  # down
        elif dr == 0 and dc == -1:
            return 3  # left
        else:
            return np.random.randint(self.action_space)


class MemoryAlgoAgent(AlgoAgent):
    def __init__(
        self,
        agent_id: str,
        action_space: int | MazeActionSpace = 4,
        search_algo: str = "bfs",
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            action_space=action_space,
            search_algo=search_algo,
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
        self.position_memory_cache = {}
        self.step_number = 0

    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        self.step_number += 1
        if self.demo_path is not None and self.demo_step < len(self.demo_path):
            action = self.demo_path[self.demo_step]
            self.demo_step += 1
            return int(action)
        obs = convert_numpy_to_python(observation)
        if not isinstance(obs, MazeObservation):
            obs = MazeObservation(**obs)
        position_key = str(obs.position)
        state_key = f"{obs.position}|{obs.target}|{obs.steps}"
        # Try to retrieve similar states from memory
        try:
            query_state = {
                "position": obs.position,
                "target": obs.target,
                "steps": obs.steps,
                "manhattan_distance": abs(obs.position[0] - obs.target[0])
                + abs(obs.position[1] - obs.target[1]),
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
                    # Store action in memory
                    self._store_action_and_interaction(
                        obs, chosen_action, None, None, None
                    )
                    return int(chosen_action)
        except Exception:
            pass
        # Otherwise, use the search algorithm
        if not self.last_plan or obs.position != self.last_plan[0]:
            self.last_plan = self._plan_path(obs)
        if len(self.last_plan) < 2:
            action = np.random.randint(self.action_space)
        else:
            current = self.last_plan[0]
            next_pos = self.last_plan[1]
            action = self._get_action_from_positions(current, next_pos)
            self.last_plan = self.last_plan[1:]
        # Store action in memory
        self._store_action_and_interaction(obs, action, None, None, None)
        return int(action)

    def _store_action_and_interaction(self, obs, action, reward, next_obs, done):
        try:
            position_key = str(obs.position)
            action_data = {
                "action": int(action),
                "position": obs.position,
                "state_key": f"{obs.position}|{obs.target}|{obs.steps}",
                "steps": obs.steps,
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

    def set_demo_path(self, path: list[int]) -> None:
        self.demo_path = path
        self.demo_step = 0
