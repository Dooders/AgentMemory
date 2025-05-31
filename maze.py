"""
Maze Environment Module
----------------------
This module provides the MazeEnvironment class, a simple grid-based environment for reinforcement learning or pathfinding experiments.

Features:
- Configurable maze size, obstacles, and maximum steps per episode.
- Agent starts at (1, 1) and aims to reach the target at (size-2, size-2).
- Step function supports four actions: up, right, down, left.
- Rewards for reaching the target, penalties for timeouts, and step/distance-based penalties.
- Observations include agent position, target, nearby obstacles, and step count.

Example usage:
    env = MazeEnvironment(size=5, obstacles=[(2,2), (3,3)])
    obs = env.reset()
    obs, reward, done = env.step(1)  # Take action 'right'
"""

from memory.api.models import MazeObservation, MazeActionSpace

class MazeEnvironment:
    """
    A simple grid-based maze environment for RL/pathfinding experiments.

    Attributes:
        size (int): Size of the maze (size x size grid).
        obstacles (list[tuple[int, int]]): List of obstacle coordinates.
        target (tuple[int, int]): Target position in the maze.
        max_steps (int): Maximum steps per episode.
        position (tuple[int, int]): Current agent position.
        steps (int): Steps taken in current episode.
        action_space (MazeActionSpace): The action space for the environment.
    """

    def __init__(
        self,
        size: int = 5,
        obstacles: list[tuple[int, int]] = None,
        max_steps: int = 15,
    ) -> None:
        """
        Initialize the maze environment.

        Args:
            size: Size of the maze (size x size grid).
            obstacles: List of (row, col) tuples for obstacle locations.
            max_steps: Maximum steps allowed per episode.
        """
        self.size = size
        self.obstacles = obstacles or []
        self.target = (size - 2, size - 2)
        self.max_steps = max_steps
        self.action_space = MazeActionSpace()
        self.reset()

    def reset(self) -> dict:
        """
        Reset the environment to the initial state.

        Returns:
            dict: Initial observation after reset.
        """
        self.position = (1, 1)
        self.steps = 0
        return self.get_observation()

    def get_observation(self) -> MazeObservation:
        """
        Get the current observation of the environment.

        Returns:
            MazeObservation: Observation containing position, target, nearby obstacles, and steps.
        """
        return MazeObservation(
            position=self.position,
            target=self.target,
            nearby_obstacles=self._get_nearby_obstacles(),
            steps=self.steps,
        )

    def _get_nearby_obstacles(self) -> list[tuple[int, int]]:
        """
        Get obstacles within a Manhattan distance of 2 from the agent.

        Returns:
            list[tuple[int, int]]: Nearby obstacle coordinates.
        """
        return [
            obs
            for obs in self.obstacles
            if abs(obs[0] - self.position[0]) <= 2
            and abs(obs[1] - self.position[1]) <= 2
        ]

    def step(self, action: int) -> tuple[dict, float, bool]:
        """
        Take an action in the environment.

        Args:
            action (int): Action to take (0=up, 1=right, 2=down, 3=left).

        Returns:
            tuple: (observation, reward, done)
                observation (dict): New observation after action.
                reward (float): Reward for the action.
                done (bool): Whether the episode has ended.
        """
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

    def get_action_space(self) -> MazeActionSpace:
        """
        Get the action space for the environment.

        Returns:
            MazeActionSpace: The action space model for the maze.
        """
        return self.action_space
