from abc import ABC, abstractmethod

from memory.api.models import MazeObservation


class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, agent_id: str, action_space, **kwargs):
        self.agent_id = agent_id
        self.action_space = action_space
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def act(self, observation: MazeObservation, epsilon: float = 0.1) -> int:
        """
        Choose and return an action for the given observation.
        """
        pass

    @abstractmethod
    def set_demo_path(self, path: list[int]) -> None:
        """
        Set a predetermined path of actions for demonstration or scripted exploration.
        """
        pass
