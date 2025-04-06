"""Agent Memory Models Module

This module defines structured data models used throughout the agent memory system,
providing validation, serialization, and consistent typing for memory operations.
These models serve as the foundation for representing agent states, actions, and
their relationships in memory.

Key components:

1. AgentState: A standardized representation of an agent's state at a point in time,
   including positional information, health, resources, and other core attributes
   common across many agent types.

2. ActionData: A comprehensive record of an agent action, capturing both the state
   before and after the action, execution metrics, and contextual information
   needed for memory retrieval and importance calculation.

3. ActionResult: A lightweight representation of an action's outcome, used as the
   return value for agent action methods, providing a consistent interface for
   memory hooks.

The models in this module are built using Pydantic, enabling automatic validation,
serialization to various formats, and schema generation. They provide a consistent
interface for memory operations while allowing for extensibility through additional
fields and inheritance.

Usage example:
```python
from memory.api.models import AgentState, ActionResult
from memory.api import AgentMemoryAPI

# Create an agent state
state = AgentState(
    agent_id="agent-001",
    step_number=42,
    health=0.8,
    position_x=10.5,
    position_y=20.3,
    extra_data={
        "inventory": ["apple", "sword"],
        "current_goal": "find the treasure"
    }
)

# Convert to dictionary for storage
state_dict = state.as_dict()

# Create an action result
result = ActionResult(
    action_type="move",
    params={"direction": "north", "distance": 1.5},
    reward=0.25
)

# Store in memory system
memory_api = AgentMemoryAPI()
memory_id = memory_api.store_state(state_dict)
```
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Standardized representation of an agent's state.

    This model provides a consistent structure for agent states,
    with common fields that are often present across different agent types.

    Attributes:
        agent_id: Unique identifier for the agent
        step_number: Current step/time in the agent's lifecycle
        health: Agent's current health level (0.0-1.0)
        reward: Current or accumulated reward value
        position_x: X coordinate in environment (if applicable)
        position_y: Y coordinate in environment (if applicable)
        position_z: Z coordinate in environment (if applicable)
        resource_level: Agent's current resources
        extra_data: Additional agent-specific state information
    """

    agent_id: str
    step_number: int
    health: Optional[float] = None
    reward: Optional[float] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_z: Optional[float] = None
    resource_level: Optional[float] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary, excluding None values."""
        # Start with the model's dict representation
        data = self.model_dump(exclude_none=True)
        # If extra_data is empty, remove it
        if not data.get("extra_data"):
            data.pop("extra_data", None)
        return data


class ActionData(BaseModel):
    """Data about an agent action with associated states and metrics.

    This model captures the full context of an agent action, including
    the states before and after the action, execution metrics, and results.

    Attributes:
        action_type: Type of action performed
        action_params: Parameters used for the action
        state_before: Agent state before action execution
        state_after: Agent state after action execution
        reward: Reward received from the action
        execution_time: Time taken to execute the action (seconds)
        step_number: Step number when the action was taken
    """

    action_type: str
    action_params: Dict[str, Any] = Field(default_factory=dict)
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    reward: float = 0.0
    execution_time: float
    step_number: int

    def get_state_difference(self) -> Dict[str, Any]:
        """Calculate the differences between before and after states."""
        differences = {}

        # Find numeric fields that exist in both states
        for key in set(self.state_before.keys()) & set(self.state_after.keys()):
            before_val = self.state_before.get(key)
            after_val = self.state_after.get(key)

            # Only consider numeric values
            if isinstance(before_val, (int, float)) and isinstance(
                after_val, (int, float)
            ):
                differences[key] = after_val - before_val

        return differences


class ActionResult(BaseModel):
    """Standardized result of an agent action.

    This model represents the outcome of an agent action,
    including the type of action performed and any parameters or rewards.

    Attributes:
        action_type: Type of action performed
        params: Parameters used in or resulting from the action
        reward: Reward value associated with this action
    """

    action_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
