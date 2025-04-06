"""Memory hooks for integrating with agent lifecycle events."""

import functools
import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

from memory.api.models import ActionData, ActionResult, AgentState
from memory.config import MemoryConfig
from memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)

# Type variable for agent classes
T = TypeVar("T", bound="BaseAgent")


def get_memory_config(config: Any) -> Optional[MemoryConfig]:
    """Retrieve memory configuration from agent config.

    Handles both dictionary-style and object-style configurations.

    Args:
        config: Agent configuration (dict or object)

    Returns:
        MemoryConfig object or None if not found
    """
    if config is None:
        return None

    # Try attribute-style access first
    memory_config = getattr(config, "memory_config", None)

    # Fall back to dictionary-style access
    if memory_config is None and isinstance(config, dict):
        memory_config = config.get("memory_config")

    # Convert dict to MemoryConfig object
    if isinstance(memory_config, dict):
        memory_config = MemoryConfig(**memory_config)

    return memory_config


def _log_throttled_error(self: Any, message: str, throttle_seconds: int = 60) -> None:
    """Log errors with throttling to avoid log spam.

    Args:
        message: Error message to log
        throttle_seconds: Minimum seconds between logging the same error
    """
    current_time = time.time()
    if current_time - getattr(self, "_memory_last_error_time", 0) > throttle_seconds:
        logger.exception(message)
        self._memory_last_error_time = current_time


def _initialize_memory_attributes(
    obj: Any, memory_config: Optional[MemoryConfig]
) -> None:
    """Initialize memory-related attributes on an object.

    Args:
        obj: Object to initialize attributes on
        memory_config: Memory configuration
    """
    try:
        obj.memory_system = AgentMemorySystem.get_instance(memory_config)
        obj._memory_last_error_time = 0
        obj._memory_recording = False
    except Exception as e:
        _handle_memory_error(obj, f"Failed to initialize memory system: {e}")
        obj.memory_system = None


def _handle_memory_error(obj: Any, error_message: str) -> None:
    """Handle memory-related errors with proper logging and error management.

    Args:
        obj: Object where the error occurred
        error_message: Error message to log
    """
    try:
        if not hasattr(obj, "_log_throttled_error"):
            obj._log_throttled_error = _log_throttled_error.__get__(obj, type(obj))
        obj._log_throttled_error(error_message)
    except (AttributeError, TypeError):
        # Fallback for tests or when method binding fails
        logger.exception(error_message)
        obj._memory_last_error_time = time.time()


class BaseAgent:
    """Base agent class with core functionality required for memory hooks."""

    def __init__(self, config=None, agent_id=None, **kwargs):
        """Initialize a base agent.

        Args:
            config: Configuration object or dict
            agent_id: Unique identifier for this agent
            **kwargs: Additional arguments
        """
        self.config = config or {}
        self.agent_id = agent_id or f"agent_{id(self)}"
        self.step_number = 0

    def act(self, observation=None, **kwargs):
        """Perform an agent action based on observation.

        Args:
            observation: Current environment observation
            **kwargs: Additional parameters

        Returns:
            Action result object
        """
        self.step_number += 1
        # Base implementation just returns a simple result
        return ActionResult(action_type="noop")

    def get_state(self) -> AgentState:
        """Get the current agent state.

        Returns:
            AgentState object containing agent state
        """
        return AgentState(agent_id=self.agent_id, step_number=self.step_number)


def install_memory_hooks(agent_class: Type[T]) -> Type[T]:
    """Install memory hooks on an agent class.

    This is a class decorator that adds memory hooks to BaseAgent subclasses.
    Hooks are only installed if enable_memory_hooks is True in the memory config.

    Args:
        agent_class: The agent class to install hooks on

    Returns:
        The modified agent class
    """
    original_init = agent_class.__init__
    original_act = agent_class.act
    original_get_state = agent_class.get_state

    @functools.wraps(original_init)
    def init_with_memory(self: T, *args: Any, **kwargs: Any) -> None:
        """Initialize with memory system support."""
        original_init(self, *args, **kwargs)

        # Get memory system
        memory_config = get_memory_config(self.config)

        # Early return if memory hooks are disabled
        if memory_config and not memory_config.enable_memory_hooks:
            logger.info(
                f"Memory hooks disabled for agent {getattr(self, 'agent_id', 'unknown')}"
            )
            return

        _initialize_memory_attributes(self, memory_config)

    @functools.wraps(original_act)
    def act_with_memory(self: T, *args: Any, **kwargs: Any) -> ActionResult:
        """Act with memory integration.

        Captures state before and after action, calculates importance,
        and handles errors gracefully with fallback behavior.
        """
        if not hasattr(self, "memory_system") or self.memory_system is None:
            return original_act(self, *args, **kwargs)

        # Get state before action
        try:
            self._memory_recording = True
            state_before = original_get_state(self)
            step_number = getattr(self, "step_number", 0)

            # Call original act method
            start_time = time.time()
            result = original_act(self, *args, **kwargs)
            execution_time = time.time() - start_time

            # Get state after action
            state_after = original_get_state(self)

            # Calculate importance score based on state change and reward
            reward = getattr(result, "reward", 0.0)
            state_diff = self._calculate_state_difference(
                state_before.as_dict(), state_after.as_dict()
            )
            importance = min(1.0, (0.5 * abs(reward) + 0.5 * state_diff))

            # Create action record with metadata
            action_data = ActionData(
                action_type=result.action_type,
                action_params=result.params,
                state_before=state_before.as_dict(),
                state_after=state_after.as_dict(),
                reward=reward,
                execution_time=execution_time,
                step_number=step_number,
            )

            # Store in memory with importance score for prioritization
            self.memory_system.store_agent_action(
                self.agent_id,
                action_data.model_dump(),
                step_number,
                priority=importance,
            )

            # Reset recording flag
            self._memory_recording = False
            return result

        except Exception as e:
            _handle_memory_error(self, f"Memory error in act_with_memory: {e}")
            self._memory_recording = False
            return original_act(self, *args, **kwargs)

    @functools.wraps(original_get_state)
    def get_state_with_memory(self: T, *args: Any, **kwargs: Any) -> AgentState:
        """Get state with memory integration.

        Retrieves state and stores in memory if not already recording.
        """
        # Call original get_state method
        state = original_get_state(self, *args, **kwargs)

        # Skip storage if we're already in a recording process or memory system is unavailable
        if not hasattr(self, "memory_system") or self.memory_system is None:
            return state

        # Store in memory if not already storing in act method
        if not getattr(self, "_memory_recording", False):
            try:
                step_number = getattr(self, "step_number", 0)
                state_dict = state.as_dict()
                importance = _calculate_state_importance(state_dict)

                self.memory_system.store_agent_state(
                    self.agent_id, state_dict, step_number, priority=importance
                )
            except Exception as e:
                _handle_memory_error(
                    self, f"Memory error in get_state_with_memory: {e}"
                )

        return state

    def _calculate_state_difference(
        self: T, state_before: Dict[str, Any], state_after: Dict[str, Any]
    ) -> float:
        """Calculate a normalized difference between two states.

        Args:
            state_before: State before action
            state_after: State after action

        Returns:
            Normalized difference score (0.0-1.0)
        """
        # Handle empty states
        if not state_before and not state_after:
            return 0.5  # Return mid-range value for empty-to-empty comparison

        if not state_before or not state_after:
            return 1.0  # Maximum difference when one state is empty

        # Find common keys with numeric values in both states
        common_keys = set(state_before.keys()) & set(state_after.keys())

        # Calculate normalized differences using list comprehension
        differences = [
            min(
                1.0,
                abs(state_after[key] - state_before[key])
                / max(1.0, abs(state_before[key])),
            )
            for key in common_keys
            if isinstance(state_before[key], (int, float))
            and isinstance(state_after[key], (int, float))
        ]

        # Return average difference, defaulting to 0.5 if no comparable values
        return sum(differences) / len(differences) if differences else 0.5

    # Replace methods only if this is the first time applying hooks to this class
    if not hasattr(agent_class, "_memory_hooks_installed"):
        agent_class.__init__ = init_with_memory
        agent_class.act = act_with_memory
        agent_class.get_state = get_state_with_memory
        agent_class._calculate_state_difference = _calculate_state_difference
        agent_class._log_throttled_error = _log_throttled_error
        agent_class._memory_hooks_installed = True

    return agent_class


def _calculate_state_importance(state: Dict[str, Any]) -> float:
    """Calculate an importance score for a state.

    Args:
        state: Agent state dictionary

    Returns:
        Importance score (0.0-1.0)
    """
    importance = 0.5  # Default importance

    if "health" in state:
        # Low health is more important to remember
        importance = max(importance, 1.0 - state["health"])

    if "reward" in state:
        # High rewards (positive or negative) are important
        # Scale so that rewards > 5.0 result in importance > 0.5
        reward_importance = min(1.0, abs(state["reward"]) / 5.0)
        importance = max(importance, reward_importance)

    return importance


def with_memory(agent_instance: T) -> T:
    """Add memory capabilities to an existing agent instance.

    Args:
        agent_instance: The agent instance to add memory to

    Returns:
        The agent with memory capabilities
    """
    # Skip if memory hooks are disabled in config
    memory_config = get_memory_config(agent_instance.config)
    if memory_config and not memory_config.enable_memory_hooks:
        logger.info(
            f"Memory hooks disabled for agent {getattr(agent_instance, 'agent_id', 'unknown')}"
        )
        return agent_instance

    # Create a dynamic subclass with memory hooks
    agent_class = type(agent_instance)
    memory_class = type(f"{agent_class.__name__}WithMemory", (agent_class,), {})

    # Install hooks on the new class
    memory_class = install_memory_hooks(memory_class)

    # Update the instance's class
    agent_instance.__class__ = memory_class

    # Explicitly initialize memory attributes since changing __class__ doesn't call __init__
    if (
        not hasattr(agent_instance, "memory_system")
        or agent_instance.memory_system is None
    ):
        _initialize_memory_attributes(agent_instance, memory_config)

    return agent_instance
