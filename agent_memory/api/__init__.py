"""API for the Agent Memory System.

This package provides the public API for integrating the Agent Memory System
with agents, including memory hooks and data models.
"""

from agent_memory.api.hooks import (
    BaseAgent, 
    install_memory_hooks,
    with_memory
)
from agent_memory.api.models import (
    AgentState,
    ActionData,
    ActionResult
)

__all__ = [
    # Hooks API
    "BaseAgent",
    "install_memory_hooks",
    "with_memory",
    
    # Models API
    "AgentState",
    "ActionData",
    "ActionResult",
]
