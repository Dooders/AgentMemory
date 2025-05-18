"""
AgentFarm DB to Memory System Converter

This module provides functionality to convert data from an AgentFarm SQLite database
into a memory system, handling the import of agents and their associated memories.
"""

__version__ = "0.1.0"

from .config import DEFAULT_CONFIG
from .converter import from_agent_farm

__all__ = ["from_agent_farm", "DEFAULT_CONFIG"] 