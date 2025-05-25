"""
Memory Type Mapping System for AgentFarm DB to Memory System Converter

This module provides a robust system for mapping between AgentFarm database models and memory system types.
It handles the conversion and validation of memory types, ensuring data consistency during the conversion process.

Key Components:
    - MemoryTypeMapping: Configuration class for memory type mappings with validation
    - MemoryTypeMapper: Main class for converting between database models and memory types

The system supports three primary memory types:
    - state: Agent state information
    - action: Agent action records
    - interaction: Social interaction data

Usage:
    >>> mapper = MemoryTypeMapper()
    >>> memory_type = mapper.get_memory_type('AgentStateModel')
    >>> model_name = mapper.get_model_name('state')
    >>> is_valid = mapper.validate_memory_data('state', {...})
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


@dataclass
class MemoryTypeMapping:
    """Configuration for memory type mapping."""

    model_to_type: Dict[str, str]
    required_models: Set[str] = None
    valid_types: Set[str] = None

    def __post_init__(self):
        """Initialize default values and validate mapping."""
        if self.required_models is None:
            self.required_models = {
                "AgentStateModel",
                "ActionModel",
                "SocialInteractionModel",
            }

        if self.valid_types is None:
            self.valid_types = {"state", "action", "interaction"}

        self._validate_mapping()

    def _validate_mapping(self):
        """Validate the memory type mapping configuration."""
        # Check for missing required models
        missing_models = self.required_models - set(self.model_to_type.keys())
        if missing_models:
            raise ValueError(f"Missing required memory type mappings: {missing_models}")

        # Check for invalid memory types
        invalid_types = {
            model: type_
            for model, type_ in self.model_to_type.items()
            if type_ not in self.valid_types
        }
        if invalid_types:
            raise ValueError(
                f"Invalid memory types in mapping: {invalid_types}. "
                f"Must be one of: {self.valid_types}"
            )

        # Check for duplicate memory types
        type_to_models = {}
        for model, type_ in self.model_to_type.items():
            if type_ in type_to_models:
                type_to_models[type_].append(model)
            else:
                type_to_models[type_] = [model]

        duplicates = {
            type_: models for type_, models in type_to_models.items() if len(models) > 1
        }
        if duplicates:
            raise ValueError(
                f"Duplicate memory types found: {duplicates}. "
                "Each memory type must be unique."
            )


class MemoryTypeMapper:
    """
    Maps AgentFarm database models to memory system types.

    This class handles the conversion between database models and memory types,
    including validation and custom mapping support.
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, str]] = None,
        required_models: Optional[Set[str]] = None,
        valid_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the memory type mapper.

        Args:
            mapping: Optional custom mapping of model names to memory types
            required_models: Optional set of required model names
            valid_types: Optional set of valid memory types
        """
        self.mapping = MemoryTypeMapping(
            model_to_type=mapping
            or {
                "AgentStateModel": "state",
                "ActionModel": "action",
                "SocialInteractionModel": "interaction",
            },
            required_models=required_models,
            valid_types=valid_types,
        )

    @property
    def required_models(self) -> Set[str]:
        """
        Get the set of required model names.

        Returns:
            Set of required model names
        """
        return self.mapping.required_models

    def get_memory_type(self, model_name: str) -> str:
        """
        Get the memory type for a given model name.

        Args:
            model_name: Name of the database model

        Returns:
            Corresponding memory type

        Raises:
            ValueError: If model_name is not in the mapping
        """
        if model_name not in self.mapping.model_to_type:
            raise ValueError(f"No memory type mapping for model: {model_name}")

        return self.mapping.model_to_type[model_name]

    def get_model_name(self, memory_type: str) -> str:
        """
        Get the model name for a given memory type.

        Args:
            memory_type: Type of memory

        Returns:
            Corresponding model name

        Raises:
            ValueError: If memory_type is not in the mapping
        """
        for model, type_ in self.mapping.model_to_type.items():
            if type_ == memory_type:
                return model

        raise ValueError(f"No model mapping for memory type: {memory_type}")

    def validate_memory_data(self, memory_type: str, data: Dict[str, Any]) -> bool:
        """
        Validate memory data for a given type.

        Args:
            memory_type: Type of memory to validate
            data: Memory data to validate

        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation based on memory type
        if memory_type == "state":
            required_fields = {"agent_id": int, "step_number": int, "state_data": dict}
        elif memory_type == "action":
            required_fields = {"agent_id": int, "step_number": int, "action_type": str}
        elif memory_type == "interaction":
            required_fields = {
                "agent_id": int,
                "step_number": int,
                "interaction_type": str,
            }
        else:
            return False

        # Check for required fields and their types
        for field, expected_type in required_fields.items():
            if field not in data:
                return False
            if not isinstance(data[field], expected_type):
                return False

        return True
