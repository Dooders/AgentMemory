"""
Memory type mapping system for the AgentFarm DB to Memory System converter.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set, Any

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
                'AgentStateModel',
                'ActionModel',
                'SocialInteractionModel'
            }
            
        if self.valid_types is None:
            self.valid_types = {'state', 'action', 'interaction'}
            
        self._validate_mapping()
        
    def _validate_mapping(self):
        """Validate the memory type mapping configuration."""
        # Check for missing required models
        missing_models = self.required_models - set(self.model_to_type.keys())
        if missing_models:
            raise ValueError(
                f"Missing required memory type mappings: {missing_models}"
            )
            
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

class MemoryTypeMapper:
    """
    Maps AgentFarm database models to memory system types.
    
    This class handles the conversion between database models and memory types,
    including validation and custom mapping support.
    """
    
    def __init__(self, mapping: Optional[Dict[str, str]] = None, 
                 required_models: Optional[Set[str]] = None,
                 valid_types: Optional[Set[str]] = None):
        """
        Initialize the memory type mapper.
        
        Args:
            mapping: Optional custom mapping of model names to memory types
            required_models: Optional set of required model names
            valid_types: Optional set of valid memory types
        """
        self.mapping = MemoryTypeMapping(
            model_to_type=mapping or {
                'AgentStateModel': 'state',
                'ActionModel': 'action',
                'SocialInteractionModel': 'interaction'
            },
            required_models=required_models,
            valid_types=valid_types
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
        if memory_type == 'state':
            return all(key in data for key in ['agent_id', 'step_number', 'state_data'])
        elif memory_type == 'action':
            return all(key in data for key in ['agent_id', 'step_number', 'action_type'])
        elif memory_type == 'interaction':
            return all(key in data for key in ['agent_id', 'step_number', 'interaction_type'])
            
        return False 