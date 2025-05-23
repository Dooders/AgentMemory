"""
Configuration system for the AgentFarm DB to Memory System converter.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union

from .tiering import TieringStrategy, create_tiering_strategy
from .mapping import MemoryTypeMapper

logger = logging.getLogger(__name__)

@dataclass
class ConverterConfig:
    """Configuration for the AgentFarm DB to Memory System converter."""
    
    # Redis configuration
    use_mock_redis: bool = True
    
    # Validation settings
    validate: bool = True
    error_handling: str = "skip"  # One of: "skip", "fail", "log"
    
    # Processing settings
    batch_size: int = 100
    show_progress: bool = True
    
    # Memory type mapping
    memory_type_mapping: Dict[str, str] = field(default_factory=lambda: {
        'AgentStateModel': 'state',
        'ActionModel': 'action',
        'SocialInteractionModel': 'interaction'
    })
    memory_type_mapper: Optional[MemoryTypeMapper] = None
    
    # Tiering strategy
    tiering_strategy_type: str = "simple"  # One of: "simple", "step_based", "importance_aware"
    tiering_strategy: Optional[TieringStrategy] = None
    
    # Import settings
    import_mode: str = "full"  # One of: "full", "incremental"
    selective_agents: Optional[List[int]] = None
    total_steps: Optional[int] = None  # Total number of steps in the simulation
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_error_handling()
        self._validate_import_mode()
        self._validate_batch_size()
        self._validate_tiering_strategy()
        
        # Initialize memory type mapper
        if self.memory_type_mapper is None:
            self.memory_type_mapper = MemoryTypeMapper(mapping=self.memory_type_mapping)
        
        # Initialize tiering strategy if not provided
        if self.tiering_strategy is None:
            self.tiering_strategy = create_tiering_strategy(self.tiering_strategy_type)
        
    def _validate_error_handling(self):
        """Validate error handling mode."""
        if self.error_handling not in ["skip", "fail", "log"]:
            raise ValueError(
                f"Invalid error_handling mode: {self.error_handling}. "
                "Must be one of: skip, fail, log"
            )
            
    def _validate_import_mode(self):
        """Validate import mode."""
        if self.import_mode not in ["full", "incremental"]:
            raise ValueError(
                f"Invalid import_mode: {self.import_mode}. "
                "Must be one of: full, incremental"
            )
            
    def _validate_batch_size(self):
        """Validate batch size."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
            
    def _validate_tiering_strategy(self):
        """Validate tiering strategy settings."""
        valid_types = ["simple", "step_based", "importance_aware"]
        if self.tiering_strategy_type not in valid_types:
            raise ValueError(
                f"Invalid tiering_strategy_type: {self.tiering_strategy_type}. "
                f"Must be one of: {valid_types}"
            )

# Default configuration
DEFAULT_CONFIG = {
    'use_mock_redis': True,
    'validate': True,
    'error_handling': 'skip',
    'batch_size': 100,
    'show_progress': True,
    'memory_type_mapping': {
        'AgentStateModel': 'state',
        'ActionModel': 'action',
        'SocialInteractionModel': 'interaction'
    },
    'tiering_strategy_type': 'simple',
    'import_mode': 'full',
    'selective_agents': None
} 