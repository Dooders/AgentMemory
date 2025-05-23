"""
Memory import handlers for the AgentFarm DB to Memory System converter.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Generator
import logging
from datetime import datetime
from sqlalchemy import func

from .db import DatabaseManager
from .config import ConverterConfig
from .tiering import TieringContext, TieringStrategy
from .mapping import MemoryTypeMapper

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetadata:
    """Metadata for an imported memory."""
    memory_id: int
    agent_id: int
    memory_type: str
    step_number: int
    tier: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

class MemoryImporter:
    """
    Handles the import of memories from AgentFarm database to memory system.
    
    This class manages the process of importing different types of memories,
    including validation, tiering, and error handling.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: ConverterConfig,
        tiering_strategy: TieringStrategy,
        memory_type_mapper: MemoryTypeMapper
    ):
        """
        Initialize the memory importer.
        
        Args:
            db_manager: Database manager instance
            config: Converter configuration
            tiering_strategy: Strategy for determining memory tiers
            memory_type_mapper: Mapper for memory types
        """
        self.db_manager = db_manager
        self.config = config
        self.tiering_strategy = tiering_strategy
        self.memory_type_mapper = memory_type_mapper
        
    def import_memories(self, agent_id: int) -> List[MemoryMetadata]:
        """
        Import memories for a specific agent.
        
        Args:
            agent_id: ID of the agent to import memories for
            
        Returns:
            List of imported memory metadata
            
        Raises:
            ValueError: If memory validation fails and error_handling is 'fail'
        """
        memories = []
        logger.debug(f"Starting memory import for agent {agent_id}")
        
        with self.db_manager.session() as session:
            # First verify the agent exists
            agent = session.query(self.db_manager.AgentModel).filter(
                self.db_manager.AgentModel.agent_id == agent_id
            ).first()
            
            if not agent:
                logger.error(f"Agent {agent_id} not found in database")
                return memories
                
            logger.info(f"Found agent {agent_id} with type {agent.agent_type}")
            
            # Get total steps from simulation steps table
            if self.config.total_steps is None:
                max_step = session.query(func.max(self.db_manager.SimulationStepModel.step_number)).scalar()
                if max_step is not None:
                    self.config.total_steps = max_step
                    logger.info(f"Found total steps: {max_step}")
                else:
                    logger.warning("Could not determine total steps, using step number as current step")
            
            # Import each memory type
            for model_name in self.memory_type_mapper.required_models:
                logger.debug(f"Processing memory type: {model_name}")
                try:
                    model_memories = self._import_memory_type(
                        session,
                        agent_id,
                        model_name
                    )
                    logger.debug(f"Found {len(model_memories)} memories for {model_name}")
                    memories.extend(model_memories)
                except Exception as e:
                    self._handle_import_error(e, agent_id, model_name)
                    
        logger.debug(f"Total memories imported for agent {agent_id}: {len(memories)}")
        return memories
        
    def _import_memory_type(
        self,
        session,
        agent_id: int,
        model_name: str
    ) -> List[MemoryMetadata]:
        """
        Import memories of a specific type.
        
        Args:
            session: Database session
            agent_id: ID of the agent
            model_name: Name of the model to import
            
        Returns:
            List of imported memory metadata
        """
        memories = []
        model = getattr(self.db_manager, model_name)
        memory_type = self.memory_type_mapper.get_memory_type(model_name)
        
        # Get query for this memory type
        query = self._get_memory_query(session, model, agent_id)
        
        # Process memories in batches
        for batch in self._batch_query(query):
            for memory in batch:
                try:
                    memory_metadata = self._import_memory(
                        memory,
                        memory_type,
                        model_name
                    )
                    if memory_metadata:  # Only add if memory was successfully created
                        memories.append(memory_metadata)
                        logger.debug(f"Created memory for {model_name} with id {memory_metadata.memory_id}")
                except Exception as e:
                    logger.error(f"Error creating memory for {model_name}: {str(e)}")
                    self._handle_import_error(e, agent_id, model_name)
                    
        logger.info(f"Successfully imported {len(memories)} memories of type {model_name} for agent {agent_id}")
        return memories
        
    def _get_memory_query(self, session, model, agent_id: int):
        """Get the appropriate memory query based on import mode."""
        if model.__name__ == 'SocialInteractionModel':
            # For social interactions, check both initiator and recipient
            query = session.query(model).filter(
                (model.initiator_id == agent_id) | (model.recipient_id == agent_id)
            )
        else:
            # For other models, just check the agent_id
            query = session.query(model).filter(model.agent_id == agent_id)
            
        # Log the actual SQL query
        logger.debug(f"SQL Query for {model.__name__}: {query.statement.compile(compile_kwargs={'literal_binds': True})}")
        
        # Log the count of results
        count = query.count()
        logger.debug(f"Found {count} records for {model.__name__} with agent_id {agent_id}")
        
        if self.config.import_mode == "incremental":
            # Add incremental import conditions
            pass
            
        return query
        
    def _batch_query(self, query) -> Generator[List[Any], None, None]:
        """Process query in batches."""
        offset = 0
        while True:
            batch = query.offset(offset).limit(self.config.batch_size).all()
            if not batch:
                break
            logger.debug(f"Retrieved batch of {len(batch)} records at offset {offset}")
            yield batch
            offset += self.config.batch_size
            
    def _import_memory(
        self,
        memory: Any,
        memory_type: str,
        model_name: str
    ) -> MemoryMetadata:
        """
        Import a single memory.
        
        Args:
            memory: Memory model instance
            memory_type: Type of memory
            model_name: Name of the model
            
        Returns:
            MemoryMetadata instance
            
        Raises:
            ValueError: If memory validation fails
        """
        # Validate memory
        if self.config.validate:
            self._validate_memory(memory, model_name)
            
        # Get step number based on model type
        step_number = getattr(memory, 'step_number', None)
        if step_number is None:
            logger.warning(f"No step_number found for {model_name} with id {getattr(memory, 'id', 'unknown')}")
            return None
            
        # Create memory metadata
        metadata = {
            'type': memory_type,
            'step_number': step_number,
            'agent_id': getattr(memory, 'agent_id', None),
            'position': {
                'x': getattr(memory, 'position_x', 0.0),
                'y': getattr(memory, 'position_y', 0.0)
            }
        }
        
        # Add model-specific metadata
        if model_name == 'AgentStateModel':
            metadata.update({
                'resource_level': getattr(memory, 'resource_level', 0.0),
                'current_health': getattr(memory, 'current_health', 0.0),
                'is_defending': getattr(memory, 'is_defending', False),
                'total_reward': getattr(memory, 'total_reward', 0.0),
                'age': getattr(memory, 'age', 0)
            })
        elif model_name == 'ActionModel':
            metadata.update({
                'action_type': getattr(memory, 'action_type', 'unknown'),
                'action_target_id': getattr(memory, 'action_target_id', None),
                'resources_before': getattr(memory, 'resources_before', 0.0),
                'resources_after': getattr(memory, 'resources_after', 0.0),
                'reward': getattr(memory, 'reward', 0.0)
            })
        elif model_name == 'SocialInteractionModel':
            metadata.update({
                'interaction_type': getattr(memory, 'interaction_type', 'unknown'),
                'subtype': getattr(memory, 'subtype', None),
                'outcome': getattr(memory, 'outcome', 'unknown'),
                'resources_transferred': getattr(memory, 'resources_transferred', 0.0),
                'distance': getattr(memory, 'distance', 0.0)
            })
            
        # Determine memory tier
        tiering_context = TieringContext(
            step_number=step_number,
            current_step=self.config.total_steps,  # Use total_steps as current_step
            total_steps=self.config.total_steps,
            importance_score=getattr(memory, 'importance_score', None),
            metadata=metadata
        )
        tier = self.tiering_strategy.determine_tier(tiering_context)
        
        # Get the correct ID field based on model type and ensure uniqueness
        memory_id = None
        if model_name == 'ActionModel':
            memory_id = f"action_{getattr(memory, 'action_id', None)}_step_{getattr(memory, 'step_number', 0)}"
        elif model_name == 'SocialInteractionModel':
            memory_id = f"interaction_{getattr(memory, 'interaction_id', None)}_step_{getattr(memory, 'step_number', 0)}"
        elif model_name == 'AgentStateModel':
            memory_id = f"state_{getattr(memory, 'id', None)}_step_{getattr(memory, 'step_number', 0)}"
            
        if memory_id is None:
            logger.warning(f"Could not find ID for {model_name} memory")
            return None
            
        # Create memory metadata
        return MemoryMetadata(
            memory_id=memory_id,
            agent_id=getattr(memory, 'agent_id', None),
            memory_type=memory_type,
            step_number=step_number,
            tier=tier,
            metadata=metadata,
            created_at=str(getattr(memory, 'timestamp', None) or datetime.now()),
            updated_at=str(getattr(memory, 'timestamp', None) or datetime.now())
        )
        
    def _validate_memory(self, memory: Any, model_name: str):
        """
        Validate a memory.
        
        Args:
            memory: Memory model instance
            model_name: Name of the model
            
        Raises:
            ValueError: If validation fails
        """
        # Mapping of model names to their ID fields
        id_field_mapping = {
            "ActionModel": "action_id",
            "SocialInteractionModel": "interaction_id"
        }
        
        # Determine the correct ID field for the model
        id_field = id_field_mapping.get(model_name, "id")
        
        # Validate the presence of the ID field
        if not getattr(memory, id_field, None):
            raise ValueError(f"{model_name} must have a valid {id_field}")
            
        if not memory.agent_id:
            raise ValueError(f"{model_name} must have an agent ID")
            
        if not hasattr(memory, 'step_number'):
            raise ValueError(f"{model_name} must have a step number")
            
    def _extract_memory_metadata(self, memory: Any) -> Dict[str, Any]:
        """
        Extract metadata from a memory.
        
        Args:
            memory: Memory model instance
            
        Returns:
            Dictionary of memory metadata
        """
        metadata = {}
        
        # Extract common fields
        for field in ['type', 'status', 'properties', 'settings']:
            if hasattr(memory, field):
                metadata[field] = getattr(memory, field)
                
        # Extract model-specific fields
        if hasattr(memory, 'action_type'):
            metadata['action_type'] = memory.action_type
        if hasattr(memory, 'interaction_type'):
            metadata['interaction_type'] = memory.interaction_type
            
        return metadata
        
    def _handle_import_error(
        self,
        error: Exception,
        agent_id: int,
        model_name: str
    ):
        """
        Handle memory import error based on configuration.
        
        Args:
            error: The error that occurred
            agent_id: ID of the agent
            model_name: Name of the model
        """
        error_msg = f"Error importing {model_name} for agent {agent_id}: {str(error)}"
        
        if self.config.error_handling == "fail":
            raise ValueError(error_msg)
        elif self.config.error_handling == "log":
            logger.error(error_msg)
        # Skip mode just continues without raising or logging 