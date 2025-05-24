"""
Database connection management for the AgentFarm DB to Memory System converter.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from .config import ConverterConfig
from data.models import (
    AgentModel,
    AgentStateModel,
    ActionModel,
    SocialInteractionModel,
    SimulationStepModel,
    Simulation,
    ExperimentModel,
    SimulationConfig,
    HealthIncident,
    LearningExperienceModel,
    ReproductionEventModel,
    ResourceModel
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and provides session management."""
    
    # Expose models
    AgentModel = AgentModel
    AgentStateModel = AgentStateModel
    ActionModel = ActionModel
    SocialInteractionModel = SocialInteractionModel
    SimulationStepModel = SimulationStepModel
    Simulation = Simulation
    ExperimentModel = ExperimentModel
    SimulationConfig = SimulationConfig
    HealthIncident = HealthIncident
    LearningExperienceModel = LearningExperienceModel
    ReproductionEventModel = ReproductionEventModel
    ResourceModel = ResourceModel
    
    def __init__(self, db_path: str, config: ConverterConfig):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database
            config: Converter configuration
        """
        self.db_path = db_path
        self.config = config
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker] = None
        
    def initialize(self) -> None:
        """Initialize the database connection and session factory."""
        try:
            # Handle in-memory database
            if self.db_path == 'sqlite:///:memory:':
                engine_url = 'sqlite:///:memory:'
            else:
                engine_url = f'sqlite:///{self.db_path}'
                
            self._engine = create_engine(
                engine_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            self._Session = sessionmaker(bind=self._engine)
            
            # Log database structure
            inspector = inspect(self._engine)
            tables = inspector.get_table_names()
            logger.info(f"Database tables: {tables}")
            
            for table in tables:
                columns = inspector.get_columns(table)
                logger.info(f"Table {table} columns: {[col['name'] for col in columns]}")
                
            logger.info(f"Database connection initialized for {self.db_path}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
            
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Yields:
            Session: SQLAlchemy session
            
        Raises:
            SQLAlchemyError: If session creation fails
        """
        if not self._Session:
            self.initialize()
            
        session = self._Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
            
    def validate_database(self) -> bool:
        """
        Validate the database schema and required tables.
        
        Returns:
            bool: True if validation passes, False otherwise
            
        Raises:
            ValueError: If validation fails and error_handling is 'fail'
        """
        if not self._engine:
            self.initialize()
            
        inspector = inspect(self._engine)
        required_tables = {
            'agents',
            'agent_states',
            'agent_actions',
            'social_interactions',
            'simulations'
        }
        
        existing_tables = set(inspector.get_table_names())
        missing_tables = required_tables - existing_tables
        
        if missing_tables:
            error_msg = f"Missing required tables: {missing_tables}"
            logger.error(error_msg)
            if self.config.error_handling == 'fail':
                raise ValueError(error_msg)
            return False
            
        # Validate table schemas
        for table in required_tables:
            columns = {col['name'] for col in inspector.get_columns(table)}
            if not columns:
                error_msg = f"Table {table} has no columns"
                logger.error(error_msg)
                if self.config.error_handling == 'fail':
                    raise ValueError(error_msg)
                return False
                
        logger.info("Database validation successful")
        return True
        
    def get_total_steps(self) -> int:
        """
        Get the total number of simulation steps.
        
        Returns:
            int: Total number of steps
            
        Raises:
            SQLAlchemyError: If query fails
        """
        try:
            with self.session() as session:
                result = session.query(func.max(SimulationStepModel.step_number)).scalar()
                return result or 0
        except OperationalError:
            # Table doesn't exist yet
            return 0
            
    def get_agent_count(self) -> int:
        """
        Get the total number of agents.
        
        Returns:
            int: Total number of agents
            
        Raises:
            SQLAlchemyError: If query fails
        """
        try:
            with self.session() as session:
                return session.query(func.count(AgentModel.agent_id)).scalar() or 0
        except OperationalError:
            # Table doesn't exist yet
            return 0
            
    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed") 