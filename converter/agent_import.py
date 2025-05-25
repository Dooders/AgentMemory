"""
Agent import system for the AgentFarm DB to Memory System converter.

This module provides functionality to import agents from the AgentFarm database into the memory system.
It handles the conversion of database records into memory system compatible metadata, including:

- Agent identification and basic information
- Position and state data
- Resource and health metrics
- Genome and generation tracking
- Action weights and behavioral data

The system supports multiple import modes:
- Full: Import all agents from the database
- Incremental: Import only new or modified agents
- Selective: Import specific agents by ID

Features:
- Batch processing for efficient memory usage
- Configurable error handling (fail/log/skip)
- Validation of required fields
- Metadata preservation and transformation
- Flexible query filtering

Example:
    ```python
    from converter.agent_import import AgentImporter
    from converter.config import ConverterConfig
    from converter.db import DatabaseManager

    config = ConverterConfig(batch_size=100, validate=True)
    db_manager = DatabaseManager()
    importer = AgentImporter(db_manager, config)
    agents = importer.import_agents()
    ```

Note:
    The system assumes the presence of specific fields in the database records
    and will raise validation errors if required fields are missing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy.orm import Query, Session

from .config import ConverterConfig
from .db import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Metadata for an imported agent."""

    agent_id: str
    name: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class AgentImporter:
    """
    Handles the import of agents from AgentFarm database to memory system.

    This class manages the process of importing agents, including validation,
    metadata preservation, and error handling.
    """

    def __init__(self, db_manager: DatabaseManager, config: ConverterConfig) -> None:
        """
        Initialize the agent importer.

        Args:
            db_manager: Database manager instance
            config: Converter configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.db_manager = db_manager
        self.config = config

        # Validate configuration
        if config.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

        if config.error_handling not in ["fail", "log", "skip"]:
            raise ValueError("Invalid error handling mode")

        if config.import_mode not in ["full", "incremental", "selective"]:
            raise ValueError("Invalid import mode")

    def import_agents(self) -> List[AgentMetadata]:
        """
        Import agents from the database.

        Returns:
            List of imported agent metadata

        Raises:
            ValueError: If agent validation fails and error_handling is 'fail'
        """
        agents = []
        with self.db_manager.session() as session:
            # Get agent query based on import mode
            query = self._get_agent_query(session)

            # Process agents in batches
            for batch in self._batch_query(query):
                for agent in batch:
                    try:
                        agent_metadata = self._import_agent(agent)
                        agents.append(agent_metadata)
                    except Exception as e:
                        self._handle_import_error(e, agent)

        return agents[0:1]  #! TODO: Remove this

    def _get_agent_query(self, session: Session) -> Query:
        """Get the appropriate agent query based on import mode."""
        query = session.query(self.db_manager.AgentModel)

        if self.config.import_mode == "incremental":
            # Add incremental import conditions
            pass

        if self.config.selective_agents:
            query = query.filter(
                self.db_manager.AgentModel.agent_id.in_(self.config.selective_agents)
            )

        return query

    def _batch_query(self, query: Query) -> Generator[List[Any], None, None]:
        """Process query in batches."""
        offset = 0
        while True:
            batch = query.offset(offset).limit(self.config.batch_size).all()
            if not batch:
                break
            yield batch
            offset += self.config.batch_size

    def _import_agent(self, agent: Any) -> AgentMetadata:
        """
        Import a single agent.

        Args:
            agent: Agent model instance

        Returns:
            AgentMetadata instance

        Raises:
            ValueError: If agent validation fails
        """
        # Validate agent
        if self.config.validate:
            self._validate_agent(agent)

        # Create agent metadata
        metadata = AgentMetadata(
            agent_id=agent.agent_id,
            name=agent.name if agent.name is not None else f"Agent-{agent.agent_id}",
            metadata=self._extract_agent_metadata(agent),
            created_at=str(agent.birth_time),
            updated_at=str(agent.death_time or agent.birth_time),
        )

        return metadata

    def _validate_agent(self, agent: Any) -> None:
        """
        Validate an agent.

        Args:
            agent: Agent model instance

        Raises:
            ValueError: If validation fails
        """
        if not agent.agent_id:
            raise ValueError("Agent must have an ID")

    def _extract_agent_metadata(self, agent: Any) -> Dict[str, Any]:
        """
        Extract metadata from an agent.

        Args:
            agent: Agent model instance

        Returns:
            Dictionary of agent metadata
        """
        return {
            "type": agent.agent_type,
            "position": {"x": agent.position_x, "y": agent.position_y},
            "initial_resources": agent.initial_resources,
            "starting_health": agent.starting_health,
            "starvation_threshold": agent.starvation_threshold,
            "genome_id": agent.genome_id,
            "generation": agent.generation,
            "action_weights": agent.action_weights,
        }

    def _handle_import_error(self, error: Exception, agent: Any) -> None:
        """
        Handle agent import error based on configuration.

        Args:
            error: The error that occurred
            agent: The agent that caused the error
        """
        error_msg = f"Error importing agent {agent.agent_id}: {str(error)}"

        if self.config.error_handling == "fail":
            raise ValueError(error_msg)
        elif self.config.error_handling == "log":
            logger.error(error_msg)
        # Skip mode just continues without raising or logging
