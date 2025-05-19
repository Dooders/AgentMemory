"""
Main converter module for importing AgentFarm data into a memory system.
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError

from memory.config import MemoryConfig
from memory.core import AgentMemorySystem

from .agent_import import AgentImporter, AgentMetadata
from .config import DEFAULT_CONFIG, ConverterConfig
from .db import DatabaseManager
from .memory_import import MemoryImporter, MemoryMetadata

logger = logging.getLogger(__name__)


def from_agent_farm(db_path: str, config: Optional[Dict] = None) -> AgentMemorySystem:
    """
    Import data from an AgentFarm SQLite database into a memory system.

    Args:
        db_path (str): Path to the AgentFarm SQLite database
        config (dict, optional): Configuration options for the import process

    Returns:
        AgentMemorySystem: Configured memory system with imported memories

    Raises:
        ValueError: If database validation fails or import verification fails (when error_handling='fail')
        SQLAlchemyError: If there are database connection issues
    """
    # Merge provided config with defaults
    merged_config = DEFAULT_CONFIG.copy()
    if config:
        merged_config.update(config)

    # Create configuration object
    converter_config = ConverterConfig(**merged_config)

    # Initialize database manager
    db_manager = DatabaseManager(db_path, converter_config)

    try:
        # Initialize database connection
        db_manager.initialize()

        # Validate database if required
        if converter_config.validate:
            if not db_manager.validate_database():
                if converter_config.error_handling == "fail":
                    raise ValueError("Database validation failed")
                logger.warning(
                    "Database validation failed, but continuing due to error_handling='skip'"
                )

        # Get total steps for tiering strategy
        total_steps = db_manager.get_total_steps()
        logger.info(f"Total simulation steps: {total_steps}")

        # Get agent count for progress tracking
        agent_count = db_manager.get_agent_count()
        logger.info(f"Total agents: {agent_count}")

        # Initialize importers
        agent_importer = AgentImporter(db_manager, converter_config)
        memory_importer = MemoryImporter(
            db_manager,
            converter_config,
            converter_config.tiering_strategy,
            converter_config.memory_type_mapper,
        )

        # Import agents
        logger.info("Starting agent import...")
        imported_agents = agent_importer.import_agents()
        logger.info(f"Successfully imported {len(imported_agents)} agents")

        # Import memories for each agent
        logger.info("Starting memory import...")
        all_memories: List[MemoryMetadata] = []
        for agent in imported_agents:
            try:
                agent_memories = memory_importer.import_memories(agent.agent_id)
                all_memories.extend(agent_memories)
                logger.info(
                    f"Imported {len(agent_memories)} memories for agent {agent.agent_id}"
                )
            except Exception as e:
                if converter_config.error_handling == "fail":
                    raise ValueError(
                        f"Failed to import memories for agent {agent.agent_id}: {e}"
                    )
                logger.error(
                    f"Failed to import memories for agent {agent.agent_id}: {e}"
                )
                continue

        logger.info(f"Successfully imported {len(all_memories)} total memories")

        # Create memory system configuration
        memory_config = MemoryConfig(
            use_mock_redis=True, logging_level="INFO"
        )

        # Create and configure memory system
        memory_system = AgentMemorySystem.get_instance(memory_config)

        logger.info("!!!!!!!!!!!!!!! Adding agents and their memories to the system... !!!!!!!!!!!!!!!")
        # Add agents and their memories to the system
        for agent in imported_agents:
            # Add agent's memories
            agent_memories = [m for m in all_memories if m.agent_id == agent.agent_id]
            for memory in agent_memories:
                memory_data = {
                    "memory_id": str(memory.memory_id),
                    "agent_id": str(memory.agent_id),
                    "type": memory.memory_type,
                    "step_number": memory.step_number,
                    "content": memory.metadata,
                    "metadata": {
                        "tier": memory.tier,
                        "created_at": memory.created_at,
                        "updated_at": memory.updated_at,
                    },
                }
                memory_system.add_memory(memory_data)

        # Verify import if required
        if converter_config.validate:
            # Check if all agents were imported
            if len(memory_system.agents) != len(imported_agents):
                if converter_config.error_handling == "fail":
                    raise ValueError("Import verification failed: agent count mismatch")
                logger.warning("Import verification failed: agent count mismatch")

            # Check if all memories were imported
            total_memories = sum(
                agent.stm_store.count(str(agent.agent_id)) + 
                agent.im_store.count(str(agent.agent_id)) + 
                agent.ltm_store.count()  # SQLiteLTMStore doesn't take agent_id
                for agent in memory_system.agents.values()
            )
            if total_memories != len(all_memories):
                if converter_config.error_handling == "fail":
                    raise ValueError(
                        "Import verification failed: memory count mismatch"
                    )
                logger.warning("Import verification failed: memory count mismatch")

        return memory_system

    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        db_manager.close()
