#!/usr/bin/env python
"""
Script to run the AgentFarm to Memory System converter.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from converter.config import DEFAULT_CONFIG
from converter.converter import from_agent_farm


def main():
    db_path = "data/simulation.db"
    output = "validation/memory_samples/agent_farm_memories.json"
    validate = False
    error_handling = "skip"
    use_mock_redis = True
    batch_size = 100
    tiering_strategy = "simple"
    log_file = None

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate default log filename if not provided
    if not log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"converter_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Set all SQLAlchemy loggers to WARNING level to suppress most logs
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    # Set all other loggers to DEBUG level
    for name in logging.root.manager.loggerDict:
        if not name.startswith("sqlalchemy"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    logger.info(f"Logging to file: {log_file}")

    # Prepare configuration
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "validate": validate,
            "error_handling": error_handling,
            "use_mock_redis": True,
            "batch_size": batch_size,
            "tiering_strategy_type": tiering_strategy,
        }
    )

    # Run converter
    logger.info(f"Converting database: {db_path}")
    memory_system = from_agent_farm(db_path, config)

    return memory_system


if __name__ == "__main__":
    memory_system = main()
    agent = memory_system.agents["nWpvyFJReoFD5Fnq7AEggt"]
    stats = agent.get_memory_statistics("nWpvyFJReoFD5Fnq7AEggt")
    print(f"STM count: {stats['tiers']['stm']['count']}")
    print(f"IM count: {stats['tiers']['im']['count']}")
    print(f"LTM count: {stats['tiers']['ltm']['count']}")
