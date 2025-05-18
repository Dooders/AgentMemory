#!/usr/bin/env python
"""
Script to run the AgentFarm to Memory System converter.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from converter.config import DEFAULT_CONFIG
from converter.converter import from_agent_farm


def main():
    """Run the converter with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert AgentFarm database to Memory System"
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        default="data/simulation.db",
        type=str,
        help="Path to the AgentFarm SQLite database (default: data/simulation.db)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save the memory system JSON file (optional)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Enable validation (default: True)"
    )
    parser.add_argument(
        "--error-handling",
        choices=["skip", "fail", "log"],
        default="skip",
        help="Error handling mode (default: skip)",
    )
    parser.add_argument(
        "--use-mock-redis",
        action="store_true",
        help="Use MockRedis instead of real Redis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--tiering-strategy",
        choices=["step_based", "importance_aware"],
        default="step_based",
        help="Memory tiering strategy (default: step_based)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to save the log file (default: logs/converter_YYYY-MM-DD_HH-MM-SS.log)",
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate default log filename if not provided
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.log_file = log_dir / f"converter_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
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

    logger.info(f"Logging to file: {args.log_file}")

    try:
        # Prepare configuration
        config = DEFAULT_CONFIG.copy()
        config.update(
            {
                "validate": args.validate,
                "error_handling": args.error_handling,
                "use_mock_redis": args.use_mock_redis,
                "batch_size": args.batch_size,
                "tiering_strategy_type": args.tiering_strategy,
            }
        )

        # Run converter
        logger.info(f"Converting database: {args.db_path}")
        memory_system = from_agent_farm(args.db_path, config)
        logger.info("Conversion completed successfully")

        # Save to JSON if output path provided
        if args.output:
            output_path = Path(args.output)
            logger.info(f"Saving memory system to: {output_path}")
            if memory_system.save_to_json(str(output_path)):
                logger.info("Memory system saved successfully")
            else:
                logger.error("Failed to save memory system")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
