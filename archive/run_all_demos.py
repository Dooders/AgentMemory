"""
Run all demos script for Agent Memory System

This script runs all demo scripts in sequence to showcase the full capabilities
of the Agent Memory System.
"""

import importlib
import logging
import sys
from typing import List, Callable

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def run_demos(demo_numbers: List[int] = None) -> None:
    """Run the specified demos or all demos if none specified.
    
    Args:
        demo_numbers: Optional list of demo numbers to run, runs all if None
    """
    # Available demos
    demos = {
        1: "01_basic_memory_operations",
        2: "02_memory_retrieval",
        3: "03_memory_tiers_compression",
        4: "04_memory_hooks_integration",
        5: "05_performance_error_handling",
        6: "06_search_capabilities",  # Add the new demo
    }
    
    # If no specific demos requested, run all
    if demo_numbers is None:
        demo_numbers = sorted(demos.keys())
    
    logger.info("=" * 80)
    logger.info(f"RUNNING {'SELECTED' if len(demo_numbers) < len(demos) else 'ALL'} DEMOS")
    logger.info("=" * 80)
    
    for demo_num in demo_numbers:
        if demo_num not in demos:
            logger.warning(f"Demo {demo_num} not found. Skipping.")
            continue
        
        demo_module_name = demos[demo_num]
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING DEMO {demo_num}: {demo_module_name}")
        logger.info("=" * 80 + "\n")
        
        try:
            # Import the demo module
            demo_module = importlib.import_module(f"demos.{demo_module_name}")
            
            # Run the demo
            if hasattr(demo_module, "run_demo"):
                run_demo_func: Callable = getattr(demo_module, "run_demo")
                run_demo_func()
            else:
                logger.error(f"Demo {demo_num} doesn't have a run_demo function. Skipping.")
        
        except Exception as e:
            logger.error(f"Error running demo {demo_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL DEMOS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Check if specific demos are requested
    if len(sys.argv) > 1:
        try:
            selected_demos = [int(arg) for arg in sys.argv[1:]]
            run_demos(selected_demos)
        except ValueError:
            logger.error("Invalid demo number provided. Please use integer values.")
    else:
        # Run all demos
        run_demos() 