#!/usr/bin/env python
"""Diagnostic script to debug memory transitions between tiers."""

import json
import logging
import time
from typing import Dict, Any

from memory.config import MemoryConfig
from memory.memory_agent import MemoryAgent

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("memory_debug")


def check_memory_content(memory: Dict[str, Any], stage: str) -> None:
    """Check memory content and log the structure."""
    logger.info(f"Memory at {stage}:")
    logger.info(f"  ID: {memory.get('memory_id')}")
    logger.info(f"  Step: {memory.get('step_number')}")
    
    # Check for content field
    if "content" in memory:
        logger.info(f"  Content field exists: {bool(memory['content'])}")
        logger.info(f"  Content type: {type(memory['content'])}")
        if isinstance(memory['content'], dict):
            logger.info(f"  Content keys: {list(memory['content'].keys())}")
            logger.info(f"  Content sample: {json.dumps(memory['content'], indent=2)[:100]}...")
    else:
        logger.info("  Content field MISSING")
    
    # Check for deprecated contents field
    if "contents" in memory:
        logger.warning(f"  DEPRECATED 'contents' field exists: {bool(memory['contents'])}")
    
    # Check metadata
    if "metadata" in memory:
        logger.info(f"  Metadata: {json.dumps(memory['metadata'], indent=2)}")
    
    logger.info("-" * 50)


def debug_memory_system():
    """Run a memory debug session."""
    config = MemoryConfig()
    agent_id = f"debug_agent_{int(time.time())}"
    memory_agent = MemoryAgent(agent_id, config)
    
    logger.info(f"Created debug agent: {agent_id}")
    
    # Create and store test memory (STM)
    test_data = {"position": [10, 20], "health": 0.8, "inventory": ["map", "key"]}
    memory_agent.store_state(test_data, 1, 0.5)
    
    # Get the memory from STM to check structure
    stm_memories = memory_agent.stm_store.get_all(agent_id)
    if stm_memories:
        check_memory_content(stm_memories[0], "STM")
        original_memory_id = stm_memories[0]["memory_id"]
    else:
        logger.error("Failed to store memory in STM")
        return
    
    # Force transition to IM
    logger.info("Forcing transition to IM...")
    memory_agent.config.stm_config.memory_limit = 0  # Force transition
    memory_agent._check_memory_transition()
    
    # Check if memory made it to IM
    im_memories = memory_agent.im_store.get_all(agent_id)
    if im_memories:
        # Find the same memory
        for memory in im_memories:
            if memory["memory_id"] == original_memory_id:
                check_memory_content(memory, "IM")
                break
    else:
        logger.error("Memory not found in IM after transition")
    
    # Force transition to LTM
    logger.info("Forcing transition to LTM...")
    memory_agent.config.im_config.memory_limit = 0  # Force transition
    memory_agent._check_memory_transition()
    
    # Check if memory made it to LTM
    ltm_memory = memory_agent.ltm_store.get(original_memory_id)
    if ltm_memory:
        check_memory_content(ltm_memory, "LTM")
    else:
        logger.error(f"Memory not found in LTM after transition: {original_memory_id}")
    
    # Check database directly
    from tests.check_db import check_database
    logger.info("Checking database content:")
    check_database()


if __name__ == "__main__":
    debug_memory_system() 