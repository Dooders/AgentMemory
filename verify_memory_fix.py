#!/usr/bin/env python
"""Verification script to confirm memory content preservation through memory tiers."""

import json
import logging
import time
import sys
from typing import Dict, Any

from memory.config import MemoryConfig
from memory.memory_agent import MemoryAgent

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("memory_verification")

def compare_memory_content(original_data: Dict[str, Any], memory: Dict[str, Any], stage: str) -> None:
    """Compare memory content with original data and log the comparison."""
    logger.info(f"\nMemory content comparison at {stage}:")
    
    if "content" not in memory:
        logger.error(f"FAIL: 'content' field missing in {stage} memory")
        return
    
    content = memory["content"]
    if isinstance(content, dict) and "_compressed" in content and "_compression_info" in content:
        # This is compressed content, decompress it first
        import base64
        import zlib
        try:
            compressed_data = base64.b64decode(content["_compressed"])
            decompressed_str = zlib.decompress(compressed_data).decode("utf-8")
            content = json.loads(decompressed_str)
            logger.info(f"Successfully decompressed content in {stage}")
        except Exception as e:
            logger.error(f"FAIL: Could not decompress content in {stage}: {e}")
            return
    
    # Compare original data with retrieved content
    original_keys = set(original_data.keys())
    content_keys = set(content.keys())
    
    # Check keys
    if original_keys == content_keys:
        logger.info(f"PASS: All keys preserved in {stage}")
    else:
        missing = original_keys - content_keys
        extra = content_keys - original_keys
        if missing:
            logger.warning(f"PARTIAL: Missing keys in {stage}: {missing}")
        if extra:
            logger.info(f"INFO: Extra keys in {stage}: {extra}")
    
    # Check values for common keys
    common_keys = original_keys.intersection(content_keys)
    all_values_match = True
    
    for key in common_keys:
        if original_data[key] != content[key]:
            logger.warning(f"FAIL: Value mismatch for key '{key}' in {stage}")
            logger.warning(f"  Original: {original_data[key]}")
            logger.warning(f"  Retrieved: {content[key]}")
            all_values_match = False
    
    if all_values_match and common_keys:
        logger.info(f"PASS: All values match for common keys in {stage}")
    
    # Overall result
    if original_keys == content_keys and all_values_match:
        logger.info(f"OVERALL: Content preserved perfectly in {stage}")
    else:
        logger.warning(f"OVERALL: Content partially preserved in {stage}")


def verify_memory_system():
    """Run a complete memory verification through all tiers."""
    config = MemoryConfig()
    agent_id = f"verify_agent_{int(time.time())}"
    memory_agent = MemoryAgent(agent_id, config)
    
    logger.info(f"Created verification agent: {agent_id}")
    
    # Create test data with various types of content
    test_data = {
        "position": [10, 20, 30],
        "health": 0.95,
        "inventory": ["map", "compass", "water bottle"],
        "status": "exploring",
        "notes": "Found interesting landmark at position",
        "nested": {
            "level1": {
                "level2": "deep value"
            }
        },
        "values": [1, 2, 3, 4, 5]
    }
    
    logger.info(f"Original test data: {json.dumps(test_data, indent=2)}")
    
    # 1. Store in STM
    memory_agent.store_state(test_data, 1, 0.7)
    logger.info("Stored memory in STM")
    
    # Get memory from STM to verify 
    stm_memories = memory_agent.stm_store.get_all(agent_id)
    if stm_memories:
        original_memory_id = stm_memories[0]["memory_id"]
        logger.info(f"Memory ID: {original_memory_id}")
        compare_memory_content(test_data, stm_memories[0], "STM")
    else:
        logger.error("Failed to store memory in STM")
        return
    
    # 2. Force transition to IM
    logger.info("\nForcing transition to IM...")
    memory_agent.config.stm_config.memory_limit = 0
    memory_agent._check_memory_transition()
    
    # Get memory from IM to verify
    im_memories = memory_agent.im_store.get_all(agent_id)
    if im_memories:
        im_memory = next((m for m in im_memories if m["memory_id"] == original_memory_id), None)
        if im_memory:
            compare_memory_content(test_data, im_memory, "IM")
        else:
            logger.error(f"Memory {original_memory_id} not found in IM")
    else:
        logger.error("No memories found in IM after transition")
        return
    
    # 3. Force transition to LTM
    logger.info("\nForcing transition to LTM...")
    memory_agent.config.im_config.memory_limit = 0
    memory_agent._check_memory_transition()
    
    # Get memory from LTM to verify
    ltm_memory = memory_agent.ltm_store.get(original_memory_id)
    if ltm_memory:
        compare_memory_content(test_data, ltm_memory, "LTM")
    else:
        logger.error(f"Memory {original_memory_id} not found in LTM")
        return
    
    # 4. Check database directly
    from check_db import check_database
    logger.info("\nFinal database check:")
    check_database(original_memory_id)
    
    logger.info("\nVerification complete!")


if __name__ == "__main__":
    verify_memory_system() 