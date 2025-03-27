"""Factory module for creating memory store instances.

This module provides a factory pattern implementation for instantiating
different memory store backends based on configuration.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from agent_memory.config import (
    BaseConfig,
    RedisIMConfig,
    RedisSTMConfig,
    SQLiteLTMConfig,
)
from agent_memory.storage.base import BaseMemoryStore
from agent_memory.storage.models import IMMemoryEntry, LTMMemoryEntry, STMMemoryEntry
from agent_memory.storage.redis_im import RedisIMStore
from agent_memory.storage.redis_stm import RedisSTMStore
from agent_memory.storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)

# Type variable for memory stores
T = TypeVar("T", STMMemoryEntry, IMMemoryEntry, LTMMemoryEntry)


class MemoryStoreFactory:
    """Factory for creating memory store instances.
    
    This class provides methods to create and configure memory store instances
    based on the specified store type and configuration.
    """
    
    @staticmethod
    def create_store(
        store_type: str,
        agent_id: str,
        config: BaseConfig
    ) -> BaseMemoryStore[Any]:
        """Create a memory store instance based on the specified type and configuration.
        
        Args:
            store_type: Type of memory store to create (STM, IM, LTM)
            agent_id: ID of the agent
            config: Configuration for the memory store
            
        Returns:
            Configured memory store instance
            
        Raises:
            ValueError: If an unsupported store type or configuration is provided
        """
        store_type = store_type.upper()
        
        if store_type == "STM":
            if not isinstance(config, RedisSTMConfig):
                raise ValueError(f"STM store requires RedisSTMConfig, got {type(config).__name__}")
            return cast(BaseMemoryStore[Any], RedisSTMStore(agent_id, config))
            
        elif store_type == "IM":
            if not isinstance(config, RedisIMConfig):
                raise ValueError(f"IM store requires RedisIMConfig, got {type(config).__name__}")
            return cast(BaseMemoryStore[Any], RedisIMStore(agent_id, config))
            
        elif store_type == "LTM":
            if not isinstance(config, SQLiteLTMConfig):
                raise ValueError(f"LTM store requires SQLiteLTMConfig, got {type(config).__name__}")
            return cast(BaseMemoryStore[Any], SQLiteLTMStore(agent_id, config))
            
        else:
            raise ValueError(f"Unsupported memory store type: {store_type}")

    @staticmethod
    def create_stm_store(
        agent_id: str,
        config: RedisSTMConfig
    ) -> BaseMemoryStore[STMMemoryEntry]:
        """Create a Short-Term Memory (STM) store.
        
        Args:
            agent_id: ID of the agent
            config: Configuration for the STM store
            
        Returns:
            Configured STM store instance
        """
        logger.debug(f"Creating STM store for agent {agent_id}")
        return RedisSTMStore(agent_id, config)

    @staticmethod
    def create_im_store(
        agent_id: str,
        config: RedisIMConfig
    ) -> BaseMemoryStore[IMMemoryEntry]:
        """Create an Intermediate Memory (IM) store.
        
        Args:
            agent_id: ID of the agent
            config: Configuration for the IM store
            
        Returns:
            Configured IM store instance
        """
        logger.debug(f"Creating IM store for agent {agent_id}")
        return RedisIMStore(agent_id, config)

    @staticmethod
    def create_ltm_store(
        agent_id: str,
        config: SQLiteLTMConfig
    ) -> BaseMemoryStore[LTMMemoryEntry]:
        """Create a Long-Term Memory (LTM) store.
        
        Args:
            agent_id: ID of the agent
            config: Configuration for the LTM store
            
        Returns:
            Configured LTM store instance
        """
        logger.debug(f"Creating LTM store for agent {agent_id}")
        return SQLiteLTMStore(agent_id, config) 