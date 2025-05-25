"""Serialization utilities for the agent memory system.

This module provides functions for serializing and deserializing memory
entries and related data structures for storage and transmission.
"""

import base64
import copy
import datetime
import json
import logging
import pickle
from typing import Any, Dict, List

from memory.embeddings.text_embeddings import TextEmbeddingEngine

logger = logging.getLogger(__name__)


class MemorySerializer:
    """Serialize and deserialize memory entries and related objects.

    This class provides methods for converting memory entries and their
    components to and from various serialized formats.
    """

    @staticmethod
    def serialize_memory(memory_entry: Dict[str, Any], format: str = "json") -> str:
        """Serialize a memory entry to a string.

        Args:
            memory_entry: Memory entry to serialize
            format: Format to use ("json" or "pickle")

        Returns:
            Serialized string representation

        Raises:
            ValueError: If serialization fails
        """
        if format == "json":
            return MemorySerializer.to_json(memory_entry)
        elif format == "pickle":
            return MemorySerializer.to_pickle(memory_entry)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    @staticmethod
    def deserialize_memory(data: str, format: str = "json") -> Dict[str, Any]:
        """Deserialize a memory entry from a string.

        Args:
            data: Serialized memory entry
            format: Format to use ("json" or "pickle")

        Returns:
            Deserialized memory entry

        Raises:
            ValueError: If deserialization fails
        """
        if format == "json":
            return MemorySerializer.from_json(data)
        elif format == "pickle":
            return MemorySerializer.from_pickle(data)
        else:
            raise ValueError(f"Unsupported deserialization format: {format}")

    @staticmethod
    def to_json(obj: Any) -> str:
        """Convert an object to a JSON string with special type handling.

        This method handles special types like datetime objects that are
        not natively serializable to JSON.

        Args:
            obj: Object to serialize

        Returns:
            JSON string representation

        Raises:
            ValueError: If serialization fails
        """
        try:
            return json.dumps(obj, cls=MemoryJSONEncoder)
        except Exception as e:
            logger.error("Failed to serialize to JSON: %s", str(e))
            raise ValueError(f"JSON serialization failed: {str(e)}")

    @staticmethod
    def from_json(json_str: str) -> Any:
        """Convert a JSON string to an object with special type handling.

        This method handles special types like datetime objects that are
        not natively deserializable from JSON.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Deserialized object

        Raises:
            ValueError: If deserialization fails
        """
        try:
            return json.loads(json_str, object_hook=memory_json_decoder)
        except Exception as e:
            logger.error("Failed to deserialize from JSON: %s", str(e))
            raise ValueError(f"JSON deserialization failed: {str(e)}")

    @staticmethod
    def to_pickle(obj: Any) -> str:
        """Convert an object to a base64-encoded pickle string.

        This method serializes an object using Python's pickle protocol
        and encodes the result as a base64 string for storage.

        Args:
            obj: Object to serialize

        Returns:
            Base64-encoded pickle string

        Raises:
            ValueError: If serialization fails
        """
        try:
            pickle_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return base64.b64encode(pickle_bytes).decode("ascii")
        except Exception as e:
            logger.error("Failed to serialize to pickle: %s", str(e))
            raise ValueError(f"Pickle serialization failed: {str(e)}")

    @staticmethod
    def from_pickle(pickle_str: str) -> Any:
        """Convert a base64-encoded pickle string to an object.

        This method decodes a base64 string and deserializes it using
        Python's pickle protocol.

        Args:
            pickle_str: Base64-encoded pickle string

        Returns:
            Deserialized object

        Raises:
            ValueError: If deserialization fails
        """
        try:
            pickle_bytes = base64.b64decode(pickle_str.encode("ascii"))
            return pickle.loads(pickle_bytes)
        except Exception as e:
            logger.error("Failed to deserialize from pickle: %s", str(e))
            raise ValueError(f"Pickle deserialization failed: {str(e)}")

    @staticmethod
    def serialize_vector(vector: List[float]) -> str:
        """Serialize a vector to a string.

        This method creates a compact string representation of a vector
        for efficient storage.

        Args:
            vector: List of float values representing a vector

        Returns:
            String representation of the vector

        Raises:
            ValueError: If serialization fails
        """
        try:
            return json.dumps(vector)
        except Exception as e:
            logger.error("Failed to serialize vector: %s", str(e))
            raise ValueError(f"Vector serialization failed: {str(e)}")

    @staticmethod
    def deserialize_vector(vector_str: str) -> List[float]:
        """Deserialize a vector from a string.

        This method converts a string representation back to a vector.

        Args:
            vector_str: String representation of a vector

        Returns:
            List of float values representing a vector

        Raises:
            ValueError: If deserialization fails
        """
        try:
            return json.loads(vector_str)
        except Exception as e:
            logger.error("Failed to deserialize vector: %s", str(e))
            raise ValueError(f"Vector deserialization failed: {str(e)}")


class MemoryJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for memory entries.

    This encoder handles special types like datetime objects and
    numpy arrays that are not natively serializable to JSON.
    """

    def default(self, obj):
        """Convert special types to JSON-serializable objects.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation of the object
        """
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}

        # Handle sets
        if isinstance(obj, set):
            return {"__type__": "set", "value": list(obj)}

        # Handle bytes
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "value": base64.b64encode(obj).decode("ascii")}

        # Handle numpy arrays if available
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return {"__type__": "ndarray", "value": obj.tolist()}
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass

        # Let the base class handle it or raise TypeError
        return super().default(obj)


def memory_json_decoder(obj):
    """Custom JSON decoder for memory entries.

    This decoder handles special types like datetime objects that
    were encoded by MemoryJSONEncoder.

    Args:
        obj: Object to decode

    Returns:
        Decoded object
    """
    if "__type__" in obj:
        obj_type = obj["__type__"]

        # Handle datetime objects
        if obj_type == "datetime":
            return datetime.datetime.fromisoformat(obj["value"])

        # Handle sets
        if obj_type == "set":
            return set(obj["value"])

        # Handle bytes
        if obj_type == "bytes":
            return base64.b64decode(obj["value"].encode("ascii"))

        # Handle numpy arrays if available
        if obj_type == "ndarray":
            try:
                import numpy as np

                return np.array(obj["value"])
            except ImportError:
                # Return as list if numpy is not available
                return obj["value"]

    # Return the object unchanged
    return obj


# Shorthand functions for external use
serialize_memory = MemorySerializer.serialize_memory
deserialize_memory = MemorySerializer.deserialize_memory
to_json = MemorySerializer.to_json
from_json = MemorySerializer.from_json


def load_memory_system_from_json(filepath: str, use_mock_redis: bool = False):
    """Load a memory system from a JSON file.

    Args:
        filepath: Path to the JSON file
        use_mock_redis: Whether to use MockRedis for Redis storage

    Returns:
        AgentMemorySystem instance or None if loading failed
    """
    import json
    import logging
    import os
    import traceback

    from memory.config import (
        MemoryConfig,
        RedisIMConfig,
        RedisSTMConfig,
        SQLiteLTMConfig,
    )
    from memory.core import AgentMemorySystem
    from memory.embeddings.vector_store import VectorStore
    from memory.schema import validate_memory_system_json

    logger = logging.getLogger(__name__)

    try:
        # Read from file
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate against schema
        if not validate_memory_system_json(data):
            logger.error(f"JSON file {filepath} does not conform to schema")
            return None

        # Create config with Redis mock settings if requested
        config_data = data.get("config", {})

        # Create clean config data without nested configs
        clean_config_data = {}
        for key, value in config_data.items():
            if key not in ["stm_config", "im_config", "ltm_config"]:
                clean_config_data[key] = value

        # Create config instance
        config = MemoryConfig(**clean_config_data)

        # Set up STM config
        if "stm_config" in config_data:
            stm_config_data = config_data["stm_config"]
            stm_config = RedisSTMConfig(**stm_config_data)
            if use_mock_redis:
                stm_config.use_mock = True
            stm_config.test_mode = True  # Enable test mode
            config.stm_config = stm_config

        # Set up IM config
        if "im_config" in config_data:
            im_config_data = config_data["im_config"]
            im_config = RedisIMConfig(**im_config_data)
            if use_mock_redis:
                im_config.use_mock = True
            im_config.test_mode = True  # Enable test mode
            config.im_config = im_config

        # Set up LTM config
        if "ltm_config" in config_data:
            ltm_config_data = config_data["ltm_config"]
            ltm_config = SQLiteLTMConfig(**ltm_config_data)
            ltm_config.test_mode = True  # Enable test mode
            config.ltm_config = ltm_config

        # Initialize vector store with dimensions from autoencoder config
        # Create Redis connection params dictionary directly instead of using connection_params attribute
        redis_connection = None
        if not use_mock_redis:
            redis_connection = {
                "host": config.stm_config.host,
                "port": config.stm_config.port,
                "db": config.stm_config.db,
                "password": config.stm_config.password if hasattr(config.stm_config, "password") else None
            }
            
        vector_store = VectorStore(
            redis_client=redis_connection,
            stm_dimension=config.autoencoder_config.stm_dim,
            im_dimension=config.autoencoder_config.im_dim,
            ltm_dimension=config.autoencoder_config.ltm_dim,
            namespace=config.stm_config.namespace,
        )
        # Create memory system
        memory_system = AgentMemorySystem(config)
        memory_system.vector_store = vector_store
        memory_system.embedding_engine = TextEmbeddingEngine(
            model_name="all-MiniLM-L6-v2"
        )

        # Load agents and their memories
        for agent_id, agent_data in data.get("agents", {}).items():
            # Get or create agent
            memory_agent = memory_system.get_memory_agent(agent_id)

            # Add memories
            for memory in agent_data.get("memories", []):
                # First check the top-level type
                memory_type = memory.get("type", "generic")

                # Also check metadata.memory_type as a fallback
                if memory_type == "generic" and "metadata" in memory:
                    metadata_type = memory["metadata"].get("memory_type")
                    if metadata_type in ["state", "interaction", "action"]:
                        memory_type = metadata_type
                        logger.debug(f"Using memory_type from metadata: {memory_type}")

                # Ensure both type fields match for consistency during loading
                memory["type"] = memory_type
                if "metadata" in memory:
                    memory["metadata"]["memory_type"] = memory_type
                    logger.debug(f"Synchronized memory type fields to {memory_type}")

                content = memory.get("content", {})
                step_number = memory.get("step_number", 0)
                priority = memory.get("metadata", {}).get("importance_score", 1.0)
                tier = memory.get("metadata", {}).get("current_tier", "stm")

                # Log memory being loaded for debugging
                logger.debug(
                    f"Loading memory of type {memory_type} for agent {agent_id}"
                )
                logger.debug(f"Memory content keys: {list(content.keys())}")
                logger.debug(f"Memory step: {step_number}, priority: {priority}")
                logger.debug(f"Memory ID: {memory.get('memory_id', 'unknown')}")
                logger.debug(f"Memory metadata: {memory.get('metadata', {})}")

                # Create a deep copy to avoid reference issues
                memory_copy = copy.deepcopy(memory)

                # Store memory in the appropriate store based on the tier
                if tier == "stm":
                    logger.debug(f"Storing memory in STM store with type {memory_type}")
                    memory_agent.stm_store.store(agent_id, memory_copy)
                elif tier == "im":
                    logger.debug(f"Storing memory in IM store with type {memory_type}")
                    memory_agent.im_store.store(agent_id, memory_copy)
                elif tier == "ltm":
                    logger.debug(f"Storing memory in LTM store with type {memory_type}")
                    memory_agent.ltm_store.store(memory_copy)
                else:
                    # Default to STM if tier is unknown
                    logger.warning(f"Unknown tier '{tier}', storing in STM")
                    logger.debug(f"Storing memory in STM store with type {memory_type}")
                    memory_agent.stm_store.store(agent_id, memory_copy)

                # Store memory vectors if embeddings exist
                if "embeddings" in memory_copy and memory_copy.get("embeddings"):
                    logger.info(f"Storing embeddings for memory {memory_copy.get('memory_id')}")
                    vector_store.store_memory_vectors(memory_copy)
                # Generate embeddings if the memory has content but no embeddings
                #! Intended to be temporary, until embeddings are automatically generated when the memory is added
                elif "content" in memory_copy and memory_copy.get("content"):
                    try:
                        logger.info(f"Generating embeddings for memory {memory_copy.get('memory_id')}")
                        content = memory_copy.get("content", {})
                        
                        # Generate embeddings using the embedding engine
                        full_vector = memory_system.embedding_engine.encode_stm(content)
                        
                        # Add embeddings to memory
                        memory_copy["embeddings"] = {
                            "full_vector": full_vector
                        }
                        
                        # Store the vector
                        
                        
                        # Determine the tier for the memory
                        metadata = memory_copy.get("metadata", {})
                        tier = metadata.get("current_tier") or metadata.get("tier")
                        if tier == "stm":
                            vector_store.store_memory_vectors(memory_copy, tier="stm")
                            memory_agent.stm_store.store(agent_id, memory_copy)
                        elif tier == "im":
                            vector_store.store_memory_vectors(memory_copy, tier="im")
                            memory_agent.im_store.store(agent_id, memory_copy)
                        elif tier == "ltm":
                            vector_store.store_memory_vectors(memory_copy, tier="ltm")
                            memory_agent.ltm_store.store(memory_copy)
                        else:
                            logger.warning(f"Unknown tier '{tier}' for memory {memory_copy.get('memory_id')}")
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings for memory {memory_copy.get('memory_id')}: {e}")
        logger.info(f"Memory system loaded from {filepath}")
        return memory_system

    except Exception as e:
        logger.error(f"Failed to load memory system from {filepath}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def save_memory_system_to_json(memory_system, filepath: str) -> bool:
    """Save a memory system to a JSON file.

    Args:
        memory_system: The AgentMemorySystem instance to save
        filepath: Path to save the JSON file

    Returns:
        True if saving was successful
    """
    import json
    import logging
    import os
    import time
    import traceback
    import uuid

    from memory.schema import validate_memory_system_json

    logger = logging.getLogger(__name__)

    try:
        # Create the data structure to save with proper config serialization
        config_dict = {}
        for key, value in memory_system.config.__dict__.items():
            # Skip complex objects that aren't JSON serializable
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                config_dict[key] = value

        # Create the data structure to save
        data = {"config": config_dict, "agents": {}}

        # Save agent data
        for agent_id, agent in memory_system.agents.items():
            # Get all memories from different tiers
            stm_memories = []
            im_memories = []
            ltm_memories = []

            try:
                stm_memories = agent.stm_store.get_all(agent_id)
                logger.info(
                    f"Retrieved {len(stm_memories)} STM memories for agent {agent_id}"
                )
            except Exception as e:
                logger.warning(f"Could not get STM memories for agent {agent_id}: {e}")

            try:
                im_memories = agent.im_store.get_all(agent_id)
                logger.info(
                    f"Retrieved {len(im_memories)} IM memories for agent {agent_id}"
                )
            except Exception as e:
                logger.warning(f"Could not get IM memories for agent {agent_id}: {e}")

            try:
                ltm_memories = agent.ltm_store.get_all(agent_id)
                logger.info(
                    f"Retrieved {len(ltm_memories)} LTM memories for agent {agent_id}"
                )
            except Exception as e:
                logger.warning(f"Could not get LTM memories for agent {agent_id}: {e}")

            # Combine all memories
            all_memories = stm_memories + im_memories + ltm_memories
            logger.info(f"Total memories for agent {agent_id}: {len(all_memories)}")

            # Clean up non-serializable objects in memories
            clean_memories = []
            for i, memory in enumerate(all_memories):
                # Make a copy of the memory to avoid modifying the original
                clean_memory = {}
                for k, v in memory.items():
                    # Skip non-serializable embeddings
                    if k == "embeddings":
                        clean_memory[k] = {}
                        for embed_key, embed_val in v.items():
                            # Convert numpy arrays to lists if needed
                            if hasattr(embed_val, "tolist"):
                                clean_memory[k][embed_key] = embed_val.tolist()
                            elif (
                                isinstance(
                                    embed_val, (list, dict, str, int, float, bool)
                                )
                                or embed_val is None
                            ):
                                clean_memory[k][embed_key] = embed_val
                    else:
                        clean_memory[k] = v

                # Ensure all required fields for schema validation are present
                if "memory_id" not in clean_memory or clean_memory["memory_id"] is None:
                    clean_memory["memory_id"] = str(f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}")
                    logger.debug(f"Added memory_id to memory {i}")

                if "agent_id" not in clean_memory or clean_memory["agent_id"] is None:
                    clean_memory["agent_id"] = str(agent_id)
                    logger.debug(f"Added agent_id to memory {i}")

                if "content" not in clean_memory:
                    clean_memory["content"] = {}
                    logger.debug(f"Added empty content to memory {i}")

                # Ensure the memory type is set correctly and consistently
                memory_type = memory.get("type", "generic")
                if memory_type is None:
                    memory_type = "generic"

                # If type is generic, try to get it from metadata
                if memory_type == "generic" and "metadata" in memory:
                    metadata_type = memory.get("metadata", {}).get("memory_type")
                    if metadata_type in ["state", "interaction", "action"]:
                        memory_type = metadata_type
                        logger.debug(
                            f"Using memory_type from metadata: {metadata_type} for memory {i}"
                        )

                # Always ensure both top-level type and metadata.memory_type are consistent
                clean_memory["type"] = str(memory_type)
                logger.debug(f"Set memory type to {memory_type} for memory {i}")

                # Ensure metadata is present with all required fields
                if "metadata" not in clean_memory:
                    clean_memory["metadata"] = {}
                    logger.debug(f"Added empty metadata to memory {i}")

                metadata = clean_memory["metadata"]
                if "creation_time" not in metadata:
                    metadata["creation_time"] = int(time.time())

                if "last_access_time" not in metadata:
                    metadata["last_access_time"] = int(time.time())

                if "importance_score" not in metadata:
                    metadata["importance_score"] = float(memory.get("priority", 1.0))

                if "retrieval_count" not in metadata:
                    metadata["retrieval_count"] = 0

                if "current_tier" not in metadata or metadata["current_tier"] is None:
                    metadata["current_tier"] = "stm"

                # Ensure memory_type in metadata matches the top-level type
                metadata["memory_type"] = str(memory_type)
                logger.debug(
                    f"Set metadata.memory_type to {memory_type} for memory {i}"
                )

                if "step_number" not in clean_memory:
                    clean_memory["step_number"] = 0

                if "timestamp" not in clean_memory:
                    clean_memory["timestamp"] = int(time.time())

                # Ensure embeddings structure is valid if present
                if "embeddings" not in clean_memory:
                    clean_memory["embeddings"] = {}

                # Log the memory structure after preparation
                logger.debug(f"Memory {i} keys: {list(clean_memory.keys())}")
                logger.debug(
                    f"Memory {i} metadata keys: {list(clean_memory.get('metadata', {}).keys())}"
                )

                # Final validation of required string fields
                required_string_fields = {
                    "memory_id": clean_memory["memory_id"],
                    "agent_id": clean_memory["agent_id"],
                    "type": clean_memory["type"],
                    "metadata.memory_type": clean_memory["metadata"]["memory_type"],
                    "metadata.current_tier": clean_memory["metadata"]["current_tier"]
                }

                for field, value in required_string_fields.items():
                    if value is None:
                        logger.error(f"Required string field {field} is None in memory {i}")
                        return False
                    if not isinstance(value, str):
                        logger.error(f"Required string field {field} is not a string in memory {i}")
                        return False

                clean_memories.append(clean_memory)

            logger.info(f"Prepared {len(clean_memories)} memories for agent {agent_id}")
            data["agents"][agent_id] = {
                "agent_id": agent_id,
                "memories": clean_memories,
            }

        # Log a summary of the data structure
        logger.info(f"Data keys: {list(data.keys())}")
        logger.info(f"Number of agents: {len(data.get('agents', {}))}")
        for a_id, a_data in data.get("agents", {}).items():
            logger.info(f"Agent {a_id} has {len(a_data.get('memories', []))} memories")

        # Validate against schema
        logger.info("Validating against schema...")
        if not validate_memory_system_json(data):
            logger.error("Generated JSON does not conform to schema")
            return False

        # Check if the directory exists
        dir_path = os.path.dirname(os.path.abspath(filepath))

        # Ensure directory exists
        try:
            # Only create directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Memory system saved to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save memory system to {filepath}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
