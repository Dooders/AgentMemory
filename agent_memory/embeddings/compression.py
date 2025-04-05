"""Compression utilities for agent memory entries at different resolution levels.

This module provides functionality for compressing memory entries as they
move through the memory hierarchy, reducing detail while preserving
essential information.
"""

import base64
import copy
import json
import logging
import zlib
from typing import Any, Dict, Set

from agent_memory.config import AutoencoderConfig
from agent_memory.embeddings.utils import filter_dict_keys

logger = logging.getLogger(__name__)


class CompressionEngine:
    """Engine for compressing memory entries at different resolution levels.

    This class provides methods to compress memory entries as they move
    from Short-Term Memory (STM) to Intermediate Memory (IM) to Long-Term
    Memory (LTM), with increasing levels of compression.

    Attributes:
        config: Configuration for compression settings
    """

    def __init__(self, config: AutoencoderConfig):
        """Initialize the compression engine.

        Args:
            config: Configuration for the autoencoder and compression
        """
        self.config = config

        # Initialize compression level configurations
        self.level_configs = {
            # Level 0: No compression (STM)
            0: {
                "attribute_precision": None,  # Full precision
                "content_filter_keys": set(),  # Keep all keys
                "binary_compression": False,  # No binary compression
            },
            # Level 1: Moderate compression (IM)
            1: {
                "attribute_precision": 3,  # Numeric precision (decimal places)
                "content_filter_keys": {  # Keys to exclude from content
                    "raw_observation",
                    "full_observation",
                    "detailed_state",
                    "debug_info",
                    "internal_state",
                    "temporary_data",
                },
                "binary_compression": False,  # No binary compression
            },
            # Level 2: High compression (LTM)
            2: {
                "attribute_precision": 1,  # Low numeric precision
                "content_filter_keys": {  # Only keep essential keys
                    "raw_observation",
                    "full_observation",
                    "detailed_state",
                    "debug_info",
                    "internal_state",
                    "temporary_data",
                    "intermediate_results",
                    "processing_steps",
                    "trace",
                    "interaction_details",
                    "sensory_data",
                    "raw_action_data",
                },
                "binary_compression": True,  # Apply binary compression
            },
        }

    def compress(self, memory: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Compress a memory entry to a specified level.

        Args:
            memory: Memory entry to compress
            level: Compression level (0-2)

        Returns:
            Compressed memory entry copy
        """
        if level == 0:
            # No compression
            return copy.deepcopy(memory)

        compressed = copy.deepcopy(memory)
        compression_config = self.level_configs[level]

        # Apply attribute precision reduction
        if compression_config["attribute_precision"] is not None:
            self._reduce_numeric_precision(
                compressed, compression_config["attribute_precision"]
            )

        # Apply content filtering
        if compression_config["content_filter_keys"] and "content" in compressed:
            filtered_content = filter_dict_keys(
                compressed["content"], compression_config["content_filter_keys"]
            )
            compressed["content"] = filtered_content

        # Apply binary compression if configured
        if compression_config["binary_compression"] and "content" in compressed:
            # Convert to string and then compress
            content_str = json.dumps(compressed["content"])
            compressed_bytes = zlib.compress(content_str.encode("utf-8"))

            # Store compressed data as base64 string
            compressed["content"] = {
                "_compressed": base64.b64encode(compressed_bytes).decode("ascii"),
                "_compression_info": {
                    "algorithm": "zlib",
                    "original_size": len(content_str),
                    "compressed_size": len(compressed_bytes),
                },
            }

        # Update metadata
        if "metadata" in compressed:
            if not isinstance(compressed["metadata"], dict):
                compressed["metadata"] = {}
            compressed["metadata"]["compression_level"] = level

        return compressed

    def decompress(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to decompress a memory entry.

        Args:
            memory: Compressed memory entry

        Returns:
            Decompressed memory entry (best effort)
        """
        if not memory:
            return {}

        # If not compressed or unknown format, return as is
        if (
            "metadata" not in memory
            or not isinstance(memory["metadata"], dict)
            or "compression_level" not in memory["metadata"]
            or memory["metadata"]["compression_level"] == 0
        ):
            return copy.deepcopy(memory)

        # Check for binary compression
        if "content" in memory and isinstance(memory["content"], dict):
            content_data = memory["content"]
            if "_compressed" in content_data and "_compression_info" in content_data:
                try:
                    # Get compressed data
                    compressed_data = base64.b64decode(content_data["_compressed"])

                    # Decompress based on algorithm
                    compression_info = content_data["_compression_info"]
                    if compression_info.get("algorithm") == "zlib":
                        decompressed_str = zlib.decompress(compressed_data).decode(
                            "utf-8"
                        )
                        memory = copy.deepcopy(memory)
                        memory["content"] = json.loads(decompressed_str)
                except Exception as e:
                    logger.warning(f"Failed to decompress memory: {str(e)}")

        return copy.deepcopy(memory)

    def _filter_content_keys(
        self, content: Dict[str, Any], filter_keys: Set[str]
    ) -> Dict[str, Any]:
        """Remove specified keys from content dictionary.

        Args:
            content: Content dictionary to filter
            filter_keys: Set of keys to remove

        Returns:
            Filtered content dictionary
        """
        if not isinstance(content, dict):
            return content

        # Create a copy to avoid modifying the original
        result = copy.deepcopy(content)

        # Get keys to remove (cannot modify during iteration)
        keys_to_remove = [key for key in result if key in filter_keys]

        # Remove keys
        for key in keys_to_remove:
            del result[key]

        # Recursively filter nested dictionaries
        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = self._filter_content_keys(value, filter_keys)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._filter_content_keys(item, filter_keys)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]

        return result

    def _reduce_numeric_precision(self, obj: Any, precision: int) -> None:
        """Recursively reduce precision of numeric values.

        Args:
            obj: Object to process
            precision: Number of decimal places to keep
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float):
                        obj[key] = round(value, precision)
                else:
                    self._reduce_numeric_precision(value, precision)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float):
                        obj[i] = round(value, precision)
                else:
                    self._reduce_numeric_precision(value, precision)

    def _binary_compress(self, obj: Any) -> str:
        """Compress an object using binary compression.

        Args:
            obj: Object to compress

        Returns:
            Base64-encoded compressed string
        """
        import base64

        # Convert to JSON
        json_str = json.dumps(obj)

        # Compress with zlib
        compressed = zlib.compress(json_str.encode("utf-8"), level=9)

        # Encode as base64
        b64_str = base64.b64encode(compressed).decode("ascii")

        return b64_str

    def _binary_decompress(self, compressed_str: str) -> Any:
        """Decompress a binary compressed string.

        Args:
            compressed_str: Base64-encoded compressed string

        Returns:
            Decompressed object
        """
        import base64

        # Decode base64
        compressed = base64.b64decode(compressed_str)

        # Decompress with zlib
        json_str = zlib.decompress(compressed).decode("utf-8")

        # Parse JSON
        return json.loads(json_str)


class AbstractionEngine:
    """Engine for creating abstract representations of memory content.

    This class provides methods to extract key information and create
    summarized versions of memory entries, preserving semantic meaning
    while reducing detail.
    """

    def __init__(self):
        """Initialize the abstraction engine."""
        pass

    def abstract_memory(
        self, memory_entry: Dict[str, Any], level: int = 1
    ) -> Dict[str, Any]:
        """Create an abstracted version of a memory entry.

        Args:
            memory_entry: Memory entry to abstract
            level: Abstraction level (1=light, 2=heavy)

        Returns:
            Abstracted memory entry
        """
        # Make a copy to avoid modifying the original
        abstract_entry = copy.deepcopy(memory_entry)

        # Extract key fields
        if "content" in abstract_entry and isinstance(abstract_entry["content"], dict):
            if level == 1:
                abstract_entry["content"] = self._light_abstraction(
                    abstract_entry["content"]
                )
            else:
                abstract_entry["content"] = self._heavy_abstraction(
                    abstract_entry["content"]
                )

        # Update metadata
        if "metadata" in abstract_entry:
            abstract_entry["metadata"]["abstraction_level"] = level

        return abstract_entry

    def _light_abstraction(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a lightly abstracted version of memory contents.

        This preserves most semantic content but removes details.

        Args:
            content: Memory content to abstract

        Returns:
            Abstracted content
        """
        # Extract a subset of key fields that represent core information
        abstract = {}

        # Extract key aspects based on content type
        if "memory_type" in content:
            abstract["memory_type"] = content["memory_type"]

        # For state memories
        if content.get("memory_type") == "state":
            # Extract key state attributes
            for key in ["location", "status", "goals", "resources", "relationships"]:
                if key in content:
                    abstract[key] = content[key]

        # For action memories
        elif content.get("memory_type") == "action":
            # Extract key action attributes
            for key in ["action_type", "target", "outcome", "success"]:
                if key in content:
                    abstract[key] = content[key]

        # For interaction memories
        elif content.get("memory_type") == "interaction":
            # Extract key interaction attributes
            for key in ["interaction_type", "entities", "dialog", "outcome"]:
                if key in content:
                    abstract[key] = content[key]

        # Copy common fields relevant for all types
        for key in ["timestamp", "step_number", "agent_id", "importance"]:
            if key in content:
                abstract[key] = content[key]

        return abstract

    def _heavy_abstraction(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a heavily abstracted version of memory contents.

        This preserves only the most essential semantic content.

        Args:
            content: Memory content to abstract

        Returns:
            Heavily abstracted content
        """
        # Start with light abstraction
        abstract = self._light_abstraction(content)

        # Further reduce content based on type
        if "memory_type" in abstract:
            if abstract["memory_type"] == "state":
                # Keep only the most essential state information
                essential_keys = ["location", "status"]
                abstract = {
                    k: v
                    for k, v in abstract.items()
                    if k in essential_keys or k == "memory_type"
                }

            elif abstract["memory_type"] == "action":
                # Keep only the most essential action information
                essential_keys = ["action_type", "outcome"]
                abstract = {
                    k: v
                    for k, v in abstract.items()
                    if k in essential_keys or k == "memory_type"
                }

            elif abstract["memory_type"] == "interaction":
                # Keep only the most essential interaction information
                essential_keys = ["interaction_type", "outcome"]
                abstract = {
                    k: v
                    for k, v in abstract.items()
                    if k in essential_keys or k == "memory_type"
                }

        # Always keep timestamp and step info
        for key in ["timestamp", "step_number"]:
            if key in content:
                abstract[key] = content[key]

        return abstract
