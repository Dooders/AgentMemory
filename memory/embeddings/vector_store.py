"""Vector store implementations for efficient retrieval of agent memory embeddings.

This module provides vector storage and similarity search capabilities for
the agent memory system, enabling efficient retrieval of memories based on
semantic similarity.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from memory.embeddings.utils import cosine_similarity

logger = logging.getLogger(__name__)


class VectorIndex:
    """Base class for vector indexing and retrieval.

    This abstract base class defines the interface for vector storage
    and similarity search implementations.
    """

    def __init__(self):
        """Initialize the vector index."""
        pass

    def add(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector to the index.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector to store
            metadata: Optional metadata to associate with the vector

        Returns:
            True if the operation was successful
        """
        raise NotImplementedError("Subclasses must implement this method")

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_fn: Optional function to filter results

        Returns:
            List of search results with scores and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")

    def delete(self, id: str) -> bool:
        """Delete a vector from the index.

        Args:
            id: Unique identifier for the vector

        Returns:
            True if the operation was successful
        """
        raise NotImplementedError("Subclasses must implement this method")

    def clear(self) -> bool:
        """Clear all vectors from the index.

        Returns:
            True if the operation was successful
        """
        raise NotImplementedError("Subclasses must implement this method")


class InMemoryVectorIndex(VectorIndex):
    """Simple in-memory vector index for development and testing.

    This implementation stores vectors in memory and performs
    brute-force similarity search. It is not optimized for
    large-scale production use.
    """

    def __init__(self):
        """Initialize the in-memory vector index."""
        super().__init__()
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector to the index.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector to store
            metadata: Optional metadata to associate with the vector

        Returns:
            True if the operation was successful
        """
        try:
            self.vectors[id] = vector
            self.metadata[id] = metadata or {}
            return True
        except Exception as e:
            logger.error("Failed to add vector to index: %s", str(e))
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity.

        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_fn: Optional function to filter results

        Returns:
            List of search results with scores and metadata
        """
        try:
            query_array = np.array(query_vector)

            results = []
            for id, vector in self.vectors.items():
                # Skip if filtered out
                if filter_fn and not filter_fn(self.metadata.get(id, {})):
                    continue

                # Calculate cosine similarity
                similarity = cosine_similarity(query_array, np.array(vector))

                results.append(
                    {
                        "id": id,
                        "score": float(similarity),
                        "metadata": self.metadata.get(id, {}),
                    }
                )

            # Sort by score (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limit results
            return results[:limit]
        except Exception as e:
            logger.error("Vector search failed: %s", str(e))
            return []

    def delete(self, id: str) -> bool:
        """Delete a vector from the index.

        Args:
            id: Unique identifier for the vector

        Returns:
            True if the operation was successful
        """
        try:
            if id in self.vectors:
                del self.vectors[id]
                del self.metadata[id]
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete vector: %s", str(e))
            return False

    def clear(self) -> bool:
        """Clear all vectors from the index.

        Returns:
            True if the operation was successful
        """
        try:
            self.vectors.clear()
            self.metadata.clear()
            return True
        except Exception as e:
            logger.error("Failed to clear vector index: %s", str(e))
            return False


class RedisVectorIndex(VectorIndex):
    """Redis-based vector index using RediSearch.

    This implementation stores vectors in Redis and uses RediSearch
    for efficient vector similarity search at scale.

    Note: Requires Redis with the RediSearch module installed.
    """

    def __init__(
        self,
        redis_client,
        index_name: str,
        vector_field: str = "embedding",
        dimension: int = 384,
        distance_metric: str = "COSINE",
    ):
        """Initialize the Redis vector index.

        Args:
            redis_client: Redis client instance
            index_name: Name of the RediSearch index
            vector_field: Name of the vector field in the index
            dimension: Dimension of the stored vectors
            distance_metric: Distance metric for similarity search
        """
        super().__init__()
        self.redis = redis_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.dimension = dimension
        self.distance_metric = distance_metric

        # Cached index existence flag to avoid repeated checks
        self._index_exists = False

        # Create index if it doesn't exist
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure the Redis index exists, creating it if necessary."""
        if self._index_exists:
            return

        try:
            # Check if index exists
            indices = self.redis.execute_command("FT._LIST")
            # Convert both the index name and list items to strings for comparison
            index_name_str = str(self.index_name)
            indices_str = [
                str(idx) if isinstance(idx, bytes) else idx for idx in indices
            ]
            if index_name_str in indices_str:
                self._index_exists = True
                return

            # Create the index
            self.redis.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                f"{self.index_name}:",
                "SCHEMA",
                self.vector_field,
                "VECTOR",
                self.distance_metric,
                f"6",
                "DIM",
                self.dimension,
                "TYPE",
                "FLOAT32",
                "metadata",
                "TEXT",
                "timestamp",
                "NUMERIC",
                "SORTABLE",
            )
            self._index_exists = True
            logger.info("Created Redis vector index %s", self.index_name)
        except Exception as e:
            logger.error("Failed to create Redis vector index: %s", str(e))

    def add(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector to the Redis index.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector to store
            metadata: Optional metadata to associate with the vector

        Returns:
            True if the operation was successful
        """
        try:
            self._ensure_index()

            # Convert vector to byte representation
            vector_bytes = self._float_list_to_bytes(vector)

            # Prepare metadata
            metadata_dict = metadata or {}
            metadata_json = json.dumps(metadata_dict)

            # Store in Redis
            key = f"{self.index_name}:{id}"
            self.redis.hset(
                key,
                mapping={
                    self.vector_field: vector_bytes,
                    "metadata": metadata_json,
                    "timestamp": int(time.time()),
                },
            )

            return True
        except Exception as e:
            logger.error("Failed to add vector to Redis index: %s", str(e))
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the Redis index.

        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_fn: Optional function to filter results

        Returns:
            List of search results with scores and metadata
        """
        try:
            self._ensure_index()

            # Convert query vector to byte representation
            query_bytes = self._float_list_to_bytes(query_vector)

            # Build the search query
            query = f"*=>[KNN {limit} @{self.vector_field} $vector AS score]"

            # Execute search
            results = self.redis.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
                "PARAMS",
                "2",
                "vector",
                query_bytes,
                "SORTBY",
                "score",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "3",
                "score",
                "metadata",
                "timestamp",
            )

            # Process results
            processed_results = []
            num_results = results[0]
            for i in range(1, min(num_results * 2 + 1, len(results)), 2):
                key = results[i].decode()
                attrs = {}
                for j in range(0, len(results[i + 1]), 2):
                    if j < len(results[i + 1]):
                        attr_name = results[i + 1][j].decode()
                        attr_value = results[i + 1][j + 1]
                        if isinstance(attr_value, bytes):
                            attr_value = attr_value.decode()
                        attrs[attr_name] = attr_value

                # Extract ID from key
                id = key.split(":", 1)[1]

                # Parse metadata if present
                metadata = {}
                if "metadata" in attrs:
                    try:
                        metadata = json.loads(attrs["metadata"])
                    except:
                        pass

                # Skip if filtered out
                if filter_fn and not filter_fn(metadata):
                    continue

                processed_results.append(
                    {
                        "id": id,
                        "score": float(attrs.get("score", 0)),
                        "metadata": metadata,
                    }
                )

            return processed_results
        except Exception as e:
            logger.error("Redis vector search failed: %s", str(e))
            return []

    def _float_list_to_bytes(self, vector: List[float]) -> bytes:
        """Convert a list of floats to byte representation for Redis.

        Args:
            vector: List of float values

        Returns:
            Byte representation of the vector
        """
        import struct

        # Ensure vector has correct dimension
        if len(vector) != self.dimension:
            if len(vector) < self.dimension:
                vector = vector + [0.0] * (self.dimension - len(vector))
            else:
                vector = vector[: self.dimension]

        # Pack as binary
        return b"".join([struct.pack("f", x) for x in vector])

    def delete(self, id: str) -> bool:
        """Delete a vector from the Redis index.

        Args:
            id: Unique identifier for the vector

        Returns:
            True if the operation was successful
        """
        try:
            key = f"{self.index_name}:{id}"
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error("Failed to delete vector from Redis: %s", str(e))
            return False

    def clear(self) -> bool:
        """Clear all vectors from the Redis index.

        Returns:
            True if the operation was successful
        """
        try:
            # Get all keys with the index prefix
            pattern = f"{self.index_name}:*"
            cursor = 0

            while True:
                cursor, keys = self.redis.scan(cursor, pattern, 100)
                if keys:
                    self.redis.delete(*keys)

                if cursor == 0:
                    break

            # Drop and recreate the index
            try:
                self.redis.execute_command("FT.DROPINDEX", self.index_name)
            except:
                pass

            self._index_exists = False
            self._ensure_index()

            return True
        except Exception as e:
            logger.error("Failed to clear Redis vector index: %s", str(e))
            return False


# Import JSON here to avoid circular imports
import json


class VectorStore:
    """High-level interface for vector storage and retrieval operations.

    This class provides a unified interface for storing and retrieving
    vectors, handling different backends and compression levels.

    Attributes:
        stm_index: Vector index for Short-Term Memory
        im_index: Vector index for Intermediate Memory
        ltm_index: Vector index for Long-Term Memory
    """

    def __init__(
        self,
        redis_client=None,
        stm_dimension: int = 384,
        im_dimension: int = 128,
        ltm_dimension: int = 32,
        namespace: str = "memory",
    ):
        """Initialize the vector store.

        Args:
            redis_client: Redis client for Redis-based indices
            stm_dimension: Dimension of STM vectors
            im_dimension: Dimension of IM vectors
            ltm_dimension: Dimension of LTM vectors
            namespace: Namespace prefix for indices
        """
        self.redis_client = redis_client

        # Initialize vector indices based on available backends
        if redis_client:
            # Use Redis-based indices if Redis is available
            self.stm_index = RedisVectorIndex(
                redis_client, f"{namespace}:stm_vectors", dimension=stm_dimension
            )
            self.im_index = RedisVectorIndex(
                redis_client, f"{namespace}:im_vectors", dimension=im_dimension
            )
            self.ltm_index = RedisVectorIndex(
                redis_client, f"{namespace}:ltm_vectors", dimension=ltm_dimension
            )
        else:
            # Fall back to in-memory indices
            self.stm_index = InMemoryVectorIndex()
            self.im_index = InMemoryVectorIndex()
            self.ltm_index = InMemoryVectorIndex()

        logger.debug(
            "VectorStore initialized with %s backend",
            "Redis" if redis_client else "in-memory",
        )

    def store_memory_vectors(
        self, memory_entry: Dict[str, Any], tier: str = "stm"
    ) -> bool:
        """Store vectors for a memory entry in appropriate indices.

        Args:
            memory_entry: Memory entry with embeddings
            tier: Tier to store the vectors in ("stm", "im", or "ltm")

        Returns:
            True if storage was successful
        """
        memory_id = memory_entry.get("memory_id")
        if not memory_id:
            logger.error("Cannot store vectors: missing memory_id")
            return False

        embeddings = memory_entry.get("embeddings", {})
        metadata = memory_entry.get("metadata", {})
        
        # Include content data in the metadata to enable filtering on content.metadata fields
        if "content" in memory_entry and isinstance(memory_entry["content"], dict):
            metadata["content"] = memory_entry["content"]
            logger.debug(f"Including content in vector metadata for memory {memory_id}")

        def store_vector(
            index,
            memory_id: str,
            vector: List[float],
            metadata: Dict[str, Any],
            tier: str,
        ) -> bool:
            """
            Store a vector in the appropriate index.

            Args:
                index: Vector index to store in
                memory_id: Unique identifier for the memory
                vector: Vector to store
                metadata: Metadata to store
                tier: Tier to store the vector in ("stm", "im", or "ltm")

            Returns:
                True if storage was successful
            """
            try:
                logger.debug(f"Storing {tier.upper()} vector for memory {memory_id}")
                return index.add(memory_id, vector, metadata)
            except Exception as e:
                logger.error(
                    f"Failed to store {tier.upper()} vector for memory {memory_id}: {e}"
                )
                return False

        success = True

        # Store in appropriate index based on tier
        if tier == "stm":
            success = store_vector(
                self.stm_index, memory_id, embeddings["full_vector"], metadata, "stm"
            )
        elif tier == "im":
            #! TODO: Use compressed vector
            logger.debug(
                f"@@@@@@@@@@@@@@@@@@@@@@@@@ Storing IM vector for memory {memory_id}"
            )
            success = store_vector(
                self.im_index, memory_id, embeddings["full_vector"], metadata, "im"
            )
            logger.debug(
                f"@@@@@@@@@@@@@@@@@@@@@@@@@ Result of storing IM vector: {success}"
            )
        elif tier == "ltm":
            #! TODO: Use abstract vector
            success = store_vector(
                self.ltm_index, memory_id, embeddings["full_vector"], metadata, "ltm"
            )

        return success

    def find_similar_memories(
        self,
        query_vector: List[float],
        tier: str = "stm",
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find memories similar to the query vector.

        Args:
            query_vector: Vector to search for
            tier: Memory tier to search ("stm", "im", or "ltm")
            limit: Maximum number of results to return
            metadata_filter: Optional metadata constraints

        Returns:
            List of similar memories with scores
        """
        # Create filter function if metadata filter is provided
        filter_fn = None
        if metadata_filter:
            logger.debug(
                "Creating filter function for metadata filter: %s", metadata_filter
            )

            def filter_fn(metadata):
                logger.debug("Checking metadata: %s", metadata)
                unmatched_keys = []  # Initialize the list before using it
                for key, value in metadata_filter.items():
                    # Try direct match in top-level metadata
                    if key in metadata and metadata[key] == value:
                        logger.debug("Found direct match for %s: %s", key, value)
                        continue

                    # Special handling for 'type' field - also check 'memory_type'
                    if (
                        key == "type"
                        and "memory_type" in metadata
                        and metadata["memory_type"] == value
                    ):
                        logger.debug("Found match for type in memory_type: %s", value)
                        continue

                    # Try match in nested content.metadata
                    if "content" in metadata and isinstance(metadata["content"], dict):
                        content = metadata["content"]
                        if "metadata" in content and isinstance(
                            content["metadata"], dict
                        ):
                            content_metadata = content["metadata"]
                            if (
                                key in content_metadata
                                and content_metadata[key] == value
                            ):
                                logger.debug(
                                    "Found nested match for %s: %s in content.metadata",
                                    key,
                                    value,
                                )
                                continue

                    # No match found for this key
                    unmatched_keys.append((key, value))

                if unmatched_keys:
                    logger.debug(
                        "No matches found for the following keys and values: %s",
                        unmatched_keys,
                    )
                    return False
                else:
                    logger.debug("All filter criteria matched")
                    return True

        # Select the appropriate index based on tier
        if tier == "im":
            index = self.im_index
        elif tier == "ltm":
            index = self.ltm_index
        else:  # Default to STM
            index = self.stm_index

        # Perform the search
        return index.search(query_vector, limit, filter_fn)

    def delete_memory_vectors(self, memory_id: str) -> bool:
        """Delete vectors for a memory entry from all indices.

        Args:
            memory_id: Unique identifier for the memory

        Returns:
            True if deletion was successful
        """
        stm_success = self.stm_index.delete(memory_id)
        im_success = self.im_index.delete(memory_id)
        ltm_success = self.ltm_index.delete(memory_id)

        return stm_success and im_success and ltm_success

    def clear_all(self) -> bool:
        """Clear all vectors from all indices.

        Returns:
            True if clearing was successful
        """
        stm_success = self.stm_index.clear()
        im_success = self.im_index.clear()
        ltm_success = self.ltm_index.clear()

        return stm_success and im_success and ltm_success
