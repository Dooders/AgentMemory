"""SQLite-based Long-Term Memory (LTM) storage for agent memory system.

This module provides a SQLite-based implementation of the Long-Term Memory
storage tier, designed for persistent, highly-compressed storage of agent memories
with comprehensive error handling and resilient operations.
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent_memory.config import SQLiteLTMConfig
from agent_memory.utils.error_handling import SQLitePermanentError, SQLiteTemporaryError

logger = logging.getLogger(__name__)


class SQLiteLTMStore:
    """SQLite-based storage for Long-Term Memory (LTM).

    This class provides storage operations for the high-compression,
    persistent agent memory tier using SQLite as the backing store.

    Attributes:
        agent_id: Unique identifier for the agent
        config: Configuration for LTM SQLite storage
        db_path: Path to the SQLite database
        table_prefix: Prefix for table names
        memory_table: Name of the main memory table
        embeddings_table: Name of the embeddings table
    """

    def __init__(self, agent_id: str, config: SQLiteLTMConfig):
        """Initialize the SQLite LTM store.

        Args:
            agent_id: ID of the agent
            config: Configuration for LTM SQLite storage
        """
        self.agent_id = agent_id
        self.config = config
        self.db_path = config.db_path
        self.table_prefix = config.table_prefix

        # Table names
        self.memory_table = f"{self.table_prefix}_memories"
        self.embeddings_table = f"{self.table_prefix}_embeddings"

        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Only try to create directory if there is a directory path
            os.makedirs(db_dir, exist_ok=True)

        # Initialize database tables if they don't exist
        self._init_database()

        logger.info(
            "Initialized SQLiteLTMStore for agent %s at %s", agent_id, self.db_path
        )

    @contextmanager
    def _get_connection(self):
        """Get a SQLite connection with proper error handling.

        Yields:
            SQLite connection

        Raises:
            SQLiteTemporaryError: For temporary errors (locks, timeouts)
            SQLitePermanentError: For permanent errors (corruption)
        """
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,  # 30 second timeout
                isolation_level="IMMEDIATE",  # Immediate transactions
            )
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Return dictionaries from queries
            conn.row_factory = sqlite3.Row

            yield conn

        except sqlite3.OperationalError as e:
            # Handle temporary errors (locks, timeouts)
            error_msg = str(e).lower()
            if "locked" in error_msg or "timeout" in error_msg:
                logger.warning("SQLite temporary error: %s", str(e))
                raise SQLiteTemporaryError(f"SQLite temporary error: {str(e)}")
            else:
                logger.error("SQLite operational error: %s", str(e))
                raise SQLitePermanentError(f"SQLite error: {str(e)}")

        except sqlite3.DatabaseError as e:
            # Handle corruption and other serious errors
            logger.error("SQLite database error: %s", str(e))
            raise SQLitePermanentError(f"SQLite database error: {str(e)}")

        except Exception as e:
            # Handle all other errors
            logger.error("Unexpected error with SQLite: %s", str(e))
            raise SQLitePermanentError(f"Unexpected SQLite error: {str(e)}")

        finally:
            if conn:
                conn.close()

    def _init_database(self):
        """Initialize the SQLite database schema if needed."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create memory table
                cursor.execute(
                    f"""
                CREATE TABLE IF NOT EXISTS {self.memory_table} (
                    memory_id TEXT,
                    agent_id TEXT NOT NULL,
                    step_number INTEGER,
                    timestamp INTEGER NOT NULL,
                    
                    content_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    
                    compression_level INTEGER DEFAULT 2,
                    importance_score REAL DEFAULT 0.0,
                    retrieval_count INTEGER DEFAULT 0,
                    memory_type TEXT,
                    
                    created_at INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, agent_id)
                )
                """
                )

                # Create indices for faster retrieval
                cursor.execute(
                    f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_agent_id 
                ON {self.memory_table} (agent_id)
                """
                )

                cursor.execute(
                    f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_step 
                ON {self.memory_table} (step_number)
                """
                )

                cursor.execute(
                    f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_type 
                ON {self.memory_table} (memory_type)
                """
                )

                cursor.execute(
                    f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_importance 
                ON {self.memory_table} (importance_score)
                """
                )

                cursor.execute(
                    f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_timestamp 
                ON {self.memory_table} (timestamp)
                """
                )

                # Vector embeddings table
                cursor.execute(
                    f"""
                CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                    memory_id TEXT,
                    agent_id TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    
                    PRIMARY KEY (memory_id, agent_id),
                    FOREIGN KEY (memory_id, agent_id) REFERENCES {self.memory_table} (memory_id, agent_id) 
                    ON DELETE CASCADE
                )
                """
                )

                conn.commit()

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            # These are already properly handled and logged
            raise
        except Exception as e:
            logger.error("Error initializing SQLite database: %s", str(e))
            raise SQLitePermanentError(
                f"Failed to initialize SQLite database: {str(e)}"
            )

    def store(self, memory_entry: Dict[str, Any]) -> bool:
        """Store a memory entry in the LTM.

        Args:
            memory_entry: Memory entry to store

        Returns:
            True if the operation succeeded, False otherwise
        """
        # Validate memory entry
        if "memory_id" not in memory_entry:
            logger.error("Cannot store memory without a memory_id")
            return False

        try:
            memory_id = memory_entry["memory_id"]

            # Extract metadata fields for direct storage
            timestamp = memory_entry.get("timestamp", int(time.time()))
            # Allow None values to be stored directly
            step_number = memory_entry.get("step_number")
            if step_number is None:
                # Keep it as None
                pass
            elif step_number == 0:
                # Default value
                step_number = 0

            memory_type = memory_entry.get("type")
            if memory_type is None:
                # Keep it as None
                pass
            elif not memory_type:
                # Empty or falsy but not None
                memory_type = "unknown"

            metadata = memory_entry.get("metadata", {})
            compression_level = metadata.get(
                "compression_level", self.config.compression_level
            )

            # Preserve None for importance score
            importance_score = metadata.get("importance_score")
            if importance_score is None:
                # Keep it as None
                pass
            elif importance_score == 0:
                # Default value
                importance_score = 0.0

            retrieval_count = metadata.get("retrieval_count", 0)
            created_at = metadata.get("creation_time", int(time.time()))
            last_accessed = metadata.get("last_access_time", int(time.time()))

            # Prepare content and metadata as JSON
            content_json = json.dumps(memory_entry.get("content", {}))
            metadata_json = json.dumps(metadata)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Store the memory entry
                cursor.execute(
                    f"""
                INSERT OR REPLACE INTO {self.memory_table} 
                (memory_id, agent_id, step_number, timestamp, 
                content_json, metadata_json, compression_level, 
                importance_score, retrieval_count, memory_type, 
                created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        memory_id,
                        self.agent_id,
                        step_number,
                        timestamp,
                        content_json,
                        metadata_json,
                        compression_level,
                        importance_score,
                        retrieval_count,
                        memory_type,
                        created_at,
                        last_accessed,
                    ),
                )

                # Store embeddings if available
                embeddings = memory_entry.get("embeddings", {})
                if embeddings and "compressed_vector" in embeddings:
                    vector = embeddings["compressed_vector"]
                    vector_dim = len(vector)

                    # Convert vector to bytes for blob storage
                    vector_blob = np.array(vector, dtype=np.float32).tobytes()

                    cursor.execute(
                        f"""
                    INSERT OR REPLACE INTO {self.embeddings_table}
                    (memory_id, agent_id, vector_blob, vector_dim)
                    VALUES (?, ?, ?, ?)
                    """,
                        (memory_id, self.agent_id, vector_blob, vector_dim),
                    )

                conn.commit()

                logger.debug(
                    "Stored memory %s for agent %s in LTM", memory_id, self.agent_id
                )
                return True

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to store memory %s for agent %s: %s",
                memory_entry.get("memory_id"),
                self.agent_id,
                str(e),
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error storing memory %s: %s",
                memory_entry.get("memory_id"),
                str(e),
            )
            return False

    def store_batch(self, memory_entries: List[Dict[str, Any]]) -> bool:
        """Store multiple memory entries in a single transaction.

        Args:
            memory_entries: List of memory entries to store

        Returns:
            True if all entries were stored successfully, False otherwise
        """
        if not memory_entries:
            return True

        # Check if any entry is missing memory_id
        all_valid = True
        for memory_entry in memory_entries:
            if not memory_entry.get("memory_id"):
                all_valid = False
                logger.warning("Found memory entry without memory_id in batch")
                break

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use a transaction for the batch
                cursor.execute("BEGIN TRANSACTION")

                for memory_entry in memory_entries:
                    memory_id = memory_entry.get("memory_id")
                    if not memory_id:
                        logger.warning("Skipping memory entry without memory_id")
                        continue

                    # Extract metadata fields for direct storage
                    timestamp = memory_entry.get("timestamp", int(time.time()))
                    # Allow None values to be stored directly
                    step_number = memory_entry.get("step_number")
                    if step_number is None:
                        # Keep it as None
                        pass
                    elif step_number == 0:
                        # Default value
                        step_number = 0

                    memory_type = memory_entry.get("type")
                    if memory_type is None:
                        # Keep it as None
                        pass
                    elif not memory_type:
                        # Empty or falsy but not None
                        memory_type = "unknown"

                    metadata = memory_entry.get("metadata", {})
                    compression_level = metadata.get(
                        "compression_level", self.config.compression_level
                    )

                    # Preserve None for importance score
                    importance_score = metadata.get("importance_score")
                    if importance_score is None:
                        # Keep it as None
                        pass
                    elif importance_score == 0:
                        # Default value
                        importance_score = 0.0

                    retrieval_count = metadata.get("retrieval_count", 0)
                    created_at = metadata.get("creation_time", int(time.time()))
                    last_accessed = metadata.get("last_access_time", int(time.time()))

                    # Prepare content and metadata as JSON
                    content_json = json.dumps(memory_entry.get("content", {}))
                    metadata_json = json.dumps(metadata)

                    # Store the memory entry
                    cursor.execute(
                        f"""
                    INSERT OR REPLACE INTO {self.memory_table} 
                    (memory_id, agent_id, step_number, timestamp, 
                    content_json, metadata_json, compression_level, 
                    importance_score, retrieval_count, memory_type, 
                    created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            memory_id,
                            self.agent_id,
                            step_number,
                            timestamp,
                            content_json,
                            metadata_json,
                            compression_level,
                            importance_score,
                            retrieval_count,
                            memory_type,
                            created_at,
                            last_accessed,
                        ),
                    )

                    # Store embeddings if available
                    embeddings = memory_entry.get("embeddings", {})
                    if embeddings and "compressed_vector" in embeddings:
                        vector = embeddings["compressed_vector"]
                        vector_dim = len(vector)

                        # Convert vector to bytes for blob storage
                        vector_blob = np.array(vector, dtype=np.float32).tobytes()

                        cursor.execute(
                            f"""
                        INSERT OR REPLACE INTO {self.embeddings_table}
                        (memory_id, agent_id, vector_blob, vector_dim)
                        VALUES (?, ?, ?, ?)
                        """,
                            (memory_id, self.agent_id, vector_blob, vector_dim),
                        )

                # Commit the transaction
                conn.commit()

                logger.debug(
                    "Stored batch of %d memories for agent %s in LTM",
                    len(memory_entries),
                    self.agent_id,
                )
                return all_valid

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to store batch of memories for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return False
        except Exception as e:
            logger.error("Unexpected error storing batch of memories: %s", str(e))
            return False

    def get(
        self, memory_id: str, agent_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by its ID.

        Args:
            memory_id: ID of the memory to retrieve
            agent_id: Optional agent ID to search for (defaults to self.agent_id)

        Returns:
            Memory entry or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use provided agent_id or the default one
                search_agent_id = agent_id if agent_id is not None else self.agent_id

                # Get the memory entry with the specified agent_id
                cursor.execute(
                    f"""
                SELECT * FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """,
                    (memory_id, search_agent_id),
                )

                row = cursor.fetchone()

                if not row:
                    return None

                # Convert row to dict
                memory_data = dict(row)
                actual_agent_id = memory_data["agent_id"]

                # Parse JSON fields
                content = json.loads(memory_data["content_json"])
                metadata = json.loads(memory_data["metadata_json"])

                # Construct memory entry
                memory_entry = {
                    "memory_id": memory_data["memory_id"],
                    "agent_id": actual_agent_id,
                    "step_number": memory_data["step_number"],
                    "timestamp": memory_data["timestamp"],
                    "type": memory_data["memory_type"],
                    "content": content,
                    "metadata": metadata,
                }

                # Get vector embeddings if available
                cursor.execute(
                    f"""
                SELECT vector_blob, vector_dim FROM {self.embeddings_table}
                WHERE memory_id = ? AND agent_id = ?
                """,
                    (memory_id, actual_agent_id),
                )

                vector_row = cursor.fetchone()
                if vector_row:
                    # Convert blob back to vector
                    vector_blob = vector_row["vector_blob"]
                    vector_dim = vector_row["vector_dim"]
                    vector = np.frombuffer(vector_blob, dtype=np.float32).tolist()

                    memory_entry["embeddings"] = {"compressed_vector": vector}

                # Update access metadata
                self._update_access_metadata(memory_id)

                return memory_entry

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to retrieve memory %s for agent %s: %s",
                memory_id,
                self.agent_id,
                str(e),
            )
            return None
        except Exception as e:
            logger.error("Unexpected error retrieving memory %s: %s", memory_id, str(e))
            return None

    def _update_access_metadata(self, memory_id: str) -> None:
        """Update access metadata for a memory entry.

        Args:
            memory_id: ID of the memory to update
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get current retrieval count
                cursor.execute(
                    f"""
                SELECT retrieval_count FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """,
                    (memory_id, self.agent_id),
                )

                row = cursor.fetchone()
                if not row:
                    return

                retrieval_count = row["retrieval_count"] + 1
                last_accessed = int(time.time())

                # Update access metadata
                cursor.execute(
                    f"""
                UPDATE {self.memory_table}
                SET retrieval_count = ?, last_accessed = ?
                WHERE memory_id = ? AND agent_id = ?
                """,
                    (retrieval_count, last_accessed, memory_id, self.agent_id),
                )

                conn.commit()

        except Exception as e:
            # Non-critical operation, just log
            logger.warning(
                "Failed to update access metadata for memory %s: %s", memory_id, str(e)
            )

    def get_by_timerange(
        self, start_time: float, end_time: float, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            limit: Maximum number of results

        Returns:
            List of memory entries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get memories within time range
                cursor.execute(
                    f"""
                SELECT memory_id FROM {self.memory_table}
                WHERE agent_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                    (self.agent_id, int(start_time), int(end_time), limit),
                )

                rows = cursor.fetchall()

                # Retrieve full memory entries
                results = []
                for row in rows:
                    memory = self.get(row["memory_id"])
                    if memory:
                        results.append(memory)

                return results

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to retrieve memories by timerange for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by timerange: %s", str(e)
            )
            return []

    def get_by_importance(
        self, min_importance: float = 0.0, max_importance: float = 1.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories by importance score.

        Args:
            min_importance: Minimum importance score (inclusive)
            max_importance: Maximum importance score (inclusive)
            limit: Maximum number of results

        Returns:
            List of memory entries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get memories by importance
                cursor.execute(
                    f"""
                SELECT memory_id FROM {self.memory_table}
                WHERE agent_id = ? AND importance_score BETWEEN ? AND ?
                ORDER BY importance_score DESC
                LIMIT ?
                """,
                    (self.agent_id, min_importance, max_importance, limit),
                )

                rows = cursor.fetchall()

                # Retrieve full memory entries
                results = []
                for row in rows:
                    memory = self.get(row["memory_id"])
                    if memory:
                        results.append(memory)

                return results

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to retrieve memories by importance for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by importance: %s", str(e)
            )
            return []

    def get_most_similar(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve memories most similar to the query vector.

        Args:
            query_vector: Query vector for similarity search
            top_k: Number of results to return

        Returns:
            List of tuples containing (memory_entry, similarity_score)
        """
        try:
            # Convert query vector to numpy array
            query_array = np.array(query_vector, dtype=np.float32)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get all vectors for the agent
                cursor.execute(
                    f"""
                SELECT e.memory_id, e.vector_blob, e.vector_dim 
                FROM {self.embeddings_table} e
                JOIN {self.memory_table} m ON e.memory_id = m.memory_id
                WHERE m.agent_id = ?
                """,
                    (self.agent_id,),
                )

                rows = cursor.fetchall()

                # Calculate similarities
                similarities = []
                for row in rows:
                    memory_id = row["memory_id"]
                    vector_blob = row["vector_blob"]
                    vector_dim = row["vector_dim"]

                    # Convert blob back to vector
                    vector = np.frombuffer(vector_blob, dtype=np.float32)

                    # Calculate cosine similarity
                    similarity = np.dot(query_array, vector) / (
                        np.linalg.norm(query_array) * np.linalg.norm(vector)
                    )

                    similarities.append((memory_id, similarity))

                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Get top-k results
                top_memories = []
                for memory_id, similarity in similarities[:top_k]:
                    memory = self.get(memory_id)
                    if memory:
                        top_memories.append((memory, similarity))

                return top_memories

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to retrieve similar memories for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error("Unexpected error retrieving similar memories: %s", str(e))
            return []

    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories with similar embeddings.

        Args:
            query_embedding: The vector embedding to use for similarity search
            k: Number of results to return
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries sorted by similarity score
        """
        try:
            # Get similar memories using the existing method
            similar_memories = self.get_most_similar(query_embedding, top_k=k)

            # Process results to match the expected format
            results = []
            for memory, similarity in similar_memories:
                # Filter by memory type if specified
                if memory_type and memory.get("memory_type") != memory_type:
                    continue

                # Add similarity score to the memory entry
                memory["similarity_score"] = float(similarity)
                results.append(memory)

            # If we've filtered by memory_type, we might need more results
            if memory_type and len(results) < k:
                # We would need to get more results and filter them
                additional_needed = k - len(results)
                more_similar = self.get_most_similar(query_embedding, top_k=k + 20)

                for memory, similarity in more_similar[k:]:
                    if memory.get("memory_type") == memory_type:
                        memory["similarity_score"] = float(similarity)
                        results.append(memory)
                        if len(results) >= k:
                            break

            return results[:k]

        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            return []

    def search_by_attributes(
        self, attributes: Dict[str, Any], memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for memories matching specific attributes.

        Args:
            attributes: Dictionary of attribute keys and values to match
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries with matching attributes
        """
        try:
            # Get all memories (with type filter if provided)
            if memory_type:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"""
                    SELECT memory_id FROM {self.memory_table}
                    WHERE agent_id = ? AND memory_type = ?
                    """,
                        (self.agent_id, memory_type),
                    )
                    rows = cursor.fetchall()
                    candidates = [self.get(row["memory_id"]) for row in rows if row]
                    candidates = [m for m in candidates if m]  # Filter out None values
            else:
                candidates = self.get_all(limit=1000)

            # Filter by attributes
            results = []
            for memory in candidates:
                if self._matches_attributes(memory, attributes):
                    results.append(memory)

            return results

        except Exception as e:
            logger.error(f"Error in search_by_attributes: {e}")
            return []

    def _matches_attributes(
        self, memory: Dict[str, Any], attributes: Dict[str, Any]
    ) -> bool:
        """Check if a memory matches the specified attributes.

        Args:
            memory: Memory entry to check
            attributes: Dictionary of attribute keys and values to match

        Returns:
            True if the memory matches all attributes, False otherwise
        """
        for attr_path, attr_value in attributes.items():
            # Handle nested attributes using dot notation (e.g., "position.location")
            parts = attr_path.split(".")

            # Start from the memory content
            current = memory.get("content", {})

            # Navigate through the nested structure
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    return False
                current = current[part]

            # Check the final attribute value
            last_part = parts[-1]
            if last_part not in current or current[last_part] != attr_value:
                return False

        return True

    def count(self) -> int:
        """Get the number of memories for the agent.

        Returns:
            Number of memories
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"""
                SELECT COUNT(*) as count FROM {self.memory_table}
                WHERE agent_id = ?
                """,
                    (self.agent_id,),
                )

                row = cursor.fetchone()
                return row["count"] if row else 0

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to count memories for agent %s: %s", self.agent_id, str(e)
            )
            return 0
        except Exception as e:
            logger.error("Unexpected error counting memories: %s", str(e))
            return 0

    def get_size(self) -> int:
        """Get the approximate size in bytes of all memories for the agent.

        Returns:
            Approximate size in bytes
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get the size of the memory data
                cursor.execute(
                    f"""
                SELECT SUM(LENGTH(content_json)) as total_size 
                FROM {self.memory_table}
                WHERE agent_id = ?
                """,
                    (self.agent_id,),
                )

                row = cursor.fetchone()
                memory_size = row["total_size"] if row and row["total_size"] else 0

                # Get the size of the embeddings
                cursor.execute(
                    f"""
                SELECT SUM(LENGTH(vector_blob)) as total_size 
                FROM {self.embeddings_table} e
                JOIN {self.memory_table} m ON e.memory_id = m.memory_id
                WHERE m.agent_id = ?
                """,
                    (self.agent_id,),
                )

                row = cursor.fetchone()
                embedding_size = row["total_size"] if row and row["total_size"] else 0

                return memory_size + embedding_size

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to calculate memory size for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return 0
        except Exception as e:
            logger.error("Unexpected error calculating memory size: %s", str(e))
            return 0

    def get_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all memories for the agent.

        Args:
            limit: Maximum number of results

        Returns:
            List of memory entries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get all memory entries directly to avoid individual get() calls
                cursor.execute(
                    f"""
                SELECT 
                    m.memory_id, m.agent_id, m.step_number, m.timestamp, 
                    m.content_json, m.metadata_json, m.memory_type, 
                    m.importance_score, m.retrieval_count,
                    m.created_at, m.last_accessed
                FROM {self.memory_table} AS m
                WHERE m.agent_id = ?
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                    (self.agent_id, limit),
                )

                rows = cursor.fetchall()

                results = []
                for row in rows:
                    # Convert row to dict and parse JSON fields
                    memory_data = dict(row)

                    # Handle explicitly to avoid any type mismatches
                    try:
                        content = json.loads(memory_data["content_json"])
                        metadata = json.loads(memory_data["metadata_json"])

                        # Ensure step_number is consistent (int or None, not mixed types)
                        step_number = memory_data["step_number"]
                        if step_number is not None:
                            step_number = int(step_number)

                        memory_entry = {
                            "memory_id": memory_data["memory_id"],
                            "agent_id": memory_data["agent_id"],
                            "step_number": step_number,
                            "timestamp": memory_data["timestamp"],
                            "type": memory_data["memory_type"],
                            "content": content,
                            "metadata": metadata,
                            "importance_score": (
                                float(memory_data["importance_score"])
                                if memory_data["importance_score"] is not None
                                else 0.0
                            ),
                            "retrieval_count": (
                                int(memory_data["retrieval_count"])
                                if memory_data["retrieval_count"] is not None
                                else 0
                            ),
                        }
                        results.append(memory_entry)
                    except Exception as e:
                        logger.warning(f"Error parsing memory data: {e}")
                        continue

                return results

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to retrieve all memories for agent %s: %s",
                self.agent_id,
                str(e),
            )
            return []
        except Exception as e:
            logger.error("Unexpected error retrieving all memories: %s", str(e))
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if the memory was deleted, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Due to foreign key constraints, deleting from the main table
                # will cascade to the embeddings table
                cursor.execute(
                    f"""
                DELETE FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """,
                    (memory_id, self.agent_id),
                )

                conn.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(
                        "Deleted memory %s for agent %s from LTM",
                        memory_id,
                        self.agent_id,
                    )

                return deleted

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to delete memory %s for agent %s: %s",
                memory_id,
                self.agent_id,
                str(e),
            )
            return False
        except Exception as e:
            logger.error("Unexpected error deleting memory %s: %s", memory_id, str(e))
            return False

    def clear(self) -> bool:
        """Clear all memories for the agent.

        Returns:
            True if the memories were cleared, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Due to foreign key constraints, deleting from the main table
                # will cascade to the embeddings table
                cursor.execute(
                    f"""
                DELETE FROM {self.memory_table}
                WHERE agent_id = ?
                """,
                    (self.agent_id,),
                )

                conn.commit()

                cleared = cursor.rowcount > 0
                if cleared:
                    logger.info(
                        "Cleared all memories for agent %s from LTM", self.agent_id
                    )

                return True

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(
                "Failed to clear memories for agent %s: %s", self.agent_id, str(e)
            )
            return False
        except Exception as e:
            logger.error("Unexpected error clearing memories: %s", str(e))
            return False

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the SQLite LTM store.

        Returns:
            Dictionary with health status information
        """
        try:
            start_time = time.time()

            with self._get_connection() as conn:
                # Check if we can execute a simple query
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()

                # Check database integrity
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]

                latency = (time.time() - start_time) * 1000  # ms

                return {
                    "status": "healthy" if integrity_result == "ok" else "unhealthy",
                    "latency_ms": latency,
                    "integrity": integrity_result,
                    "client": "sqlite-ltm",
                    "db_path": self.db_path,
                }

        except Exception as e:
            logger.error("LTM health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "client": "sqlite-ltm",
                "db_path": self.db_path,
            }

    def search_by_step_range(
        self, start_step: int, end_step: int, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for memories within a specific step range.

        Args:
            start_step: Beginning of step range (inclusive)
            end_step: End of step range (inclusive)
            memory_type: Optional filter for specific memory types

        Returns:
            List of memory entries with step numbers in the range
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Construct query based on whether memory_type is specified
                if memory_type:
                    cursor.execute(
                        f"""
                    SELECT memory_id FROM {self.memory_table}
                    WHERE agent_id = ? AND step_number BETWEEN ? AND ? AND memory_type = ?
                    ORDER BY step_number
                    """,
                        (self.agent_id, start_step, end_step, memory_type),
                    )
                else:
                    cursor.execute(
                        f"""
                    SELECT memory_id FROM {self.memory_table}
                    WHERE agent_id = ? AND step_number BETWEEN ? AND ?
                    ORDER BY step_number
                    """,
                        (self.agent_id, start_step, end_step),
                    )

                rows = cursor.fetchall()

                # Get full memory entries
                results = []
                for row in rows:
                    memory = self.get(row["memory_id"])
                    if memory:
                        results.append(memory)

                return results

        except Exception as e:
            logger.error(f"Error in search_by_step_range: {e}")
            return []
