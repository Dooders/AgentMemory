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
from agent_memory.storage.models import LTMMemoryEntry
from agent_memory.storage.sql_base import SQLMemoryStore
from agent_memory.utils.error_handling import (
    LTMError,
    MemoryError,
    Priority,
    SQLitePermanentError,
    SQLiteTemporaryError,
)

logger = logging.getLogger(__name__)


class SQLiteLTMStore(SQLMemoryStore[LTMMemoryEntry]):
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
        """Initialize SQLite LTM memory store.
        
        Args:
            agent_id: ID of the agent
            config: Configuration for LTM SQLite storage
        """
        # Convert :memory: to a shared memory path for in-memory databases
        db_path = config.db_path
        if db_path == ":memory:":
            # Use the special file: syntax with a unique identifier to create a 
            # named in-memory database that can be shared between connections
            # See: https://www.sqlite.org/inmemorydb.html
            db_path = f"file:{agent_id}-ltm-sqlite?mode=memory&cache=shared"
            logger.info(f"Using shared in-memory database: {db_path}")
            
        # Initialize the base class
        super().__init__(
            store_type="LTM",
            agent_id=agent_id,
            db_path=db_path,
            table_prefix=config.table_prefix
        )
        
        # Store the config for LTM-specific settings
        self.config = config
        
        # Explicitly initialize the database to ensure tables exist
        self._init_database()
        
        logger.info(f"Initialized SQLiteLTMStore for agent {agent_id} at {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database schema if needed."""
        logger.info(f"Initializing SQLite database tables for {self.agent_id} with table_prefix={self.table_prefix}")
        logger.debug(f"Memory table name: {self.memory_table}")
        logger.debug(f"Embeddings table name: {self.embeddings_table}")
        
        # Use direct connection instead of context manager
        conn = None
        try:
            # Check if this is a URI-based connection string
            use_uri = self.db_path.startswith("file:")
            
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,  # 30 second timeout
                isolation_level="IMMEDIATE",  # Immediate transactions
                uri=use_uri,  # Enable URI mode if needed
            )
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set row factory
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()

            # Create memory table
            logger.debug(f"Creating memory table: {self.memory_table}")
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
            logger.debug(f"Creating indices for {self.memory_table}")
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
            logger.debug(f"Creating embeddings table: {self.embeddings_table}")
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
            logger.info(f"Successfully initialized SQLite database tables for {self.agent_id}")

        except sqlite3.OperationalError as e:
            logger.error(f"SQLite operational error during initialization: {str(e)}")
            raise MemoryError(f"Failed to initialize database: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error initializing SQLite database: {str(e)}")
            raise MemoryError(f"Failed to initialize database: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def store(self, memory_entry: LTMMemoryEntry, priority: Priority = Priority.NORMAL) -> bool:
        """Store a memory entry in the LTM.

        Args:
            memory_entry: Memory entry to store
            priority: Priority level for this operation (not used in SQLite implementation)

        Returns:
            True if the operation succeeded, False otherwise
            
        Raises:
            MemoryError: If the store operation fails
        """
        # Validate memory entry
        if "memory_id" not in memory_entry:
            logger.error("Cannot store memory without a memory_id")
            return False

        try:
            logger.info(f"Storing memory {memory_entry.get('memory_id')} in LTM")
            
            # Ensure tables are initialized
            self._init_database()
            
            memory_id = memory_entry["memory_id"]

            # Set LTM-specific defaults if needed
            if "metadata" not in memory_entry:
                memory_entry["metadata"] = {}
            
            # LTM-specific defaults
            memory_entry["metadata"]["compression_level"] = self.config.compression_level
            
            # Ensure we have creation time
            if "creation_time" not in memory_entry["metadata"]:
                memory_entry["metadata"]["creation_time"] = float(time.time())
            
            # Set default access time if not present
            if "last_access_time" not in memory_entry["metadata"]:
                memory_entry["metadata"]["last_access_time"] = float(time.time())
            
            # Set default retrieval count if not present
            if "retrieval_count" not in memory_entry["metadata"]:
                memory_entry["metadata"]["retrieval_count"] = 0

            # Extract metadata fields for direct storage
            timestamp = memory_entry.get("timestamp", float(time.time()))
            
            # Extract step number
            step_number = memory_entry.get("step_number")
            
            # Extract memory type
            memory_type = memory_entry.get("memory_type")
            
            # Extract metadata fields
            metadata = memory_entry["metadata"]
            compression_level = metadata.get("compression_level", self.config.compression_level)
            importance_score = metadata.get("importance_score", 0.0)
            retrieval_count = metadata.get("retrieval_count", 0)
            created_at = metadata.get("creation_time", float(time.time()))
            last_accessed = metadata.get("last_access_time", float(time.time()))

            # Prepare content and metadata as JSON
            content_json = self._serialize_content(memory_entry.get("content", {}))
            metadata_json = self._serialize_content(metadata)

            # Use connection from base class
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
                if "embeddings" in memory_entry and "compressed_vector" in memory_entry["embeddings"]:
                    vector = memory_entry["embeddings"]["compressed_vector"]
                    vector_dim = len(vector)
                    
                    # Convert vector to bytes for blob storage
                    vector_blob = self._vector_to_blob(vector)
                    
                    cursor.execute(
                        f"""
                    INSERT OR REPLACE INTO {self.embeddings_table} 
                    (memory_id, agent_id, vector_blob, vector_dim)
                    VALUES (?, ?, ?, ?)
                    """,
                        (memory_id, self.agent_id, vector_blob, vector_dim),
                    )

                conn.commit()
                logger.debug(f"Stored memory {memory_id} in LTM")
                return True
            
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(f"SQLite error during store: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to store memory {memory_entry.get('memory_id')}: {str(e)}")
            raise MemoryError(f"Failed to store memory: {str(e)}")
    
    def store_batch(self, memory_entries: List[LTMMemoryEntry], priority: Priority = Priority.NORMAL) -> bool:
        """Store multiple memory entries in a single transaction.

        Args:
            memory_entries: List of memory entries to store
            priority: Priority level for this operation (not used in SQLite implementation)

        Returns:
            True if all entries were stored successfully, False otherwise
            
        Raises:
            MemoryError: If the batch store operation fails
        """
        if not memory_entries:
            return True

        # Check if any entry is missing memory_id
        for memory_entry in memory_entries:
            if not memory_entry.get("memory_id"):
                logger.warning("Found memory entry without memory_id in batch")
                return False

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use a transaction for the batch
                cursor.execute("BEGIN TRANSACTION")

                for memory_entry in memory_entries:
                    memory_id = memory_entry.get("memory_id")
                    
                    # Set LTM-specific defaults if needed
                    if "metadata" not in memory_entry:
                        memory_entry["metadata"] = {}
                    
                    # LTM-specific defaults
                    memory_entry["metadata"]["compression_level"] = self.config.compression_level
                    
                    # Ensure we have creation time
                    if "creation_time" not in memory_entry["metadata"]:
                        memory_entry["metadata"]["creation_time"] = float(time.time())
                    
                    # Set default access time if not present
                    if "last_access_time" not in memory_entry["metadata"]:
                        memory_entry["metadata"]["last_access_time"] = float(time.time())
                    
                    # Set default retrieval count if not present
                    if "retrieval_count" not in memory_entry["metadata"]:
                        memory_entry["metadata"]["retrieval_count"] = 0

                    # Extract metadata fields for direct storage
                    timestamp = memory_entry.get("timestamp", float(time.time()))
                    
                    # Extract step number
                    step_number = memory_entry.get("step_number")
                    
                    # Extract memory type
                    memory_type = memory_entry.get("memory_type")
                    
                    # Extract metadata fields
                    metadata = memory_entry["metadata"]
                    compression_level = metadata.get("compression_level", self.config.compression_level)
                    importance_score = metadata.get("importance_score", 0.0)
                    retrieval_count = metadata.get("retrieval_count", 0)
                    created_at = metadata.get("creation_time", float(time.time()))
                    last_accessed = metadata.get("last_access_time", float(time.time()))

                    # Prepare content and metadata as JSON
                    content_json = self._serialize_content(memory_entry.get("content", {}))
                    metadata_json = self._serialize_content(metadata)

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
                    if "embeddings" in memory_entry and "compressed_vector" in memory_entry["embeddings"]:
                        vector = memory_entry["embeddings"]["compressed_vector"]
                        vector_dim = len(vector)

                        # Convert vector to bytes for blob storage
                        vector_blob = self._vector_to_blob(vector)

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
                
                logger.debug(f"Stored batch of {len(memory_entries)} memories in LTM")
                return True

        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.warning(f"Failed to store memory batch: {str(e)}")
            raise MemoryError(f"Failed to store memory batch: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error storing memory batch: {str(e)}")
            raise MemoryError(f"Failed to store memory batch: {str(e)}")
    
    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[LTMMemoryEntry]:
        """Search memories by vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries ordered by similarity
            
        Raises:
            MemoryError: If the search fails
        """
        try:
            # Convert query to numpy array for faster calculations
            query_array = np.array(query_embedding, dtype=np.float32)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all embeddings
                if memory_type:
                    query = f"""
                    SELECT e.memory_id, e.vector_blob, e.vector_dim 
                    FROM {self.embeddings_table} e
                    JOIN {self.memory_table} m ON e.memory_id = m.memory_id AND e.agent_id = m.agent_id
                    WHERE e.agent_id = ? AND m.memory_type = ?
                    """
                    cursor.execute(query, (self.agent_id, memory_type))
                else:
                    query = f"""
                    SELECT memory_id, vector_blob, vector_dim 
                    FROM {self.embeddings_table}
                    WHERE agent_id = ?
                    """
                    cursor.execute(query, (self.agent_id,))
                
                rows = cursor.fetchall()
                
                # Calculate similarities
                memories_with_scores = []
                
                for row in rows:
                    memory_id = row["memory_id"]
                    vector_blob = row["vector_blob"]
                    vector_dim = row["vector_dim"]
                    
                    # Convert blob to vector
                    vector = self._blob_to_vector(vector_blob, vector_dim)
                    vector_array = np.array(vector, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_array, vector_array)
                    
                    # Get the full memory
                    memory = self.get(memory_id)
                    if memory:
                        memories_with_scores.append((memory, similarity))
                
                # Sort by similarity (descending)
                memories_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Return top k results
                return [memory for memory, _ in memories_with_scores[:k]]
            
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during similarity search: {str(e)}")
            raise MemoryError(f"Failed to search by similarity: {str(e)}")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector (numpy array)
            b: Second vector (numpy array)
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if a.shape != b.shape:
            raise ValueError(f"Vector dimensions don't match: {a.shape} vs {b.shape}")
        
        # Calculate dot product
        dot_product = np.dot(a, b)
        
        # Calculate magnitudes
        a_magnitude = np.linalg.norm(a)
        b_magnitude = np.linalg.norm(b)
        
        # Calculate similarity
        if a_magnitude == 0 or b_magnitude == 0:
            return 0.0
        
        similarity = dot_product / (a_magnitude * b_magnitude)
        
        # Handle numerical errors that might push similarity outside [-1, 1]
        return float(max(-1.0, min(1.0, similarity)))

    def search_by_attributes(
        self,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None
    ) -> List[LTMMemoryEntry]:
        """Search memories by attribute matching.
        
        Args:
            attributes: Attributes to match
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries matching the attributes
            
        Raises:
            MemoryError: If the search fails
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
                # Get all memories
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"""
                    SELECT memory_id FROM {self.memory_table}
                    WHERE agent_id = ?
                    LIMIT 1000
                    """,
                        (self.agent_id,),
                    )
                    rows = cursor.fetchall()
                    candidates = [self.get(row["memory_id"]) for row in rows if row]
                    candidates = [m for m in candidates if m]  # Filter out None values
            
            # Filter by attributes
            results = []
            for memory in candidates:
                if self._matches_attributes(memory, attributes):
                    results.append(memory)
            
            return results
            
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during attribute search: {str(e)}")
            raise MemoryError(f"Failed to search by attributes: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during attribute search: {str(e)}")
            raise MemoryError(f"Failed to search by attributes: {str(e)}")

    def get(self, memory_id: str) -> Optional[LTMMemoryEntry]:
        """Retrieve a memory entry by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry or None if not found
            
        Raises:
            MemoryError: If retrieval fails
        """
        # Ensure tables are initialized before retrieval
        try:
            self._init_database()
            return super().get(memory_id)
        except Exception as e:
            logger.error(f"Error during get operation: {str(e)}")
            raise MemoryError(f"Failed to retrieve memory: {str(e)}")

    def clear(self) -> bool:
        """Clear all memories.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemoryError: If clear operation fails
        """
        try:
            # Ensure tables are initialized
            self._init_database()
            return super().clear()
        except Exception as e:
            logger.error(f"Failed to clear memories: {str(e)}")
            raise MemoryError(f"Failed to clear memories: {str(e)}")
