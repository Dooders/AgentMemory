"""SQL base class for memory store implementations.

This module provides a base class for SQL-based memory store implementations,
abstracting common SQL operations and patterns.
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast

import numpy as np

from agent_memory.storage.base import BaseMemoryStore
from agent_memory.storage.models import MemoryEntry
from agent_memory.utils.error_handling import (
    MemoryError,
    Priority,
    SQLitePermanentError,
    SQLiteTemporaryError,
)

logger = logging.getLogger(__name__)

# Type variable for memory entry
M = TypeVar('M', bound=Dict[str, Any])


class SQLMemoryStore(BaseMemoryStore[M]):
    """Base class for SQL-based memory stores.
    
    This class implements common SQL operations and patterns
    for memory storage, providing a foundation for LTM and other
    SQL-based implementations.
    
    Attributes:
        store_type: Type of memory store (LTM)
        agent_id: ID of the agent
        db_path: Path to the SQLite database
        table_prefix: Prefix for table names
        memory_table: Name of the main memory table
        embeddings_table: Name of the embeddings table
    """
    
    def __init__(
        self,
        store_type: str,
        agent_id: str,
        db_path: str,
        table_prefix: str
    ):
        """Initialize the SQL memory store.
        
        Args:
            store_type: Type of memory store (LTM)
            agent_id: ID of the agent
            db_path: Path to the SQLite database
            table_prefix: Prefix for table names
        """
        super().__init__(store_type)
        self.agent_id = agent_id
        self.db_path = db_path
        self.table_prefix = table_prefix
        
        # Table names
        self.memory_table = f"{self.table_prefix}_memories"
        self.embeddings_table = f"{self.table_prefix}_embeddings"
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Only try to create directory if there is a directory path
            os.makedirs(db_dir, exist_ok=True)
        
        # Initialize database tables if they don't exist
        self._init_database()
        
        logger.info(f"Initialized {store_type} SQL memory store for agent {agent_id}")
    
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
        """Initialize the SQLite database schema if needed.
        
        This should be implemented by subclasses with their specific schema.
        """
        pass
    
    def _serialize_content(self, content: Any) -> str:
        """Serialize content to JSON string.
        
        Args:
            content: Content to serialize
            
        Returns:
            JSON string representation of the content
            
        Raises:
            MemoryError: If serialization fails
        """
        try:
            return json.dumps(content)
        except Exception as e:
            logger.error(f"Failed to serialize content: {str(e)}")
            raise MemoryError(f"Failed to serialize content: {str(e)}")
    
    def _deserialize_content(self, serialized_content: str) -> Any:
        """Deserialize a JSON string to content.
        
        Args:
            serialized_content: JSON string to deserialize
            
        Returns:
            Content deserialized from JSON
            
        Raises:
            MemoryError: If deserialization fails
        """
        try:
            return json.loads(serialized_content)
        except Exception as e:
            logger.error(f"Failed to deserialize content: {str(e)}")
            raise MemoryError(f"Failed to deserialize content: {str(e)}")
    
    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """Convert a vector to a binary blob for storage.
        
        Args:
            vector: Vector to convert
            
        Returns:
            Binary blob representation of the vector
            
        Raises:
            MemoryError: If conversion fails
        """
        try:
            return np.array(vector, dtype=np.float32).tobytes()
        except Exception as e:
            logger.error(f"Failed to convert vector to blob: {str(e)}")
            raise MemoryError(f"Failed to convert vector to blob: {str(e)}")
    
    def _blob_to_vector(self, blob: bytes, dim: int) -> List[float]:
        """Convert a binary blob to a vector.
        
        Args:
            blob: Binary blob to convert
            dim: Dimension of the vector
            
        Returns:
            Vector converted from binary blob
            
        Raises:
            MemoryError: If conversion fails
        """
        try:
            vector = np.frombuffer(blob, dtype=np.float32)
            return vector.tolist()
        except Exception as e:
            logger.error(f"Failed to convert blob to vector: {str(e)}")
            raise MemoryError(f"Failed to convert blob to vector: {str(e)}")
    
    def _get_memory_from_row(self, row: sqlite3.Row) -> M:
        """Convert a database row to a memory entry.
        
        Args:
            row: Database row
            
        Returns:
            Memory entry converted from database row
            
        Raises:
            MemoryError: If conversion fails
        """
        try:
            # Convert SQLite Row to dict
            row_dict = dict(row)
            
            # Extract basic fields
            memory_entry = {
                "memory_id": row_dict["memory_id"],
                "agent_id": row_dict["agent_id"],
                "timestamp": row_dict["timestamp"],
                "memory_type": row_dict["memory_type"]
            }
            
            # Add step number if present
            if "step_number" in row_dict and row_dict["step_number"] is not None:
                memory_entry["step_number"] = row_dict["step_number"]
            
            # Deserialize content
            memory_entry["content"] = self._deserialize_content(row_dict["content_json"])
            
            # Create metadata
            metadata = self._deserialize_content(row_dict["metadata_json"])
            if not isinstance(metadata, dict):
                metadata = {}
                
            # Add explicit metadata fields if they exist
            for field in ["compression_level", "importance_score", "retrieval_count"]:
                if field in row_dict and row_dict[field] is not None:
                    metadata[field] = row_dict[field]
                    
            # Add timestamps
            if "created_at" in row_dict:
                metadata["creation_time"] = row_dict["created_at"]
            if "last_accessed" in row_dict:
                metadata["last_access_time"] = row_dict["last_accessed"]
                
            memory_entry["metadata"] = metadata
            
            return cast(M, memory_entry)
            
        except Exception as e:
            logger.error(f"Failed to convert row to memory: {str(e)}")
            raise MemoryError(f"Failed to convert row to memory: {str(e)}")
    
    def get(self, memory_id: str) -> Optional[M]:
        """Retrieve a memory entry by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry or None if not found
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Query the memory table
                query = f"""
                SELECT * FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """
                cursor.execute(query, (memory_id, self.agent_id))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Convert row to memory entry
                memory_entry = self._get_memory_from_row(row)
                
                # Update access metadata
                self._update_access_metadata(memory_id, memory_entry)
                
                # Get vector embedding if available
                try:
                    query = f"""
                    SELECT vector_blob, vector_dim FROM {self.embeddings_table}
                    WHERE memory_id = ? AND agent_id = ?
                    """
                    cursor.execute(query, (memory_id, self.agent_id))
                    vector_row = cursor.fetchone()
                    
                    if vector_row:
                        vector = self._blob_to_vector(vector_row["vector_blob"], vector_row["vector_dim"])
                        if "embeddings" not in memory_entry:
                            memory_entry["embeddings"] = {}
                        memory_entry["embeddings"]["compressed_vector"] = vector
                except Exception as e:
                    # Non-critical error - log but continue
                    logger.warning(f"Failed to retrieve vector for memory {memory_id}: {str(e)}")
                
                return memory_entry
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during memory retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memory: {str(e)}")
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if the memory was deleted, False otherwise
            
        Raises:
            MemoryError: If deletion fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if memory exists
                query = f"""
                SELECT 1 FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """
                cursor.execute(query, (memory_id, self.agent_id))
                if not cursor.fetchone():
                    return False
                
                # Delete the memory (cascades to embeddings due to foreign key)
                query = f"""
                DELETE FROM {self.memory_table}
                WHERE memory_id = ? AND agent_id = ?
                """
                cursor.execute(query, (memory_id, self.agent_id))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during memory deletion: {str(e)}")
            raise MemoryError(f"Failed to delete memory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory deletion: {str(e)}")
            raise MemoryError(f"Failed to delete memory: {str(e)}")
    
    def count(self) -> int:
        """Count memories.
        
        Returns:
            Number of memories in the store
            
        Raises:
            MemoryError: If count operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = f"""
                SELECT COUNT(*) as count FROM {self.memory_table}
                WHERE agent_id = ?
                """
                cursor.execute(query, (self.agent_id,))
                row = cursor.fetchone()
                
                return row["count"] if row else 0
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during memory count: {str(e)}")
            raise MemoryError(f"Failed to count memories: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory count: {str(e)}")
            raise MemoryError(f"Failed to count memories: {str(e)}")
    
    def clear(self) -> bool:
        """Clear all memories.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemoryError: If clear operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if the table exists
                try:
                    cursor.execute(
                        f"""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='{self.memory_table}'
                        """
                    )
                    
                    if not cursor.fetchone():
                        # Table doesn't exist yet, nothing to clear
                        return True
                    
                    # Delete all memories for this agent
                    query = f"""
                    DELETE FROM {self.memory_table}
                    WHERE agent_id = ?
                    """
                    cursor.execute(query, (self.agent_id,))
                    conn.commit()
                    
                    return True
                except sqlite3.OperationalError as e:
                    logger.warning(f"Table check failed during clear: {str(e)}")
                    # Return True since there's nothing to clear if we can't even check tables
                    return True
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during memory clear: {str(e)}")
            raise MemoryError(f"Failed to clear memories: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during memory clear: {str(e)}")
            raise MemoryError(f"Failed to clear memories: {str(e)}")
    
    def get_size(self) -> int:
        """Get the size of the memory store in bytes.
        
        Returns:
            Memory usage in bytes
            
        Raises:
            MemoryError: If size calculation fails
        """
        try:
            # Check if the file exists
            if not os.path.exists(self.db_path):
                return 0
                
            # Get the file size
            return os.path.getsize(self.db_path)
            
        except Exception as e:
            logger.error(f"Error getting database size: {str(e)}")
            raise MemoryError(f"Failed to get memory store size: {str(e)}")
    
    def get_by_timerange(
        self, start_time: float, end_time: float, limit: int = 100
    ) -> List[M]:
        """Get memories in a time range.
        
        Args:
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entries in the time range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = f"""
                SELECT * FROM {self.memory_table}
                WHERE agent_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """
                cursor.execute(query, (self.agent_id, start_time, end_time, limit))
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    memory = self._get_memory_from_row(row)
                    memories.append(memory)
                    
                return memories
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during timerange retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by timerange: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during timerange retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by timerange: {str(e)}")
    
    def get_by_importance(
        self, min_importance: float = 0.0, max_importance: float = 1.0, limit: int = 100
    ) -> List[M]:
        """Get memories by importance score.
        
        Args:
            min_importance: Minimum importance score
            max_importance: Maximum importance score
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entries in the importance range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = f"""
                SELECT * FROM {self.memory_table}
                WHERE agent_id = ? AND importance_score >= ? AND importance_score <= ?
                ORDER BY importance_score DESC
                LIMIT ?
                """
                cursor.execute(query, (self.agent_id, min_importance, max_importance, limit))
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    memory = self._get_memory_from_row(row)
                    memories.append(memory)
                    
                return memories
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during importance retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by importance: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during importance retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by importance: {str(e)}")
    
    def search_by_step_range(
        self,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None
    ) -> List[M]:
        """Get memories in a step range.
        
        Args:
            start_step: Start step number
            end_step: End step number
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory entries in the step range
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if memory_type:
                    query = f"""
                    SELECT * FROM {self.memory_table}
                    WHERE agent_id = ? AND step_number >= ? AND step_number <= ?
                    AND memory_type = ?
                    ORDER BY step_number ASC
                    LIMIT 100
                    """
                    cursor.execute(query, (self.agent_id, start_step, end_step, memory_type))
                else:
                    query = f"""
                    SELECT * FROM {self.memory_table}
                    WHERE agent_id = ? AND step_number >= ? AND step_number <= ?
                    ORDER BY step_number ASC
                    LIMIT 100
                    """
                    cursor.execute(query, (self.agent_id, start_step, end_step))
                
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    memory = self._get_memory_from_row(row)
                    memories.append(memory)
                    
                return memories
                
        except (SQLiteTemporaryError, SQLitePermanentError) as e:
            logger.error(f"SQLite error during step range retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by step range: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during step range retrieval: {str(e)}")
            raise MemoryError(f"Failed to retrieve memories by step range: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the memory store.
        
        Returns:
            Dictionary with health status information
            
        Raises:
            MemoryError: If health check fails
        """
        try:
            # Check if database file exists
            db_exists = os.path.exists(self.db_path)
            
            # Check if we can connect and run a query
            connection_ok = False
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    connection_ok = cursor.fetchone() is not None
            except Exception:
                connection_ok = False
            
            # Get memory count
            memory_count = 0
            try:
                memory_count = self.count()
            except Exception:
                pass
            
            # Get database size
            db_size = 0
            try:
                db_size = self.get_size()
            except Exception:
                pass
            
            return {
                "status": "healthy" if (db_exists and connection_ok) else "unhealthy",
                "store_type": self.store_type,
                "db_exists": db_exists,
                "connection_ok": connection_ok,
                "memory_count": memory_count,
                "db_size_bytes": db_size,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            return {
                "status": "error",
                "store_type": self.store_type,
                "error": str(e),
                "timestamp": time.time()
            } 