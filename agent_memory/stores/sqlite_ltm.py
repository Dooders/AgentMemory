import copy
import json
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SQLiteLTMStore:
    def store(self, agent_id: str, memory: Dict[str, Any]) -> bool:
        """Store a memory entry in the LTM database.

        Args:
            agent_id: Agent ID
            memory: Memory entry to store

        Returns:
            True if storage was successful
        """
        # Update agent_id if not set
        if "agent_id" not in memory:
            memory["agent_id"] = agent_id

        # Make a copy to avoid modifying the original
        memory_copy = copy.deepcopy(memory)

        try:
            # Extract fields for storage
            memory_id = memory_copy.get("memory_id")
            if not memory_id:
                logger.error("Cannot store memory without memory_id")
                return False

            # Extract JSON fields
            content_json = json.dumps(memory_copy.get("content", {}))
            metadata_json = json.dumps(memory_copy.get("metadata", {}))

            # Extract embedding if available
            embedding_json = None
            if "embeddings" in memory_copy and memory_copy["embeddings"]:
                embedding_json = json.dumps(memory_copy["embeddings"])

            # Prepare SQL parameters
            params = (
                memory_id,
                agent_id,
                memory_copy.get("type", "unknown"),
                memory_copy.get("subtype", None),
                memory_copy.get("step_number", None),
                memory_copy.get("timestamp", int(time.time())),
                content_json,
                metadata_json,
                embedding_json,
            )

            # Execute SQL to insert the memory
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO agent_ltm_memories
                    (memory_id, agent_id, memory_type, memory_subtype, 
                     step_number, timestamp, content_json, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
                conn.commit()

            return True
        except Exception as e:
            logger.error("Failed to store memory in LTM: %s", str(e))
            return False

    def store_batch(self, agent_id: str, memories: List[Dict[str, Any]]) -> bool:
        """Store multiple memory entries in a single transaction.

        Args:
            agent_id: Agent ID
            memories: List of memory entries to store

        Returns:
            True if storage was successful
        """
        if not memories:
            return True

        try:
            # Prepare parameters for all memories
            params = []
            for memory in memories:
                # Update agent_id if not set
                if "agent_id" not in memory:
                    memory["agent_id"] = agent_id

                # Make a copy to avoid modifying the original
                memory_copy = copy.deepcopy(memory)

                # Extract fields for storage
                memory_id = memory_copy.get("memory_id")
                if not memory_id:
                    logger.error("Cannot store memory without memory_id, skipping")
                    continue

                # Extract JSON fields
                content_json = json.dumps(memory_copy.get("content", {}))
                metadata_json = json.dumps(memory_copy.get("metadata", {}))

                # Extract embedding if available
                embedding_json = None
                if "embeddings" in memory_copy and memory_copy["embeddings"]:
                    embedding_json = json.dumps(memory_copy["embeddings"])

                # Add to parameters list
                params.append(
                    (
                        memory_id,
                        agent_id,
                        memory_copy.get("type", "unknown"),
                        memory_copy.get("subtype", None),
                        memory_copy.get("step_number", None),
                        memory_copy.get("timestamp", int(time.time())),
                        content_json,
                        metadata_json,
                        embedding_json,
                    )
                )

            # Execute SQL to insert all memories in a single transaction
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO agent_ltm_memories
                    (memory_id, agent_id, memory_type, memory_subtype, 
                     step_number, timestamp, content_json, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
                conn.commit()

            return True
        except Exception as e:
            logger.error("Failed to store batch of memories in LTM: %s", str(e))
            return False
