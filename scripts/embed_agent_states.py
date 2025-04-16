"""Script to embed all agent states from the simulation database.

This script extracts agent states from the SQLite database, converts them to text embeddings,
and saves them either back to the database or in a vector store format.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Import your existing modules
from memory.embeddings.text_embeddings import TextEmbeddingEngine
from memory.embeddings.vector_store import InMemoryVectorIndex, VectorIndex, VectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import FAISS if available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "FAISS not available. To use FAISS, install with 'pip install faiss-cpu' or 'pip install faiss-gpu'"
    )


class FaissVectorIndex(VectorIndex):
    """FAISS-based vector index for efficient similarity search.

    This implementation uses FAISS for high-performance vector similarity search.
    """

    def __init__(
        self,
        dimension: int = 384,
        metric: str = "cosine",
        index_type: str = "Flat",
    ):
        """Initialize the FAISS vector index.

        Args:
            dimension: Dimension of vectors to be stored
            metric: Distance metric ('cosine', 'l2')
            index_type: Type of FAISS index ('Flat', 'IVF', etc.)
        """
        super().__init__()

        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not available. Install with 'pip install faiss-cpu'"
            )

        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type

        # Create FAISS index
        if metric == "cosine":
            # For cosine similarity, we need to normalize vectors
            self.index = faiss.IndexFlatIP(
                dimension
            )  # Inner product for normalized vectors = cosine
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(dimension)

        # Storage for IDs and metadata since FAISS only stores vectors
        self.ids = []
        self.metadata = []

    def add(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector to the FAISS index.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector to store
            metadata: Optional metadata to associate with the vector

        Returns:
            True if the operation was successful
        """
        try:
            # Convert to numpy array
            vector_np = np.array([vector], dtype=np.float32)

            # Normalize for cosine similarity if needed
            if self.metric == "cosine":
                faiss.normalize_L2(vector_np)

            # Add to FAISS index
            self.index.add(vector_np)

            # Store ID and metadata
            self.ids.append(id)
            self.metadata.append(metadata or {})

            return True
        except Exception as e:
            logger.error(f"Failed to add vector to FAISS index: {str(e)}")
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the FAISS index.

        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_fn: Optional function to filter results

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Convert query to numpy array
            query_np = np.array([query_vector], dtype=np.float32)

            # Normalize for cosine similarity if needed
            if self.metric == "cosine":
                faiss.normalize_L2(query_np)

            # Perform search
            distances, indices = self.index.search(query_np, limit)

            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices (happens if we have fewer items than limit)
                if idx == -1 or idx >= len(self.ids):
                    continue

                # Calculate the actual similarity score
                score = 1.0 - distance if self.metric == "l2" else distance

                # Get ID and metadata
                id = self.ids[idx]
                metadata = self.metadata[idx]

                # Apply filter if provided
                if filter_fn and not filter_fn(metadata):
                    continue

                # Add to results
                results.append({"id": id, "score": float(score), "metadata": metadata})

            return results
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {str(e)}")
            return []

    def save(self, filepath: str) -> bool:
        """Save the FAISS index to a file.

        Args:
            filepath: Path to save the index

        Returns:
            True if successful
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")

            # Save IDs and metadata
            data = {
                "ids": self.ids,
                "metadata": self.metadata,
                "dimension": self.dimension,
                "metric": self.metric,
                "index_type": self.index_type,
            }

            with open(f"{filepath}.json", "w") as f:
                json.dump(data, f)

            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
            return False

    @classmethod
    def load(cls, filepath: str) -> "FaissVectorIndex":
        """Load a FAISS index from a file.

        Args:
            filepath: Path to load the index from

        Returns:
            FaissVectorIndex instance
        """
        try:
            # Load metadata
            with open(f"{filepath}.json", "r") as f:
                data = json.load(f)

            # Create instance
            instance = cls(
                dimension=data["dimension"],
                metric=data["metric"],
                index_type=data["index_type"],
            )

            # Load FAISS index
            instance.index = faiss.read_index(f"{filepath}.faiss")

            # Load IDs and metadata
            instance.ids = data["ids"]
            instance.metadata = data["metadata"]

            return instance
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise


class AgentStateEmbedder:
    """Class to handle embedding agent states from the simulation database."""

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        output_format: str = "vector_store",
        batch_size: int = 100,
        output_dir: Optional[str] = None,
    ):
        """Initialize the agent state embedder.

        Args:
            db_path: Path to the SQLite database file
            embedding_model: Name of the embedding model to use
            output_format: Format to save embeddings ('vector_store', 'sqldb', or 'faiss')
            batch_size: Number of states to process in one batch
            output_dir: Directory to save output files (if not saving to database)
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.output_format = output_format
        self.batch_size = batch_size
        self.output_dir = output_dir

        # Initialize database connection
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Initialize the embedding engine
        self.embedding_engine = TextEmbeddingEngine(model_name=self.embedding_model)

        # Initialize vector store based on output format
        if self.output_format == "vector_store":
            self.vector_store = VectorStore()
        elif self.output_format == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError(
                    "FAISS is not available. Install with 'pip install faiss-cpu'"
                )
            self.faiss_index = FaissVectorIndex(
                dimension=self.embedding_engine.embedding_dim
            )

        # If output directory is specified, create it if it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_total_agent_states(self) -> int:
        """Get the total number of agent states in the database.

        Returns:
            Total count of agent states
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_states")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_agent_states_batch(self, offset: int) -> List[Dict[str, Any]]:
        """Get a batch of agent states from the database.

        Args:
            offset: Starting offset for the batch

        Returns:
            List of agent states as dictionaries
        """
        cursor = self.conn.cursor()
        query = """
        SELECT 
            as1.id, as1.simulation_id, as1.step_number, as1.agent_id,
            as1.position_x, as1.position_y, as1.position_z,
            as1.resource_level, as1.current_health, as1.is_defending,
            as1.total_reward, as1.age,
            a.agent_type, a.genome_id, a.generation
        FROM 
            agent_states as1
        JOIN 
            agents a ON as1.agent_id = a.agent_id
        ORDER BY 
            as1.id
        LIMIT ? OFFSET ?
        """
        cursor.execute(query, (self.batch_size, offset))

        states = []
        for row in cursor.fetchall():
            state = dict(row)
            states.append(state)

        cursor.close()
        return states

    def create_embedding_table(self) -> None:
        """Create a new table in the database to store embeddings."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_state_embeddings (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            step_number INTEGER NOT NULL,
            simulation_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (id) REFERENCES agent_states (id)
        )
        """
        )

        # Create indices for faster lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_state_embeddings_agent_id ON agent_state_embeddings (agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_state_embeddings_step_number ON agent_state_embeddings (step_number)"
        )

        self.conn.commit()
        cursor.close()

    def convert_state_to_dict_for_embedding(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert agent state to a structured dictionary format for embedding.

        Args:
            state: Raw agent state dictionary from database

        Returns:
            Formatted dictionary ready for embedding
        """
        # Create a structured dictionary that captures the agent state in a format
        # that will produce meaningful embeddings
        return {
            "agent_id": state["agent_id"],
            "step_number": state["step_number"],
            "simulation_id": state["simulation_id"],
            "position": {
                "x": state["position_x"],
                "y": state["position_y"],
                "z": state["position_z"] if state["position_z"] is not None else 0,
            },
            "resources": state["resource_level"],
            "health": state["current_health"],
            "is_defending": state["is_defending"],
            "total_reward": state["total_reward"],
            "age": state["age"],
            "agent_type": state["agent_type"],
            "genome_id": state["genome_id"],
            "generation": state["generation"],
        }

    def save_embeddings_to_db(
        self, state_id: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> None:
        """Save embeddings to the SQLite database.

        Args:
            state_id: ID of the agent state
            embedding: Embedding vector
            metadata: Additional metadata about the embedding
        """
        cursor = self.conn.cursor()

        # Convert embedding to bytes for storage
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        cursor.execute(
            """
        INSERT OR REPLACE INTO agent_state_embeddings
        (id, agent_id, step_number, simulation_id, embedding, embedding_model, embedding_dimension, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                state_id,
                metadata["agent_id"],
                metadata["step_number"],
                metadata["simulation_id"],
                embedding_bytes,
                self.embedding_model,
                len(embedding),
                int(time.time()),
            ),
        )

        self.conn.commit()
        cursor.close()

    def save_vector_store_to_file(self, output_file: str) -> None:
        """Save the in-memory vector store to a file.

        Args:
            output_file: Path to save the vector store
        """
        # For simplicity, we'll just save as a JSON file
        # In production, you might want to use a more efficient format

        # Extract vectors and metadata from the vector store
        if isinstance(self.vector_store.stm_index, InMemoryVectorIndex):
            index = self.vector_store.stm_index

            data = {
                "vectors": index.vectors,
                "metadata": index.metadata,
                "model": self.embedding_model,
                "dimension": self.embedding_engine.embedding_dim,
                "created_at": int(time.time()),
            }

            with open(output_file, "w") as f:
                json.dump(data, f)

            logger.info(f"Saved vector store to {output_file}")

    def save_faiss_index_to_file(self, output_file: str) -> None:
        """Save the FAISS index to a file.

        Args:
            output_file: Path to save the FAISS index
        """
        if not hasattr(self, "faiss_index") or self.faiss_index is None:
            logger.error("No FAISS index available to save")
            return

        success = self.faiss_index.save(output_file)
        if success:
            logger.info(f"Saved FAISS index to {output_file}")
        else:
            logger.error(f"Failed to save FAISS index to {output_file}")

    def process_all_agent_states(self) -> None:
        """Process and embed all agent states from the database."""
        # Get total count for progress tracking
        total_states = self.get_total_agent_states()
        logger.info(f"Found {total_states} agent states to process")

        # Create embedding table if saving to the database
        if self.output_format == "sqldb":
            self.create_embedding_table()

        # Process in batches
        offset = 0
        total_processed = 0

        with tqdm(total=total_states, desc="Embedding agent states") as pbar:
            while True:
                # Get a batch of states
                states = self.get_agent_states_batch(offset)
                if not states:
                    break

                # Process each state in the batch
                for state in states:
                    # Convert to format suitable for embedding
                    state_dict = self.convert_state_to_dict_for_embedding(state)

                    # Create embedding
                    embedding = self.embedding_engine.encode(state_dict)

                    # Create metadata for the embedding
                    metadata = {
                        "agent_id": state["agent_id"],
                        "step_number": state["step_number"],
                        "simulation_id": state["simulation_id"],
                        "original_state": state_dict,
                    }

                    # Save based on output format
                    if self.output_format == "sqldb":
                        self.save_embeddings_to_db(state["id"], embedding, metadata)
                    elif self.output_format == "vector_store":
                        self.vector_store.stm_index.add(
                            id=state["id"], vector=embedding, metadata=metadata
                        )
                    elif self.output_format == "faiss":
                        self.faiss_index.add(
                            id=state["id"], vector=embedding, metadata=metadata
                        )

                # Update progress
                total_processed += len(states)
                pbar.update(len(states))

                # Move to next batch
                offset += self.batch_size

        logger.info(f"Successfully processed {total_processed} agent states")

        # If using vector store format and output directory is specified, save to file
        if self.output_format == "vector_store" and self.output_dir:
            output_file = os.path.join(self.output_dir, "agent_state_embeddings.json")
            self.save_vector_store_to_file(output_file)
        elif self.output_format == "faiss" and self.output_dir:
            output_file = os.path.join(self.output_dir, "agent_state_embeddings")
            self.save_faiss_index_to_file(output_file)

    def close(self) -> None:
        """Clean up resources."""
        if self.conn:
            self.conn.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Embed agent states from simulation database"
    )

    parser.add_argument(
        "--db_path",
        type=str,
        default="data/simulation.db",
        help="Path to the SQLite database file",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use (smaller='all-MiniLM-L6-v2', better='all-mpnet-base-v2')",
    )

    parser.add_argument(
        "--output_format",
        type=str,
        choices=["vector_store", "sqldb", "faiss"],
        default="faiss",
        help="Format to save embeddings ('vector_store', 'sqldb', or 'faiss')",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of states to process in one batch",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="memory/embeddings/data",
        help="Directory to save output files (if not saving to database)",
    )

    args = parser.parse_args()

    # Create and run the embedder
    embedder = AgentStateEmbedder(
        db_path=args.db_path,
        embedding_model=args.embedding_model,
        output_format=args.output_format,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    try:
        embedder.process_all_agent_states()
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
