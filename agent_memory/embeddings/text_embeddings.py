"""Text embedding engine using sentence-transformers models."""

import logging
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbeddingEngine:
    """Embedding engine using sentence-transformers models.

    This provides a simpler alternative to the autoencoder approach,
    using pre-trained text embedding models to generate vector
    representations of memory content.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize the text embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-mpnet-base-v2" which provides better quality (420MB).
                For smaller size, try "all-MiniLM-L6-v2" which is (80MB).
        """
        try:
            self.model = SentenceTransformer(model_name)
            # Capture embedding dimensions
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Initialized TextEmbeddingEngine with model {model_name} (dim={self.embedding_dim})"
            )
        except NameError:
            logger.error(
                "SentenceTransformer not available. Install with 'pip install sentence-transformers'"
            )
            raise

    def _object_to_text(self, obj: Any) -> str:
        """Convert any object to an enhanced text representation.

        Args:
            obj: Any Python object to convert to text

        Returns:
            String representation of the object
        """
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            # Better handling of nested structures
            parts = []
            for key, value in obj.items():
                # Format based on value type
                if isinstance(value, dict):
                    # For nested dictionaries like position
                    if key == "position" and "location" in value:
                        # Make location more prominent in the text representation
                        formatted = f"{key}: location is {value['location']}"
                        if "x" in value and "y" in value:
                            formatted += (
                                f", coordinates x={value['x']:.1f} y={value['y']:.1f}"
                            )
                    else:
                        formatted = f"{key}: " + ", ".join(
                            f"{k}={v}" for k, v in value.items()
                        )
                elif isinstance(value, list):
                    if key == "inventory":
                        # Make inventory items more prominent
                        formatted = f"{key}: has " + ", ".join(
                            str(item) for item in value
                        )
                    else:
                        formatted = f"{key}: " + ", ".join(str(item) for item in value)
                else:
                    formatted = f"{key}: {value}"
                parts.append(formatted)
            return " | ".join(parts)
        elif isinstance(obj, list):
            # Handle lists with better formatting
            return "items: " + ", ".join(self._object_to_text(item) for item in obj)
        else:
            # Convert other types to string
            return str(obj)

    def encode(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data into an embedding vector with optional context weighting.

        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights
                for context-aware embedding generation

        Returns:
            Embedding vector as a list of floats
        """
        # Check for context-aware weighting
        if context_weights and isinstance(data, dict):
            weighted_text = ""
            # Process standard representation
            standard_text = self._object_to_text(data)

            # Add weighted components
            for key, weight in context_weights.items():
                if key in data:
                    # Special case for position to extract location
                    if (
                        key == "position"
                        and isinstance(data[key], dict)
                        and "location" in data[key]
                    ):
                        location_text = f"location is {data[key]['location']}"
                        # Repeat text based on weight for emphasis (integer multiplier)
                        repeat_count = max(1, int(weight * 5))
                        weighted_text += f" {location_text}" * repeat_count
                    # Special case for inventory to emphasize items
                    elif key == "inventory" and isinstance(data[key], list):
                        for item in data[key]:
                            item_text = f"has {item}"
                            repeat_count = max(1, int(weight * 3))
                            weighted_text += f" {item_text}" * repeat_count
                    else:
                        # Extract and repeat important components based on weight
                        value_text = self._object_to_text({key: data[key]})
                        # Repeat text based on weight for emphasis (integer multiplier)
                        repeat_count = max(1, int(weight * 3))
                        weighted_text += f" {value_text}" * repeat_count

            # Combine standard and weighted text with more weight on the emphasized parts
            combined_text = f"{standard_text} {weighted_text} {weighted_text}"
            embedding = self.model.encode(combined_text)
            return embedding.tolist()

        # Default encoding without weighting
        text = self._object_to_text(data)
        embedding = self.model.encode(text)
        return embedding.tolist()

    def encode_stm(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for STM tier with optional context weighting.

        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights

        Returns:
            Embedding vector for STM
        """
        return self.encode(data, context_weights)

    def encode_im(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for IM tier with optional context weighting.

        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights

        Returns:
            Embedding vector for IM
        """
        return self.encode(data, context_weights)

    def encode_ltm(
        self, data: Any, context_weights: Dict[str, float] = None
    ) -> List[float]:
        """Encode data for LTM tier with optional context weighting.

        Args:
            data: Any data structure to encode
            context_weights: Optional dictionary mapping keys to importance weights

        Returns:
            Embedding vector for LTM
        """
        return self.encode(data, context_weights)

    def configure(self, config: Any) -> None:
        """Update configuration of the embedding engine.

        Args:
            config: New configuration parameters
        """
        # Nothing to configure for this engine
        pass
