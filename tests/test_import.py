import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch not installed")
except Exception as e:
    print(f"Error importing PyTorch: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")
except Exception as e:
    print(f"Error importing transformers: {e}")

try:
    import sentence_transformers
    print(f"Sentence Transformers version: {sentence_transformers.__version__}")
    
    # Try to create a sentence transformer model with minimal resources
    from sentence_transformers import SentenceTransformer
    print("Initializing a small SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded successfully. Dimension: {model.get_sentence_embedding_dimension()}")
    
    # Test encoding
    text = "This is a test sentence."
    embedding = model.encode(text)
    print(f"Encoded text to vector of dimension {len(embedding)}")
    
except ImportError as e:
    print(f"Sentence Transformers not installed properly: {e}")
except Exception as e:
    print(f"Error with Sentence Transformers: {e}")
    import traceback
    traceback.print_exc() 