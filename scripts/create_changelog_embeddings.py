#!/usr/bin/env python3
"""
Create embeddings for changelog context files.

This script generates embeddings for various context files that are used
to provide relevant information when generating changelog entries.
"""
import os
import json
import math
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from openai import OpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("🔑 Loaded environment variables from .env file")
except ImportError:
    print("⚠️ dotenv package not installed, skipping .env loading")

# List of context files to embed and their max character lengths
# Format: (file_name, max_chars_to_include)
context_files = [
    ("readme_content.txt", 10000),
    ("module_info.txt", 10000),
    ("project_structure.txt", 5000),
    ("changelog_history.txt", 15000),
    ("pr_commits.txt", 5000),
    ("pr_files_changed.txt", 5000)
]

def simulate_embedding(text: str, embedding_dim: int = 1536) -> List[float]:
    """
    Simulate an embedding vector based on the text.
    Used as a fallback when OpenAI API is not available.
    
    Args:
        text: The text to create an embedding for
        embedding_dim: The dimension of the embedding vector
    
    Returns:
        A list of floats representing the embedding vector
    """
    # Create a deterministic hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Use the hash to seed a simple PRNG
    def simple_prng(seed: str) -> float:
        """Simple PRNG based on hash value"""
        value = int(seed, 16)
        return (value % 10000) / 10000.0  # value between 0 and 1
    
    # Generate embedding values from hash fragments
    values = []
    hash_fragments = math.ceil(embedding_dim / 8)  # Each hex char is 4 bits
    
    for i in range(hash_fragments):
        # Generate a new hash for each fragment
        fragment_hash = hashlib.md5(f"{text_hash}{i}".encode()).hexdigest()
        
        # Use each character of the hash to generate a value
        for j in range(0, min(32, embedding_dim - len(values))):
            seed = fragment_hash[j % len(fragment_hash):] + fragment_hash[:j % len(fragment_hash)]
            values.append(simple_prng(seed) * 2 - 1)  # Scale to (-1, 1)
    
    # Normalize the vector (unit length)
    magnitude = math.sqrt(sum(v*v for v in values))
    return [v/magnitude for v in values]

def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embeddings for text using the OpenAI API.
    
    Args:
        text: The text to create an embedding for
        model: The OpenAI model to use
    
    Returns:
        A list of floats representing the embedding vector
    """
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Call the OpenAI API
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        # Get the embedding from the response
        embedding = response.data[0].embedding
        
        print(f"✅ Successfully generated REAL OpenAI embedding using model: {model}")
        return embedding
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        print("⚠️ Using SIMULATED embeddings instead")
        return simulate_embedding(text)

def truncate_text(text: str, max_chars: Optional[int]) -> str:
    """
    Truncate text to a maximum number of characters.
    
    Args:
        text: The text to truncate
        max_chars: Maximum number of characters to include
    
    Returns:
        Truncated text with "..." appended if truncation occurred
    """
    if max_chars is None or len(text) <= max_chars:
        return text
    
    return text[:max_chars] + "..."

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the text using tiktoken.
    
    Args:
        text: The text to count tokens for
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: estimate 1 token per 4 characters
        return len(text) // 4

def create_changelog_context_embeddings() -> None:
    """
    Process context files, generate embeddings, and save them to a JSON file.
    
    This function:
    1. Reads the content of each context file
    2. Truncates the content if needed
    3. Generates embeddings for each file
    4. Saves the embeddings, token counts, and content samples to a JSON file
    """
    # Initialize dictionaries for embeddings and metadata
    embeddings: Dict[str, List[float]] = {}
    token_counts: Dict[str, int] = {}
    content_samples: Dict[str, str] = {}
    
    # Process PR metadata from environment variables
    pr_metadata = ""
    pr_env_vars = ["PR_NUMBER", "PR_TITLE", "PR_BODY", "REPO_NAME"]
    
    for var in pr_env_vars:
        if var in os.environ:
            pr_metadata += f"{var}: {os.environ[var]}\n"
    
    if pr_metadata:
        truncated_pr_metadata = truncate_text(pr_metadata, 5000)
        content_samples["pr_metadata"] = truncated_pr_metadata
        token_counts["pr_metadata"] = count_tokens(truncated_pr_metadata)
        embeddings["pr_metadata"] = get_openai_embedding(truncated_pr_metadata)
    
    # Process each context file
    for file_name, max_chars in context_files:
        file_path = Path(file_name)
        
        # Skip if file doesn't exist
        if not file_path.exists():
            print(f"Warning: Context file '{file_name}' not found. Skipping.")
            continue
        
        # Read file content
        content = file_path.read_text(encoding="utf-8")
        
        # Truncate if necessary
        truncated_content = truncate_text(content, max_chars)
        
        # Save content sample
        content_samples[file_name] = truncated_content
        
        # Count tokens
        token_count = count_tokens(truncated_content)
        token_counts[file_name] = token_count
        
        # Generate embedding
        embedding = get_openai_embedding(truncated_content)
        embeddings[file_name] = embedding
        
        print(f"Processed {file_name}: {token_count} tokens")
    
    # Create a sample prompt for generating changelog entries
    sample_prompt = """
    Given the PR information and the context of the repository,
    generate a clear and concise changelog entry for this PR
    in markdown format.
    """
    
    # Save embeddings and metadata to JSON file
    output = {
        "embeddings": embeddings,
        "token_counts": token_counts,
        "content_samples": content_samples,
        "sample_prompt": sample_prompt
    }
    
    with open("changelog_embeddings.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved embeddings for {len(embeddings)} context files to changelog_embeddings.json")

if __name__ == "__main__":
    create_changelog_context_embeddings() 