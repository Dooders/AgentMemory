import os
import json
import tiktoken
import numpy as np
from pathlib import Path

# Using a simulated embedding function since we can't directly access the OpenAI API
# In a real scenario, you would use OpenAI's embedding API or another embedding service
def simulate_embedding(text, embedding_dim=1536):
    """Simulate an embedding vector for demonstration purposes"""
    # This is just a deterministic hash-based simulation
    # In reality, you would use a proper embedding API
    import hashlib
    
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).digest()
    
    # Convert hash to a seed for numpy
    seed = int.from_bytes(text_hash, byteorder='big') % (2**32 - 1)
    np.random.seed(seed)
    
    # Generate a normalized random vector
    vector = np.random.randn(embedding_dim)
    vector = vector / np.linalg.norm(vector)
    
    return vector.tolist()

def get_openai_embedding(text, model="text-embedding-3-small"):
    """
    Get embeddings from the OpenAI API
    Note: You would need to have the OpenAI Python library installed and API key set
    """
    # Commented out code that would be used in production
    """
    from openai import OpenAI
    
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
    """
    # For demonstration, we'll use our simulation function
    return simulate_embedding(text)

def truncate_text(text, max_chars):
    """Truncate text to maximum character length"""
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def create_changelog_context_embeddings():
    """Create embeddings for changelog context files"""
    # Context files with their maximum character limits
    context_files = [
        ('readme_content.txt', 800),          # README content
        ('module_info.txt', 1000),            # Module information
        ('project_structure.txt', 500),       # Project structure
        ('changelog_history.txt', 500),       # Previous changelog entries
        ('commit_categories.txt', None),      # Conventional commits analysis
        ('impact_analysis.txt', None),        # Code impact analysis
        ('pr_commits.txt', None),             # PR commits
        ('pr_files_changed.txt', None),       # Files changed in PR
        ('pr_diff_stats.txt', None),          # Diff statistics
    ]
    
    # Store results
    embeddings = {}
    token_counts = {}
    content_samples = {}
    
    # Process each file
    for file_name, max_chars in context_files:
        if os.path.exists(file_name):
            # Read and truncate content
            with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            truncated_content = truncate_text(content, max_chars)
            token_count = count_tokens(truncated_content)
            
            # Generate embedding
            embedding = get_openai_embedding(truncated_content)
            
            # Store results
            embeddings[file_name] = embedding
            token_counts[file_name] = token_count
            content_samples[file_name] = truncated_content[:100] + "..." if len(truncated_content) > 100 else truncated_content
            
            print(f"Processed {file_name}: {token_count} tokens")
        else:
            print(f"{file_name}: File not found")
    
    # Save embeddings and metadata to file
    output = {
        "embeddings": embeddings,
        "token_counts": token_counts,
        "content_samples": content_samples
    }
    
    with open('changelog_embeddings.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nEmbeddings saved to changelog_embeddings.json")
    
    # Generate a sample prompt that incorporates embeddings
    sample_prompt = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates changelog entries based on PR information."},
            {"role": "user", "content": "Generate a changelog entry for this PR"}
        ],
        "context_embeddings": embeddings
    }
    
    with open('sample_embedding_prompt.json', 'w') as f:
        json.dump(sample_prompt, f, indent=2)
    
    print(f"Sample embedding prompt saved to sample_embedding_prompt.json")
    
    # Show how to use this in a GitHub workflow
    workflow_example = """
# Example GitHub workflow snippet that uses embeddings
- name: Generate embeddings for context
  run: |
    python scripts/create_changelog_embeddings.py
  
- name: Generate changelog with embeddings
  id: generate_changelog
  run: |
    # Read the embeddings
    EMBEDDINGS=$(cat changelog_embeddings.json)
    
    # Prepare the API call with embeddings
    curl -X POST https://api.openai.com/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer $OPENAI_API_KEY" \\
      -d '{
        "model": "gpt-4o",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant that generates changelog entries."},
          {"role": "user", "content": "Generate a changelog entry for PR #${{ github.event.pull_request.number }}"}
        ],
        "context_embeddings": '$EMBEDDINGS'
      }'
    """
    
    print("\nExample of how to use embeddings in a GitHub workflow:")
    print(workflow_example)

if __name__ == "__main__":
    # Check if tiktoken is installed, install if not
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        import subprocess
        subprocess.check_call(["pip", "install", "tiktoken"])
        import tiktoken
    
    create_changelog_context_embeddings() 