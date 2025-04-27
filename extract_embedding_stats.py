import json

# Load the embeddings file
with open('changelog_embeddings.json', 'r') as f:
    data = json.load(f)

# Display token counts
print("Token counts for each context file:")
for file_name, token_count in data['token_counts'].items():
    print(f"{file_name}: {token_count} tokens")

print("\nContent samples:")
for file_name, sample in data['content_samples'].items():
    print(f"\n{file_name}:")
    print(f"{sample}")

# Calculate embedding dimensions
first_embedding_key = next(iter(data['embeddings'].keys()))
embedding_size = len(data['embeddings'][first_embedding_key])
print(f"\nEmbedding dimensions: {embedding_size}")

# Count total tokens
total_tokens = sum(data['token_counts'].values())
print(f"Total tokens across all files: {total_tokens}")

# Calculate approximate size reduction
raw_tokens = total_tokens
embedding_tokens = len(data['embeddings']) * (embedding_size / 6)  # Approx 6 float values per token
print(f"\nOriginal tokens: {raw_tokens}")
print(f"Embedding size in token equivalent: ~{int(embedding_tokens)}")
print(f"Compression ratio: {raw_tokens/embedding_tokens:.2f}x") 