from demos.demo_utils import create_memory_system

agent_id = "retrieval_agent"
memory_file = "demos/memory_samples/retrieval_demo_memory.json"

memory_system = create_memory_system(
    description="retrieval demo",
    use_embeddings=True,  # Enable embeddings for similarity search
    embedding_type="text-embedding-ada-002",  # Specify an embedding model
    memory_file=memory_file,  # Use our new memory sample file
    clear_db=True,  # Clear any existing database
    use_mock_redis=True
)

if not memory_system:
    print("Failed to load memory system")

stats = memory_system.get_memory_statistics(agent_id, simplified=True)
print(stats)
