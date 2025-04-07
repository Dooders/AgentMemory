# AgentMemory System Demos

This directory contains demonstration scripts that showcase the main features of the AgentMemory system as outlined in the design document. These demos are designed to both demonstrate functionality and validate that the system is working as expected.

## Available Demos

1. **Basic Memory Operations** (`01_basic_memory_operations.py`)
   - Initialization of the memory system
   - Storage of state, action, and interaction memories
   - Basic memory statistics and maintenance

2. **Memory Retrieval** (`02_memory_retrieval.py`)
   - Similarity-based retrieval
   - Attribute-based retrieval
   - Temporal retrieval
   - Cross-tier retrieval

3. **Memory Tiers and Compression** (`03_memory_tiers_compression.py`)
   - Memory transitions between STM, IM, and LTM tiers
   - Memory compression during tier transitions
   - Information retention across compression

4. **Memory Hooks and Integration** (`04_memory_hooks_integration.py`)
   - Memory event hooks for automatic memory formation
   - Integration with external systems via event triggers
   - Custom memory filtering and processing

5. **Performance and Error Handling** (`05_performance_error_handling.py`)
   - Batch operations for efficient storage
   - Performance benchmarking
   - Error recovery mechanisms and fallbacks

## Running the Demos

The demos use MockRedis by default, so no real Redis server is required to run them.

You can run individual demos directly:

```bash
python -m demos.01_basic_memory_operations
```

Or run all demos in sequence:

```bash
python -m demos.run_all_demos
```

### Redis Configuration

If you want to use a real Redis server instead of MockRedis, you can modify the configuration in each demo:

```python
# To use a real Redis server instead of MockRedis
memory_system = create_memory_system(use_mock_redis=False)
```

## Acceptance Criteria Validation

These demos collectively validate the key acceptance criteria from the design document:

- **Performance**: Demos 1 and 5 validate that memory operations meet latency requirements
- **Storage Efficiency**: Demo 3 validates compression ratios and tier transitions
- **Retrieval Quality**: Demo 2 validates the various retrieval methods
- **Integration**: Demo 4 validates API completeness and event hooks
- **Reliability**: Demo 5 validates error handling and recovery mechanisms

## Adding New Demos

When adding new demos, please follow these guidelines:

1. Use a numbered prefix for ordering (e.g., `06_new_feature.py`)
2. Include a clear docstring explaining what the demo showcases
3. Implement a `run_demo()` function that can be called from the main runner
4. Update this README.md to include the new demo 