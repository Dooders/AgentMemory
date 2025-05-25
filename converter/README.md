# AgentFarm DB to Memory System Converter

A comprehensive module for converting data from AgentFarm SQLite databases into a structured memory system. This converter handles the extraction, transformation, and loading of agent data and their associated memories with support for tiering strategies, validation, and configurable error handling.

## Overview

The converter module provides functionality to import agent metadata and memories from an AgentFarm simulation database into a memory system that supports different memory tiers (Short-Term Memory, Intermediate Memory, Long-Term Memory). It's designed to handle large datasets efficiently with batch processing and comprehensive error handling.

## Features

### Core Functionality
- **Agent Import**: Extracts agent metadata including position, resources, health, genome data, and behavioral parameters
- **Memory Import**: Processes different types of memories (states, actions, social interactions) with metadata preservation
- **Memory Tiering**: Intelligent assignment of memories to different tiers based on configurable strategies
- **Batch Processing**: Efficient handling of large datasets with configurable batch sizes
- **Validation**: Comprehensive validation of database structure and imported data
- **Error Handling**: Configurable error handling modes (fail, skip, log)

### Memory Types Supported
- **State Memories**: Agent state information including position, resources, health, and status
- **Action Memories**: Agent action records with type, targets, rewards, and resource changes
- **Interaction Memories**: Social interaction data including type, outcome, and resource transfers

### Tiering Strategies
- **Simple**: Places all memories in Short-Term Memory (STM)
- **Step-Based**: Uses time decay to distribute memories across tiers based on simulation timeline
- **Importance-Aware**: Combines time decay with importance scores for intelligent tier assignment

## Installation

The converter module is part of the AgentMemory project. Ensure you have the required dependencies:

```bash
pip install sqlalchemy redis
```

## Quick Start

```python
from converter import from_agent_farm

# Basic usage with default configuration
memory_system = from_agent_farm("path/to/agentfarm.db")

# Custom configuration
memory_system = from_agent_farm(
    db_path="path/to/agentfarm.db",
    config={
        "validate": True,
        "error_handling": "fail",
        "batch_size": 200,
        "tiering_strategy_type": "importance_aware",
        "memory_config": {
            "use_mock_redis": False,
            "stm_config": {
                "memory_limit": 1000,
                "ttl": 3600
            }
        }
    }
)
```

## Configuration

### Basic Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `validate` | bool | `True` | Enable database and data validation |
| `error_handling` | str | `"skip"` | Error handling mode: "skip", "fail", or "log" |
| `batch_size` | int | `100` | Number of records to process in each batch |
| `show_progress` | bool | `True` | Display progress information during import |
| `use_mock_redis` | bool | `True` | Use mock Redis for testing |

### Memory Type Mapping

```python
config = {
    "memory_type_mapping": {
        "AgentStateModel": "state",
        "ActionModel": "action", 
        "SocialInteractionModel": "interaction"
    }
}
```

### Tiering Strategy Configuration

```python
config = {
    "tiering_strategy_type": "importance_aware"  # "simple", "step_based", "importance_aware"
}
```

### Import Modes

```python
config = {
    "import_mode": "full",  # "full", "incremental"
    "selective_agents": [1, 2, 3]  # Import specific agents only
}
```

## Architecture

### Core Components

#### DatabaseManager (`db.py`)
- Manages SQLite database connections and sessions
- Provides schema validation and query utilities
- Handles connection pooling and error recovery

#### AgentImporter (`agent_import.py`)
- Extracts agent metadata from the database
- Supports batch processing and validation
- Handles agent-specific data transformation

#### MemoryImporter (`memory_import.py`)
- Processes different memory types (states, actions, interactions)
- Applies tiering strategies to determine memory placement
- Manages memory metadata extraction and transformation

#### MemoryTypeMapper (`mapping.py`)
- Maps database models to memory system types
- Validates memory data consistency
- Supports custom type mappings

#### TieringStrategy (`tiering.py`)
- Implements different strategies for memory tier assignment
- Supports time-based and importance-based tiering
- Extensible architecture for custom strategies

#### ConverterConfig (`config.py`)
- Centralized configuration management
- Type-safe configuration with validation
- Default values and extensible options

### Data Flow

1. **Database Connection**: DatabaseManager establishes connection to AgentFarm SQLite database
2. **Validation**: Optional schema and data validation
3. **Agent Import**: AgentImporter extracts agent metadata in batches
4. **Memory Import**: MemoryImporter processes memories for each agent
5. **Type Mapping**: MemoryTypeMapper converts database models to memory types
6. **Tiering**: TieringStrategy determines appropriate memory tier
7. **Memory System**: Imported data is added to the AgentMemorySystem

## Usage Examples

### Basic Import

```python
from converter import from_agent_farm

# Import with default settings
memory_system = from_agent_farm("simulation.db")
print(f"Imported {len(memory_system.agents)} agents")
```

### Custom Configuration

```python
from converter import from_agent_farm

config = {
    "validate": True,
    "error_handling": "fail",
    "batch_size": 500,
    "tiering_strategy_type": "step_based",
    "memory_config": {
        "use_mock_redis": False,
        "stm_config": {
            "memory_limit": 2000,
            "ttl": 7200
        }
    }
}

memory_system = from_agent_farm("large_simulation.db", config)
```

### Selective Import

```python
# Import only specific agents
config = {
    "import_mode": "selective",
    "selective_agents": [1, 5, 10, 15],
    "error_handling": "log"
}

memory_system = from_agent_farm("simulation.db", config)
```

### Custom Memory Type Mapping

```python
config = {
    "memory_type_mapping": {
        "AgentStateModel": "state",
        "ActionModel": "action",
        "SocialInteractionModel": "interaction",
        "CustomModel": "custom"
    }
}

memory_system = from_agent_farm("simulation.db", config)
```

## Error Handling

The converter supports three error handling modes:

- **`"skip"`** (default): Continue processing, skip problematic records
- **`"fail"`**: Stop processing and raise exception on first error
- **`"log"`**: Log errors but continue processing

```python
# Strict error handling
config = {"error_handling": "fail"}

# Permissive error handling with logging
config = {"error_handling": "log"}
```

## Memory Tiering

### Simple Strategy
Places all memories in Short-Term Memory (STM).

### Step-Based Strategy
Distributes memories across tiers based on simulation timeline:
- Most recent 10% of steps → STM
- Next 30% of steps → Intermediate Memory (IM)
- Remaining steps → Long-Term Memory (LTM)

### Importance-Aware Strategy
Combines time decay with importance scores:
- High importance (>0.8) → Promoted to STM
- Medium importance (>0.5) → Promoted to IM
- Low importance → Uses step-based tiering

## Performance Considerations

### Batch Processing
- Default batch size: 100 records
- Increase for better performance with large datasets
- Decrease if memory usage is a concern

### Memory Usage
- The converter processes data in batches to manage memory usage
- Large simulations may require tuning batch sizes and connection pooling

### Database Optimization
- Ensure proper indexing on agent_id and step_number columns
- Consider using WAL mode for SQLite databases

## Validation

The converter includes comprehensive validation:

### Database Validation
- Checks for required tables and columns
- Validates database schema integrity
- Ensures data consistency

### Data Validation
- Validates agent and memory records
- Checks for required fields and data types
- Ensures referential integrity

## Logging

The converter uses Python's logging module with configurable levels:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable debug logging for detailed information
logging.getLogger('converter').setLevel(logging.DEBUG)
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify database path and permissions
   - Check SQLite database integrity

2. **Memory Import Failures**
   - Enable validation to identify data issues
   - Check error logs for specific problems
   - Use "log" error handling mode for debugging

3. **Performance Issues**
   - Increase batch size for large datasets
   - Monitor memory usage during import
   - Consider selective import for testing

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = {
    "validate": True,
    "error_handling": "log",
    "show_progress": True
}

memory_system = from_agent_farm("simulation.db", config)
```

## API Reference

### Main Function

#### `from_agent_farm(db_path: str, config: Optional[Dict] = None) -> AgentMemorySystem`

Imports data from an AgentFarm SQLite database into a memory system.

**Parameters:**
- `db_path` (str): Path to the AgentFarm SQLite database
- `config` (dict, optional): Configuration options for the import process

**Returns:**
- `AgentMemorySystem`: Configured memory system with imported memories

**Raises:**
- `ValueError`: If database validation fails or import verification fails
- `SQLAlchemyError`: If there are database connection issues

### Configuration Classes

#### `ConverterConfig`
Main configuration class with validation and default values.

#### `MemoryTypeMapper`
Handles mapping between database models and memory types.

#### `TieringStrategy`
Base class for memory tiering strategies.

## Contributing

When contributing to the converter module:

1. Ensure all new features include comprehensive tests
2. Update documentation for any API changes
3. Follow the existing error handling patterns
4. Add appropriate logging for debugging
5. Validate configuration parameters in `__post_init__` methods

## License

This module is part of the AgentMemory project. See the main project license for details. 