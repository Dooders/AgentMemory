# Changelog

## [0.1.1] - 2025-04-25

### Added
- Added checksum functionality to memory metadata:
  - New `checksum` field in `MemoryMetadata` for data integrity verification
  - Added utility functions in `memory/utils/checksums.py` for generating and validating checksums
  - Implemented automatic checksum generation when storing memories
  - Added checksum validation when retrieving memories
  - Added integrity verification flags in memory metadata

### Improved
- Enhanced data integrity through checksum verification across all memory tiers
- Added support for different hashing algorithms in checksum generation
- Implemented comprehensive test suite for checksum functionality

## [0.1.0] - 2025-04-05

### Refactoring
- Extracted duplicate code into common utilities in `utils.py`:
  - `cosine_similarity`: Moved from `vector_store.py` for vector comparison
  - `flatten_dict`: Moved from `autoencoder.py` for dictionary flattening
  - `object_to_text`: Moved from `text_embeddings.py` for object text conversion
  - `filter_dict_keys`: Moved from `compression.py` for dictionary filtering

### Updated
- Modified `text_embeddings.py` to use the common `object_to_text` function
- Updated `vector_store.py` to use the shared `cosine_similarity` function
- Revised `autoencoder.py` to use the centralized `flatten_dict` function
- Changed `compression.py` to use the common `filter_dict_keys` function
- Enhanced `object_to_text` function for better handling of empty values and nested structures
- Refactored embeddings module by removing direct `TextEmbeddingEngine` import

### Added
- Added comprehensive module documentation in `__init__.py` for LLM context
- Added detailed documentation file `docs/Embeddings.md` explaining the Embeddings module components and usage
- Exposed utility functions in `__all__` list for easy importing
- Added comprehensive unit tests for the embeddings module:
  - Tests for `AutoEncoder` and dimensionality reduction capabilities
  - Tests for `NumericExtractor` functionality 
  - Tests for vector compression operations
  - Tests for text embedding operations
  - Tests for vector store functionality
- Added vector compression utilities:
  - `quantize_vector` and `dequantize_vector` for bit-depth reduction
  - `compress_vector_rp` and `decompress_vector_rp` for random projection compression
  - `CompressionConfig` for managing compression settings
- Added MockRedis implementation with pipeline and pubsub support for testing
- Enhanced MockRedis implementation for development and testing:
  - Added configuration options in `RedisSTMConfig` and `RedisIMConfig`
  - Created `RedisFactory` to streamline Redis client instantiation
  - Added comprehensive integration tests for Redis stores with MockRedis
  - Updated README with MockRedis usage instructions

### Improved
- Eliminated code duplication across the embeddings module
- Enhanced maintainability by centralizing common functionality
- Improved consistency in utility functions across the codebase
- Enhanced RedisFactory to support MockRedis with custom ResilientRedisClient for better testing
- Improved MockRedis module documentation with usage examples and implementation details

### Fixed
- Updated random number generation in vector compression to use local `RandomState` instead of global seed to prevent interference with other code