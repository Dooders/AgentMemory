# Changelog

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

### Added
- Added comprehensive module documentation in `__init__.py` for LLM context
- Exposed utility functions in `__all__` list for easy importing

### Improved
- Eliminated code duplication across the embeddings module
- Enhanced maintainability by centralizing common functionality
- Improved consistency in utility functions across the codebase