"""Search model implementation for agent memory retrieval.

This package provides an extensible search model with different strategies for 
retrieving agent memories based on various criteria.
"""

from memory.search.model import SearchModel
from memory.search.strategies.base import SearchStrategy
from memory.search.strategies.similarity import SimilaritySearchStrategy
from memory.search.strategies.temporal import TemporalSearchStrategy
from memory.search.strategies.attribute import AttributeSearchStrategy
from memory.search.strategies.combined import CombinedSearchStrategy

__all__ = [
    "SearchModel",
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "TemporalSearchStrategy",
    "AttributeSearchStrategy",
    "CombinedSearchStrategy",
] 