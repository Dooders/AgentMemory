"""Search model implementation for agent memory retrieval.

This package provides an extensible search model with different strategies for 
retrieving agent memories based on various criteria.
"""

from agent_memory.search.model import SearchModel
from agent_memory.search.strategies.base import SearchStrategy
from agent_memory.search.strategies.similarity import SimilaritySearchStrategy
from agent_memory.search.strategies.temporal import TemporalSearchStrategy
from agent_memory.search.strategies.attribute import AttributeSearchStrategy
from agent_memory.search.strategies.combined import CombinedSearchStrategy

__all__ = [
    "SearchModel",
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "TemporalSearchStrategy",
    "AttributeSearchStrategy",
    "CombinedSearchStrategy",
] 