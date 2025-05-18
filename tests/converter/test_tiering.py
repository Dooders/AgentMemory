"""
Tests for memory tiering strategies.
"""

import pytest
from converter.tiering import (
    TieringContext,
    StepBasedTieringStrategy,
    ImportanceAwareTieringStrategy,
    create_tiering_strategy
)

def test_step_based_tiering():
    """Test step-based tiering strategy."""
    strategy = StepBasedTieringStrategy()
    
    # Test empty simulation
    context = TieringContext(0, 0, 0)
    assert strategy.determine_tier(context) == "stm"
    
    # Test STM (most recent 10%)
    context = TieringContext(90, 100, 100)
    assert strategy.determine_tier(context) == "stm"
    
    # Test IM (next 30%)
    context = TieringContext(60, 100, 100)
    assert strategy.determine_tier(context) == "im"
    
    # Test LTM (remaining)
    context = TieringContext(10, 100, 100)
    assert strategy.determine_tier(context) == "ltm"

def test_importance_aware_tiering():
    """Test importance-aware tiering strategy."""
    strategy = ImportanceAwareTieringStrategy()
    
    # Test without importance score (falls back to step-based)
    context = TieringContext(10, 100, 100)
    assert strategy.determine_tier(context) == "ltm"
    
    # Test high importance promotion
    context = TieringContext(10, 100, 100, importance_score=0.9)
    assert strategy.determine_tier(context) == "stm"
    
    # Test medium importance promotion
    context = TieringContext(10, 100, 100, importance_score=0.6)
    assert strategy.determine_tier(context) == "im"
    
    # Test low importance (no promotion)
    context = TieringContext(10, 100, 100, importance_score=0.3)
    assert strategy.determine_tier(context) == "ltm"

def test_tiering_strategy_factory():
    """Test tiering strategy factory function."""
    # Test step-based strategy creation
    strategy = create_tiering_strategy("step_based")
    assert isinstance(strategy, StepBasedTieringStrategy)
    
    # Test importance-aware strategy creation
    strategy = create_tiering_strategy("importance_aware")
    assert isinstance(strategy, ImportanceAwareTieringStrategy)
    
    # Test invalid strategy type
    with pytest.raises(ValueError, match="Invalid strategy_type"):
        create_tiering_strategy("invalid")

def test_tiering_context_metadata():
    """Test tiering context with metadata."""
    context = TieringContext(
        step_number=10,
        current_step=100,
        total_steps=100,
        metadata={"custom_field": "value"}
    )
    assert context.metadata["custom_field"] == "value" 