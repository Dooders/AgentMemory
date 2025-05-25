"""
Tests for memory tiering strategies.
"""

import pytest

from converter.tiering import (
    ImportanceAwareTieringStrategy,
    SimpleTieringStrategy,
    StepBasedTieringStrategy,
    TieringContext,
    create_tiering_strategy,
)


def test_simple_tiering():
    """Test simple tiering strategy that always returns STM."""
    strategy = SimpleTieringStrategy()

    # Test various contexts
    contexts = [
        TieringContext(0, 0, 0),
        TieringContext(10, 100, 100),
        TieringContext(90, 100, 100),
        TieringContext(10, 100, 100, importance_score=0.9),
        TieringContext(10, 100, 100, metadata={"test": "value"}),
    ]

    for context in contexts:
        assert strategy.determine_tier(context) == "stm"


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

    # Test edge cases
    with pytest.raises(ValueError, match="step_number cannot be negative"):
        TieringContext(-1, 100, 100)

    with pytest.raises(
        ValueError, match="step_number cannot be greater than total_steps"
    ):
        TieringContext(101, 100, 100)


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

    # Test edge cases for importance scores
    with pytest.raises(ValueError, match="importance_score must be between 0 and 1"):
        TieringContext(10, 100, 100, importance_score=1.1)

    with pytest.raises(ValueError, match="importance_score must be between 0 and 1"):
        TieringContext(10, 100, 100, importance_score=-0.1)


def test_tiering_strategy_factory():
    """Test tiering strategy factory function."""
    # Test all available strategies
    strategy_types = ["simple", "step_based", "importance_aware"]
    strategy_classes = [
        SimpleTieringStrategy,
        StepBasedTieringStrategy,
        ImportanceAwareTieringStrategy,
    ]

    for strategy_type, strategy_class in zip(strategy_types, strategy_classes):
        strategy = create_tiering_strategy(strategy_type)
        assert isinstance(strategy, strategy_class)

    # Test invalid strategy type
    with pytest.raises(ValueError, match="Invalid strategy_type"):
        create_tiering_strategy("invalid")


def test_tiering_context_metadata():
    """Test tiering context with metadata."""
    # Test with different metadata types
    test_metadata = [
        {"custom_field": "value"},
        {"number": 42, "boolean": True, "list": [1, 2, 3]},
        {},  # Empty metadata
        None,  # No metadata
    ]

    for metadata in test_metadata:
        context = TieringContext(
            step_number=10, current_step=100, total_steps=100, metadata=metadata
        )
        if metadata and metadata != {}:
            assert context.metadata == metadata
        else:
            assert context.metadata is None


def test_tiering_context_validation():
    """Test tiering context parameter validation."""
    # Test step number validation
    with pytest.raises(ValueError, match="step_number cannot be negative"):
        TieringContext(-1, 100, 100)

    with pytest.raises(
        ValueError, match="step_number cannot be greater than total_steps"
    ):
        TieringContext(101, 100, 100)

    # Test current step validation
    with pytest.raises(ValueError, match="current_step cannot be negative"):
        TieringContext(10, -1, 100)

    with pytest.raises(
        ValueError, match="current_step cannot be greater than total_steps"
    ):
        TieringContext(10, 101, 100)

    # Test total steps validation
    with pytest.raises(ValueError, match="total_steps cannot be negative"):
        TieringContext(10, 100, -1)

    # Test valid cases
    context = TieringContext(0, 0, 0)  # Should not raise
    assert context.step_number == 0
    assert context.current_step == 0
    assert context.total_steps == 0
