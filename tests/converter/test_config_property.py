"""
Property-based tests for the configuration system using hypothesis.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import settings, HealthCheck

from converter.config import ConverterConfig


# Strategy for valid error handling modes
error_handling_strategy = st.sampled_from(["skip", "fail", "log"])

# Strategy for valid import modes
import_mode_strategy = st.sampled_from(["full", "incremental"])

# Strategy for valid tiering strategy types
tiering_strategy_strategy = st.sampled_from(
    ["simple", "step_based", "importance_aware"]
)

# Strategy for valid memory type mappings
memory_type_mapping_strategy = st.lists(
    st.sampled_from(["state", "action", "interaction"]),
    min_size=3,
    max_size=3,
    unique=True
).map(lambda types: dict(zip(
    ["AgentStateModel", "ActionModel", "SocialInteractionModel"],
    types
)))

# Strategy for valid batch sizes
batch_size_strategy = st.integers(min_value=1, max_value=1000)

# Strategy for valid selective agents
selective_agents_strategy = st.lists(st.integers(min_value=1), min_size=0, max_size=100)


@given(
    use_mock_redis=st.booleans(),
    validate=st.booleans(),
    error_handling=error_handling_strategy,
    batch_size=batch_size_strategy,
    show_progress=st.booleans(),
    import_mode=import_mode_strategy,
    selective_agents=selective_agents_strategy,
    tiering_strategy_type=tiering_strategy_strategy,
    memory_type_mapping=memory_type_mapping_strategy,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_config_properties(
    use_mock_redis,
    validate,
    error_handling,
    batch_size,
    show_progress,
    import_mode,
    selective_agents,
    tiering_strategy_type,
    memory_type_mapping,
):
    """Test that any valid combination of configuration parameters creates a valid config."""
    config = ConverterConfig(
        use_mock_redis=use_mock_redis,
        validate=validate,
        error_handling=error_handling,
        batch_size=batch_size,
        show_progress=show_progress,
        import_mode=import_mode,
        selective_agents=selective_agents,
        tiering_strategy_type=tiering_strategy_type,
        memory_type_mapping=memory_type_mapping,
    )

    # Verify all properties are set correctly
    assert config.use_mock_redis == use_mock_redis
    assert config.validate == validate
    assert config.error_handling == error_handling
    assert config.batch_size == batch_size
    assert config.show_progress == show_progress
    assert config.import_mode == import_mode
    assert config.selective_agents == selective_agents
    assert config.tiering_strategy_type == tiering_strategy_type
    assert config.memory_type_mapping == memory_type_mapping


@given(
    error_handling=st.text().filter(lambda x: x not in ["skip", "fail", "log"]),
)
def test_invalid_error_handling(error_handling):
    """Test that invalid error handling modes raise ValueError."""
    with pytest.raises(ValueError, match="Invalid error_handling mode"):
        ConverterConfig(error_handling=error_handling)


@given(
    import_mode=st.text().filter(lambda x: x not in ["full", "incremental"]),
)
def test_invalid_import_mode(import_mode):
    """Test that invalid import modes raise ValueError."""
    with pytest.raises(ValueError, match="Invalid import_mode"):
        ConverterConfig(import_mode=import_mode)


@given(
    batch_size=st.integers(max_value=0),
)
def test_invalid_batch_size(batch_size):
    """Test that invalid batch sizes raise ValueError."""
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        ConverterConfig(batch_size=batch_size)


@given(
    tiering_strategy_type=st.text().filter(
        lambda x: x not in ["simple", "step_based", "importance_aware"]
    ),
)
def test_invalid_tiering_strategy(tiering_strategy_type):
    """Test that invalid tiering strategy types raise ValueError."""
    with pytest.raises(ValueError, match="Invalid tiering_strategy_type"):
        ConverterConfig(tiering_strategy_type=tiering_strategy_type)


@given(
    memory_type_mapping=st.dictionaries(
        keys=st.text(), values=st.text(), min_size=1, max_size=2
    )
)
def test_invalid_memory_type_mapping(memory_type_mapping):
    """Test that invalid memory type mappings raise ValueError."""
    with pytest.raises(ValueError):
        ConverterConfig(memory_type_mapping=memory_type_mapping)


@given(
    memory_type_mapping=st.dictionaries(
        keys=st.sampled_from(
            ["AgentStateModel", "ActionModel", "SocialInteractionModel"]
        ),
        values=st.text().filter(lambda x: x not in ["state", "action", "interaction"]),
        min_size=3,
        max_size=3,
    )
)
def test_invalid_memory_types(memory_type_mapping):
    """Test that invalid memory types in mapping raise ValueError."""
    with pytest.raises(ValueError, match="Invalid memory types in mapping"):
        ConverterConfig(memory_type_mapping=memory_type_mapping)
