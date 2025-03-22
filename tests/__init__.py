"""Unit tests for the Agent Memory system."""

# Import test modules to ensure they're discovered
from tests import test_memory_api
from tests import test_hooks
from tests import test_attribute_retrieval
from tests import test_attribute_retrieval_edge_cases
from tests import test_attribute_retrieval_combined
from tests import test_attribute_retrieval_integration
from tests import test_similarity_retrieval
from tests import test_temporal_retrieval
from tests import test_redis_client
from tests import test_redis_client_integration
from tests import test_redis_client_circuit_breaker
from tests import test_redis_recovery_queue
