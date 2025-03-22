"""Unit tests for the Agent Memory system."""

# Import test modules to ensure they're discovered
from tests.api import test_memory_api
from tests.api import test_hooks
from tests.retrieval import test_attribute_retrieval
from tests.retrieval import test_attribute_retrieval_edge_cases
from tests.retrieval import test_attribute_retrieval_combined
from tests.retrieval import test_attribute_retrieval_integration
from tests.retrieval import test_similarity_retrieval
from tests.retrieval import test_temporal_retrieval
from tests.storage import test_redis_client
from tests.storage import test_redis_client_integration
from tests.storage import test_redis_client_circuit_breaker
from tests.storage import test_redis_recovery_queue
