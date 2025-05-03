"""
Example Validation for a Search Strategy.

This file demonstrates how to use the validation framework
to create validation tests for a new search strategy.
"""

import os
import sys
from typing import Any, Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from validation.framework.validation_framework import ValidationTest, create_validation_test
from validation.demo_utils import log_print

# Constants
AGENT_ID = "test-agent-example"
MEMORY_SAMPLE = os.path.join("memory_samples", "example_validation_memory.json")

class ExampleValidationTest(ValidationTest):
    """Example validation test implementation."""
    
    def setup_memory_system(self, memory_count: int = 1000) -> None:
        """Initialize memory system with test data."""
        log_print(self.logger, f"Setting up memory system with {memory_count} test memories")
        
        # Create memory system
        self.memory_system = create_memory_system(
            logging_level="INFO",
            memory_file=MEMORY_SAMPLE,
            use_mock_redis=True,
        )
        
        if not self.memory_system:
            log_print(self.logger, "Failed to load memory system")
            return
        
        # Setup search strategy
        agent = self.memory_system.get_memory_agent(AGENT_ID)
        self.search_strategy = ExampleSearchStrategy(
            agent.stm_store,
            agent.im_store,
            agent.ltm_store
        )
        
        log_print(self.logger, f"Testing search strategy: {self.search_strategy.name()}")
        log_print(self.logger, f"Description: {self.search_strategy.description()}")
    
    def run_test(
        self,
        test_name: str,
        query: Any,
        expected_checksums: Set[str] = None,
        expected_memory_ids: List[str] = None,
        **search_params
    ) -> Dict[str, Any]:
        """Run a test case."""
        log_print(self.logger, f"\n=== Test: {test_name} ===")
        
        # Log test parameters
        if isinstance(query, dict):
            log_print(self.logger, f"Query (dict): {query}")
        else:
            log_print(self.logger, f"Query: '{query}'")
        
        for param, value in search_params.items():
            log_print(self.logger, f"{param}: {value}")
        
        # Run search
        results = self.search_strategy.search(
            query=query,
            agent_id=AGENT_ID,
            **search_params
        )
        
        log_print(self.logger, f"Found {len(results)} results")
        
        # Validate results
        test_passed, validation_details = self.validate_results(
            results,
            expected_checksums,
            expected_memory_ids
        )
        
        # Log validation results
        if validation_details["missing_checksums"]:
            log_print(self.logger, f"Missing expected memories: {validation_details['missing_checksums']}")
        
        if validation_details["unexpected_checksums"]:
            log_print(self.logger, f"Found unexpected memories: {validation_details['unexpected_checksums']}")
        
        # Create test result
        test_result = {
            "test_name": test_name,
            "passed": test_passed,
            "result_count": len(results),
            "validation_details": validation_details,
            "query": query,
            "search_params": search_params
        }
        
        self.results.append(test_result)
        return test_result

def validate_example_search():
    """Run validation tests for the example search strategy."""
    # Create validation test instance
    validation_test = create_validation_test("example", ExampleSearchStrategy)
    
    # Test 1: Basic content search
    validation_test.run_test(
        "Basic Content Search",
        "example",
        content_fields=["content.content"],
        expected_memory_ids=["example-1", "example-2"]
    )
    
    # Test 2: Case sensitive search
    validation_test.run_test(
        "Case Sensitive Search",
        "Example",
        case_sensitive=True,
        content_fields=["content.content"],
        expected_memory_ids=["example-1"]
    )
    
    # Test 3: Search by metadata
    validation_test.run_test(
        "Search by Metadata",
        {"metadata": {"type": "example"}},
        expected_memory_ids=["example-1", "example-2"]
    )
    
    # Save results
    validation_test.save_results()
    
    # Display summary
    log_print(validation_test.logger, "\n=== VALIDATION SUMMARY ===")
    for result in validation_test.results:
        status = "PASS" if result["passed"] else "FAIL"
        log_print(
            validation_test.logger,
            f"{result['test_name']}: {status} ({result['result_count']} results)"
        )

if __name__ == "__main__":
    validate_example_search() 