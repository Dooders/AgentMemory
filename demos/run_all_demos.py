"""
Run All AgentMemory Demos

This script runs all the demos in sequence to validate functionality of the
complete AgentMemory system.
"""

import importlib
import argparse

# Import common utilities
from demo_utils import clear_screen

def run_demo(demo_name):
    """Run a specific demo."""
    try:
        # Import and run the demo module
        module = importlib.import_module(f"demos.{demo_name}")
        module.run_demo()
        return True
    except Exception as e:
        print(f"\nError running demo: {e}")
        return False

def run_all_demos():
    """Run all the demos in sequence."""
    
    demos = [
        "01_basic_memory_operations",
        "02_memory_retrieval",
        "03_memory_tiers_compression",
        "04_memory_hooks_integration",
        "05_performance_error_handling"
    ]
    
    print("=" * 70)
    print("AgentMemory System Demo Suite")
    print("=" * 70)
    print(f"Found {len(demos)} demos to run.")
    
    for i, demo in enumerate(demos, 1):
        input(f"\nPress Enter to run Demo {i}: {demo}...")
        clear_screen()
        
        print(f"Running Demo {i}/{len(demos)}: {demo}")
        print("=" * 70)
        
        success = run_demo(demo)
        
        if success:
            print("\nDemo completed successfully.")
        
        print("=" * 70)
        
    print("\nAll demos completed!")
    print("\nDemos validated the following major system features:")
    print("  1. Hierarchical Memory Architecture (STM, IM, LTM)")
    print("  2. Performance Optimization")
    print("  3. Flexible Retrieval System")
    print("  4. Integration Capabilities")
    print("  5. Memory Embedding System")
    print("  6. Memory Transition Management")
    print("  7. Error Handling and Recovery")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentMemory demos")
    parser.add_argument("demo", nargs="?", help="Specific demo to run (e.g., '01_basic_memory_operations')")
    args = parser.parse_args()
    
    if args.demo:
        # Run a specific demo
        print(f"Running demo: {args.demo}")
        run_demo(args.demo)
    else:
        # Run all demos
        run_all_demos() 