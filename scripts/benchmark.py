#!/usr/bin/env python3
"""
Entry script for running Agent Memory System benchmarks.

This script provides a command-line interface for running, analyzing, and 
comparing benchmarks for the Agent Memory System.
"""

import sys
from memory.benchmarking.cli import main

if __name__ == "__main__":
    sys.exit(main()) 