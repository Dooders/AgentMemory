#!/usr/bin/env python
"""
Main entry point for running AgentMemory system validations.

This script provides a convenient way to run validations for various
components of the AgentMemory system.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.framework.cli import main

if __name__ == "__main__":
    main() 