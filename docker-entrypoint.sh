#!/bin/bash
set -e

# Execute the agent function and capture the output
# We'll use environment variables to configure the agent
# - AGENT_TYPE: The type of agent to run (simple, memory)
# - EPISODES: Number of episodes to run
# - SEED: Random seed for reproducibility
# Default to a debug run if no args specified

if [ "$AGENT_TYPE" == "memory" ]; then
    echo "Running memory-enhanced agent..."
    MEMORY_ENABLED="True"
else
    echo "Running simple agent..."
    MEMORY_ENABLED="False"
fi

# Default values
EPISODES=${EPISODES:-10}
SEED=${SEED:-42}

# Run the agent experiment and save results to output file
echo "Starting experiment with $EPISODES episodes, memory_enabled=$MEMORY_ENABLED, seed=$SEED"
python -c "
import json
import sys
from main_demo import run_debug_experiment
import numpy as np

# Configure NumPy to use a specific random seed for reproducibility
np.random.seed($SEED)

# Run the experiment
results = run_debug_experiment(episodes=$EPISODES, memory_enabled=$MEMORY_ENABLED, random_seed=$SEED)

# Convert NumPy types to Python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

# Print results to stdout
print(json.dumps(convert_numpy(results), indent=2))
"

# Exit with success
exit 0 