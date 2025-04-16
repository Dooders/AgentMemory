#!/usr/bin/env python
"""
Script to list all agents in the database and help identify the target agent.
"""

import os
import sqlite3
import argparse
from collections import defaultdict

# Set data directory
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "simulation.db")

def list_all_agents(db_path, prefix=None, limit=None):
    """List all agents in the database, optionally filtering by prefix."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Base query
    query = """
        SELECT a.agent_id, a.agent_type, a.generation, 
               COUNT(s.id) as state_count, 
               COUNT(act.action_id) as action_count
        FROM agents a
        LEFT JOIN agent_states s ON a.agent_id = s.agent_id
        LEFT JOIN agent_actions act ON a.agent_id = act.agent_id
    """
    
    # Add prefix filter if specified
    if prefix:
        query += " WHERE a.agent_id LIKE ? "
        params = (f"{prefix}%",)
    else:
        params = ()
    
    # Complete the query with grouping
    query += """
        GROUP BY a.agent_id
        ORDER BY a.agent_id
    """
    
    # Add limit if specified
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    
    # Print agent info
    print(f"Found {len(rows)} agents:")
    print("=" * 80)
    print(f"{'Agent ID':<40} {'Type':<15} {'Gen':<5} {'States':<8} {'Actions':<8}")
    print("-" * 80)
    
    for row in rows:
        print(f"{row['agent_id']:<40} {row['agent_type']:<15} {row['generation']:<5} {row['state_count']:<8} {row['action_count']:<8}")
    
    # Additional statistics
    print("\nAgent ID Prefixes:")
    prefixes = defaultdict(int)
    for row in rows:
        # Count occurrences of first 3 characters
        prefix = row['agent_id'][:3]
        prefixes[prefix] += 1
    
    # Print prefix counts
    for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}*: {count} agents")
    
    conn.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="List agents in the simulation database"
    )
    
    parser.add_argument(
        "--prefix", 
        type=str,
        help="Filter agents by ID prefix"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit the number of agents to display"
    )
    
    args = parser.parse_args()
    
    list_all_agents(DB_PATH, args.prefix, args.limit)

if __name__ == "__main__":
    main() 