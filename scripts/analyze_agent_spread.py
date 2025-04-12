#!/usr/bin/env python
"""
Script to analyze why certain agents have more spread in their embedding visualization.

This script loads the same agent states used in the embedding visualization, 
analyzes their state variables, and computes statistics to determine what
might cause differences in embedding spread between agents.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Tuple

def load_agent_data(limit_per_agent: int = 100) -> pd.DataFrame:
    """
    Load agent state data from the simulation database.
    
    Args:
        limit_per_agent: Maximum number of states to load per agent
        
    Returns:
        DataFrame containing agent state data
    """
    conn = sqlite3.connect('data/simulation.db')
    
    # Get unique agent_ids
    query = "SELECT DISTINCT agent_id FROM agent_states LIMIT 50"
    agent_ids = pd.read_sql_query(query, conn)['agent_id'].tolist()
    
    # Load data for each agent
    all_data = []
    for agent_id in agent_ids:
        query = f"""
        SELECT * FROM agent_states 
        WHERE agent_id = ?
        ORDER BY step_number
        LIMIT {limit_per_agent}
        """
        agent_data = pd.read_sql_query(query, conn, params=(agent_id,))
        all_data.append(agent_data)
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    conn.close()
    
    return df

def analyze_agent_variation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the variation in state variables for each agent.
    
    Args:
        df: DataFrame containing agent state data
        
    Returns:
        DataFrame containing statistics about each agent's state variables
    """
    # List of numeric columns to analyze
    numeric_cols = ['position_x', 'position_y', 'resource_level', 'current_health', 
                    'step_number', 'energy_level', 'age']
    
    # Ensure all these columns exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Group by agent_id and calculate statistics
    agent_stats = []
    
    for agent_id, group in df.groupby('agent_id'):
        agent_stat = {'agent_id': agent_id}
        
        # Calculate range and standard deviation for each numeric column
        for col in numeric_cols:
            if col in group.columns:
                agent_stat[f'{col}_range'] = group[col].max() - group[col].min()
                agent_stat[f'{col}_std'] = group[col].std()
                agent_stat[f'{col}_mean'] = group[col].mean()
        
        # Calculate the total movement distance (path length)
        if 'position_x' in group.columns and 'position_y' in group.columns:
            sorted_group = group.sort_values('step_number')
            dx = sorted_group['position_x'].diff().dropna()
            dy = sorted_group['position_y'].diff().dropna()
            distances = np.sqrt(dx**2 + dy**2)
            agent_stat['total_distance'] = distances.sum()
            agent_stat['avg_step_distance'] = distances.mean()
        
        # Add to results
        agent_stats.append(agent_stat)
    
    return pd.DataFrame(agent_stats)

def plot_position_traces(df: pd.DataFrame, top_n: int = 5):
    """
    Plot the position traces for agents with the highest and lowest position ranges.
    
    Args:
        df: DataFrame containing agent state data
        top_n: Number of agents to plot from each category
    """
    # Group the data by agent_id
    agent_groups = df.groupby('agent_id')
    
    # Calculate position range for each agent
    position_ranges = []
    for agent_id, group in agent_groups:
        x_range = group['position_x'].max() - group['position_x'].min()
        y_range = group['position_y'].max() - group['position_y'].min()
        total_range = x_range + y_range
        position_ranges.append((agent_id, total_range))
    
    # Sort by position range
    position_ranges.sort(key=lambda x: x[1], reverse=True)
    
    # Get top_n agents with highest position range
    highest_range_agents = [agent_id for agent_id, _ in position_ranges[:top_n]]
    
    # Get top_n agents with lowest position range
    lowest_range_agents = [agent_id for agent_id, _ in position_ranges[-top_n:]]
    
    # Create a plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot highest range agents
    for agent_id in highest_range_agents:
        group = df[df['agent_id'] == agent_id]
        ax1.plot(group['position_x'], group['position_y'], 'o-', label=f"{agent_id[:6]}...")
    
    ax1.set_title(f'Position Traces: {top_n} Agents with Highest Range')
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')
    ax1.legend()
    ax1.grid(True)
    
    # Plot lowest range agents
    for agent_id in lowest_range_agents:
        group = df[df['agent_id'] == agent_id]
        ax2.plot(group['position_x'], group['position_y'], 'o-', label=f"{agent_id[:6]}...")
    
    ax2.set_title(f'Position Traces: {top_n} Agents with Lowest Range')
    ax2.set_xlabel('Position X')
    ax2.set_ylabel('Position Y')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('agent_position_traces.png', dpi=300)
    print("Saved position traces to 'agent_position_traces.png'")

def analyze_variable_impact(stats_df: pd.DataFrame):
    """
    Analyze which variables have the most impact on agent spread.
    
    Args:
        stats_df: DataFrame containing agent statistics
    """
    # Create a correlation plot for the statistics
    plt.figure(figsize=(14, 10))
    
    # Get columns that represent ranges or standard deviations
    range_cols = [col for col in stats_df.columns if 'range' in col or 'std' in col or 'distance' in col]
    
    if range_cols:
        # Calculate correlation matrix
        corr_matrix = stats_df[range_cols].corr()
        
        # Plot heatmap
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Correlation coefficient')
        plt.xticks(np.arange(len(range_cols)), range_cols, rotation=90)
        plt.yticks(np.arange(len(range_cols)), range_cols)
        plt.title('Correlation Between Agent State Variable Ranges')
        
        # Add correlation values
        for i in range(len(range_cols)):
            for j in range(len(range_cols)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig('variable_correlation.png', dpi=300)
        print("Saved variable correlation to 'variable_correlation.png'")
    
    # Create a bar chart showing range of each variable across agents
    plt.figure(figsize=(14, 8))
    
    # Identify likely high-spread agent (assumes the highest position range agent is the one with most spread)
    high_spread_agent = stats_df.sort_values('position_x_range', ascending=False).iloc[0]['agent_id']
    low_spread_agent = stats_df.sort_values('position_x_range').iloc[0]['agent_id']
    
    print(f"High spread agent ID: {high_spread_agent}")
    print(f"Low spread agent ID: {low_spread_agent}")
    
    # Get data for high and low spread agents
    high_spread_data = stats_df[stats_df['agent_id'] == high_spread_agent].iloc[0]
    low_spread_data = stats_df[stats_df['agent_id'] == low_spread_agent].iloc[0]
    
    # Get range columns
    range_cols = [col for col in stats_df.columns if 'range' in col]
    
    # Prepare data for plotting
    high_values = [high_spread_data[col] for col in range_cols]
    low_values = [low_spread_data[col] for col in range_cols]
    
    # Plot
    x = np.arange(len(range_cols))
    width = 0.35
    
    plt.bar(x - width/2, high_values, width, label=f'High Spread Agent ({high_spread_agent[:6]}...)')
    plt.bar(x + width/2, low_values, width, label=f'Low Spread Agent ({low_spread_agent[:6]}...)')
    
    plt.xlabel('Variables')
    plt.ylabel('Range')
    plt.title('Comparison of Variable Ranges Between High and Low Spread Agents')
    plt.xticks(x, [col.replace('_range', '') for col in range_cols], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('agent_variable_comparison.png', dpi=300)
    print("Saved agent variable comparison to 'agent_variable_comparison.png'")

def main():
    print("Loading agent data...")
    df = load_agent_data(limit_per_agent=100)
    print(f"Loaded data for {df['agent_id'].nunique()} agents ({len(df)} total states)")
    
    # Print column names to see available state variables
    print("\nAvailable state variables:")
    print(df.columns.tolist())
    
    # Analyze the variation in state variables for each agent
    print("\nAnalyzing agent variation...")
    agent_stats = analyze_agent_variation(df)
    
    # Print the agents with highest position ranges
    print("\nAgents with highest position_x range:")
    print(agent_stats.sort_values('position_x_range', ascending=False)[['agent_id', 'position_x_range']].head())
    
    print("\nAgents with highest position_y range:")
    print(agent_stats.sort_values('position_y_range', ascending=False)[['agent_id', 'position_y_range']].head())
    
    print("\nAgents with highest total movement distance:")
    print(agent_stats.sort_values('total_distance', ascending=False)[['agent_id', 'total_distance']].head())
    
    # Plot position traces
    print("\nPlotting position traces...")
    plot_position_traces(df, top_n=5)
    
    # Analyze which variables have the most impact
    print("\nAnalyzing variable impact...")
    analyze_variable_impact(agent_stats)
    
    # Save the statistics to CSV
    agent_stats.to_csv('agent_statistics.csv', index=False)
    print("\nSaved agent statistics to 'agent_statistics.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 