"""
Configuration settings for benchmarks.

This module defines the configuration structure and default settings
for running benchmarks on the AgentMemory system.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class StorageConfig:
    """Configuration for storage benchmarks."""
    batch_sizes: List[int] = field(default_factory=lambda: [10, 100, 1000, 10000])
    data_complexity_levels: List[int] = field(default_factory=lambda: [1, 5, 10])
    num_samples: int = 1000


@dataclass
class CompressionConfig:
    """Configuration for compression benchmarks."""
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    target_dimensions: List[int] = field(default_factory=lambda: [128, 256, 512])
    training_iterations: int = 1000


@dataclass
class MemoryTransitionConfig:
    """Configuration for memory transition benchmarks."""
    transition_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "recency": 0.5, 
        "relevance": 0.3, 
        "frequency": 0.2
    })


@dataclass
class RetrievalConfig:
    """Configuration for retrieval benchmarks."""
    dataset_size: int = 10000
    query_counts: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    vector_dimensions: List[int] = field(default_factory=lambda: [128, 256, 512])
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])


@dataclass
class ScalabilityConfig:
    """Configuration for scalability benchmarks."""
    agent_counts: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])
    memory_sizes: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    parallel_operations: List[int] = field(default_factory=lambda: [1, 4, 8, 16])


@dataclass
class IntegrationConfig:
    """Configuration for integration benchmarks."""
    frameworks: List[str] = field(default_factory=lambda: ["langchain", "llama_index", "custom"])
    operations_per_test: int = 1000
    concurrent_clients: List[int] = field(default_factory=lambda: [1, 5, 10, 20])


@dataclass
class BenchmarkConfig:
    """Main configuration for benchmarks."""
    storage: StorageConfig = field(default_factory=StorageConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory_transition: MemoryTransitionConfig = field(default_factory=MemoryTransitionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scalability: ScalabilityConfig = field(default_factory=ScalabilityConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    output_dir: str = "benchmarks/results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: Optional[str] = None) -> str:
        """Save configuration to a JSON file.
        
        Args:
            path: Optional file path to save to. If not provided, uses default.
            
        Returns:
            The path where the config was saved.
        """
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, "benchmark_config.json")
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> "BenchmarkConfig":
        """Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON configuration file.
            
        Returns:
            Loaded BenchmarkConfig instance.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        storage = StorageConfig(**config_dict.get("storage", {}))
        compression = CompressionConfig(**config_dict.get("compression", {}))
        memory_transition = MemoryTransitionConfig(**config_dict.get("memory_transition", {}))
        retrieval = RetrievalConfig(**config_dict.get("retrieval", {}))
        scalability = ScalabilityConfig(**config_dict.get("scalability", {}))
        integration = IntegrationConfig(**config_dict.get("integration", {}))
        
        return cls(
            storage=storage,
            compression=compression,
            memory_transition=memory_transition,
            retrieval=retrieval,
            scalability=scalability,
            integration=integration,
            output_dir=config_dict.get("output_dir", "benchmarks/results")
        ) 