"""
Memory tiering strategies for the AgentFarm DB to Memory System converter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

@dataclass
class TieringContext:
    """Context information for tiering decisions."""
    step_number: int
    current_step: int
    total_steps: int
    importance_score: Optional[float] = None
    metadata: Optional[Dict] = None

class TieringStrategy(ABC):
    """Base class for memory tiering strategies."""
    
    @abstractmethod
    def determine_tier(self, context: TieringContext) -> str:
        """
        Determine which memory tier a memory should be placed in.
        
        Args:
            context: TieringContext containing information for tiering decision
            
        Returns:
            String indicating the tier: "stm", "im", or "ltm"
        """
        pass

class SimpleTieringStrategy(TieringStrategy):
    """
    Simple tiering strategy that puts all memories in STM.
    """
    
    def determine_tier(self, context: TieringContext) -> str:
        """Always return STM tier."""
        return "stm"

class StepBasedTieringStrategy(TieringStrategy):
    """
    Step-based time decay tiering strategy.
    
    Uses the relative position in the simulation to determine memory tier:
    - Most recent 10% of steps -> STM
    - Next 30% of steps -> IM
    - Remaining steps -> LTM
    """
    
    def determine_tier(self, context: TieringContext) -> str:
        """Determine tier based on step position."""
        if context.total_steps == 0:
            return "stm"
            
        # Calculate relative position in simulation
        relative_position = (context.current_step - context.step_number) / context.total_steps
        
        # Most recent 10% of steps -> STM
        if relative_position <= 0.1:
            return "stm"
        # Next 30% of steps -> IM
        elif relative_position <= 0.4:
            return "im"
        # Remaining steps -> LTM
        else:
            return "ltm"

class ImportanceAwareTieringStrategy(StepBasedTieringStrategy):
    """
    Importance-aware tiering strategy that considers both time decay and importance.
    
    Uses importance scores to potentially promote memories to higher tiers:
    - High importance (>0.8) -> Promotes to STM
    - Medium importance (>0.5) -> Promotes to IM
    - Low importance -> Uses step-based tiering
    """
    
    def determine_tier(self, context: TieringContext) -> str:
        """Determine tier based on step position and importance."""
        # If no importance score, fall back to step-based tiering
        if context.importance_score is None:
            return super().determine_tier(context)
            
        # Get base tier from step-based strategy
        base_tier = super().determine_tier(context)
        
        # Promote based on importance
        if context.importance_score > 0.8:
            return "stm"
        elif context.importance_score > 0.5 and base_tier == "ltm":
            return "im"
            
        return base_tier

def create_tiering_strategy(strategy_type: str = "simple") -> TieringStrategy:
    """
    Factory function to create tiering strategies.
    
    Args:
        strategy_type: Type of strategy to create ("simple", "step_based", or "importance_aware")
        
    Returns:
        Configured TieringStrategy instance
        
    Raises:
        ValueError: If strategy_type is invalid
    """
    strategies = {
        "simple": SimpleTieringStrategy,
        "step_based": StepBasedTieringStrategy,
        "importance_aware": ImportanceAwareTieringStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(
            f"Invalid strategy_type: {strategy_type}. "
            f"Must be one of: {list(strategies.keys())}"
        )
        
    return strategies[strategy_type]() 