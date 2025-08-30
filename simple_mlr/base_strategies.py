"""
Strategy system for SimpleMLR - Smart hyperparameter optimization made simple.

Hyperparameter optimization can be overwhelming with hundreds of possible settings.
This module provides "strategies" - pre-configured sets of parameter ranges that work
well for different situations:

- 'fast': Quick optimization for experimentation and prototyping
- 'stable': Reliable settings for production use
- 'aggressive': Tries more extreme values for maximum performance
- 'balanced': Good all-around choice for most datasets

Each algorithm (XGBoost, LightGBM, sklearn GBM) implements these strategies with
algorithm-specific parameter ranges, but they all follow the same concepts.
This means once you understand strategies for one algorithm, you know them all.

Most users will just pick a strategy name when using auto-tuners like:
    model = xgb_auto(X, y, strategy='fast')

Advanced users can override specific parameters or create custom strategies.
"""

from typing import Dict, Any, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
import optuna


class BaseStrategyConfig(ABC):
    """
    The blueprint for how each algorithm defines its optimization strategies.
    
    This is an abstract class that each algorithm (XGBoost, LightGBM, sklearn GBM)
    implements to define its own parameter ranges and constraints. It ensures
    all algorithms provide the same strategy names and interface while allowing
    algorithm-specific parameter mappings.
    
    Most users won't work with this directly - it's used internally by the
    auto-tuning classes.
    """
    
    @abstractmethod
    def get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for the specified strategy."""
        pass
    
    @abstractmethod
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get complete strategy information including description and use case."""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies for this algorithm."""
        pass
    
    
    @abstractmethod
    def suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced parameter suggestion with categorical support for this algorithm."""
        pass
    
    def get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get categorical parameter choices for the specified strategy."""
        # Default implementation returns empty dict for backward compatibility
        return {}
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the algorithm name (e.g., 'XGBoost', 'LightGBM', 'sklearn GBM')."""
        pass




class ParameterValidator:
    """
    Ensures parameter overrides are valid and won't cause errors.
    
    When you override parameters (like max_depth=5 instead of a range),
    this class checks that:
    - Parameter names are strings
    - Numerical ranges have two values and min < max
    - Categorical choices are unique strings
    - Fixed values are the right data type
    
    This prevents common mistakes and gives helpful error messages.
    """
    
    @staticmethod
    def validate_override_params(override_params: Optional[Dict[str, Union[Tuple[float, float], float, int, List[str], str]]]) -> None:
        """Validate override parameters format."""
        if override_params is None:
            return
            
        if not isinstance(override_params, dict):
            raise TypeError("override_params must be a dictionary")
        
        for param_name, param_value in override_params.items():
            if not isinstance(param_name, str):
                raise TypeError(f"Parameter name must be string, got {type(param_name)}")
            
            if isinstance(param_value, tuple):
                # Numerical range validation (existing)
                if len(param_value) != 2:
                    raise ValueError(f"Parameter range tuple for '{param_name}' must have exactly 2 values")
                if not all(isinstance(v, (int, float)) for v in param_value):
                    raise TypeError(f"Parameter range tuple for '{param_name}' must contain numbers only")
                if param_value[0] >= param_value[1]:
                    raise ValueError(f"Invalid range for '{param_name}': min_val must be < max_val")
            elif isinstance(param_value, list):
                # Categorical choices validation (new)
                if len(param_value) == 0:
                    raise ValueError(f"Categorical parameter '{param_name}' must have at least one choice")
                if not all(isinstance(choice, str) for choice in param_value):
                    raise TypeError(f"Categorical choices for '{param_name}' must be strings")
                if len(param_value) != len(set(param_value)):
                    raise ValueError(f"Categorical choices for '{param_name}' must be unique")
            elif isinstance(param_value, str):
                # Fixed categorical value validation (new) - just ensure it's a string
                pass
            elif not isinstance(param_value, (int, float)):
                # Fixed numerical value validation (existing)
                raise TypeError(f"Parameter value for '{param_name}' must be number, tuple of 2 numbers, list of strings, or string")


class StrategyMerger:
    """
    Combines strategy defaults with your custom parameter overrides.
    
    When you use a strategy like 'fast' but want to override one parameter
    (like setting max_depth=5), this class intelligently merges them:
    - Uses your override values where specified
    - Keeps strategy defaults for everything else
    - Handles both numerical ranges and categorical choices
    - Validates the final combination makes sense
    
    This lets you customize strategies without starting from scratch.
    """
    
    def __init__(self, strategy_config: BaseStrategyConfig):
        """
        Initialize with algorithm-specific strategy configuration.
        
        Args:
            strategy_config: Algorithm-specific strategy configuration instance
        """
        self.strategy_config = strategy_config
    
    def merge_strategy_with_overrides(self, strategy: str, 
                                    override_params: Optional[Dict[str, Union[Tuple[float, float], float, int, List[str], str]]] = None) -> Dict[str, Any]:
        """
        Merge strategy ranges with user overrides.
        
        Args:
            strategy: Strategy name
            override_params: User parameter overrides
            
        Returns:
            Merged configuration with 'ranges' and 'fixed_params' keys
        """
        # Get base strategy ranges
        base_ranges = self.strategy_config.get_strategy_ranges(strategy)
        
        # Get base categorical parameters
        base_categorical = self.strategy_config.get_strategy_categorical_params(strategy)
        
        # Initialize merged configuration with categorical support
        merged_config = {
            'ranges': {},
            'fixed_params': {},
            'categorical_params': {}
        }
        
        if override_params is None:
            # No overrides - use only numerical strategy ranges (no categorical params by default)
            merged_config['ranges'] = base_ranges.copy()
            # categorical_params remains empty - only add when explicitly requested
            return merged_config
        
        # Validate override parameters
        ParameterValidator.validate_override_params(override_params)
        
        # Process numerical parameters from base strategy
        for param_name, param_range in base_ranges.items():
            if param_name in override_params:
                override_value = override_params[param_name]
                if isinstance(override_value, tuple):
                    # Override with new numerical range
                    merged_config['ranges'][param_name] = override_value
                elif isinstance(override_value, list):
                    # Convert numerical to categorical - move to categorical_params
                    merged_config['categorical_params'][param_name] = override_value
                else:
                    # Fix parameter to specific value (numerical or categorical)
                    merged_config['fixed_params'][param_name] = override_value
            else:
                # Use strategy range
                merged_config['ranges'][param_name] = param_range
        
        # Process categorical parameters from base strategy - ONLY if explicitly requested
        for param_name, param_choices in base_categorical.items():
            if param_name in override_params:
                override_value = override_params[param_name]
                if isinstance(override_value, list):
                    # Override with new categorical choices
                    merged_config['categorical_params'][param_name] = override_value
                else:
                    # Fix categorical parameter to specific value
                    merged_config['fixed_params'][param_name] = override_value
            # Note: Do NOT automatically include base categorical params unless explicitly requested
        
        # Add any additional override parameters not in base strategy
        for param_name, param_value in override_params.items():
            if param_name not in base_ranges and param_name not in base_categorical:
                if isinstance(param_value, tuple):
                    merged_config['ranges'][param_name] = param_value
                elif isinstance(param_value, list):
                    merged_config['categorical_params'][param_name] = param_value
                else:
                    merged_config['fixed_params'][param_name] = param_value
        
        return merged_config


class UniversalStrategyTypes:
    """
    Defines the strategy names that work across all algorithms.
    
    This class lists the standard strategy names like 'fast', 'stable', etc.
    that every algorithm should support. It ensures consistency - when you
    use strategy='fast' with XGBoost, LightGBM, or sklearn GBM, you get
    similar behavior even though the underlying parameters are different.
    
    Required strategies that every algorithm must implement:
    - 'default': Balanced, general-purpose optimization
    - 'fast': Speed-optimized for quick experiments  
    - 'stable': Conservative, reliable settings
    - 'balanced': Good all-around choice
    """
    
    # Core strategy types - should be implemented by all algorithms
    REQUIRED_STRATEGIES = [
        'default',    # General-purpose balanced optimization
        'fast',       # Speed-optimized for quick experimentation
        'stable',     # Production-ready reliability focus
        'balanced'    # Balanced approach for unknown data characteristics
    ]
    
    # Extended strategy types - optional but recommended
    EXTENDED_STRATEGIES = [
        'aggressive',     # High-variance complex pattern detection
        'conservative',   # Maximum stability with heavy regularization
        'regularized',    # Strong overfitting prevention
        'diversity',      # Maximum parameter variation for ensemble building
        'competition'     # State-of-the-art competition settings
    ]
    
    @classmethod
    def get_all_strategy_types(cls) -> List[str]:
        """Get all defined strategy types."""
        return cls.REQUIRED_STRATEGIES + cls.EXTENDED_STRATEGIES
    
    @classmethod
    def get_required_strategies(cls) -> List[str]:
        """Get strategies that must be implemented by all algorithms."""
        return cls.REQUIRED_STRATEGIES.copy()
    
    @classmethod
    def validate_algorithm_strategies(cls, algorithm_strategies: List[str]) -> None:
        """Validate that an algorithm implements all required strategies."""
        missing = set(cls.REQUIRED_STRATEGIES) - set(algorithm_strategies)
        if missing:
            missing_str = ', '.join(missing)
            raise ValueError(f"Algorithm must implement required strategies: {missing_str}")




class MultiAlgorithmStrategyManager:
    """
    Manages strategies across different algorithms for advanced use cases.
    
    This class helps when you want to:
    - Compare how different algorithms implement the same strategy
    - Build ensembles with complementary parameter settings
    - Ensure all algorithms support required strategies
    - Get recommendations for diverse ensemble members
    
    Most users won't need this - it's for advanced ensemble building
    and algorithm comparison workflows.
    """
    
    def __init__(self):
        """Initialize strategy manager."""
        self.registered_algorithms = {}
    
    def register_algorithm(self, algorithm_name: str, strategy_config: BaseStrategyConfig) -> None:
        """
        Register a new algorithm with its strategy configuration.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., 'xgboost', 'lightgbm')
            strategy_config: Strategy configuration instance for the algorithm
        """
        # Validate that algorithm implements required strategies
        available_strategies = strategy_config.get_available_strategies()
        UniversalStrategyTypes.validate_algorithm_strategies(available_strategies)
        
        self.registered_algorithms[algorithm_name] = strategy_config
    
    def get_algorithm_strategies(self, algorithm_name: str) -> List[str]:
        """Get available strategies for a specific algorithm."""
        if algorithm_name not in self.registered_algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not registered. "
                           f"Available: {list(self.registered_algorithms.keys())}")
        return self.registered_algorithms[algorithm_name].get_available_strategies()
    
    def get_common_strategies(self) -> List[str]:
        """Get strategies available across all registered algorithms."""
        if not self.registered_algorithms:
            return []
        
        algorithm_strategies = [
            set(config.get_available_strategies()) 
            for config in self.registered_algorithms.values()
        ]
        
        return list(set.intersection(*algorithm_strategies))
    
    def compare_strategy_ranges(self, strategy: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Compare parameter ranges for a strategy across all algorithms.
        Useful for understanding algorithm differences.
        
        Args:
            strategy: Strategy name to compare
            
        Returns:
            Dictionary mapping algorithm names to their parameter ranges
        """
        comparison = {}
        
        for algorithm_name, config in self.registered_algorithms.items():
            try:
                ranges = config.get_strategy_ranges(strategy)
                comparison[algorithm_name] = ranges
            except ValueError:
                # Algorithm doesn't support this strategy
                comparison[algorithm_name] = None
        
        return comparison
    
    def recommend_ensemble_strategies(self, diversity_level: str = 'medium') -> Dict[str, str]:
        """
        Recommend complementary strategies for ensemble building.
        
        Args:
            diversity_level: 'low', 'medium', or 'high' diversity
            
        Returns:
            Dictionary mapping algorithm names to recommended strategies
        """
        recommendations = {}
        common_strategies = self.get_common_strategies()
        
        if diversity_level == 'low':
            # Use same strategy across algorithms for consistency
            strategy = 'stable' if 'stable' in common_strategies else common_strategies[0]
            for algorithm in self.registered_algorithms:
                recommendations[algorithm] = strategy
                
        elif diversity_level == 'medium':
            # Use complementary strategies
            strategy_rotation = ['stable', 'balanced', 'default']
            available_rotation = [s for s in strategy_rotation if s in common_strategies]
            
            for i, algorithm in enumerate(self.registered_algorithms):
                strategy_idx = i % len(available_rotation)
                recommendations[algorithm] = available_rotation[strategy_idx]
                
        else:  # high diversity
            # Use maximum diversity strategies
            diversity_strategies = ['diversity', 'aggressive', 'fast', 'regularized']
            available_diversity = [s for s in diversity_strategies if s in common_strategies]
            
            if not available_diversity:
                available_diversity = common_strategies
            
            for i, algorithm in enumerate(self.registered_algorithms):
                strategy_idx = i % len(available_diversity)
                recommendations[algorithm] = available_diversity[strategy_idx]
        
        return recommendations


# Global strategy manager instance
global_strategy_manager = MultiAlgorithmStrategyManager()


def get_universal_strategy_info(strategy_type: str) -> Dict[str, str]:
    """
    Get universal information about a strategy type that applies across algorithms.
    
    Args:
        strategy_type: One of the universal strategy types
        
    Returns:
        Dictionary with description and use_case
    """
    strategy_descriptions = {
        'default': {
            'description': 'General-purpose optimization with balanced parameter ranges',
            'use_case': 'Standard optimization for most datasets and unknown characteristics'
        },
        'fast': {
            'description': 'Speed-optimized ranges for rapid experimentation',
            'use_case': 'Quick prototyping, hyperparameter exploration, time-constrained optimization'
        },
        'stable': {
            'description': 'Production-ready, reliable performance',
            'use_case': 'Production deployment, reliable predictions, ensemble base models'
        },
        'balanced': {
            'description': 'Balanced approach for unknown data characteristics',
            'use_case': 'General-purpose modeling, baseline establishment, educational use'
        },
        'aggressive': {
            'description': 'High-variance, complex pattern detection',
            'use_case': 'Complex relationships, high-variance ensemble members, competition settings'
        },
        'conservative': {
            'description': 'Maximum stability with heavy regularization focus',
            'use_case': 'Production stability, ensemble base learners, risk-averse applications'
        },
        'regularized': {
            'description': 'Strong overfitting prevention with high regularization',
            'use_case': 'Noisy data, small datasets, overfitting-prone scenarios'
        },
        'diversity': {
            'description': 'Maximum parameter variation for ensemble building',
            'use_case': 'Ensemble building, model diversity maximization, exploration'
        },
        'competition': {
            'description': 'State-of-the-art configuration for maximum performance',
            'use_case': 'Kaggle competitions, maximum performance scenarios, research'
        }
    }
    
    if strategy_type not in strategy_descriptions:
        available = ', '.join(strategy_descriptions.keys())
        raise ValueError(f"Unknown universal strategy type '{strategy_type}'. Available: {available}")
    
    return strategy_descriptions[strategy_type]