"""
Sklearn Gradient Boosting Machine strategy configurations for SimpleMLR multi-algorithm framework.

This module provides GBM-specific strategy implementations, translating the universal
strategy concepts to sklearn GBM's parameter space while maintaining mathematical relationships
and constraint enforcement.

Key GBM-specific considerations:
- Limited regularization options compared to XGBoost/LightGBM
- No GPU support (CPU only)
- Different parameter names: max_features vs colsample_bytree
- Simpler parameter space but still needs constraint management
- Strong emphasis on learning_rate * n_estimators relationship
"""

from typing import Dict, Any, Tuple, Union, Optional, List
import optuna

# GBM Strategy Configurations - Adapted for sklearn GradientBoostingRegressor
GBM_STRATEGY_CONFIGS = {
    "default": {
        "description": "General-purpose optimization with balanced parameter ranges",
        "use_case": "Standard optimization for most datasets",
        "ranges": {
            'n_estimators': (100, 500),
            'max_depth': (3, 6),  # Shallower than XGBoost due to no L1/L2 regularization
            'learning_rate': (0.05, 0.15),  # More conservative than XGBoost
            'subsample': (0.8, 1.0),
            'max_features': (0.3, 0.7),  # Replaces colsample_bytree
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'min_impurity_decrease': (0.0, 0.01)  # Similar to gamma but different scale
        }
    },

    "fast": {
        "description": "Speed-optimized ranges for rapid experimentation",
        "use_case": "Quick prototyping, hyperparameter exploration, time-constrained optimization",
        "estimated_speedup": "2-3x faster than default",
        "ranges": {
            'n_estimators': (50, 200),
            'max_depth': (2, 4),  # Shallow trees for speed
            'learning_rate': (0.1, 0.3),
            'subsample': (0.7, 0.9),
            'max_features': (0.5, 0.8),
            'min_samples_split': (10, 50),  # Higher values for faster splits
            'min_samples_leaf': (5, 20),
            'min_impurity_decrease': (0.0, 0.001)
        }
    },

    "aggressive": {
        "description": "High-variance, complex pattern detection",
        "use_case": "Complex relationships, high-variance ensemble members, competition settings",
        "ranges": {
            'n_estimators': (100, 500),
            'max_depth': (5, 12),  # Deeper trees for complex patterns
            'learning_rate': (0.05, 0.15),  # More conservative than XGBoost aggressive
            'subsample': (0.85, 1.0),
            'max_features': (0.7, 1.0),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5),
            'min_impurity_decrease': (0.0, 0.0001)
        }
    },

    "balanced": {
        "description": "Balanced approach for unknown data characteristics",
        "use_case": "General-purpose modeling, baseline establishment, educational use",
        "ranges": {
            'n_estimators': (200, 600),
            'max_depth': (3, 6),
            'learning_rate': (0.05, 0.1),
            'subsample': (0.8, 1.0),
            'max_features': (0.5, 0.8),
            'min_samples_split': (5, 30),
            'min_samples_leaf': (2, 15),
            'min_impurity_decrease': (0.0, 0.005)
        }
    },

    "stable": {
        "description": "Production-ready, reliable performance",
        "use_case": "Production deployment, reliable predictions, ensemble members",
        "ranges": {
            'n_estimators': (500, 1500),
            'max_depth': (3, 5),
            'learning_rate': (0.02, 0.08),
            'subsample': (0.8, 0.95),
            'max_features': (0.6, 0.9),
            'min_samples_split': (10, 40),
            'min_samples_leaf': (5, 20),
            'min_impurity_decrease': (0.0, 0.002),
            'validation_fraction': (0.1, 0.2),  # For early stopping
            'n_iter_no_change': (5, 20)
        }
    },

    "conservative": {
        "description": "Maximum stability with very slow learning for reliability",
        "use_case": "Production stability, ensemble base learners, risk-averse applications",
        "ranges": {
            'n_estimators': (2000, 5000),
            'max_depth': (2, 4),
            'learning_rate': (0.005, 0.03),  # Very low learning rate
            'subsample': (0.7, 0.85),
            'max_features': (0.5, 0.7),
            'min_samples_split': (20, 60),
            'min_samples_leaf': (10, 30),
            'min_impurity_decrease': (0.001, 0.01),
            'validation_fraction': (0.15, 0.25),
            'n_iter_no_change': (10, 30)
        }
    },

    "regularized": {
        "description": "Strong structural regularization for overfitting prevention",
        "use_case": "Noisy data, small datasets, overfitting-prone scenarios",
        "note": "Uses structural constraints since sklearn lacks L1/L2 regularization",
        "ranges": {
            'n_estimators': (100, 400),
            'max_depth': (2, 4),  # Very shallow trees
            'learning_rate': (0.01, 0.08),
            'subsample': (0.6, 0.8),
            'max_features': (0.3, 0.5),  # Aggressive feature subsampling
            'min_samples_split': (20, 100),  # High minimum for splits
            'min_samples_leaf': (10, 50),  # Large leaf size
            'min_impurity_decrease': (0.001, 0.02)  # High threshold
        }
    },

    "diversity": {
        "description": "Maximum parameter variation for ensemble building",
        "use_case": "Ensemble building, model diversity maximization, exploration",
        "ranges": {
            'n_estimators': (50, 2000),
            'max_depth': (2, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1.0),
            'max_features': (0.2, 1.0),
            'min_samples_split': (2, 100),
            'min_samples_leaf': (1, 50),
            'min_impurity_decrease': (0.0, 0.02)
        }
    },

    "competition": {
        "description": "Competition-optimized based on 2024-2025 insights",
        "use_case": "Kaggle competitions, maximum performance with ensemble potential",
        "note": "Best used with HistGradientBoostingRegressor for large datasets",
        "ranges": {
            'n_estimators': (1000, 3000),
            'max_depth': (2, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1.0),
            'max_features': (0.2, 1.0),
            'min_samples_split': (2, 100),
            'min_samples_leaf': (1, 50),
            'min_impurity_decrease': (0.0, 0.02)
        }
    }
}


# GBM Categorical Parameter Configurations
GBM_CATEGORICAL_PARAMETER_CHOICES = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_features': ['auto', 'sqrt', 'log2'],  # Note: can also be numerical
    'validation_fraction': ['0.1', '0.15', '0.2'],  # Common fixed values
    'warm_start': ['False', 'True'],
    'ccp_alpha': ['0.0'],  # Usually fixed to 0.0 for GBM
}


# Strategy-specific categorical parameter configurations for GBM
GBM_STRATEGY_CATEGORICAL_CONFIGS = {
    "default": {
        'loss': ['squared_error', 'absolute_error'],
        'criterion': ['friedman_mse', 'squared_error']
    },
    
    "fast": {
        'loss': ['squared_error'],  # Fastest loss function
        'criterion': ['friedman_mse']  # Fastest criterion
    },
    
    "aggressive": {
        'loss': ['squared_error', 'absolute_error', 'huber'],
        'criterion': ['friedman_mse', 'squared_error']
    },
    
    "balanced": {
        'loss': ['squared_error', 'absolute_error'],
        'criterion': ['friedman_mse']
    },
    
    "stable": {
        'loss': ['squared_error'],  # Most stable
        'criterion': ['friedman_mse']  # Most stable
    },
    
    "conservative": {
        'loss': ['squared_error'],
        'criterion': ['friedman_mse']
    },
    
    "regularized": {
        'loss': ['squared_error', 'huber'],  # Robust losses
        'criterion': ['friedman_mse']
    },
    
    "diversity": {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'criterion': ['friedman_mse', 'squared_error']
    },
    
    "competition": {
        'loss': ['squared_error', 'absolute_error', 'huber'],
        'criterion': ['friedman_mse', 'squared_error']
    }
}


# GBM Progressive Optimization Groups - Adapted for sklearn GBM parameters
GBM_PROGRESSIVE_GROUPS = {
    "default": {
        "description": "Research-proven 4-stage sequence adapted for GBM",
        "stages": [
            {
                "name": "learning_dynamics",
                "display_name": "Learning Dynamics",
                "params": ['learning_rate', 'n_estimators'],  # Core GBM relationship
                "trials_ratio": 0.35,
                "description": "Establishes fundamental model capacity with GBM"
            },
            {
                "name": "tree_architecture",
                "display_name": "Tree Architecture",
                "params": ['max_depth'],  # GBM tree complexity
                "trials_ratio": 0.35,
                "description": "Controls GBM tree complexity and depth"
            },
            {
                "name": "regularization",
                "display_name": "Regularization",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.20,
                "description": "GBM-specific regularization tuning"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],  # GBM sampling strategies
                "trials_ratio": 0.10,
                "description": "GBM data and feature sampling"
            }
        ]
    },

    "fast": {
        "description": "Speed-optimized 3-stage sequence for GBM",
        "stages": [
            {
                "name": "tree_architecture",
                "display_name": "Tree Structure",
                "params": ['max_depth', 'n_estimators'],
                "trials_ratio": 0.50,
                "description": "GBM tree structure optimization priority"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Rate",
                "params": ['learning_rate'],
                "trials_ratio": 0.35,
                "description": "Speed vs accuracy balance for GBM"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.15,
                "description": "Final GBM sampling optimizations"
            }
        ]
    },

    "aggressive": {
        "description": "Complex pattern detection with GBM's stable gradient boosting",
        "stages": [
            {
                "name": "tree_architecture",
                "display_name": "Tree Complexity",
                "params": ['max_depth', 'n_estimators'],
                "trials_ratio": 0.55,
                "description": "Maximizing GBM tree complexity"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Rate",
                "params": ['learning_rate'],
                "trials_ratio": 0.30,
                "description": "Aggressive learning with GBM"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.15,
                "description": "Data sampling for complex GBM patterns"
            }
        ]
    },

    "balanced": {
        "description": "Comprehensive 4-stage GBM optimization",
        "stages": [
            {
                "name": "learning_dynamics",
                "display_name": "Learning Dynamics",
                "params": ['learning_rate', 'n_estimators'],
                "trials_ratio": 0.35,
                "description": "GBM learning foundation"
            },
            {
                "name": "tree_architecture",
                "display_name": "Tree Architecture",
                "params": ['max_depth'],
                "trials_ratio": 0.30,
                "description": "Balancing GBM complexity"
            },
            {
                "name": "regularization",
                "display_name": "Regularization",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.25,
                "description": "GBM overfitting prevention"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.10,
                "description": "Final GBM sampling refinements"
            }
        ]
    },

    "stable": {
        "description": "Production-ready 4-stage GBM sequence",
        "stages": [
            {
                "name": "regularization",
                "display_name": "Regularization",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.35,
                "description": "GBM stability through regularization"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Dynamics",
                "params": ['learning_rate', 'n_estimators'],
                "trials_ratio": 0.30,
                "description": "Stable GBM learning progression"
            },
            {
                "name": "tree_architecture",
                "display_name": "Tree Architecture",
                "params": ['max_depth'],
                "trials_ratio": 0.25,
                "description": "Conservative GBM complexity"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.10,
                "description": "Robust GBM sampling"
            }
        ]
    },

    "conservative": {
        "description": "Maximum stability with heavy GBM regularization",
        "stages": [
            {
                "name": "regularization",
                "display_name": "Regularization",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.45,
                "description": "Maximum GBM regularization"
            },
            {
                "name": "tree_architecture",
                "display_name": "Tree Architecture",
                "params": ['max_depth'],
                "trials_ratio": 0.25,
                "description": "Conservative GBM tree structure"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Rate",
                "params": ['learning_rate', 'n_estimators'],
                "trials_ratio": 0.20,
                "description": "Slow, stable GBM learning"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.10,
                "description": "Conservative GBM sampling"
            }
        ]
    },

    "regularized": {
        "description": "Heavy regularization sequence for noisy data with GBM",
        "stages": [
            {
                "name": "regularization",
                "display_name": "Regularization",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.50,
                "description": "Combat overfitting with GBM regularization"
            },
            {
                "name": "tree_architecture",
                "display_name": "Tree Architecture",
                "params": ['max_depth'],
                "trials_ratio": 0.25,
                "description": "Simple GBM structures"
            },
            {
                "name": "sampling",
                "display_name": "Sampling",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.15,
                "description": "Robust sampling for noisy data"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Rate",
                "params": ['learning_rate', 'n_estimators'],
                "trials_ratio": 0.10,
                "description": "Conservative GBM learning"
            }
        ]
    },

    "diversity": {
        "description": "Maximum parameter variation for GBM ensemble building",
        "stages": [
            {
                "name": "tree_architecture",
                "display_name": "Tree Diversity",
                "params": ['max_depth', 'n_estimators'],
                "trials_ratio": 0.40,
                "description": "Maximum GBM structure variation"
            },
            {
                "name": "learning_dynamics",
                "display_name": "Learning Diversity",
                "params": ['learning_rate'],
                "trials_ratio": 0.25,
                "description": "Diverse GBM learning rates"
            },
            {
                "name": "regularization",
                "display_name": "Regularization Diversity",
                "params": ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.20,
                "description": "Varied GBM regularization"
            },
            {
                "name": "sampling",
                "display_name": "Sampling Diversity",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.15,
                "description": "Diverse GBM sampling strategies"
            }
        ]
    },

    "competition": {
        "description": "Optimized GBM competition optimization",
        "stages": [
            {
                "name": "learning_dynamics",
                "display_name": "Learning Rate Focus",
                "params": ['learning_rate', 'n_estimators'],
                "trials_ratio": 0.45,
                "description": "Optimal GBM learning foundation"
            },
            {
                "name": "structure_and_regularization",
                "display_name": "Structure & Regularization",
                "params": ['max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf'],
                "trials_ratio": 0.40,
                "description": "Joint GBM complexity and generalization"
            },
            {
                "name": "sampling_refinement",
                "display_name": "Sampling Refinement",
                "params": ['subsample', 'max_features'],
                "trials_ratio": 0.15,
                "description": "Final GBM sampling optimization"
            }
        ]
    }
}

# GBM parameters that should use log-scale sampling
GBM_LOG_SCALE_PARAMS = {
    'learning_rate': True,
    'min_weight_fraction_leaf': True,  # Small fractions benefit from log scale
    'alpha': False,  # Quantile parameter, linear scale is better
}

# GBM parameters that should use integer sampling
GBM_INTEGER_PARAMS = {
    'n_estimators': True,
    'max_depth': True,
    'min_samples_split': True,
    'min_samples_leaf': True,
    'max_leaf_nodes': True,
    'random_state': True,
    'verbose': True,
    'warm_start': False,  # Boolean parameter
    'validation_fraction': False,  # Float parameter
    'n_iter_no_change': True,
    'tol': False,  # Float tolerance parameter
}


# Utility Functions
def get_gbm_strategy_ranges(strategy: str) -> Dict[str, tuple]:
    """Get GBM parameter ranges for a strategy."""
    if strategy not in GBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown GBM strategy '{strategy}'. Available strategies: {list(GBM_STRATEGY_CONFIGS.keys())}")
    return GBM_STRATEGY_CONFIGS[strategy]['ranges']


def get_gbm_strategy_info(strategy: str) -> Dict[str, Any]:
    """Get complete GBM strategy information."""
    if strategy not in GBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown GBM strategy '{strategy}'. Available strategies: {list(GBM_STRATEGY_CONFIGS.keys())}")
    return GBM_STRATEGY_CONFIGS[strategy]


def list_available_gbm_strategies() -> list:
    """Get list of available GBM strategies."""
    return list(GBM_STRATEGY_CONFIGS.keys())


def get_gbm_progressive_groups(strategy: str) -> Dict[str, Any]:
    """Get GBM progressive optimization groups."""
    if strategy not in GBM_PROGRESSIVE_GROUPS:
        raise ValueError(f"GBM strategy '{strategy}' does not support progressive optimization. "
                         f"Available progressive strategies: {list(GBM_PROGRESSIVE_GROUPS.keys())}")
    return GBM_PROGRESSIVE_GROUPS[strategy]


def get_all_gbm_progressive_strategies() -> list:
    """Get all GBM progressive strategies."""
    return list(GBM_PROGRESSIVE_GROUPS.keys())


def is_gbm_log_scale_param(param_name: str) -> bool:
    """Check if a GBM parameter should use log-scale sampling."""
    return GBM_LOG_SCALE_PARAMS.get(param_name, False)


def is_gbm_integer_param(param_name: str) -> bool:
    """Check if a GBM parameter should use integer sampling."""
    return GBM_INTEGER_PARAMS.get(param_name, False)


def suggest_gbm_parameter(trial, param_name: str, param_info):
    """
    Enhanced GBM parameter suggestion with categorical support.
    
    Args:
        trial: Optuna trial object
        param_name: Parameter name
        param_info: Either (min_val, max_val) tuple for numerical or list of choices for categorical
    """
    # Handle categorical parameters
    if isinstance(param_info, list):
        return trial.suggest_categorical(param_name, param_info)
    
    # Handle numerical parameters (existing logic)
    min_val, max_val = param_info
    if is_gbm_integer_param(param_name):
        if is_gbm_log_scale_param(param_name):
            return trial.suggest_int(param_name, int(min_val), int(max_val), log=True)
        else:
            return trial.suggest_int(param_name, int(min_val), int(max_val))
    else:
        if is_gbm_log_scale_param(param_name):
            # For log scale, ensure min_val > 0 to avoid optuna error
            if float(min_val) <= 0:
                # Use linear scale if min_val is 0 or negative
                return trial.suggest_float(param_name, float(min_val), float(max_val))
            else:
                return trial.suggest_float(param_name, float(min_val), float(max_val), log=True)
        else:
            return trial.suggest_float(param_name, float(min_val), float(max_val))


def validate_gbm_override_params(override_params: Dict[str, Union[Tuple[float, float], float, int, List[str], str]]) -> None:
    """Validate GBM override parameters format."""
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


def merge_gbm_strategy_with_overrides(strategy: str, override_params: Optional[
    Dict[str, Union[Tuple[float, float], float, int]]] = None) -> Dict[str, Any]:
    """Merge GBM strategy with user overrides."""
    if strategy not in GBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown GBM strategy '{strategy}'. Available strategies: {list(GBM_STRATEGY_CONFIGS.keys())}")

    # Get base strategy ranges
    base_ranges = GBM_STRATEGY_CONFIGS[strategy]['ranges'].copy()

    # Initialize merged configuration
    merged_config = {
        'ranges': {},
        'fixed_params': {}
    }

    if override_params is None:
        # No overrides - use all strategy ranges
        merged_config['ranges'] = base_ranges
        return merged_config

    # Validate override parameters
    validate_gbm_override_params(override_params)

    # Process each parameter
    for param_name, param_value in base_ranges.items():
        if param_name in override_params:
            override_value = override_params[param_name]
            if isinstance(override_value, tuple):
                # Override with new range
                merged_config['ranges'][param_name] = override_value
            else:
                # Fix parameter to specific value
                merged_config['fixed_params'][param_name] = override_value
        else:
            # Use strategy range
            merged_config['ranges'][param_name] = param_value

    # Add any additional override parameters not in strategy
    for param_name, param_value in override_params.items():
        if param_name not in base_ranges:
            if isinstance(param_value, tuple):
                merged_config['ranges'][param_name] = param_value
            else:
                merged_config['fixed_params'][param_name] = param_value

    return merged_config


def recommend_gbm_strategy(dataset_size: int = None, target_speed: str = None,
                          has_categorical: bool = None, is_noisy: bool = None) -> str:
    """Recommend a GBM strategy based on dataset characteristics."""
    # GBM is naturally slower, so adjust recommendations accordingly
    if target_speed == "fast" or (dataset_size and dataset_size < 500):
        return "fast"  # GBM is slower, so fast strategy for small datasets
    elif is_noisy:
        return "regularized"  # GBM has strong natural regularization
    elif target_speed == "stable" or (dataset_size and dataset_size > 50000):
        return "stable"  # GBM excels at stable performance
    elif has_categorical:
        return "balanced"  # GBM handles categoricals reasonably well
    else:
        return "default"  # General purpose


# ========================================
# GBM Categorical Parameter Helper Functions
# ========================================

def get_gbm_categorical_parameter_choices() -> Dict[str, List[str]]:
    """Get all available categorical parameter choices for GBM."""
    return GBM_CATEGORICAL_PARAMETER_CHOICES.copy()


def get_gbm_strategy_categorical_params(strategy: str) -> Dict[str, List[str]]:
    """Get categorical parameter choices for a specific GBM strategy."""
    if strategy not in GBM_STRATEGY_CATEGORICAL_CONFIGS:
        return {}
    return GBM_STRATEGY_CATEGORICAL_CONFIGS[strategy].copy()


def is_gbm_categorical_param(param_name: str) -> bool:
    """Check if a parameter is a GBM categorical parameter."""
    return param_name in GBM_CATEGORICAL_PARAMETER_CHOICES


def get_gbm_categorical_choices(param_name: str) -> List[str]:
    """Get the available choices for a GBM categorical parameter."""
    if param_name not in GBM_CATEGORICAL_PARAMETER_CHOICES:
        raise ValueError(f"'{param_name}' is not a known GBM categorical parameter. "
                        f"Available: {list(GBM_CATEGORICAL_PARAMETER_CHOICES.keys())}")
    return GBM_CATEGORICAL_PARAMETER_CHOICES[param_name].copy()


def validate_gbm_categorical_choice(param_name: str, choice: str) -> bool:
    """Validate that a choice is valid for a GBM categorical parameter."""
    if not is_gbm_categorical_param(param_name):
        return False
    return choice in GBM_CATEGORICAL_PARAMETER_CHOICES[param_name]


def suggest_gbm_categorical_parameter(trial, param_name: str, choices: List[str]):
    """Suggest a GBM categorical parameter value for optimization."""
    return trial.suggest_categorical(param_name, choices)