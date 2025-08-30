"""
LightGBM strategy configurations for SimpleMLR multi-algorithm framework.

This module provides LightGBM-specific strategy implementations, translating the universal
strategy concepts to LightGBM's parameter space while maintaining mathematical relationships
and constraint enforcement.

Key LightGBM-specific considerations:
- num_leaves is the primary complexity controller (not max_depth)
- Critical constraint: num_leaves < 2^max_depth
- Different parameter names: feature_fraction vs colsample_bytree
- LightGBM-specific regularization: lambda_l1, lambda_l2, min_sum_hessian_in_leaf
"""

from typing import Dict, Any, Tuple, Union, Optional, List
import optuna


LGBM_STRATEGY_CONFIGS = {
    "default": {
        "description": "General-purpose LightGBM optimization with research-proven ranges",
        "use_case": "Standard optimization for most datasets with LightGBM",
        'ranges': {
            'num_boost_round': (100, 1000),
            'num_leaves': (31, 127),
            'max_depth': (4, 10),
            'learning_rate': (0.01, 0.2),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'lambda_l1': (1e-8, 10.0),
            'lambda_l2': (1e-8, 10.0),
            'min_data_in_leaf': (20, 100),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    },

    "fast": {
        "description": "Speed-optimized for rapid experimentation",
        "use_case": "Quick prototyping, hyperparameter exploration, time-constrained optimization",
        "estimated_speedup": "2-3x faster than default",
        'ranges': {
            'num_boost_round': (50, 200),
            'num_leaves': (10, 100),
            'max_depth': (3, 6),
            'learning_rate': (0.01, 0.2),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'lambda_l1': (1e-8, 2.0),
            'lambda_l2': (1e-8, 2.0),
            'min_data_in_leaf': (5, 50),
            'min_sum_hessian_in_leaf': (1e-3, 1.0)
        }
    },

    "aggressive": {
        "description": "Complex pattern detection with controlled risk",
        "use_case": "Complex relationships, high-variance ensemble members, competition settings",
        'ranges': {
            'num_boost_round': (100, 1000),
            'num_leaves': (63, 1024),
            'max_depth': (6, 16),
            'learning_rate': (0.001, 0.3),
            'feature_fraction': (0.1, 1.0),
            'bagging_fraction': (0.1, 1.0),
            'lambda_l1': (1e-8, 100.0),
            'lambda_l2': (1e-8, 100.0),
            'min_data_in_leaf': (1, 300),
            'min_sum_hessian_in_leaf': (1e-3, 10.0),
            'bagging_freq': (1, 10)
        }
    },

    "balanced": {
        "description": "Comprehensive approach for unknown data characteristics",
        "use_case": "General-purpose modeling, baseline establishment, educational use",
        'ranges': {
            'num_boost_round': (100, 500),
            'num_leaves': (8, 256),
            'max_depth': (3, 12),
            'learning_rate': (0.005, 0.2),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'lambda_l1': (0.01, 10.0),
            'lambda_l2': (0.01, 10.0),
            'min_data_in_leaf': (1, 100),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    },

    "stable": {
        "description": "Production-ready with reliable performance",
        "use_case": "Production deployment, reliable predictions, ensemble members",
        'ranges': {
            'num_boost_round': (200, 1000),
            'num_leaves': (16, 64),
            'max_depth': (4, 8),
            'learning_rate': (0.01, 0.05),
            'feature_fraction': (0.8, 1.0),
            'bagging_fraction': (0.8, 1.0),
            'lambda_l1': (0.1, 1.0),
            'lambda_l2': (0.1, 1.0),
            'min_data_in_leaf': (20, 100),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    },

    "conservative": {
        "description": "Maximum stability with very slow learning for reliability",
        "use_case": "Production stability, ensemble base learners, risk-averse applications",
        'ranges': {
            'num_boost_round': (1000, 5000),
            'num_leaves': (8, 32),
            'max_depth': (3, 6),
            'learning_rate': (0.001, 0.02),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'lambda_l1': (0.1, 1.5),
            'lambda_l2': (0.1, 1.5),
            'min_data_in_leaf': (50, 200),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    },

    "regularized": {
        "description": "Highest regularization for strong overfitting prevention",
        "use_case": "Noisy data, small datasets, overfitting-prone scenarios",
        'ranges': {
            'num_boost_round': (100, 500),
            'num_leaves': (8, 32),
            'max_depth': (3, 5),
            'learning_rate': (0.005, 0.05),
            'feature_fraction': (0.5, 0.8),
            'bagging_fraction': (0.5, 0.8),
            'lambda_l1': (0.5, 50.0),
            'lambda_l2': (0.5, 50.0),
            'min_data_in_leaf': (50, 200),
            'min_sum_hessian_in_leaf': (0.1, 5.0),
            #'boosting_type': 'dart'    #Special case for regularized
        }
    },

    "diversity": {
        "description": "Maximum parameter variation for ensemble building",
        "use_case": "Ensemble building, model diversity maximization, exploration",
        'ranges': {
            'num_boost_round': (50, 1000),
            'num_leaves': (2, 1024),
            'max_depth': (2, 16),
            'learning_rate': (0.001, 0.3),
            'feature_fraction': (0.1, 1.0),
            'bagging_fraction': (0.1, 1.0),
            'lambda_l1': (1e-8, 100.0),
            'lambda_l2': (1e-8, 100.0),
            'min_data_in_leaf': (1, 300),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    },

    "competition": {
        "description": "State-of-the-art configuration based on 2024-2025 winning solutions",
        "use_case": "Kaggle competitions, maximum performance scenarios, ensemble base models",
        'ranges': {
            'num_boost_round': (1000, 5000),
            'num_leaves': (31, 127),
            'max_depth': (2, 16),
            'learning_rate': (0.001, 0.3),
            'feature_fraction': (0.1, 1.0),
            'bagging_fraction': (0.1, 1.0),
            'lambda_l1': (1e-8, 100.0),
            'lambda_l2': (1e-8, 100.0),
            'min_data_in_leaf': (1, 300),
            'min_sum_hessian_in_leaf': (1e-3, 10.0)
        }
    }
}


# LightGBM Categorical Parameter Configurations
LGBM_CATEGORICAL_PARAMETER_CHOICES = {
    'boosting_type': ['gbdt', 'dart', 'rf'],
    'objective': ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape'],
    'metric': ['l2', 'l1', 'huber', 'fair', 'poisson', 'quantile', 'mape'],
    'tree_learner': ['serial', 'feature', 'data', 'voting'],
    'device_type': ['cpu', 'gpu'],
    'deterministic': ['false', 'true'],
    'first_metric_only': ['false', 'true'],
    'boost_from_average': ['true', 'false'],
    'is_unbalance': ['false', 'true'],
    'linear_tree': ['false', 'true'],
    'monotone_constraints_method': ['basic', 'intermediate', 'advanced'],
    'feature_pre_filter': ['true', 'false']
}


# Strategy-specific categorical parameter configurations for LightGBM
LGBM_STRATEGY_CATEGORICAL_CONFIGS = {
    "default": {
        'boosting_type': ['gbdt', 'dart'],
        'tree_learner': ['serial', 'feature']
    },
    
    "fast": {
        'boosting_type': ['gbdt'],  # Fastest boosting method
        'tree_learner': ['serial']  # Fastest tree learner
    },
    
    "aggressive": {
        'boosting_type': ['gbdt', 'dart'],
        'tree_learner': ['serial', 'feature', 'data'],
        'linear_tree': ['false', 'true']
    },
    
    "balanced": {
        'boosting_type': ['gbdt'],
        'tree_learner': ['serial', 'feature']
    },
    
    "stable": {
        'boosting_type': ['gbdt'],  # Most stable
        'tree_learner': ['serial']  # Most stable
    },
    
    "conservative": {
        'boosting_type': ['gbdt'],
        'tree_learner': ['serial']
    },
    
    "regularized": {
        'boosting_type': ['gbdt'],
        'tree_learner': ['serial']
    },
    
    "diversity": {
        'boosting_type': ['gbdt', 'dart', 'rf'],
        'tree_learner': ['serial', 'feature', 'data', 'voting'],
        'linear_tree': ['false', 'true']
    },
    
    "competition": {
        'boosting_type': ['gbdt', 'dart'],
        'tree_learner': ['serial', 'feature', 'data'],
        'linear_tree': ['false', 'true']
    }
}



# LightGBM parameters that should use log-scale sampling
LGBM_LOG_SCALE_PARAMS = {
    'learning_rate': True,
    'lambda_l1': True,
    'lambda_l2': True,
    'min_sum_hessian_in_leaf': True,
    'min_gain_to_split': True,
    'reg_alpha': True,  # Alternative name
    'reg_lambda': True,  # Alternative name
}

# LightGBM parameters that should use integer sampling
LGBM_INTEGER_PARAMS = {
    'num_boost_round': True,
    'num_leaves': True,
    'max_depth': True,
    'min_data_in_leaf': True,
    'max_bin': True,
    'bagging_freq': True,
    'min_child_weight': True,  # FIXED: Added missing integer parameter
    'feature_fraction_seed': True,
    'bagging_seed': True,
    'verbosity': True,
    'num_threads': True,
    'random_state': True,
    'seed': True,
    'data_random_seed': True,
    'n_estimators': True,  # FIXED: Added for sklearn interface compatibility
}


# Utility Functions
def get_lgbm_strategy_ranges(strategy: str) -> Dict[str, tuple]:
    """Get LightGBM parameter ranges for a strategy."""
    if strategy not in LGBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown LightGBM strategy '{strategy}'. Available strategies: {list(LGBM_STRATEGY_CONFIGS.keys())}")
    return LGBM_STRATEGY_CONFIGS[strategy]['ranges']


def get_lgbm_strategy_info(strategy: str) -> Dict[str, Any]:
    """Get complete LightGBM strategy information."""
    if strategy not in LGBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown LightGBM strategy '{strategy}'. Available strategies: {list(LGBM_STRATEGY_CONFIGS.keys())}")
    return LGBM_STRATEGY_CONFIGS[strategy]


def list_available_lgbm_strategies() -> list:
    """Get list of available LightGBM strategies."""
    return list(LGBM_STRATEGY_CONFIGS.keys())


def get_lgbm_progressive_groups(strategy: str) -> Dict[str, Any]:
    """Get LightGBM progressive optimization groups (progressive optimization removed)."""
    raise ValueError("Progressive optimization has been removed from SimpleMLR.")


def get_all_lgbm_progressive_strategies() -> list:
    """Get all LightGBM progressive strategies (progressive optimization removed)."""
    return []


def is_lgbm_log_scale_param(param_name: str) -> bool:
    """Check if a LightGBM parameter should use log-scale sampling."""
    return LGBM_LOG_SCALE_PARAMS.get(param_name, False)


def is_lgbm_integer_param(param_name: str) -> bool:
    """Check if a LightGBM parameter should use integer sampling."""
    return LGBM_INTEGER_PARAMS.get(param_name, False)


def suggest_lgbm_parameter(trial, param_name: str, param_info):
    """
    Enhanced LightGBM parameter suggestion with categorical support.
    
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
    if is_lgbm_integer_param(param_name):
        if is_lgbm_log_scale_param(param_name):
            return trial.suggest_int(param_name, int(min_val), int(max_val), log=True)
        else:
            return trial.suggest_int(param_name, int(min_val), int(max_val))
    else:
        if is_lgbm_log_scale_param(param_name):
            return trial.suggest_float(param_name, float(min_val), float(max_val), log=True)
        else:
            return trial.suggest_float(param_name, float(min_val), float(max_val))


def validate_lgbm_override_params(override_params: Dict[str, Union[Tuple[float, float], float, int]]) -> None:
    """Validate LightGBM override parameters format."""
    if not isinstance(override_params, dict):
        raise TypeError("override_params must be a dictionary")

    for param_name, param_value in override_params.items():
        if not isinstance(param_name, str):
            raise TypeError(f"Parameter name must be string, got {type(param_name)}")

        if isinstance(param_value, tuple):
            if len(param_value) != 2:
                raise ValueError(f"Parameter range tuple for '{param_name}' must have exactly 2 values")
            if param_value[0] >= param_value[1]:
                raise ValueError(f"Invalid range for '{param_name}': min_val must be < max_val")
        elif not isinstance(param_value, (int, float)):
            raise TypeError(f"Parameter value for '{param_name}' must be number or tuple of 2 numbers")


def merge_lgbm_strategy_with_overrides(strategy: str, override_params: Optional[
    Dict[str, Union[Tuple[float, float], float, int]]] = None) -> Dict[str, Any]:
    """Merge LightGBM strategy with user overrides."""
    if strategy not in LGBM_STRATEGY_CONFIGS:
        raise ValueError(
            f"Unknown LightGBM strategy '{strategy}'. Available strategies: {list(LGBM_STRATEGY_CONFIGS.keys())}")

    # Get base strategy ranges
    base_ranges = LGBM_STRATEGY_CONFIGS[strategy]['ranges'].copy()

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
    validate_lgbm_override_params(override_params)

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


def recommend_lgbm_strategy(dataset_size: int = None, target_speed: str = None,
                            has_categorical: bool = None, is_noisy: bool = None) -> str:
    """Recommend a LightGBM strategy based on dataset characteristics."""
    # LightGBM is naturally fast, so we can be more aggressive with complex strategies
    if target_speed == "fast" or (dataset_size and dataset_size < 1000):
        return "fast"  # LightGBM's natural speed advantage
    elif is_noisy:
        return "regularized"  # Strong regularization for noisy data
    elif target_speed == "stable" or (dataset_size and dataset_size > 100000):
        return "stable"  # Production stability
    elif has_categorical:
        return "balanced"  # LightGBM handles categoricals well
    else:
        return "default"  # General purpose


# ========================================
# LightGBM Categorical Parameter Helper Functions
# ========================================

def get_lgbm_categorical_parameter_choices() -> Dict[str, List[str]]:
    """Get all available categorical parameter choices for LightGBM."""
    return LGBM_CATEGORICAL_PARAMETER_CHOICES.copy()


def get_lgbm_strategy_categorical_params(strategy: str) -> Dict[str, List[str]]:
    """Get categorical parameter choices for a specific LightGBM strategy."""
    if strategy not in LGBM_STRATEGY_CATEGORICAL_CONFIGS:
        return {}
    return LGBM_STRATEGY_CATEGORICAL_CONFIGS[strategy].copy()


def is_lgbm_categorical_param(param_name: str) -> bool:
    """Check if a parameter is a LightGBM categorical parameter."""
    return param_name in LGBM_CATEGORICAL_PARAMETER_CHOICES


def get_lgbm_categorical_choices(param_name: str) -> List[str]:
    """Get the available choices for a LightGBM categorical parameter."""
    if param_name not in LGBM_CATEGORICAL_PARAMETER_CHOICES:
        raise ValueError(f"'{param_name}' is not a known LightGBM categorical parameter. "
                        f"Available: {list(LGBM_CATEGORICAL_PARAMETER_CHOICES.keys())}")
    return LGBM_CATEGORICAL_PARAMETER_CHOICES[param_name].copy()


def validate_lgbm_categorical_choice(param_name: str, choice: str) -> bool:
    """Validate that a choice is valid for a LightGBM categorical parameter."""
    if not is_lgbm_categorical_param(param_name):
        return False
    return choice in LGBM_CATEGORICAL_PARAMETER_CHOICES[param_name]


def suggest_lgbm_categorical_parameter(trial, param_name: str, choices: List[str]):
    """Suggest a LightGBM categorical parameter value for optimization."""
    return trial.suggest_categorical(param_name, choices)