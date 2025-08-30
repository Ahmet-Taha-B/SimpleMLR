"""
XGBoost Strategy Configurations - Pre-built optimization strategies for different needs.

This file defines all the XGBoost optimization strategies available in SimpleMLR.
Each strategy is a pre-configured set of parameter ranges optimized for different
situations:

- 'default': Balanced optimization, good for most use cases
- 'fast': Quick optimization for experimentation and prototyping
- 'stable': Conservative settings for production environments
- 'aggressive': Tries extreme values for maximum performance
- 'competition': Optimized for Kaggle-style competitions

Each strategy specifies parameter ranges that work well together based on
machine learning best practices and empirical testing. You can also override
individual parameters while keeping the rest of the strategy.

Most users will just pick a strategy name when using XGBAutoTuner:
    model = xgb_auto(X, y, strategy='fast')
"""

from typing import Dict, Any, Tuple, Union, Optional, List
import optuna


# XGBoost Strategy Configurations - Each strategy is optimized for different use cases
STRATEGY_CONFIGS = {
    "default": {
        "description": "General-purpose optimization with balanced parameter ranges",
        "use_case": "Standard optimization for most datasets",
        "ranges": {
            'n_estimators': (50, 500),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 10.0),
            'reg_lambda': (1e-8, 10.0),
            'gamma': (1e-8, 5.0)
        }
    },
    
    "fast": {
        "description": "Speed-optimized ranges for rapid experimentation",
        "use_case": "Quick prototyping, hyperparameter exploration, time-constrained optimization",
        "estimated_speedup": "2-3x faster than default",
        "ranges": {
            'n_estimators': (40, 120),
            'max_depth': (4, 6),
            'learning_rate': (0.08, 0.25),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 2.0),
            'reg_lambda': (1.0, 5.0),
            'gamma': (1e-8, 1.0)
        }
    },
    
    "aggressive": {
        "description": "High-variance, complex pattern detection",
        "use_case": "Complex relationships, high-variance ensemble members, competition settings",
        "ranges": {
            'n_estimators': (50, 200),
            'learning_rate': (0.1, 0.3),
            'max_depth': (6, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 1.0),
            'reg_lambda': (1e-8, 3.0),
            'gamma': (1e-8, 0.5)
        }
    },
    
    "balanced": {
        "description": "Balanced approach for unknown data characteristics",
        "use_case": "General-purpose modeling, baseline establishment, educational use",
        "ranges": {
            'n_estimators': (100, 300),
            'learning_rate': (0.05, 0.15),
            'max_depth': (4, 7),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 2.0),
            'reg_lambda': (1.0, 5.0),
            'gamma': (1e-8, 2.0)
        }
    },
    
    "stable": {
        "description": "Production-ready, reliable performance",
        "use_case": "Production deployment, reliable predictions, ensemble members",
        "ranges": {
            'n_estimators': (500, 1000),
            'learning_rate': (0.02, 0.08),
            'max_depth': (4, 6),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 1.5),
            'reg_lambda': (1.0, 5.0),
            'gamma': (1e-8, 1.0)
        }
    },
    
    "conservative": {
        "description": "Stable regularization with very slow learning for maximum reliability",
        "use_case": "Production stability, ensemble base learners, risk-averse applications",
        "ranges": {
            'n_estimators': (1500, 3500),
            'learning_rate': (0.0021, 0.02),
            'max_depth': (4, 6),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 1.5),
            'reg_lambda': (1.0, 5.0),
            'gamma': (1e-8, 1.0)
        }
    },

    "regularized": {
        "description": "Highest regularization for strong overfitting prevention",
        "use_case": "Noisy data, small datasets, overfitting-prone scenarios",
        "ranges": {
            'n_estimators': (100, 400),
            'learning_rate': (0.01, 0.1),
            'max_depth': (3, 5),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0.09, 13.9),
            'reg_lambda': (1.4, 13.9),
            'gamma': (0.09, 6.9)
        }
    },
    
    "diversity": {
        "description": "Maximum parameter variation for ensemble building",
        "use_case": "Ensemble building, model diversity maximization, exploration",
        "ranges": {
            'n_estimators': (50, 1000),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 10.0),
            'reg_lambda': (1e-8, 10.0),
            'gamma': (1e-8, 5.0)
        }
    },

    "competition": {
        "description": "State-of-the-art configuration based on 2024-2025 winning solutions",
        "use_case": "Kaggle competitions, maximum performance scenarios, ensemble base models",
        "ranges": {
            'n_estimators': (1000, 5000),
            'learning_rate': (0.01, 0.05),
            'max_depth': (3, 8),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (1e-8, 10.0),
            'reg_lambda': (1e-8, 10.0),
            'gamma': (1e-8, 5.0)
        }
    }
}


# XGBoost Categorical Parameter Configurations
CATEGORICAL_PARAMETER_CHOICES = {
    'booster': ['gbtree', 'gblinear', 'dart'],
    'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
    'grow_policy': ['depthwise', 'lossguide'],
    'eval_metric': ['rmse', 'mae', 'mape', 'logloss', 'error'],
    'objective': ['reg:squarederror', 'reg:squaredlogerror', 'reg:absoluteerror', 'reg:pseudohubererror'],
    'sampling_method': ['uniform', 'gradient_based'],
    'normalize_type': ['tree', 'forest'],
    'rate_drop': ['uniform', 'none'],
    'one_drop': ['0', '1'],
    'skip_drop': ['uniform', 'none']
}


# Strategy-specific categorical parameter configurations
# These are the categorical parameters that are commonly optimized for each strategy
STRATEGY_CATEGORICAL_CONFIGS = {
    "default": {
        'booster': ['gbtree', 'dart'],
        'tree_method': ['auto', 'hist']
    },
    
    "fast": {
        'booster': ['gbtree'],
        'tree_method': ['hist']  # Fastest tree method
    },
    
    "aggressive": {
        'booster': ['gbtree', 'dart'],
        'tree_method': ['auto', 'exact', 'hist'],
        'grow_policy': ['depthwise', 'lossguide']
    },
    
    "balanced": {
        'booster': ['gbtree'],
        'tree_method': ['auto', 'hist']
    },
    
    "stable": {
        'booster': ['gbtree'],
        'tree_method': ['hist'],  # Most stable
    },
    
    "conservative": {
        'booster': ['gbtree'],
        'tree_method': ['hist']
    },
    
    "regularized": {
        'booster': ['gbtree'],
        'tree_method': ['hist']
    },
    
    "diversity": {
        'booster': ['gbtree', 'dart', 'gblinear'],
        'tree_method': ['auto', 'exact', 'hist'],
        'grow_policy': ['depthwise', 'lossguide']
    },
    
    "competition": {
        'booster': ['gbtree', 'dart'],
        'tree_method': ['auto', 'exact', 'hist'],
        'grow_policy': ['depthwise', 'lossguide']
    }
}



# Parameters that should use log-scale sampling
LOG_SCALE_PARAMS = {
    'learning_rate': True,
    'reg_alpha': True,
    'reg_lambda': True,
    'gamma': True,
    'max_delta_step': True,
    'base_score': True,
    'scale_pos_weight': True,
    'sketch_eps': True,
    'refresh_leaf': True,
    'process_type': True,
}

# Parameters that should use integer sampling
INTEGER_PARAMS = {
    'n_estimators': True,
    'max_depth': True,
    'min_child_weight': True,
    'max_leaves': True,
    'max_bin': True,
    'verbosity': True,
    'num_parallel_tree': True,
    'max_cat_to_onehot': True,
    'max_cat_threshold': True,
    'num_class': True,
    'nthread': True,
    'random_state': True,
    'seed': True
}


# Utility Functions
def get_strategy_ranges(strategy: str) -> Dict[str, tuple]:
    """Get XGBoost parameter ranges for a strategy."""
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: {list(STRATEGY_CONFIGS.keys())}")
    return STRATEGY_CONFIGS[strategy]['ranges']


def get_strategy_info(strategy: str) -> Dict[str, Any]:
    """Get complete XGBoost strategy information."""
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: {list(STRATEGY_CONFIGS.keys())}")
    return STRATEGY_CONFIGS[strategy]


def list_available_strategies() -> list:
    """Get list of available XGBoost strategies."""
    return list(STRATEGY_CONFIGS.keys())


def get_progressive_groups(strategy: str) -> Dict[str, Any]:
    """Get XGBoost progressive optimization groups (progressive optimization removed)."""
    raise ValueError("Progressive optimization has been removed from SimpleMLR.")


def get_all_progressive_strategies() -> list:
    """Get all XGBoost progressive strategies (progressive optimization removed)."""
    return []


def is_log_scale_param(param_name: str) -> bool:
    """Check if an XGBoost parameter should use log-scale sampling."""
    return LOG_SCALE_PARAMS.get(param_name, False)


def is_integer_param(param_name: str) -> bool:
    """Check if an XGBoost parameter should use integer sampling."""
    return INTEGER_PARAMS.get(param_name, False)


def suggest_parameter(trial, param_name: str, param_info):
    """
    Enhanced XGBoost parameter suggestion with categorical support.
    
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
    if is_integer_param(param_name):
        if is_log_scale_param(param_name):
            return trial.suggest_int(param_name, int(min_val), int(max_val), log=True)
        else:
            return trial.suggest_int(param_name, int(min_val), int(max_val))
    else:
        if is_log_scale_param(param_name):
            return trial.suggest_float(param_name, float(min_val), float(max_val), log=True)
        else:
            return trial.suggest_float(param_name, float(min_val), float(max_val))


def validate_override_params(override_params: Dict[str, Union[Tuple[float, float], float, int]]) -> None:
    """Validate override parameters format."""
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


def merge_strategy_with_overrides(strategy: str, override_params: Optional[Dict[str, Union[Tuple[float, float], float, int]]] = None) -> Dict[str, Any]:
    """Merge XGBoost strategy with user overrides."""
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: {list(STRATEGY_CONFIGS.keys())}")
    
    # Get base strategy ranges
    base_ranges = STRATEGY_CONFIGS[strategy]['ranges'].copy()
    
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
    validate_override_params(override_params)
    
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


def recommend_strategy(dataset_size: int = None, target_speed: str = None, 
                      has_categorical: bool = None, is_noisy: bool = None) -> str:
    """Recommend an XGBoost strategy based on dataset characteristics."""
    if target_speed == "fast" or (dataset_size and dataset_size < 1000):
        return "fast"
    elif is_noisy:
        return "regularized"
    elif target_speed == "stable" or (dataset_size and dataset_size > 100000):
        return "stable"
    elif has_categorical:
        return "balanced"
    else:
        return "default"


# ========================================
# Categorical Parameter Helper Functions
# ========================================

def get_categorical_parameter_choices() -> Dict[str, List[str]]:
    """Get all available categorical parameter choices for XGBoost."""
    return CATEGORICAL_PARAMETER_CHOICES.copy()


def get_strategy_categorical_params(strategy: str) -> Dict[str, List[str]]:
    """Get categorical parameter choices for a specific strategy."""
    if strategy not in STRATEGY_CATEGORICAL_CONFIGS:
        return {}
    return STRATEGY_CATEGORICAL_CONFIGS[strategy].copy()


def is_categorical_param(param_name: str) -> bool:
    """Check if a parameter is a categorical parameter."""
    return param_name in CATEGORICAL_PARAMETER_CHOICES


def get_categorical_choices(param_name: str) -> List[str]:
    """Get the available choices for a categorical parameter."""
    if param_name not in CATEGORICAL_PARAMETER_CHOICES:
        raise ValueError(f"'{param_name}' is not a known categorical parameter. "
                        f"Available: {list(CATEGORICAL_PARAMETER_CHOICES.keys())}")
    return CATEGORICAL_PARAMETER_CHOICES[param_name].copy()


def validate_categorical_choice(param_name: str, choice: str) -> bool:
    """Validate that a choice is valid for a categorical parameter."""
    if not is_categorical_param(param_name):
        return False
    return choice in CATEGORICAL_PARAMETER_CHOICES[param_name]


def suggest_categorical_parameter(trial, param_name: str, choices: List[str]):
    """Suggest a categorical parameter value for optimization."""
    return trial.suggest_categorical(param_name, choices)