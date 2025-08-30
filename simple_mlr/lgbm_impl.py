"""
LightGBM implementation for SimpleMLR - Fast, memory-efficient gradient boosting.

LightGBM (Light Gradient Boosting Machine) is Microsoft's high-performance gradient
boosting framework. It's especially known for:
- Very fast training speed compared to other boosting algorithms
- Low memory usage, great for large datasets
- High accuracy with proper tuning
- Excellent GPU acceleration support
- Built-in categorical feature handling

This module provides two main classes:
- LGBMRegressor: Direct LightGBM model with SimpleMLR's easy interface
- LGBMAutoTuner: Automatically finds the best LightGBM settings for your data

Both classes handle data preprocessing, provide beautiful visualizations,
and work seamlessly with the rest of the SimpleMLR ecosystem.
"""

import pandas as pd
import numpy as np
import warnings
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from collections import deque

# Core ML libraries
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import optuna

# Import base classes and mixins
try:
    from .base import BaseBoostingRegressor, BaseAutoTuner
    from .base_strategies import BaseStrategyConfig
    from .utils import DataValidator, ModelEvaluator
except ImportError:
    # Fallback for direct script execution
    from base import BaseBoostingRegressor, BaseAutoTuner
    from base_strategies import BaseStrategyConfig
    from utils import DataValidator, ModelEvaluator


class LightGBMStrategyConfig(BaseStrategyConfig):
    """
    Defines how LightGBM implements optimization strategies.
    
    This class maps universal strategy names like 'fast' and 'stable' to
    LightGBM-specific parameter ranges. LightGBM has different parameters
    than XGBoost (like num_leaves instead of max_depth), so this handles
    the translation while keeping the same strategy concepts.
    
    Users don't typically work with this directly - it's used internally
    by LGBMAutoTuner when you specify a strategy name.
    """
    
    def __init__(self):
        """Initialize with LightGBM strategy configurations."""
        # Import LightGBM strategies from lgbm_strategies.py
        try:
            from .lgbm_strategies import (
                LGBM_STRATEGY_CONFIGS,
                suggest_lgbm_parameter as lgb_suggest_parameter,
                get_lgbm_strategy_categorical_params
            )
        except ImportError:
            # Fallback for direct script execution
            from lgbm_strategies import (
                LGBM_STRATEGY_CONFIGS,
                suggest_lgbm_parameter as lgb_suggest_parameter,
                get_lgbm_strategy_categorical_params
            )
        self._strategy_configs = LGBM_STRATEGY_CONFIGS
        self._suggest_parameter_func = lgb_suggest_parameter
        self._get_categorical_params_func = get_lgbm_strategy_categorical_params
    
    def get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get LightGBM parameter ranges for the specified strategy."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown LightGBM strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]['ranges']
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get complete LightGBM strategy information."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown LightGBM strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available LightGBM strategies."""
        return list(self._strategy_configs.keys())
    
    
    def get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get LightGBM categorical parameter choices for the specified strategy."""
        return self._get_categorical_params_func(strategy)
    
    def suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced LightGBM parameter suggestion with categorical support."""
        return self._suggest_parameter_func(trial, param_name, param_info)
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "LightGBM"


class LGBMRegressor(BaseBoostingRegressor):
    """
    LightGBM regression model with SimpleMLR's easy-to-use interface.
    
    This class provides all the speed and efficiency of LightGBM with SimpleMLR's convenience:
    - Automatic data preprocessing (scaling, encoding)
    - Simple fit/predict interface like scikit-learn
    - Built-in plotting with quick_graph() and plot_analysis()
    - Excellent GPU support for fast training
    - Memory-efficient processing for large datasets
    
    Great choice when you need fast training or are working with large datasets.
    LightGBM often trains 2-10x faster than XGBoost with similar accuracy.
    
    Example:
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        model.quick_graph()  # Instant performance analysis!
    """
    
    def __init__(self,
                 num_boost_round: int = 100,
                 num_leaves: int = 31,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 feature_fraction: float = 1.0,
                 bagging_fraction: float = 1.0,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0,
                 min_data_in_leaf: int = 20,
                 min_sum_hessian_in_leaf: float = 1e-3,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initialize LightGBM Regressor.
        
        Args:
            num_boost_round: Number of boosting iterations (LightGBM equivalent of n_estimators)
            num_leaves: Maximum number of leaves in one tree (primary complexity controller)
            max_depth: Maximum tree depth (-1=no limit, constraint: num_leaves < 2^max_depth)
            learning_rate: Boosting learning rate
            feature_fraction: Fraction of features to be used in each tree (LightGBM's colsample_bytree)
            bagging_fraction: Fraction of data to be used for each iteration (LightGBM's subsample)
            lambda_l1: L1 regularization weight
            lambda_l2: L2 regularization weight  
            min_data_in_leaf: Minimum number of data points in one leaf
            min_sum_hessian_in_leaf: Minimum sum of hessians in one leaf (gradient-based regularization)
            random_state: Random seed for reproducibility
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to handle categorical features
            verbose: Whether to print progress information
            n_jobs: Number of CPU cores to use (-1=all cores)
            use_gpu: Whether to use GPU acceleration
            gpu_id: Which GPU to use (0=first GPU, 1=second GPU, etc.)
            **kwargs: Additional LightGBM parameters
        """
        # Store LightGBM-specific parameters
        self.num_boost_round = num_boost_round
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.gpu_id = gpu_id
        
        # Initialize base class
        super().__init__(
            random_state=random_state,
            scale_features=scale_features,
            handle_categorical=handle_categorical,
            verbose=verbose,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            **kwargs
        )
    
    def _create_model(self, **params):
        """Create LightGBM model instance."""
        return lgb.LGBMRegressor(verbose=-1, **params)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get LightGBM-specific model parameters."""
        params = {
            'n_estimators': self.num_boost_round,  # sklearn interface uses n_estimators
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'min_data_in_leaf': self.min_data_in_leaf,
            'min_sum_hessian_in_leaf': self.min_sum_hessian_in_leaf,
            'random_state': self.random_state
        }
        
        # Enforce critical LightGBM constraint: num_leaves < 2^max_depth
        if self.max_depth > 0:
            max_allowed_leaves = 2 ** self.max_depth
            if self.num_leaves >= max_allowed_leaves:
                if self.verbose:
                    print(f"WARNING: Adjusting num_leaves from {self.num_leaves} to {max_allowed_leaves - 1} "
                          f"to satisfy constraint num_leaves < 2^max_depth ({max_allowed_leaves})")
                params['num_leaves'] = max_allowed_leaves - 1
        
        return params
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get LightGBM device-specific parameters."""
        device_params = {'n_jobs': self.n_jobs}
        
        if self.use_gpu:
            if self._test_gpu_availability():
                device_params.update({
                    'device_type': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': self.gpu_id
                })
                if self.verbose:
                    print(f"Using GPU acceleration (gpu_id={self.gpu_id})")
            else:
                if self.verbose:
                    print("GPU requested but not available, using CPU")
                    
        return device_params
    
    def _test_gpu_availability(self) -> bool:
        """Test if GPU is available for LightGBM."""
        try:
            # Test GPU availability by creating a small LightGBM model
            test_model = lgb.LGBMRegressor(
                n_estimators=1,
                device_type='gpu',
                gpu_platform_id=0,
                gpu_device_id=self.gpu_id,
                verbose=-1
            )
            
            # Create minimal test data - simplified to avoid pandas overhead
            X_test = np.array([[1], [2], [3], [4], [5]])
            y_test = np.array([1, 2, 3, 4, 5])
            
            # Try to fit
            test_model.fit(X_test, y_test)
            return True
            
        except Exception:
            return False
    
    def _get_feature_importance_values(self) -> Optional[np.ndarray]:
        """Get feature importance values from LightGBM model."""
        if self.model_ is not None:
            return self.model_.feature_importances_
        return None
    
    def _get_algorithm_display_name(self) -> str:
        """Return display name for this algorithm."""
        return "LightGBM"


class LGBMAutoTuner(BaseAutoTuner):
    """
    LightGBM Auto-Tuner implementation using the multi-algorithm framework.
    Preserves all existing multi-fidelity functionality.
    """
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: int = 1,
                 strategy: str = 'default',
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 gpu_id: int = 0,
                 gpu_optimize: bool = False,
                 validation_split: Optional[float] = None,
                 early_stopping_rounds: Optional[int] = None,
                 override_params: Optional[Dict[str, Union[Tuple[float, float], float, int]]] = None,
                 multi_fidelity: bool = False,
                 fidelity_stages: Optional[List] = None,
                 fidelity_percentile: float = 0.7,
                 **kwargs):
        """
        Initialize LightGBM auto-tuner.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to handle categorical features
            verbose: Verbosity level (-1=silent, 0=minimal, 1=progress, 2=detailed)
            strategy: Optimization strategy
            n_jobs: Number of CPU cores to use (-1=all cores)
            use_gpu: Whether to use GPU acceleration
            gpu_id: Which GPU to use
            gpu_optimize: Enable GPU speed optimizations
            validation_split: Use hold-out validation instead of cross-validation
            early_stopping_rounds: Enable early stopping with specified patience rounds
            override_params: Dictionary of parameter overrides
            multi_fidelity: Enable multi-fidelity early stopping evaluation
            fidelity_stages: List of budget stages for multi-fidelity
            fidelity_percentile: Percentile threshold for advancing to next fidelity stage
            **kwargs: Additional LightGBM-specific parameters
        """
        # Check for removed parameters
        if 'progressive' in kwargs:
            raise TypeError("LGBMAutoTuner.__init__() got an unexpected keyword argument 'progressive'. "
                          "Progressive optimization has been removed from SimpleMLR.")
        
        # Store LightGBM-specific parameters
        self.gpu_id = gpu_id
        self.gpu_optimize = gpu_optimize
        
        # Create strategy config BEFORE parent init (performance optimization)
        self._strategy_config = LightGBMStrategyConfig()
        
        # Initialize base class
        super().__init__(
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            scale_features=scale_features,
            handle_categorical=handle_categorical,
            verbose=verbose,
            strategy=strategy,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            validation_split=validation_split,
            early_stopping_rounds=early_stopping_rounds,
            override_params=override_params,
            multi_fidelity=multi_fidelity,
            fidelity_stages=fidelity_stages,
            fidelity_percentile=fidelity_percentile,
            **kwargs
        )
    
    def _create_model_instance(self, **params):
        """Create LightGBM model instance for optimization."""
        return lgb.LGBMRegressor(verbose=-1, **params)
    
    def _get_algorithm_name(self) -> str:
        """Return algorithm name for display."""
        return "LightGBM"
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available LightGBM strategies."""
        return self._strategy_config.get_available_strategies()
    
    
    def _get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get LightGBM parameter ranges for the specified strategy."""
        return self._strategy_config.get_strategy_ranges(strategy)
    
    def _get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get LightGBM categorical parameter choices for the specified strategy."""
        return self._strategy_config.get_strategy_categorical_params(strategy)
    
    def _suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced LightGBM parameter suggestion with categorical support."""
        return self._strategy_config.suggest_parameter(trial, param_name, param_info)
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get LightGBM device-specific parameters."""
        device_params = {'n_jobs': self.n_jobs}
        
        if self.use_gpu:
            if self._test_gpu_availability():
                device_params.update({
                    'device_type': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': self.gpu_id
                })
            else:
                if self.verbose >= 1:
                    print("GPU requested but not available, using CPU")
                    
        return device_params
    
    def _test_gpu_availability(self) -> bool:
        """Test if GPU is available for LightGBM."""
        try:
            # Test GPU availability by creating a small LightGBM model
            test_model = lgb.LGBMRegressor(
                n_estimators=1,
                device_type='gpu',
                gpu_platform_id=0,
                gpu_device_id=self.gpu_id,
                verbose=-1
            )
            
            # Create minimal test data - simplified to avoid pandas overhead
            X_test = np.array([[1], [2], [3], [4], [5]])
            y_test = np.array([1, 2, 3, 4, 5])
            
            # Try to fit
            test_model.fit(X_test, y_test)
            return True
            
        except Exception:
            return False
    
    def _get_default_fidelity_stages(self) -> List:
        """Get default fidelity stages for LightGBM multi-fidelity optimization."""
        return [50, 150, None]  # None means use full num_boost_round from optimization
    
    def _check_algorithm_constraints(self, params: dict, n_estimators: int = None,
                                   optimization_context: dict = None) -> bool:
        """LightGBM-specific constraint checking with context awareness."""
        if optimization_context is None:
            optimization_context = {}
        
        mode = optimization_context.get('mode', 'traditional')
        
        # Route to appropriate LightGBM constraint checker
        if mode == 'traditional':
            return self._check_traditional_lgbm_constraints(params, n_estimators)
        elif mode == 'multi_fidelity':
            return self._check_multifidelity_lgbm_constraints(params, n_estimators, optimization_context)
        else:
            return self._check_traditional_lgbm_constraints(params, n_estimators)

    def _check_traditional_lgbm_constraints(self, params: dict, n_estimators: int = None) -> bool:
        """Traditional LightGBM constraint checking - RELAXED constraints for better exploration."""

        # Use n_estimators from params if not provided separately
        if n_estimators is None:
            n_estimators = params.get('n_estimators') or params.get('num_boost_round')

        # RELAXED: Learning rate * num_boost_round relationship
        if 'learning_rate' in params and n_estimators is not None:
            product = params['learning_rate'] * n_estimators
            # FIXED: More reasonable range - was (10, 50), now (2, 100)
            if product < 2 or product > 100:  # Much wider acceptable range
                return False

        # RELAXED: feature_fraction * bagging_fraction relationship
        if 'feature_fraction' in params and 'bagging_fraction' in params:
            data_fraction = params['feature_fraction'] * params['bagging_fraction']
            # FIXED: More reasonable minimum - was 0.36, now 0.16 (matches min possible values)
            if data_fraction < 0.16:  # Allow exploration of lower sampling combinations
                return False

        # Critical LightGBM constraint: num_leaves < 2^max_depth (keep strict - mathematically required)
        if 'num_leaves' in params and 'max_depth' in params:
            max_depth = params['max_depth']
            num_leaves = params['num_leaves']
            if max_depth > 0:  # -1 means no limit
                max_allowed_leaves = 2 ** max_depth
                if num_leaves >= max_allowed_leaves:
                    return False

        return True

    def _check_multifidelity_lgbm_constraints(self, params: dict, n_estimators: int = None,
                                           context: dict = None) -> bool:
        """Multi-fidelity LightGBM constraint checking - flexible exploration with intelligent validation."""
        # Check data fraction (always strict - this affects underfitting directly)
        if 'feature_fraction' in params and 'bagging_fraction' in params:
            if params['feature_fraction'] * params['bagging_fraction'] < 0.36:
                return False
        
        # Critical LightGBM constraint: num_leaves < 2^max_depth (always strict)
        if 'num_leaves' in params and 'max_depth' in params:
            max_depth = params['max_depth']
            num_leaves = params['num_leaves']
            if max_depth > 0:  # -1 means no limit
                max_allowed_leaves = 2 ** max_depth
                if num_leaves >= max_allowed_leaves:
                    return False
        
        # For learning_rate * num_boost_round, use intelligent multi-stage validation
        if 'learning_rate' in params and n_estimators is not None:
            lr = params['learning_rate']
            
            # Extreme learning rates are always problematic
            if lr < 0.001 or lr > 1.0:
                return False
            
            # Get fidelity stage information from context
            fidelity_stages = context.get('fidelity_stages', []) if context else []
            current_stage_idx = context.get('stage_idx', 0) if context else 0
            
            # Multi-fidelity exploration strategy: Be more permissive in early stages
            if fidelity_stages:
                # Strategy 1: Check if learning_rate works with ANY fidelity stage
                for stage_budget in fidelity_stages:
                    if stage_budget and 5 <= lr * stage_budget <= 100:  # Relaxed range for exploration
                        return True
                
                # Strategy 2: For very early stages, allow even more exploration
                if current_stage_idx < len(fidelity_stages) // 2:
                    # In early stages, check if it would work with geometric mean of all stages
                    valid_stages = [s for s in fidelity_stages if s is not None]
                    if valid_stages:
                        geometric_mean = (np.prod(valid_stages)) ** (1.0 / len(valid_stages))
                        if 5 <= lr * geometric_mean <= 100:
                            return True
                
                # Strategy 3: Final stage should be more restrictive
                if current_stage_idx >= len(fidelity_stages) - 1:
                    final_stage = fidelity_stages[-1] if fidelity_stages[-1] else 200
                    return 8 <= lr * final_stage <= 60  # Tighter bounds for final stage
                
                # Strategy 4: Fallback validation for intermediate stages
                return 5 <= lr * 150 <= 100  # Use reasonable intermediate baseline
            else:
                # No fidelity stages defined, use permissive range for exploration
                return 5 <= lr * 100 <= 100
        
        return True
    
        
    def optimize(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]) -> LGBMRegressor:
        """
        Optimize LightGBM hyperparameters with all advanced features.
        
        This method implements the complete optimization logic with all existing
        functionality from the original XGBAutoTuner, adapted for LightGBM.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Optimized LGBMRegressor instance
        """
        # Initialize progress tracking using mixin
        self._reset_progress_state()

        # Suppress warnings and set proper logging levels
        import logging
        if self.verbose <= 0:
            logging.getLogger('lightgbm').setLevel(logging.ERROR)
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Test GPU availability early if requested
        if self.use_gpu:
            gpu_available = self._test_gpu_availability()
            if not gpu_available:
                if self.verbose >= 0:
                    print("WARNING: GPU requested but not available, switching to optimized CPU mode")
                self.use_gpu = False

        # Enhanced initial messages
        if self.verbose >= 1:
            config = LightGBMStrategyConfig()
            strategy_info = config.get_strategy_info(self.strategy)

            if self.validation_split is not None:
                mode_str = f"Validation Split ({self.validation_split:.1%})"
            else:
                mode_str = "Traditional"
            early_str = f" + Early Stopping ({self.early_stopping_rounds} rounds)" if self.early_stopping_rounds else ""
            multi_str = f" + Multi-Fidelity" if self.multi_fidelity else ""

            print(f"{mode_str} hyperparameter optimization{early_str}{multi_str} with {self.n_trials} trials...")
            print(f"Strategy: '{self.strategy}'")

        # Validate and preprocess
        X, y = DataValidator.validate_inputs(X, y)
        X_processed, self.preprocessing_artifacts_ = DataValidator.prepare_features(
            X, self.scale_features, self.handle_categorical
        )

        # Run optimization
        best_params = self._run_traditional_optimization(X_processed, y)

        # Create final model
        final_model = self._create_final_model(X, y, X_processed, best_params)

        # Final completion messages using base class method
        opt_time = time.time() - self._start_time
        self._show_final_summary(best_params, self._overall_best_rmse, opt_time)

        # Reset logging levels
        if self.verbose <= 0:
            logging.getLogger('lightgbm').setLevel(logging.WARNING)

        return final_model

    def _run_traditional_optimization(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run traditional simultaneous optimization."""
        config = LightGBMStrategyConfig()
        
        # Create study with optional early stopping
        if self.early_stopping_rounds:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=max(5, self.n_trials // 5),
                n_warmup_steps=max(3, self.n_trials // 10)
            )
            self.study_ = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=pruner
            )
        else:
            self.study_ = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )

        # Optimize with appropriate verbosity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.study_.optimize(
                lambda trial: self._objective(trial, X, y, config),
                n_trials=self.n_trials,
                callbacks=None,
                show_progress_bar=False
            )

        self._clear_progress_line()
        return self.study_.best_params

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, config: LightGBMStrategyConfig) -> float:
        """Objective function with proper progress tracking."""
        # Route to appropriate objective function based on multi-fidelity setting
        if self.multi_fidelity:
            return self._multi_fidelity_objective(trial, X, y, config)
        else:
            return self._single_fidelity_objective(trial, X, y, config)

    def _single_fidelity_objective(self, trial, X: pd.DataFrame, y: pd.Series, config: LightGBMStrategyConfig) -> float:
        """Traditional single-fidelity objective function."""
        # Record trial start time
        trial_start_time = time.time()
        self._trial_count += 1

        # Build parameters using merged ranges (strategy + overrides)
        params = {}
        params.update(self._fixed_params)
        
        # Add optimizable numerical parameters from merged ranges
        for param_name, param_range in self._merged_ranges.items():
            # Handle LightGBM parameter name mapping
            if param_name == 'num_boost_round':
                params['n_estimators'] = config.suggest_parameter(trial, param_name, param_range)
            else:
                params[param_name] = config.suggest_parameter(trial, param_name, param_range)
        
        # Add optimizable categorical parameters
        for param_name, choices in self._categorical_params.items():
            params[param_name] = trial.suggest_categorical(param_name, choices)

        # Add device-specific parameters
        device_params = self._get_device_params()
        params.update(device_params)

        # Check algorithm-specific mathematical constraints and prune invalid combinations
        context = self._get_optimization_context()
        try:
            constraint_check = self._check_algorithm_constraints(params, optimization_context=context)
            if not constraint_check:
                trial.set_user_attr("pruned_reason", "constraint_violation")
                raise optuna.TrialPruned()
        except Exception as constraint_error:
            # If constraint checking fails, just continue with the trial
            pass

        # Create model using the framework's model creation method
        model = self._create_model_instance(**params)

        # Evaluate model
        if self.validation_split is not None:
            rmse = self._validation_split_evaluation(model, X, y)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                model, X, y,
                cv=kf,
                scoring='neg_root_mean_squared_error',
                n_jobs=self.n_jobs
            )
            rmse = -cv_scores.mean()

        # Update progress using mixin method
        self._update_standard_progress(trial_start_time, rmse, params)

        return rmse

    def _multi_fidelity_objective(self, trial, X: pd.DataFrame, y: pd.Series, config: LightGBMStrategyConfig) -> float:
        """Multi-fidelity objective with staged evaluation."""
        trial_start_time = time.time()
        self._trial_count += 1
        
        # Build base parameters
        params = {}
        params.update(self._fixed_params)
        
        # Add optimizable numerical parameters (excluding num_boost_round for multi-fidelity)
        for param_name, param_range in self._merged_ranges.items():
            if param_name != 'num_boost_round':
                params[param_name] = config.suggest_parameter(trial, param_name, param_range)
        
        # Add optimizable categorical parameters
        for param_name, choices in self._categorical_params.items():
            params[param_name] = trial.suggest_categorical(param_name, choices)
        
        device_params = self._get_device_params()
        params.update(device_params)
        
        # Multi-fidelity evaluation through stages
        trial_scores = []
        
        for stage_idx, stage_budget in enumerate(self.fidelity_stages):
            # Determine n_estimators for this stage
            if stage_budget is None:
                if 'num_boost_round' in self._merged_ranges:
                    min_est, max_est = self._merged_ranges['num_boost_round']
                    n_estimators = config.suggest_parameter(trial, 'num_boost_round', min_est, max_est)
                else:
                    n_estimators = 100
            else:
                n_estimators = stage_budget
            
            # Create model with current fidelity budget
            stage_params = params.copy()
            stage_params['n_estimators'] = n_estimators
            
            # Check algorithm-specific constraints for this fidelity stage
            context = self._get_optimization_context({'stage_idx': stage_idx, 'fidelity_budget': n_estimators})
            if not self._check_algorithm_constraints(stage_params, n_estimators, context):
                # Calculate penalty for constraint violation
                penalty_rmse = self._calculate_constraint_penalty(stage_params, n_estimators, context)
                trial_scores.append(penalty_rmse)
                
                # Continue to next stage unless this is a severe violation
                if penalty_rmse > 1000:  # Only extreme violations stop progression
                    self._fidelity_scores[trial.number] = trial_scores
                    return penalty_rmse
                else:
                    continue  # Allow progression to next fidelity stage
            
            model = self._create_model_instance(**stage_params)
            
            # Evaluate with current budget
            if self.validation_split is not None:
                rmse = self._validation_split_evaluation(model, X, y)
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=kf,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=self.n_jobs
                )
                rmse = -cv_scores.mean()
            
            trial_scores.append(rmse)
            
            # Early stopping check (except for final stage)
            if stage_idx < len(self.fidelity_stages) - 1:
                # Calculate threshold for this stage
                if stage_idx not in self._fidelity_thresholds:
                    if len(self._fidelity_scores) < 5:
                        self._fidelity_thresholds[stage_idx] = float('inf')
                    else:
                        stage_scores = [scores[stage_idx] for scores in self._fidelity_scores.values() 
                                      if len(scores) > stage_idx]
                        if stage_scores:
                            self._fidelity_thresholds[stage_idx] = np.percentile(stage_scores, 
                                                                                self.fidelity_percentile * 100)
                        else:
                            self._fidelity_thresholds[stage_idx] = float('inf')
                
                # Check if current score meets threshold
                threshold = self._fidelity_thresholds[stage_idx]
                if rmse > threshold:
                    penalty = rmse * 1.5
                    self._fidelity_scores[trial.number] = trial_scores
                    return penalty
        
        # Configuration survived all stages
        final_rmse = trial_scores[-1]
        self._fidelity_scores[trial.number] = trial_scores
        
        # Update progress using mixin method
        # For multi-fidelity, we assume all stages were completed unless early stopped
        total_stages = len(self.fidelity_stages)
        self._update_multi_fidelity_progress(trial_start_time, final_rmse, total_stages, total_stages)
        
        return final_rmse

    def _calculate_constraint_penalty(self, params: dict, n_estimators: int = None, context: dict = None) -> float:
        """Calculate penalty score for constraint violations in multi-fidelity optimization."""
        base_penalty = 50.0  # Base penalty value
        penalty_multiplier = 1.0
        
        # Get context information
        fidelity_stages = context.get('fidelity_stages', []) if context else []
        current_stage_idx = context.get('stage_idx', 0) if context else 0
        
        # Analyze learning_rate * num_boost_round constraint violation
        if 'learning_rate' in params and n_estimators is not None:
            lr = params['learning_rate']
            product = lr * n_estimators
            
            # Calculate how far we are from valid ranges
            if product < 5:
                penalty_multiplier += (5 - product) * 2  # Penalize very low products
            elif product > 100:
                penalty_multiplier += (product - 100) * 0.5  # Penalize very high products
            
            # Check if it would work with any other fidelity stage
            works_with_other_stage = False
            if fidelity_stages:
                for stage_budget in fidelity_stages:
                    if stage_budget and 5 <= lr * stage_budget <= 100:
                        works_with_other_stage = True
                        break
            
            # Reduce penalty if it works with future stages
            if works_with_other_stage:
                penalty_multiplier *= 0.3  # Much lower penalty if valid for future stages
        
        # Analyze feature_fraction * bagging_fraction constraint violation
        if 'feature_fraction' in params and 'bagging_fraction' in params:
            data_fraction = params['feature_fraction'] * params['bagging_fraction']
            if data_fraction < 0.36:
                # This is always a serious violation
                penalty_multiplier += (0.36 - data_fraction) * 10
        
        # Analyze num_leaves < 2^max_depth constraint violation
        if 'num_leaves' in params and 'max_depth' in params:
            max_depth = params['max_depth']
            num_leaves = params['num_leaves']
            if max_depth > 0:  # -1 means no limit
                max_allowed_leaves = 2 ** max_depth
                if num_leaves >= max_allowed_leaves:
                    # This is always a serious violation for LightGBM
                    penalty_multiplier += (num_leaves - max_allowed_leaves + 1) * 5
        
        # Early stages get lower penalties to encourage exploration
        if current_stage_idx < len(fidelity_stages) // 2:
            penalty_multiplier *= 0.5
        
        return base_penalty * penalty_multiplier


    def _create_final_model(self, X: pd.DataFrame, y: pd.Series,
                            X_processed: pd.DataFrame, best_params: dict) -> LGBMRegressor:
        """Create the final optimized model."""
        # Create best model using the new framework
        self.best_model_ = LGBMRegressor(
            random_state=self.random_state,
            scale_features=self.scale_features,
            handle_categorical=self.handle_categorical,
            verbose=0,
            **{k: v for k, v in best_params.items()
               if k not in ['device_type', 'n_jobs', 'gpu_platform_id', 'gpu_device_id']}
        )

        # Store training data for quick_graph() auto-detection
        self.best_model_.training_X_ = X.copy()
        self.best_model_.training_y_ = y.copy()

        # Set preprocessing artifacts
        self.best_model_.preprocessing_artifacts_ = self.preprocessing_artifacts_

        
        if hasattr(self, '_validation_split_info') and self._validation_split_info is not None:
            self.best_model_._validation_split_info = self._validation_split_info
        
        if hasattr(self, '_fidelity_scores') and self._fidelity_scores:
            self.best_model_._fidelity_scores = self._fidelity_scores
            self.best_model_._fidelity_thresholds = self._fidelity_thresholds

        # Add device parameters for final model
        device_params = self._get_device_params()
        final_params = best_params.copy()
        final_params.update(device_params)

        # Fit on full dataset
        self.best_model_.model_ = self._create_model_instance(**final_params)
        self.best_model_.model_.fit(X_processed, y)

        # Calculate metrics and feature importance
        y_pred = self.best_model_.model_.predict(X_processed)
        self.best_model_.training_metrics_ = ModelEvaluator.calculate_metrics(y.values, y_pred)
        self.best_model_.feature_importance_ = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': self.best_model_.model_.feature_importances_
        }).sort_values('importance', ascending=False)

        return self.best_model_

    def _validation_split_evaluation(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model using hold-out validation split."""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Store split info
        self._validation_split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'split_ratio': self.validation_split
        }
        
        # Train and evaluate
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        except Exception:
            return float('inf')

    def _get_strategy_defaults(self, strategy_ranges: dict) -> dict:
        """Get intelligent default parameter values."""
        defaults = {}

        for param, (min_val, max_val) in strategy_ranges.items():
            if param == 'num_boost_round':
                defaults[param] = int(min_val + (max_val - min_val) * 0.3)
            elif param == 'learning_rate':
                if 'num_boost_round' in defaults:
                    target_product = 20
                    optimal_lr = target_product / defaults['num_boost_round']
                    defaults[param] = max(min_val, min(optimal_lr, max_val))
                else:
                    defaults[param] = min_val + (max_val - min_val) * 0.3
            elif param == 'num_leaves':
                defaults[param] = int(min_val + (max_val - min_val) * 0.3)
            elif param == 'max_depth':
                defaults[param] = int(min_val + (max_val - min_val) * 0.3)
            elif param in ['lambda_l1', 'lambda_l2']:
                defaults[param] = min_val + (max_val - min_val) * 0.1
            elif param in ['feature_fraction', 'bagging_fraction']:
                defaults[param] = min_val + (max_val - min_val) * 0.5
            elif param == 'min_sum_hessian_in_leaf':
                defaults[param] = min_val + (max_val - min_val) * 0.1
            else:
                # FIXED: Use proper integer parameter detection from lgbm_strategies
                try:
                    from lgbm_strategies import is_lgbm_integer_param
                    if is_lgbm_integer_param(param):
                        defaults[param] = int((min_val + max_val) / 2)
                    else:
                        defaults[param] = (min_val + max_val) / 2.0
                except ImportError:
                    # Fallback: check common integer parameters
                    if param in ['num_boost_round', 'num_leaves', 'max_depth', 'min_data_in_leaf', 
                                'bagging_freq', 'min_child_weight', 'max_bin']:
                        defaults[param] = int((min_val + max_val) / 2)
                    else:
                        defaults[param] = (min_val + max_val) / 2.0

        defaults.update(self._fixed_params)
        return defaults


# Convenience functions to maintain backward compatibility
def lgbm_regressor(X: Union[pd.DataFrame, np.ndarray],
                   y: Union[pd.Series, np.ndarray],
                   **kwargs) -> LGBMRegressor:
    """
    One-line LightGBM regression with automatic preprocessing.
    
    Args:
        X: Training features
        y: Training targets
        **kwargs: Additional parameters for LGBMRegressor
        
    Returns:
        Fitted LGBMRegressor instance
    """
    model = LGBMRegressor(**kwargs)
    model.fit(X, y)
    return model


def lgbm_auto(X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              **kwargs) -> LGBMRegressor:
    """
    One-line auto-tuned LightGBM with hyperparameter optimization.
    
    Args:
        X: Training features
        y: Training targets
        **kwargs: Additional parameters for LGBMAutoTuner
        
    Returns:
        Optimized LGBMRegressor instance
    """
    tuner = LGBMAutoTuner(**kwargs)
    return tuner.optimize(X, y)


# Information and utility functions
def show_lgbm_progressive_strategies():
    """Display information about available LightGBM strategies (progressive optimization removed)."""
    strategy_config = LightGBMStrategyConfig()
    strategies = strategy_config.get_available_strategies()
    
    print("Available LightGBM Optimization Strategies:")
    print("=" * 60)
    
    for strategy in strategies:
        info = strategy_config.get_strategy_info(strategy)
        
        print(f"\n{strategy.upper()}")
        print(f"Description: {info['description']}")
        print(f"Use case: {info['use_case']}")
    
    print(f"\nUsage:")
    print(f"   model = lgbm_auto(X, y, strategy='fast')")


def get_lgbm_progressive_results(model):
    """Get progressive optimization results from a fitted model (progressive optimization removed)."""
    print("Progressive optimization has been removed from SimpleMLR.")
    return None


def show_lgbm_optimization_summary(model):
    """Show optimization summary for a fitted auto-tuned model."""
    if hasattr(model, 'study_') and model.study_ is not None:
        print("LightGBM Optimization Summary:")
        print("=" * 50)
        print(f"Best value (RMSE): {model.study_.best_value:.4f}")
        print(f"Best parameters: {model.study_.best_params}")
        print(f"Number of trials: {len(model.study_.trials)}")
        
    else:
        print("No optimization study found in model.")


def show_lgbm_plot(model, *args, **kwargs):
    """Alias for model.quick_graph() method."""
    return model.graph(*args, **kwargs)