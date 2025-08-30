"""
XGBoost implementation for SimpleMLR - Fast, accurate gradient boosting.

XGBoost (eXtreme Gradient Boosting) is one of the most popular and successful
machine learning algorithms, especially for structured/tabular data. It's known for:
- Excellent performance on many types of problems
- Built-in regularization to prevent overfitting
- Efficient parallel processing and GPU support
- Robust handling of missing values

This module provides two main classes:
- XGBRegressor: Direct XGBoost model with SimpleMLR's easy interface
- XGBAutoTuner: Automatically finds the best XGBoost settings for your data

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
import xgboost as xgb
import optuna

# Import base classes and mixins
from .base import BaseBoostingRegressor, BaseAutoTuner
from .base_strategies import BaseStrategyConfig
from .utils import DataValidator, ModelEvaluator


class XGBoostStrategyConfig(BaseStrategyConfig):
    """
    Defines how XGBoost implements optimization strategies.
    
    This class maps universal strategy names like 'fast' and 'stable' to
    XGBoost-specific parameter ranges. For example, 'fast' might use:
    - Lower n_estimators for speed
    - Smaller max_depth to avoid overfitting
    - Higher learning_rate to converge quickly
    
    Users don't typically work with this directly - it's used internally
    by XGBAutoTuner when you specify a strategy name.
    """
    
    def __init__(self):
        """Initialize with XGBoost strategy configurations."""
        # Import XGBoost strategies from xgboost_strategies.py
        from .xgboost_strategies import (
            STRATEGY_CONFIGS,
            suggest_parameter as xgb_suggest_parameter,
            get_strategy_categorical_params
        )
        self._strategy_configs = STRATEGY_CONFIGS
        self._suggest_parameter_func = xgb_suggest_parameter
        self._get_categorical_params_func = get_strategy_categorical_params
    
    def get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get XGBoost parameter ranges for the specified strategy."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown XGBoost strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]['ranges']
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get complete XGBoost strategy information."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown XGBoost strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available XGBoost strategies."""
        return list(self._strategy_configs.keys())
    
    
    def get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get XGBoost categorical parameter choices for the specified strategy."""
        return self._get_categorical_params_func(strategy)
    
    def suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced XGBoost parameter suggestion with categorical support."""
        return self._suggest_parameter_func(trial, param_name, param_info)
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "XGBoost"


class XGBRegressor(BaseBoostingRegressor):
    """
    XGBoost regression model with SimpleMLR's easy-to-use interface.
    
    This class provides all the power of XGBoost with SimpleMLR's convenience:
    - Automatic data preprocessing (scaling, encoding)
    - Simple fit/predict interface like scikit-learn
    - Built-in plotting with quick_graph() and plot_analysis()
    - GPU support when available
    - Comprehensive metrics calculation
    
    Perfect for both beginners who want simplicity and experts who need
    full XGBoost functionality with better usability.
    
    Example:
        model = XGBRegressor()
        model.fit(X_train, y_train)
        model.quick_graph()  # Instant performance analysis!
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 gamma: float = 0,
                 min_child_weight: float = 1,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initialize XGBoost Regressor.
        
        Args:
            n_estimators: Number of gradient boosted trees
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            gamma: Minimum loss reduction required to make a further partition
            min_child_weight: Minimum sum of instance weight needed in a child
            random_state: Random seed for reproducibility
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to handle categorical features
            verbose: Whether to print progress information
            n_jobs: Number of CPU cores to use (-1=all cores)
            use_gpu: Whether to use GPU acceleration
            gpu_id: Which GPU to use (0=first GPU, 1=second GPU, etc.)
            **kwargs: Additional XGBoost parameters
        """
        # Store XGBoost-specific parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
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
        """Create XGBoost model instance."""
        return xgb.XGBRegressor(verbosity=0, **params)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get XGBoost-specific model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'random_state': self.random_state
        }
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get XGBoost device-specific parameters."""
        device_params = {'n_jobs': self.n_jobs}
        
        if self.use_gpu:
            if self._test_gpu_availability():
                device_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': self.gpu_id
                })
                if self.verbose:
                    print(f"Using GPU acceleration (gpu_id={self.gpu_id})")
            else:
                if self.verbose:
                    print("GPU requested but not available, using CPU")
                    
        return device_params
    
    def _test_gpu_availability(self) -> bool:
        """Test if GPU is available for XGBoost."""
        try:
            # Test GPU availability by creating a small XGBoost model
            test_model = xgb.XGBRegressor(
                n_estimators=1,
                tree_method='gpu_hist',
                gpu_id=self.gpu_id,
                verbosity=0
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
        """Get feature importance values from XGBoost model."""
        if self.model_ is not None:
            return self.model_.feature_importances_
        return None
    
    def _get_algorithm_display_name(self) -> str:
        """Return display name for this algorithm."""
        return "XGBoost"


class XGBAutoTuner(BaseAutoTuner):
    """
    XGBoost Auto-Tuner implementation using the multi-algorithm framework.
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
        Initialize XGBoost auto-tuner.
        
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
            **kwargs: Additional XGBoost-specific parameters
        """
        # Check for removed parameters
        if 'progressive' in kwargs:
            raise TypeError("XGBAutoTuner.__init__() got an unexpected keyword argument 'progressive'. "
                          "Progressive optimization has been removed from SimpleMLR.")
        
        # Store XGBoost-specific parameters
        self.gpu_id = gpu_id
        self.gpu_optimize = gpu_optimize
        
        # Create strategy config BEFORE parent init (performance optimization)
        self._strategy_config = XGBoostStrategyConfig()
        
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
        """Create XGBoost model instance for optimization."""
        return xgb.XGBRegressor(verbosity=0, **params)
    
    def _get_algorithm_name(self) -> str:
        """Return algorithm name for display."""
        return "XGBoost"
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available XGBoost strategies."""
        return self._strategy_config.get_available_strategies()
    
    
    def _get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get XGBoost parameter ranges for the specified strategy."""
        return self._strategy_config.get_strategy_ranges(strategy)
    
    def _get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get XGBoost categorical parameter choices for the specified strategy."""
        return self._strategy_config.get_strategy_categorical_params(strategy)
    
    def _suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced XGBoost parameter suggestion with categorical support."""
        return self._strategy_config.suggest_parameter(trial, param_name, param_info)
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get XGBoost device-specific parameters."""
        device_params = {'n_jobs': self.n_jobs}
        
        if self.use_gpu:
            if self._test_gpu_availability():
                device_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': self.gpu_id
                })
            else:
                if self.verbose >= 1:
                    print("GPU requested but not available, using CPU")
                    
        return device_params
    
    def _test_gpu_availability(self) -> bool:
        """Test if GPU is available for XGBoost."""
        try:
            # Test GPU availability by creating a small XGBoost model
            test_model = xgb.XGBRegressor(
                n_estimators=1,
                tree_method='gpu_hist',
                gpu_id=self.gpu_id,
                verbosity=0
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
        """Get default fidelity stages for XGBoost multi-fidelity optimization."""
        return [50, 150, None]  # None means use full n_estimators from optimization
    
    def _check_algorithm_constraints(self, params: dict, n_estimators: int = None,
                                   optimization_context: dict = None) -> bool:
        """XGBoost-specific constraint checking with context awareness."""
        if optimization_context is None:
            optimization_context = {}
        
        mode = optimization_context.get('mode', 'traditional')
        
        # Route to appropriate XGBoost constraint checker
        if mode == 'traditional':
            return self._check_traditional_xgb_constraints(params, n_estimators)
        elif mode == 'multi_fidelity':
            return self._check_multifidelity_xgb_constraints(params, n_estimators, optimization_context)
        else:
            return self._check_traditional_xgb_constraints(params, n_estimators)
    
    def _check_traditional_xgb_constraints(self, params: dict, n_estimators: int = None) -> bool:
        """Traditional XGBoost constraint checking - keep existing logic exactly as is."""
        # Use n_estimators from params if not provided separately
        if n_estimators is None:
            n_estimators = params.get('n_estimators')
        
        # XGBoost-specific: Learning rate * n_estimators relationship
        if 'learning_rate' in params and n_estimators is not None:
            product = params['learning_rate'] * n_estimators
            if product < 10 or product > 50:  # Outside research-validated range for XGBoost
                return False
        
        # XGBoost-specific: Subsample * colsample_bytree relationship
        if 'subsample' in params and 'colsample_bytree' in params:
            data_fraction = params['subsample'] * params['colsample_bytree']
            if data_fraction < 0.36:  # Too little data causes underfitting in XGBoost
                return False
        
        return True
    
    def _check_multifidelity_xgb_constraints(self, params: dict, n_estimators: int = None, 
                                           context: dict = None) -> bool:
        """Multi-fidelity XGBoost constraint checking - flexible exploration with intelligent validation."""
        # Check data fraction (always strict - this affects underfitting directly)
        if 'subsample' in params and 'colsample_bytree' in params:
            if params['subsample'] * params['colsample_bytree'] < 0.36:
                return False
        
        # For learning_rate * n_estimators, use intelligent multi-stage validation
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
                 y: Union[pd.Series, np.ndarray]) -> XGBRegressor:
        """
        Optimize XGBoost hyperparameters with all advanced features.
        
        This method implements the complete optimization logic with all existing
        functionality from the original XGBAutoTuner.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Optimized XGBRegressor instance
        """
        # Initialize progress tracking using mixin
        self._reset_progress_state()

        # Suppress warnings and set proper logging levels
        import logging
        if self.verbose <= 0:
            logging.getLogger('xgboost').setLevel(logging.ERROR)
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
            config = XGBoostStrategyConfig()
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
            logging.getLogger('xgboost').setLevel(logging.WARNING)

        return final_model

    def _run_traditional_optimization(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run traditional simultaneous optimization."""
        config = XGBoostStrategyConfig()
        
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

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, config: XGBoostStrategyConfig) -> float:
        """Objective function with proper progress tracking."""
        # Route to appropriate objective function based on multi-fidelity setting
        if self.multi_fidelity:
            return self._multi_fidelity_objective(trial, X, y, config)
        else:
            return self._single_fidelity_objective(trial, X, y, config)

    def _single_fidelity_objective(self, trial, X: pd.DataFrame, y: pd.Series, config: XGBoostStrategyConfig) -> float:
        """Traditional single-fidelity objective function."""
        # Record trial start time
        trial_start_time = time.time()
        self._trial_count += 1

        # Build parameters using merged ranges (strategy + overrides)
        params = {}
        params.update(self._fixed_params)
        
        # Add optimizable numerical parameters from merged ranges
        for param_name, param_range in self._merged_ranges.items():
            params[param_name] = config.suggest_parameter(trial, param_name, param_range)
        
        # Add optimizable categorical parameters
        for param_name, choices in self._categorical_params.items():
            params[param_name] = trial.suggest_categorical(param_name, choices)

        # Add device-specific parameters
        device_params = self._get_device_params()
        params.update(device_params)

        # Check algorithm-specific mathematical constraints and prune invalid combinations
        context = self._get_optimization_context()
        if not self._check_algorithm_constraints(params, optimization_context=context):
            trial.set_user_attr("pruned_reason", "constraint_violation")
            raise optuna.TrialPruned()

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

    def _multi_fidelity_objective(self, trial, X: pd.DataFrame, y: pd.Series, config: XGBoostStrategyConfig) -> float:
        """Multi-fidelity objective with staged evaluation."""
        trial_start_time = time.time()
        self._trial_count += 1
        
        # Build base parameters
        params = {}
        params.update(self._fixed_params)
        
        # Add optimizable numerical parameters (excluding n_estimators for multi-fidelity)
        for param_name, param_range in self._merged_ranges.items():
            if param_name != 'n_estimators':
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
                if 'n_estimators' in self._merged_ranges:
                    min_est, max_est = self._merged_ranges['n_estimators']
                    n_estimators = config.suggest_parameter(trial, 'n_estimators', (min_est, max_est))
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
        
        # Analyze learning_rate * n_estimators constraint violation
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
        
        # Analyze subsample * colsample_bytree constraint violation
        if 'subsample' in params and 'colsample_bytree' in params:
            data_fraction = params['subsample'] * params['colsample_bytree']
            if data_fraction < 0.36:
                # This is always a serious violation
                penalty_multiplier += (0.36 - data_fraction) * 10
        
        # Early stages get lower penalties to encourage exploration
        if current_stage_idx < len(fidelity_stages) // 2:
            penalty_multiplier *= 0.5
        
        return base_penalty * penalty_multiplier

    def _get_final_params(self, best_params: dict) -> dict:
        """
        Get final parameters for model creation, ensuring consistency with CV evaluation.
        
        This method merges best_params from optimization with _fixed_params (override parameters)
        in the same way as the CV evaluation, fixing the parameter inconsistency bug.
        
        Args:
            best_params: Parameters from study.best_params
            
        Returns:
            dict: Complete parameter set for final model creation
        """
        # Start with fixed parameters (override parameters from initialization)
        final_params = {}
        final_params.update(self._fixed_params)
        
        # Add optimized parameters from the study
        final_params.update(best_params)
        
        # Add device-specific parameters (GPU/CPU settings)
        device_params = self._get_device_params()
        final_params.update(device_params)
        
        return final_params



    def _create_final_model(self, X: pd.DataFrame, y: pd.Series,
                            X_processed: pd.DataFrame, best_params: dict) -> XGBRegressor:
        """Create the final optimized model."""
        # Get final parameters using the same logic as CV evaluation (CRITICAL BUG FIX)
        final_params = self._get_final_params(best_params)
        
        # Filter out device-specific parameters that should not be passed to XGBRegressor
        model_params = {k: v for k, v in final_params.items()
                       if k not in ['tree_method', 'n_jobs', 'gpu_id', 'predictor']}
        
        # Create best model using the new framework with consistent parameters
        self.best_model_ = XGBRegressor(
            random_state=self.random_state,
            scale_features=self.scale_features,
            handle_categorical=self.handle_categorical,
            verbose=0,
            **model_params
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

        # Fit on full dataset using the same final_params as the model creation
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
            if param == 'n_estimators':
                defaults[param] = int(min_val + (max_val - min_val) * 0.3)
            elif param == 'learning_rate':
                if 'n_estimators' in defaults:
                    target_product = 20
                    optimal_lr = target_product / defaults['n_estimators']
                    defaults[param] = max(min_val, min(optimal_lr, max_val))
                else:
                    defaults[param] = min_val + (max_val - min_val) * 0.3
            elif param == 'max_depth':
                defaults[param] = int(min_val + (max_val - min_val) * 0.3)
            elif param in ['reg_alpha', 'reg_lambda']:
                defaults[param] = min_val + (max_val - min_val) * 0.1
            elif param == 'gamma':
                defaults[param] = min_val
            elif param in ['subsample', 'colsample_bytree']:
                defaults[param] = min_val + (max_val - min_val) * 0.5
            else:
                if param in ['n_estimators', 'max_depth']:
                    defaults[param] = int((min_val + max_val) / 2)
                else:
                    defaults[param] = (min_val + max_val) / 2.0

        defaults.update(self._fixed_params)
        return defaults


# Convenience functions to maintain backward compatibility
def xgb_regressor(X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  **kwargs) -> XGBRegressor:
    """
    One-line XGBoost regression with automatic preprocessing.
    
    Args:
        X: Training features
        y: Training targets
        **kwargs: Additional parameters for XGBRegressor
        
    Returns:
        Fitted XGBRegressor instance
    """
    model = XGBRegressor(**kwargs)
    model.fit(X, y)
    return model


def xgb_auto(X: Union[pd.DataFrame, np.ndarray],
             y: Union[pd.Series, np.ndarray],
             **kwargs) -> XGBRegressor:
    """
    One-line auto-tuned XGBoost with hyperparameter optimization.
    
    Args:
        X: Training features
        y: Training targets
        **kwargs: Additional parameters for XGBAutoTuner
        
    Returns:
        Optimized XGBRegressor instance
    """
    tuner = XGBAutoTuner(**kwargs)
    return tuner.optimize(X, y)


# Information and utility functions
def show_progressive_strategies():
    """Display information about available XGBoost strategies (progressive optimization removed)."""
    strategy_config = XGBoostStrategyConfig()
    strategies = strategy_config.get_available_strategies()
    
    print("Available XGBoost Optimization Strategies:")
    print("=" * 60)
    
    for strategy in strategies:
        info = strategy_config.get_strategy_info(strategy)
        
        print(f"\n{strategy.upper()}")
        print(f"Description: {info['description']}")
        print(f"Use case: {info['use_case']}")
    
    print(f"\nUsage:")
    print(f"   model = xgb_auto(X, y, strategy='fast')")


def get_progressive_results(model):
    """Get progressive optimization results from a fitted model (progressive optimization removed)."""
    print("Progressive optimization has been removed from SimpleMLR.")
    return None


def show_optimization_summary(model):
    """Show optimization summary for a fitted auto-tuned model."""
    if hasattr(model, 'study_') and model.study_ is not None:
        print("Optimization Summary:")
        print("=" * 50)
        print(f"Best value (RMSE): {model.study_.best_value:.4f}")
        print(f"Best parameters: {model.study_.best_params}")
        print(f"Number of trials: {len(model.study_.trials)}")
        
    else:
        print("No optimization study found in model.")


def show_plot(model, *args, **kwargs):
    """Alias for model.quick_graph() method."""
    return model.graph(*args, **kwargs)