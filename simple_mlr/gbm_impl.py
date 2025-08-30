"""
Sklearn GBM implementation for SimpleMLR - Reliable, well-tested gradient boosting.

Sklearn's Gradient Boosting Machine is the traditional, well-established gradient
boosting algorithm that comes with scikit-learn. While not as fast as XGBoost or
LightGBM, it offers:
- Rock-solid reliability and stability
- Extensive testing and validation over many years
- Consistent behavior across different platforms
- No external dependencies beyond scikit-learn
- Excellent documentation and community support

This module provides two main classes:
- GBMRegressor: Direct sklearn GBM model with SimpleMLR's easy interface
- GBMAutoTuner: Automatically finds the best sklearn GBM settings for your data

Perfect when you need maximum reliability and stability, or when you want
to avoid external dependencies beyond scikit-learn.
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
from sklearn.ensemble import GradientBoostingRegressor
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


class GBMStrategyConfig(BaseStrategyConfig):
    """
    Defines how sklearn GBM implements optimization strategies.
    
    This class maps universal strategy names like 'fast' and 'stable' to
    sklearn GBM-specific parameter ranges. Sklearn GBM has simpler parameters
    than XGBoost/LightGBM but still offers good control over the training process.
    
    Users don't typically work with this directly - it's used internally
    by GBMAutoTuner when you specify a strategy name.
    """
    
    def __init__(self):
        """Initialize with GBM strategy configurations."""
        # Import GBM strategies from gbm_strategies.py
        try:
            from .gbm_strategies import (
                GBM_STRATEGY_CONFIGS, 
                suggest_gbm_parameter,
                get_gbm_strategy_categorical_params
            )
        except ImportError:
            # Fallback for direct script execution
            from gbm_strategies import (
                GBM_STRATEGY_CONFIGS, 
                suggest_gbm_parameter,
                get_gbm_strategy_categorical_params
            )
        self._strategy_configs = GBM_STRATEGY_CONFIGS
        self._suggest_parameter_func = suggest_gbm_parameter
        self._get_categorical_params_func = get_gbm_strategy_categorical_params
    
    def get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get GBM parameter ranges for the specified strategy."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown GBM strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]['ranges']
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get complete GBM strategy information."""
        if strategy not in self._strategy_configs:
            available = list(self._strategy_configs.keys())
            raise ValueError(f"Unknown GBM strategy '{strategy}'. Available: {available}")
        return self._strategy_configs[strategy]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of all available GBM strategies."""
        return list(self._strategy_configs.keys())
    
    
    def get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get GBM categorical parameter choices for the specified strategy."""
        return self._get_categorical_params_func(strategy)
    
    def suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced GBM parameter suggestion with categorical support."""
        return self._suggest_parameter_func(trial, param_name, param_info)
    
    def get_algorithm_name(self) -> str:
        """Return algorithm name for GBM."""
        return "GBM"


class GBMRegressor(BaseBoostingRegressor):
    """
    Sklearn GBM regression model with SimpleMLR's easy-to-use interface.
    
    This class provides the reliability of sklearn's gradient boosting with SimpleMLR's convenience:
    - Automatic data preprocessing (scaling, encoding)
    - Simple fit/predict interface like scikit-learn
    - Built-in plotting with quick_graph() and plot_analysis()
    - Maximum stability and reproducibility
    - No external dependencies beyond scikit-learn
    
    Best choice when you need maximum reliability, are working in environments
    where external dependencies are restricted, or want the most stable and
    predictable behavior.
    
    Example:
        model = GBMRegressor()
        model.fit(X_train, y_train)
        model.quick_graph()  # Instant performance analysis!
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 max_features: Union[str, float] = 1.0,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 alpha: float = 0.9,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 **kwargs):
        """
        Initialize GBM regressor.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of individual trees
            learning_rate: Learning rate shrinks contribution of each tree
            subsample: Fraction of samples used for fitting individual trees
            max_features: Number of features to consider for best split
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required to be at leaf node
            min_weight_fraction_leaf: Minimum weighted fraction of samples at leaf
            alpha: Alpha-quantile for Huber and Quantile loss functions
            random_state: Random seed for reproducibility
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to handle categorical features
            verbose: Whether to print progress information
            n_jobs: Number of CPU cores to use (ignored - GBM doesn't support n_jobs)
            use_gpu: Whether to use GPU acceleration (ignored - GBM is CPU only)
            **kwargs: Additional GBM parameters
        """
        # GBM-specific parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.alpha = alpha
        
        # Warn about GPU usage - GBM doesn't support GPU
        if use_gpu:
            warnings.warn("sklearn GradientBoostingRegressor does not support GPU acceleration. "
                         "Using CPU computation.", UserWarning)
            use_gpu = False
        
        # Initialize base class with use_gpu=False for GBM
        super().__init__(
            random_state=random_state,
            scale_features=scale_features,
            handle_categorical=handle_categorical,
            verbose=verbose,
            n_jobs=n_jobs,
            use_gpu=False,  # GBM always uses CPU
            **kwargs
        )
    
    def _create_model(self, **params):
        """Create sklearn GradientBoostingRegressor instance."""
        params['verbose'] = 0  # Force sklearn to be silent
        return GradientBoostingRegressor(**params)
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get GBM-specific model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'max_features': self.max_features,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'verbose': 0  # Suppress sklearn's verbose output
        }
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get device-specific parameters (empty for GBM - no GPU support)."""
        return {}
    
    def _test_gpu_availability(self) -> bool:
        """Test GPU availability (always False for GBM)."""
        return False
    
    def _get_algorithm_display_name(self) -> str:
        """Return display name for GBM."""
        return "Gradient Boosting Machine"
    
    def _get_feature_importance_values(self) -> Optional[np.ndarray]:
        """Get feature importance values from fitted GBM model."""
        if hasattr(self.model_, 'feature_importances_'):
            return self.model_.feature_importances_
        return None


class GBMAutoTuner(BaseAutoTuner):
    """
    Auto-tuner for sklearn Gradient Boosting Machine with advanced optimization features.
    
    Provides all SimpleMLR optimization features including multi-fidelity evaluation
    and constraint checking specifically for GBM.
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
                 validation_split: Optional[float] = None,
                 early_stopping_rounds: Optional[int] = None,
                 override_params: Optional[Dict[str, Union[Tuple[float, float], float, int]]] = None,
                 multi_fidelity: bool = False,
                 fidelity_stages: Optional[List] = None,
                 fidelity_percentile: float = 0.7,
                 **kwargs):
        """
        Initialize GBM auto-tuner.
        
        Args follow the same pattern as base class with GBM-specific adjustments.
        """
        # Check for removed parameters
        if 'progressive' in kwargs:
            raise TypeError("GBMAutoTuner.__init__() got an unexpected keyword argument 'progressive'. "
                          "Progressive optimization has been removed from SimpleMLR.")
        
        # Warn about GPU usage - GBM doesn't support GPU
        if use_gpu:
            warnings.warn("sklearn GradientBoostingRegressor does not support GPU acceleration. "
                         "Using CPU computation.", UserWarning)
            use_gpu = False
        
        # Initialize GBM strategy config BEFORE parent init
        self._strategy_config = GBMStrategyConfig()
        
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
            use_gpu=False,  # GBM always uses CPU
            validation_split=validation_split,
            early_stopping_rounds=early_stopping_rounds,
            override_params=override_params,
            multi_fidelity=multi_fidelity,
            fidelity_stages=fidelity_stages,
            fidelity_percentile=fidelity_percentile,
            **kwargs
        )
    
    def _create_model_instance(self, **params):
        """Create GBM model instance for optimization."""
        params['verbose'] = 0  # Force sklearn to be silent
        return GradientBoostingRegressor(**params)
    
    def _get_algorithm_name(self) -> str:
        """Return algorithm name for display."""
        return "GBM"
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available GBM strategies."""
        return self._strategy_config.get_available_strategies()
    
    
    def _get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for the specified GBM strategy."""
        return self._strategy_config.get_strategy_ranges(strategy)
    
    def _get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get GBM categorical parameter choices for the specified strategy."""
        return self._strategy_config.get_strategy_categorical_params(strategy)
    
    def _suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced GBM parameter suggestion with categorical support."""
        return self._strategy_config.suggest_parameter(trial, param_name, param_info)
    
    def _get_default_fidelity_stages(self) -> List:
        """Get default fidelity stages for GBM multi-fidelity optimization."""
        # GBM is slower, so use smaller default stages
        return [25, 75, None]
    
    def _get_device_params(self) -> Dict[str, Any]:
        """Get device-specific parameters (empty for GBM - no GPU support)."""
        return {}
    
    def _test_gpu_availability(self) -> bool:
        """Test GPU availability (always False for GBM)."""
        return False
    
    def _check_algorithm_constraints(self, params: dict, n_estimators: int = None,
                                   optimization_context: dict = None) -> bool:
        """
        Check GBM-specific parameter constraints with RELAXED bounds for better exploration.
        
        FIXED: Updated constraints to follow working LightGBM patterns and enable effective 
        parameter space exploration while maintaining sklearn GBM requirements.
        
        Args:
            params: Parameter dictionary to validate
            n_estimators: Specific n_estimators value (for multi-fidelity)
            optimization_context: Context information for optimization
        
        Returns:
            bool: True if constraints are satisfied
        """
        if optimization_context is None:
            optimization_context = {}
        
        mode = optimization_context.get('mode', 'traditional')
        
        # Use provided n_estimators or extract from params
        n_est = n_estimators if n_estimators is not None else params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 3)
        subsample = params.get('subsample', 1.0)
        max_features = params.get('max_features', 1.0)
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        min_weight_fraction_leaf = params.get('min_weight_fraction_leaf', 0.0)
        
        # FIXED: GBM Constraint 1 - More reasonable learning_rate * n_estimators relationship
        lr_n_est_product = learning_rate * n_est
        if mode in ['progressive', 'multi_fidelity']:
            # RELAXED: Even more permissive for progressive/multi-fidelity modes
            if not (1 <= lr_n_est_product <= 150):
                return False
        else:
            # FIXED: Relaxed from (3, 200) to (2, 120) - follows LightGBM pattern
            if not (2 <= lr_n_est_product <= 120):
                return False
        
        # ESSENTIAL: GBM Constraint 2 - sklearn requirement (keep strict)
        if min_samples_split < 2 * min_samples_leaf:
            return False
        
        # ESSENTIAL: GBM Constraint 3 - sklearn requirement (keep strict) 
        if min_samples_split < 2:
            return False
        
        # ESSENTIAL: GBM Constraint 4 - sklearn requirement (keep strict)
        if min_samples_leaf < 1:
            return False
        
        # FIXED: GBM Constraint 5 - Relaxed subsample * max_features relationship
        if isinstance(max_features, (int, float)) and isinstance(subsample, (int, float)):
            if max_features <= 1.0 and subsample <= 1.0:  # Both are fractions
                sample_product = subsample * max_features
                # FIXED: Relaxed from 0.3 to 0.16 (following LightGBM pattern)
                if sample_product < 0.16:
                    return False
        
        # ESSENTIAL: GBM Constraint 6 - max_features validation (keep strict)
        if isinstance(max_features, str):
            if max_features not in ['sqrt', 'log2', 'auto']:
                return False
        elif isinstance(max_features, (int, float)):
            if max_features <= 0:
                return False
        
        # ESSENTIAL: GBM Constraint 7 - weight fraction constraints (keep strict)
        if not (0.0 <= min_weight_fraction_leaf < 0.5):
            return False
            
        # RELAXED: GBM Constraint 8 - Allow deeper trees for complex patterns
        if max_depth > 15:  # Increased from 10 to 15
            return False
            
        # RELAXED: GBM Constraint 9 - More permissive subsample range
        if not (0.05 <= subsample <= 1.0):  # Lowered from 0.1 to 0.05
            return False
        
        # ENHANCED: Multi-fidelity stage awareness
        if mode == 'multi_fidelity':
            fidelity_stages = optimization_context.get('fidelity_stages', [])
            current_stage_idx = optimization_context.get('stage_idx', 0)
            
            if fidelity_stages and n_est:
                # Check if learning rate works with ANY fidelity stage (like LightGBM)
                works_with_any_stage = False
                for stage_budget in fidelity_stages:
                    if stage_budget and 1 <= learning_rate * stage_budget <= 150:
                        works_with_any_stage = True
                        break
                
                if not works_with_any_stage:
                    return False
                
                # Early stages get more permissive constraints
                if current_stage_idx < len(fidelity_stages) // 2:
                    # Allow more exploration in early stages
                    if learning_rate > 0.5:  # Still prevent extreme values
                        return False
        
        # ENHANCED: Progressive mode with stage-aware constraints  
        elif mode == 'progressive':
            stage_info = optimization_context.get('stage_info', {})
            current_stage = stage_info.get('stage_name', '')
            
            if current_stage == 'learning_dynamics':
                # RELAXED: Less restrictive for learning dynamics stage
                if lr_n_est_product > 100:  # Increased from 150
                    return False
                    
            elif current_stage == 'regularization':
                # ENHANCED: More flexible regularization validation
                regularization_strength = 0
                if min_samples_split >= 5:
                    regularization_strength += 1
                if min_samples_leaf >= 2:
                    regularization_strength += 1
                if min_weight_fraction_leaf >= 0.01:
                    regularization_strength += 1
                
                # At least some regularization should be present
                if regularization_strength == 0 and subsample >= 0.95:
                    return False
        
        return True
    
    def _calculate_constraint_penalty(self, params: dict, n_estimators: int = None, context: dict = None) -> float:
        """
        Calculate graduated penalty for constraint violations in GBM optimization.
        
        FIXED: Added missing penalty calculation method following LightGBM patterns.
        Ensures penalties never exceed reasonable bounds and provides context-aware
        penalty calculation for multi-fidelity optimization.
        
        Args:
            params: Parameter dictionary 
            n_estimators: Specific n_estimators value (for multi-fidelity)
            context: Optimization context information
            
        Returns:
            float: Penalty score (always <= 100 to avoid breaking optimization)
        """
        if context is None:
            context = {}
            
        base_penalty = 25.0  # Reasonable base penalty
        penalty_multiplier = 1.0
        
        # Get context information
        mode = context.get('mode', 'traditional')
        fidelity_stages = context.get('fidelity_stages', [])
        current_stage_idx = context.get('stage_idx', 0)
        
        # Use provided n_estimators or extract from params
        n_est = n_estimators if n_estimators is not None else params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        subsample = params.get('subsample', 1.0)
        max_features = params.get('max_features', 1.0)
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        
        # GRADUATED: Analyze learning_rate * n_estimators constraint violation
        if learning_rate and n_est:
            lr_product = learning_rate * n_est
            target_min = 2 if mode in ['progressive', 'multi_fidelity'] else 1
            target_max = 150 if mode in ['progressive', 'multi_fidelity'] else 120
            
            if lr_product < target_min:
                penalty_multiplier += (target_min - lr_product) * 0.5
            elif lr_product > target_max:
                penalty_multiplier += (lr_product - target_max) * 0.1
            
            # MULTI-FIDELITY: Check if it works with other stages  
            if mode == 'multi_fidelity' and fidelity_stages:
                works_with_other_stage = False
                for stage_budget in fidelity_stages:
                    if stage_budget and 1 <= learning_rate * stage_budget <= 150:
                        works_with_other_stage = True
                        break
                
                # Reduce penalty if valid for future stages
                if works_with_other_stage:
                    penalty_multiplier *= 0.4  # Significant reduction
        
        # GRADUATED: Analyze subsample * max_features constraint violation
        if isinstance(max_features, (int, float)) and isinstance(subsample, (int, float)):
            if max_features <= 1.0 and subsample <= 1.0:
                sample_product = subsample * max_features
                if sample_product < 0.16:
                    # This affects model quality significantly
                    penalty_multiplier += (0.16 - sample_product) * 5
        
        # ESSENTIAL: Heavy penalty for sklearn mathematical violations
        if min_samples_split < 2 * min_samples_leaf:
            penalty_multiplier += 3  # Heavy penalty for sklearn requirement violation
        
        if min_samples_split < 2:
            penalty_multiplier += 5  # Very heavy penalty for invalid sklearn parameter
            
        if min_samples_leaf < 1:
            penalty_multiplier += 5  # Very heavy penalty for invalid sklearn parameter
        
        # CONTEXT-AWARE: Adjust penalties based on optimization stage
        if mode == 'multi_fidelity':
            # Early stages get lower penalties to encourage exploration
            if current_stage_idx < len(fidelity_stages) // 2:
                penalty_multiplier *= 0.6
        elif mode == 'progressive':
            # Progressive optimization gets slightly lower penalties
            penalty_multiplier *= 0.8
        
        # BOUNDED: Ensure penalty never exceeds reasonable bounds
        final_penalty = base_penalty * penalty_multiplier
        return min(final_penalty, 100.0)  # Cap at 100 to avoid breaking optimization
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]):
        """
        Algorithm-agnostic hyperparameter optimization for GBM.
        
        Updated to have consistent output formatting with LGBM.
        """
        # Initialize progress tracking (from ProgressTrackingMixin)
        self._initialize_progress_tracking()
        
        # Suppress warnings and set proper logging levels
        import logging
        if self.verbose <= 0:
            logging.getLogger('optuna').setLevel(logging.ERROR)
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Validate and preprocess inputs using base class patterns
        X_validated, y_validated = DataValidator.validate_inputs(X, y)
        X_processed, self.preprocessing_artifacts_ = DataValidator.prepare_features(
            X_validated,
            scale_features=self.scale_features,
            handle_categorical=self.handle_categorical
        )
        
        # Merge strategy configuration with overrides
        merged_config = self._merge_strategy_with_overrides(self.strategy, self.override_params)
        self._merged_ranges = merged_config['ranges']
        self._fixed_params = merged_config['fixed_params']
        
        # UPDATED: Custom header display (like LGBM) instead of using base class verbose method
        if self.verbose >= 1:
            # Build mode string
            if self.validation_split is not None:
                mode_str = f"Validation Split ({self.validation_split:.1%})"
            else:
                mode_str = "Traditional"
            
            # Build additional info strings
            early_str = f" + Early Stopping ({self.early_stopping_rounds} rounds)" if self.early_stopping_rounds else ""
            multi_str = f" + Multi-Fidelity" if self.multi_fidelity else ""

            # Display concise header (matching LGBM format)
            print(f"{mode_str} hyperparameter optimization{early_str}{multi_str} with {self.n_trials} trials...")
            print(f"Strategy: '{self.strategy}'")
        
        # Run traditional optimization
        best_params = self._run_traditional_optimization(X_processed, y_validated)
        
        # Create final model
        final_model = self._create_final_model(X_validated, y_validated, X_processed, best_params)
        
        # Show final summary (from ProgressTrackingMixin)
        total_time = time.time() - self._start_time
        self._show_final_summary(best_params, self._overall_best_rmse, total_time)
        
        return final_model
    
    def _run_traditional_optimization(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run traditional simultaneous optimization using same pattern as XGBoost/LightGBM."""
        config = GBMStrategyConfig()
        
        # Create study with optional early stopping
        if hasattr(self, 'early_stopping_rounds') and self.early_stopping_rounds:
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

    def _get_strategy_defaults(self, merged_ranges: dict) -> dict:
        """Get default parameter values from strategy ranges."""
        defaults = {}
        for param, (min_val, max_val) in merged_ranges.items():
            # Use midpoint for most parameters, but use more conservative defaults for some
            if param == 'learning_rate':
                defaults[param] = 0.1  # Conservative default
            elif param == 'n_estimators':
                defaults[param] = 200  # Moderate default
            elif param == 'max_depth':
                defaults[param] = 4    # Conservative for GBM
            elif param == 'subsample':
                defaults[param] = 0.9  # High sampling
            elif param == 'max_features':
                defaults[param] = 0.5  # Half features
            elif param in ['min_samples_split', 'min_samples_leaf']:
                # Integer parameters - use integer midpoint
                defaults[param] = int((min_val + max_val) / 2)
            else:
                # Use geometric mean for other parameters
                if min_val > 0:
                    defaults[param] = (min_val * max_val) ** 0.5
                else:
                    defaults[param] = (min_val + max_val) / 2
        return defaults


    def _merge_strategy_with_overrides(self, strategy: str, override_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Merge strategy configuration with user overrides."""
        try:
            from .gbm_strategies import merge_gbm_strategy_with_overrides
        except ImportError:
            from gbm_strategies import merge_gbm_strategy_with_overrides
        return merge_gbm_strategy_with_overrides(strategy, override_params)

    def _clear_progress_line(self):
        """Clear progress line - delegated to mixin."""
        if hasattr(super(), '_clear_progress_line'):
            super()._clear_progress_line()

    def _show_final_summary(self, best_params: dict, best_rmse: float, total_time: float):
        """Show final optimization summary with training metrics (matching LGBM format)."""
        if self.verbose == -1:
            pass
        elif self.verbose == 0:
            print(f">> Model ready for use!")
        elif self.verbose >= 1:
            print(f"\n>> Model ready for use!")
            print(f"-> CV RMSE: {best_rmse:.4f}")
            print(f"-- Total Time: {total_time:.1f}s")
            
            # UPDATED: Show training metrics if available (like LGBM does)
            if hasattr(self, 'best_model_') and hasattr(self.best_model_, 'training_metrics_'):
                metrics = self.best_model_.training_metrics_
                print(f"Training R2: {metrics['r2_score']:.6f}")
                print(f"Training RMSE: {metrics['rmse']:.6f}")

            # Show best parameters as Python dictionary
            display_params = self._get_display_params(best_params)
            print(f"Best parameters: {display_params}")

    def _get_display_params(self, params: dict) -> dict:
        """Format parameters for display, rounding floats appropriately."""
        display = {}
        for key, value in params.items():
            if isinstance(value, float):
                if key in ['learning_rate']:
                    display[key] = round(value, 6)
                elif key in ['subsample', 'max_features', 'min_weight_fraction_leaf', 'alpha']:
                    display[key] = round(value, 4)
                else:
                    display[key] = round(value, 8)
            else:
                display[key] = value
        return display

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, config: GBMStrategyConfig) -> float:
        """Objective function with proper progress tracking - same pattern as XGBoost/LightGBM."""
        # Route to single fidelity (GBM doesn't support multi-fidelity typically)
        return self._single_fidelity_objective(trial, X, y, config)

    def _single_fidelity_objective(self, trial, X: pd.DataFrame, y: pd.Series, config: GBMStrategyConfig) -> float:
        """Traditional single-fidelity objective function."""
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

        # Add base parameters
        params.update({
            'random_state': self.random_state,
            'verbose': 0
        })

        # Check algorithm-specific mathematical constraints and prune invalid combinations
        context = self._get_optimization_context()
        if not self._check_algorithm_constraints(params, optimization_context=context):
            trial.set_user_attr("pruned_reason", "constraint_violation")
            raise optuna.TrialPruned()

        # Create model using the framework's model creation method
        model = self._create_model_instance(**params)

        # Evaluate model
        if hasattr(self, 'validation_split') and self.validation_split is not None:
            rmse = self._validation_split_evaluation(model, X, y)
        else:
            rmse = self._cross_validation_evaluation(model, X, y)

        # Update progress display using mixin
        self._update_standard_progress(trial_start_time, rmse, params)

        return rmse

    def _validation_split_evaluation(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model using hold-out validation split."""
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Train and evaluate
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        except Exception:
            return float('inf')

    def _cross_validation_evaluation(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model using cross-validation."""
        try:
            from sklearn.model_selection import KFold, cross_val_score
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=1  # Avoid nested parallelism
            )
            return -scores.mean()
        except Exception:
            return float('inf')

    def _create_final_model(self, X_original: pd.DataFrame, y_original: pd.Series, 
                          X_processed: pd.DataFrame, best_params: dict):
        """
        Create final GBM model with best parameters and calculate training metrics.
        
        Updated to properly calculate and store training metrics for display.
        """
        # Create model with best parameters
        regressor = GBMRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', 3),
            learning_rate=best_params.get('learning_rate', 0.1),
            subsample=best_params.get('subsample', 1.0),
            max_features=best_params.get('max_features', 1.0),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=best_params.get('min_weight_fraction_leaf', 0.0),
            alpha=best_params.get('alpha', 0.9),
            random_state=self.random_state,
            scale_features=self.scale_features,
            handle_categorical=self.handle_categorical,
            verbose=0,
            n_jobs=self.n_jobs
        )
        
        # Fit the model
        regressor.fit(X_original, y_original)
        
        # UPDATED: Calculate and store training metrics (like LGBM does)
        if hasattr(regressor, 'model_') and regressor.model_ is not None:
            try:
                # Get predictions on training data
                if hasattr(regressor, 'training_X_') and hasattr(regressor, 'training_y_'):
                    y_pred = regressor.predict(regressor.training_X_)
                    y_true = regressor.training_y_.values if hasattr(regressor.training_y_, 'values') else regressor.training_y_
                    
                    # Calculate training metrics
                    from sklearn.metrics import r2_score, mean_squared_error
                    import numpy as np
                    
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    # Store training metrics in the model
                    regressor.training_metrics_ = {
                        'r2_score': r2,
                        'rmse': rmse
                    }
                    
                    # Also store in the auto-tuner for summary display
                    self.best_model_ = regressor
                    
            except Exception as e:
                # If training metrics calculation fails, continue without them
                if self.verbose >= 1:
                    print(f"Warning: Could not calculate training metrics: {e}")
        
        # Store study for compatibility
        regressor.study_ = getattr(self, 'study_', None)
        
        
        return regressor
    
    def _suggest_parameters_for_trial(self, trial):
        """Suggest parameters for a trial using base class patterns."""
        params = {}
        # Add numerical parameters
        for param_name, param_range in self._merged_ranges.items():
            params[param_name] = self._suggest_parameter(trial, param_name, param_range)
        
        # Add categorical parameters
        for param_name, choices in self._categorical_params.items():
            params[param_name] = trial.suggest_categorical(param_name, choices)
            
        return params
    

# Convenience functions
def gbm_regressor(X, y, **kwargs) -> GBMRegressor:
    """
    One-line GBM regression with automatic preprocessing.
    
    Args:
        X: Training features
        y: Training targets  
        **kwargs: GBMRegressor parameters
        
    Returns:
        Fitted GBMRegressor instance
    """
    model = GBMRegressor(**kwargs)
    return model.fit(X, y)


def gbm_auto(X, y, **kwargs) -> GBMRegressor:
    """
    One-line auto-tuned GBM with hyperparameter optimization.
    
    Args:
        X: Training features
        y: Training targets
        **kwargs: GBMAutoTuner parameters
        
    Returns:
        Fitted GBMRegressor with optimized parameters
    """
    tuner = GBMAutoTuner(**kwargs)
    return tuner.optimize(X, y)


def show_gbm_progressive_strategies():
    """Display information about available GBM strategies (progressive optimization removed)."""
    strategy_config = GBMStrategyConfig()
    strategies = strategy_config.get_available_strategies()
    
    print("Available GBM Optimization Strategies:")
    print("=" * 60)
    
    for strategy in strategies:
        info = strategy_config.get_strategy_info(strategy)
        
        print(f"\n{strategy.upper()}")
        print(f"Description: {info['description']}")
        print(f"Use case: {info['use_case']}")
    
    print(f"\nUsage:")
    print(f"   model = gbm_auto(X, y, strategy='fast')")


def get_gbm_progressive_results(model):
    """Get progressive optimization results from a fitted model (progressive optimization removed)."""
    print("Progressive optimization has been removed from SimpleMLR.")
    return None


def show_gbm_optimization_summary(model):
    """
    Show optimization summary for fitted auto-tuned model.
    
    Args:
        model: Fitted model from gbm_auto()
    """
    if not hasattr(model, 'study_'):
        print("ERROR: Model was not auto-tuned. Use gbm_auto() for optimization.")
        return
    
    study = model.study_
    print(f"GBM Optimization Summary")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    


def show_gbm_plot(model, *args, **kwargs):
    """
    Alias for model.quick_graph() method.
    
    Args:
        model: Fitted GBM model
        *args, **kwargs: Arguments passed to model.quick_graph()
    """
    return model.graph(*args, **kwargs)