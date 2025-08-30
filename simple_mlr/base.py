"""
Base classes for SimpleMLR - The foundation of the library.

This module contains the core building blocks that make all boosting algorithms in
SimpleMLR work consistently. It provides:
- Common functionality shared by all algorithms (XGBoost, LightGBM, sklearn GBM)
- Data preprocessing that works the same way for every algorithm
- Plotting capabilities that generate consistent visualizations
- Progress tracking for hyperparameter optimization

Think of this as the "engine" that powers all the different algorithms while keeping
them simple to use. Most users won't need to work with these classes directly -
they're used internally by XGBRegressor, LGBMRegressor, etc.
"""

import pandas as pd
import numpy as np
import warnings
import time
from typing import Dict, Any, Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from collections import deque
from abc import ABC, abstractmethod

# Core ML libraries
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

# Import utilities
try:
    from .utils import DataValidator, ModelEvaluator
except ImportError:
    # Fallback for direct script execution
    from utils import DataValidator, ModelEvaluator

# Characters used to make progress displays look nice in the terminal
DISPLAY_CHARS = {
    'progress_active': '->',
    'progress_complete': '=>', 
    'start_msg': '>>',
    'current_result': '->',
    'best_result': '**',
    'timing': '--',
    'improvement': '++',
    'stage_start': '::',
    'stage_complete': '==',
    'final_summary': '>>'
}


class PreprocessingMixin:
    """
    Handles data cleaning and preparation automatically.
    
    This class takes care of common data preprocessing tasks:
    - Converts text/categorical data to numbers that algorithms can use
    - Scales numerical features so they're on similar ranges
    - Validates that your data doesn't have missing values or other issues
    
    All SimpleMLR algorithms use this automatically, so you don't need to
    prepare your data manually.
    """
    
    def _validate_and_preprocess_inputs(self, X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
        """
        Validate inputs and preprocess features for training.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Tuple of (original_X, original_y, processed_X, preprocessing_artifacts)
        """
        # Validate inputs
        X_validated, y_validated = DataValidator.validate_inputs(X, y)
        
        # Store original data for auto-graphing
        X_original = X_validated.copy()
        y_original = y_validated.copy()
        
        # Preprocess features
        X_processed, preprocessing_artifacts = DataValidator.prepare_features(
            X_validated, 
            scale_features=getattr(self, 'scale_features', True),
            handle_categorical=getattr(self, 'handle_categorical', True)
        )
        
        return X_original, y_original, X_processed, preprocessing_artifacts
    
    def _preprocess_test_data(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply same preprocessing as training data to test data."""
        if not hasattr(self, 'preprocessing_artifacts_') or self.preprocessing_artifacts_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        X_validated, _ = DataValidator.validate_inputs(X, pd.Series([0] * len(X)))
        X_processed = X_validated.copy()
        
        # Handle categorical features
        if 'label_encoders' in self.preprocessing_artifacts_:
            for col, encoder in self.preprocessing_artifacts_['label_encoders'].items():
                if col in X_processed.columns:
                    # Handle unseen categories
                    X_processed[col] = X_processed[col].astype(str)
                    mask = X_processed[col].isin(encoder.classes_)
                    X_processed.loc[~mask, col] = encoder.classes_[0]
                    X_processed[col] = encoder.transform(X_processed[col])
        
        # Scale numerical features
        if 'scaler' in self.preprocessing_artifacts_:
            numeric_cols = self.preprocessing_artifacts_['numeric_features']
            available_cols = [col for col in numeric_cols if col in X_processed.columns]
            if available_cols:
                X_processed[available_cols] = self.preprocessing_artifacts_['scaler'].transform(
                    X_processed[available_cols]
                )
        
        return X_processed


class PlottingMixin:
    """
    Creates beautiful plots to help you understand your model's performance.
    
    This class provides two main plotting functions:
    - quick_graph(): Simple two-panel plot showing prediction accuracy and residuals
    - plot_analysis(): Comprehensive 6-panel analysis with detailed metrics
    
    Both work the same way across all algorithms, so once you learn one,
    you can use them all.
    """
    
    def quick_graph(self, X: Union[pd.DataFrame, np.ndarray] = None,
                    y: Union[pd.Series, np.ndarray] = None) -> None:
        """
        Simple plotting method with automatic training data detection.
        Returns None to prevent Jupyter auto-display duplication.

        Args:
            X: Features to evaluate on (optional - uses training data if not provided)
            y: True target values (optional - uses training data if not provided)

        Returns:
            None (displays plot directly)
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Clear any existing figures to prevent state issues
        plt.close('all')

        # Auto-detect: Use training data if X,y not provided
        if X is None and y is None:
            if not hasattr(self, 'training_X_') or not hasattr(self, 'training_y_'):
                raise ValueError("No training data stored and no X,y provided.")
            if self.training_X_ is None or self.training_y_ is None:
                raise ValueError("No training data stored and no X,y provided.")
            X_to_use = self.training_X_
            y_to_use = self.training_y_
        elif X is None or y is None:
            raise ValueError("Both X and y must be provided, or both can be omitted.")
        else:
            X_to_use = X
            y_to_use = y

        # Get predictions
        y_pred = self.predict(X_to_use)
        _, y_validated = DataValidator.validate_inputs(X_to_use, y_to_use)
        y_true = y_validated.values

        # Create a new figure explicitly
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)

        # Residuals
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        # Add metrics to title
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        algorithm_name = self._get_algorithm_display_name()
        fig.suptitle(f"{algorithm_name} Performance (R2 = {r2:.3f}, RMSE = {rmse:.3f})",
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Display once, close immediately, return nothing
        plt.show()
        plt.close(fig)
        return None

    def graph(self, X: Union[pd.DataFrame, np.ndarray] = None,
              y: Union[pd.Series, np.ndarray] = None) -> None:
        """Backward compatibility alias for quick_graph()."""
        return self.quick_graph(X, y)

    def plot_analysis(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      title: Optional[str] = None,
                      save_path: Optional[str] = None,
                      style: str = 'modern',
                      **kwargs) -> Tuple[None, Dict[str, float]]:
        """
        Create comprehensive model analysis plot.
        Returns None to prevent Jupyter auto-display duplication.

        Args:
            X: Features to evaluate on
            y: True target values
            title: Plot title (auto-generated if None)
            save_path: File path to save plot
            style: Plot style ('modern', 'classic', 'minimal')
            **kwargs: Additional arguments for plot_model_analysis

        Returns:
            Tuple of (None, detailed_metrics) - plot is displayed directly
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        plt.close('all')

        X, y = DataValidator.validate_inputs(X, y)
        X_processed = self._preprocess_test_data(X)

        # Auto-generate title if not provided
        if title is None:
            algorithm_name = self._get_algorithm_display_name()
            title = f"{algorithm_name} Model Analysis"

        # Call ModelEvaluator but ensure no display conflicts
        fig, metrics = ModelEvaluator.plot_model_analysis(
            model=self.model_,
            X=X_processed,
            y=y,
            title=title,
            save_path=save_path,
            style=style,
            show=False,  # Prevent ModelEvaluator from showing
            **kwargs
        )

        # Single display call, then immediate cleanup
        plt.show()
        plt.close(fig)

        return None, metrics
    
    @abstractmethod
    def _get_algorithm_display_name(self) -> str:
        """Return the display name for this algorithm (e.g., 'XGBoost', 'LightGBM')."""
        pass


class MetricsMixin:
    """
    Calculates performance metrics to help you evaluate your model.
    
    Computes standard regression metrics like R², RMSE, MAE automatically
    and stores them so you can access them later. Also handles feature
    importance calculations to show which features matter most.
    """
    
    def _calculate_and_store_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate and store comprehensive training metrics."""
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        self.training_metrics_ = metrics
        return metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N most important features."""
        if not hasattr(self, 'feature_importance_') or self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.feature_importance_.head(top_n)


class DeviceMixin:
    """
    Handles GPU vs CPU training automatically.
    
    This class manages whether to use your computer's GPU (graphics card)
    for faster training or stick with CPU. Each algorithm has its own
    way of using GPU, so this provides a consistent interface.
    """
    
    @abstractmethod
    def _get_device_params(self) -> Dict[str, Any]:
        """Get device-specific parameters for the algorithm."""
        pass
    
    @abstractmethod
    def _test_gpu_availability(self) -> bool:
        """Test if GPU is available for this algorithm."""
        pass


class ProgressTrackingMixin:
    """
    Shows progress during hyperparameter optimization.
    
    When you use auto-tuning (like XGBAutoTuner), this class shows:
    - How many trials have completed
    - Current best performance
    - Estimated time remaining
    - Detailed progress information if you want it
    
    You can control how much information you see with the verbose parameter.
    """

    def _initialize_progress_tracking(self):
        """Initialize all progress tracking state variables."""
        # Enhanced progress tracking attributes (from core_backup.py lines 475-478)
        self._trial_times = deque(maxlen=10)  # For accurate ETA calculation
        self._last_progress_line = ""  # Track progress line for clearing
        self._progress_shown = False  # Track if progress is displayed
        
        # Progress state (reset in optimize())
        self._trial_count = 0
        self._start_time = time.time()
        self._best_rmse = float('inf')
        self._overall_best_rmse = float('inf')
        self._previous_best_rmse = float('inf')
        
        
        # Validation split tracking
        self._validation_split_info = None
        
        # Multi-fidelity tracking
        self._fidelity_scores = {}
        self._fidelity_thresholds = {}
        self._current_fidelity_stage = 0
        
        # Parameter tracking for verbose display
        self._best_params = {}
        self._current_trial_params = {}

    def _reset_progress_state(self):
        """EXACT reset logic from core_backup.py line 907-920."""
        self._start_time = time.time()
        self._trial_count = 0
        self._best_rmse = float('inf')
        self._overall_best_rmse = float('inf')
        self._previous_best_rmse = float('inf')
        self._trial_times.clear()  # Clear previous trial times
        self._last_progress_line = ""
        self._progress_shown = False
        
        # Initialize multi-fidelity tracking
        self._fidelity_scores.clear()
        self._fidelity_thresholds.clear()
        self._current_fidelity_stage = 0
        
        # Reset parameter tracking
        self._best_params = {}
        self._current_trial_params = {}

    def _create_progress_bar(self, current: int, total: int) -> str:
        """Create progress display with percentage and done/total count (no visual bar)."""
        if total == 0:
            return f"{DISPLAY_CHARS['progress_active']} 0/0 0.0%"

        progress = current / total
        percentage = progress * 100

        # Choose character based on completion
        char = DISPLAY_CHARS['progress_complete'] if current >= total else DISPLAY_CHARS['progress_active']

        # DEBUG: Print actual values for troubleshooting
        # print(f"\nDEBUG: current={current}, total={total}, percentage={percentage:.1f}%", file=sys.stderr)

        # Return just percentage and done/total count
        return f"{char} {current}/{total} {percentage:.1f}%"

    def _calculate_eta(self, current: int, total: int, elapsed: float) -> str:
        """Simple and reliable ETA calculation (proven approach from older version)."""
        if current == 0:
            return "ETA: --"
        
        # If completed, show completion status
        if current >= total:
            return "ETA: Complete"

        # Simple overall average - this is what actually worked well!
        avg_time_per_trial = elapsed / current
        remaining_trials = total - current
        eta_seconds = avg_time_per_trial * remaining_trials

        # Clean formatting
        if eta_seconds < 60:
            return f"ETA: {eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"ETA: {eta_seconds / 60:.1f}m"
        else:
            return f"ETA: {eta_seconds / 3600:.1f}h"

    def _format_params_dict(self, params_dict: dict) -> str:
        """Format parameters dictionary for clean display with each parameter on new line."""
        import json
        # Use json.dumps with indentation for multi-line format
        return json.dumps(params_dict, indent=4)

    def _format_rmse_with_color(self, rmse: float, is_new_best: bool = False) -> str:
        """Color-code RMSE GREEN only when it's a new best."""
        if is_new_best:
            return f"\033[32m{rmse:.4f}\033[0m"  # Green for new best
        else:
            return f"{rmse:.4f}"  # Colorless for normal

    def _clear_progress_line(self):
        """FIXED: Proper clearing without artifacts and color reset."""
        if self._progress_shown:
            # Clear the entire line and reset all formatting, then move to new line
            line_length = max(80, len(self._last_progress_line) + 20)  # Ensure adequate clearing
            print(f"\r\033[0m{' ' * line_length}\r\033[0m", end="", flush=True)
            self._last_progress_line = ""
            self._progress_shown = False

    def _show_progress_line(self, progress_text: str):
        """FIXED: Show progress line with proper color handling."""
        # Clear any existing line completely first
        if self._last_progress_line:
            clear_length = max(len(self._last_progress_line), len(progress_text)) + 10
            print(f"\r{' ' * clear_length}\r", end="", flush=True)
        
        # Show the progress text and ensure clean color reset at end
        print(f"\r{progress_text}", end="", flush=True)

        self._last_progress_line = progress_text
        self._progress_shown = True

    def _update_standard_progress(self, trial_start_time: float, rmse: float, params: dict = None):
        """Update progress display for standard optimization - EXACT from core_backup.py."""
        # Record trial time for accurate ETA calculation
        trial_time = time.time() - trial_start_time
        self._trial_times.append(trial_time)
        
        # Store current trial parameters
        if params is not None:
            self._current_trial_params = params.copy()

        # FIXED: Update best RMSE tracking
        is_new_best = False
        if self._trial_count == 1:
            self._best_rmse = rmse
            self._overall_best_rmse = rmse
            if params is not None:
                self._best_params = params.copy()
        elif rmse < self._best_rmse:
            self._previous_best_rmse = self._best_rmse
            self._best_rmse = rmse
            self._overall_best_rmse = rmse
            is_new_best = True
            if params is not None:
                self._best_params = params.copy()

        # Progress display with simple approach (like the older working version)
        if self.verbose == 1:
            elapsed = time.time() - self._start_time
            progress_bar = self._create_progress_bar(self._trial_count, self.n_trials)
            eta = self._calculate_eta(self._trial_count, self.n_trials, elapsed)
            rmse_display = self._format_rmse_with_color(self._best_rmse, is_new_best)

            progress_text = (f"{progress_bar} | "
                             f"Best: {rmse_display} | {elapsed:.1f}s | {eta}")
            self._show_progress_line(progress_text)

        elif self.verbose == 2:
            # Clear progress line for detailed output
            if self._progress_shown:
                self._clear_progress_line()

            elapsed = time.time() - self._start_time
            progress_bar = self._create_progress_bar(self._trial_count, self.n_trials)
            eta = self._calculate_eta(self._trial_count, self.n_trials, elapsed)

            # Current trial info
            current_rmse_display = f"{rmse:.4f}"
            
            # Best RMSE display (color only if new best)
            best_rmse_display = self._format_rmse_with_color(self._best_rmse, is_new_best)

            print(f"{DISPLAY_CHARS['current_result']} Current RMSE: {current_rmse_display}")
            print(f"{DISPLAY_CHARS['best_result']} Best RMSE:    {best_rmse_display}")
            print(f"{DISPLAY_CHARS['timing']} Elapsed:     {elapsed:.1f}s")

            # Show improvement if it's a new best
            if is_new_best and self._trial_count > 1:
                improvement = self._previous_best_rmse - self._best_rmse
                print(f"{DISPLAY_CHARS['improvement']} New best! Improved by {improvement:.6f}")
            
            # Show current trial parameters
            if self._current_trial_params:
                print(f"Current trial params: {self._current_trial_params}")
            
            # Show best parameters 
            if self._best_params:
                print(f"Best parameters: {self._best_params}")

            print(f"{'=' * 65}")


    def _update_multi_fidelity_progress(self, trial_start_time: float, rmse: float, 
                                      completed_stages: int, total_stages: int, 
                                      early_stopped: bool = False):
        """Update progress display for multi-fidelity evaluation - EXACT from core_backup.py."""
        # Record trial time for accurate ETA calculation
        trial_time = time.time() - trial_start_time
        self._trial_times.append(trial_time)
        
        # Update best RMSE tracking
        is_new_best = False
        if self._trial_count == 1:
            self._best_rmse = rmse
            self._overall_best_rmse = rmse
        elif rmse < self._best_rmse:
            self._previous_best_rmse = self._best_rmse
            self._best_rmse = rmse
            self._overall_best_rmse = rmse
            is_new_best = True
        
        # Progress display
        if self.verbose == 1:
            elapsed = time.time() - self._start_time
            progress_bar = self._create_progress_bar(self._trial_count, self.n_trials)
            eta = self._calculate_eta(self._trial_count, self.n_trials, elapsed)
            rmse_display = self._format_rmse_with_color(self._best_rmse, is_new_best)
            
            # Show fidelity stage info
            if early_stopped:
                stage_info = f"Fidelity: {completed_stages}/{total_stages} (early stop)"
            else:
                stage_info = f"Fidelity: {completed_stages}/{total_stages}"
            
            progress_text = (f"{progress_bar} | "
                           f"Best: {rmse_display} | {elapsed:.1f}s | {eta} | {stage_info}")
            self._show_progress_line(progress_text)
            
        elif self.verbose == 2:
            # Clear progress line for detailed output
            if self._progress_shown:
                self._clear_progress_line()
            
            elapsed = time.time() - self._start_time
            progress_bar = self._create_progress_bar(self._trial_count, self.n_trials)
            eta = self._calculate_eta(self._trial_count, self.n_trials, elapsed)
            
            # Current trial info
            current_rmse_display = f"{rmse:.4f}"
            
            # Best RMSE display (color only if new best)  
            best_rmse_display = self._format_rmse_with_color(self._best_rmse, is_new_best)
            
            print(f"{DISPLAY_CHARS['current_result']} Current RMSE: {current_rmse_display}")
            print(f"{DISPLAY_CHARS['best_result']} Best RMSE:    {best_rmse_display}")
            print(f"{DISPLAY_CHARS['timing']} Elapsed:     {elapsed:.1f}s")
            
            # Show improvement if it's a new best
            if is_new_best and self._trial_count > 1:
                improvement = self._previous_best_rmse - self._best_rmse
                print(f"{DISPLAY_CHARS['improvement']} New best! Improved by {improvement:.6f}")
            
            print(f"{'=' * 65}")


    def _show_optimization_header(self):
        """Show optimization start message based on verbose level - EXACT from core_backup.py."""
        if self.verbose == -1:
            pass
        elif self.verbose == 0:
            if hasattr(self, 'validation_split') and self.validation_split is not None:
                mode_msg = f"validation split ({self.validation_split:.1%})"
            else:
                mode_msg = "traditional"
            early_msg = f" with early stopping ({self.early_stopping_rounds} rounds)" if hasattr(self, 'early_stopping_rounds') and self.early_stopping_rounds else ""
            multi_msg = f" + multi-fidelity" if hasattr(self, 'multi_fidelity') and self.multi_fidelity else ""
            print(f"Hyperparameter optimization started using '{self.strategy}' strategy ({mode_msg}{early_msg}{multi_msg})...")
        elif self.verbose >= 1:
            # Import strategy info dynamically to work with all algorithms
            try:
                if hasattr(self, '_get_algorithm_name'):
                    alg_name = self._get_algorithm_name()
                else:
                    alg_name = self.__class__.__name__.replace('AutoTuner', '')
                
                # Try to get strategy info, fall back to algorithm name if not available
                try:
                    from .base_strategies import get_strategy_info
                    strategy_info = get_strategy_info(self.strategy)
                    strategy_desc = strategy_info.get('description', f'{self.strategy} strategy')
                except (ImportError, KeyError):
                    strategy_desc = f'{self.strategy} strategy'
                
                print(f"{DISPLAY_CHARS['start_msg']} {alg_name} Hyperparameter Optimization")
                print(f"Strategy: {strategy_desc}")
                print(f"{DISPLAY_CHARS['current_result']} Trials: {self.n_trials}")
                
                if hasattr(self, 'validation_split') and self.validation_split is not None:
                    print(f"Validation: {self.validation_split:.1%} split")
                else:
                    print(f"{DISPLAY_CHARS['stage_start']} Mode: Traditional optimization")
                    
                if self.use_gpu:
                    print(f"Device: GPU acceleration enabled")
                else:
                    print(f"Device: CPU ({self.n_jobs} cores)")
                    
                print(f"{'=' * 65}")
            except ImportError:
                # Fallback if strategy info not available
                print(f"{DISPLAY_CHARS['start_msg']} Hyperparameter Optimization Starting...")
                print(f"{DISPLAY_CHARS['current_result']} Strategy: {self.strategy} | Trials: {self.n_trials}")
                print(f"{'=' * 65}")

    def _show_final_summary(self, best_params: Dict[str, Any], best_rmse: float, total_time: float):
        """Show final optimization summary based on verbose level - EXACT from core_backup.py."""
        if self.verbose == -1:
            pass
        elif self.verbose == 0:
            print(f"{DISPLAY_CHARS['progress_complete']} Optimization finished. Model is ready.")
        elif self.verbose == 1:
            print(f"\n{DISPLAY_CHARS['final_summary']} Model ready for use!")
            print(f"{DISPLAY_CHARS['current_result']} CV RMSE: {self._format_rmse_with_color(best_rmse, False)}")
            print(f"{DISPLAY_CHARS['timing']} Total Time: {total_time:.1f}s")
            
            # Show training metrics if available
            if hasattr(self, 'best_model_') and hasattr(self.best_model_, 'training_metrics_'):
                metrics = self.best_model_.training_metrics_
                print(f"Training R2: {metrics['r2_score']:.6f}")
                print(f"Training RMSE: {metrics['rmse']:.6f}")

            # Show best parameters as Python dictionary
            display_params = self._get_display_params(best_params)
            print(f"Best parameters: {display_params}")

        elif self.verbose == 2:
            # For verbose=2, show training metrics and final best parameters
            if hasattr(self, 'best_model_') and hasattr(self.best_model_, 'training_metrics_'):
                metrics = self.best_model_.training_metrics_
                print(f"\nTraining R2: {metrics['r2_score']:.6f}")
                print(f"Training RMSE: {metrics['rmse']:.6f}")
            
            # Show final best parameters as Python dictionary
            display_params = self._get_display_params(best_params)
            print(f"Final best parameters: {display_params}")
            print(f"{DISPLAY_CHARS['final_summary']} Model ready for predictions!")

    def _get_display_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for display, including fixed overrides."""
        display_params = params.copy()
        
        # Add fixed parameters from overrides if available
        if hasattr(self, '_fixed_params') and self._fixed_params:
            display_params.update(self._fixed_params)
            
        return display_params

    def _get_algorithm_name(self) -> str:
        """Get algorithm name for display purposes."""
        class_name = self.__class__.__name__
        if 'XGB' in class_name:
            return 'XGBoost'
        elif 'LGBM' in class_name or 'LGB' in class_name:
            return 'LightGBM'
        elif 'GBM' in class_name:
            return 'Gradient Boosting'
        else:
            return class_name.replace('AutoTuner', '')


class ParameterValidationMixin:
    """
    Validates and manages hyperparameters safely.
    
    This class ensures that:
    - Parameter values are in valid ranges
    - Override parameters are formatted correctly
    - Strategy configurations are properly merged with user preferences
    
    This prevents common errors and makes the optimization process more reliable.
    """
    
    def _validate_override_params(self, override_params: Optional[Dict[str, Union[Tuple[float, float], float, int, List[str], str]]]) -> None:
        """Validate override parameters format - Enhanced for categorical support."""
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
    
    def _merge_strategy_with_overrides(self, strategy: str, override_params: Optional[Dict[str, Union[Tuple[float, float], float, int, List[str], str]]]) -> Dict[str, Any]:
        """Merge strategy ranges with user overrides - Enhanced for categorical support."""
        # Get base strategy ranges from algorithm-specific implementation
        base_ranges = self._get_strategy_ranges(strategy)
        
        # Get base categorical parameters if available
        base_categorical = {}
        if hasattr(self, '_get_strategy_categorical_params'):
            try:
                base_categorical = self._get_strategy_categorical_params(strategy)
            except (AttributeError, NotImplementedError):
                base_categorical = {}
        
        # Initialize merged configuration with new categorical_params section
        merged_config = {
            'ranges': {},
            'fixed_params': {},
            'categorical_params': {}
        }
        
        if override_params is None:
            # No overrides - use only numerical strategy ranges (no categorical params by default)
            merged_config['ranges'] = base_ranges
            # categorical_params remains empty - only add when explicitly requested
            return merged_config
        
        # Validate override parameters
        self._validate_override_params(override_params)
        
        # Process numerical parameters from base strategy
        for param_name, param_value in base_ranges.items():
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
                merged_config['ranges'][param_name] = param_value
        
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
    
    @abstractmethod
    def _get_strategy_ranges(self, strategy: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for the specified strategy."""
        pass
    
    def _get_strategy_categorical_params(self, strategy: str) -> Dict[str, List[str]]:
        """Get categorical parameter choices for the specified strategy."""
        # Default implementation returns empty dict for backward compatibility
        return {}
    
    @abstractmethod
    def _suggest_parameter(self, trial, param_name: str, param_info):
        """Enhanced parameter suggestion with categorical support."""
        pass


class BaseBoostingRegressor(ABC, PreprocessingMixin, PlottingMixin, MetricsMixin, DeviceMixin):
    """
    The blueprint for all SimpleMLR regression models.
    
    This is the foundation that XGBRegressor, LGBMRegressor, and GBMRegressor
    all build upon. It provides the common interface like fit(), predict(),
    and quick_graph() while letting each algorithm implement its specific
    machine learning logic.
    
    Users typically don't create this class directly - use XGBRegressor,
    LGBMRegressor, or GBMRegressor instead.
    """
    
    def __init__(self,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 **kwargs):
        """
        Initialize base boosting regressor.
        
        Args:
            random_state: Random seed for reproducibility
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to handle categorical features
            verbose: Whether to print progress information
            n_jobs: Number of CPU cores to use (-1=all cores)
            use_gpu: Whether to use GPU acceleration
            **kwargs: Algorithm-specific parameters
        """
        self.random_state = random_state
        self.scale_features = scale_features
        self.handle_categorical = handle_categorical
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.extra_params = kwargs
        
        # Model state
        self.model_ = None
        self.preprocessing_artifacts_ = None
        self.feature_importance_ = None
        self.training_metrics_ = None
        
        # Store training data for auto-graphing
        self.training_X_ = None
        self.training_y_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'BaseBoostingRegressor':
        """
        Fit the boosting model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        if self.verbose:
            algorithm_name = self._get_algorithm_display_name()
            print(f"Fitting {algorithm_name} Regressor...")
        
        # Validate and preprocess inputs
        X_original, y_original, X_processed, preprocessing_artifacts = self._validate_and_preprocess_inputs(X, y)
        
        # Store for auto-graphing
        self.training_X_ = X_original
        self.training_y_ = y_original
        self.preprocessing_artifacts_ = preprocessing_artifacts
        
        # Get algorithm-specific parameters
        model_params = self._get_model_params()
        device_params = self._get_device_params()
        model_params.update(device_params)
        model_params.update(self.extra_params)
        
        # Create and fit model
        self.model_ = self._create_model(**model_params)
        self.model_.fit(X_processed, y_original)
        
        # Calculate training metrics
        y_pred = self.model_.predict(X_processed)
        self._calculate_and_store_metrics(y_original.values, y_pred)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_processed)
        
        if self.verbose:
            fit_time = time.time() - start_time
            print(f"Training completed in {fit_time:.2f} seconds")
            print(f"Training R2: {self.training_metrics_['r2_score']:.4f}")
            print(f"Training RMSE: {self.training_metrics_['rmse']:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_processed = self._preprocess_test_data(X)
        return self.model_.predict(X_processed)
    
    def _calculate_feature_importance(self, X_processed: pd.DataFrame):
        """Calculate and store feature importance."""
        importance_values = self._get_feature_importance_values()
        if importance_values is not None:
            self.feature_importance_ = pd.DataFrame({
                'feature': X_processed.columns,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance_ = None
    
    @abstractmethod
    def _create_model(self, **params):
        """Create the underlying algorithm-specific model instance."""
        pass
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Get algorithm-specific model parameters."""
        pass
    
    @abstractmethod
    def _get_feature_importance_values(self) -> Optional[np.ndarray]:
        """Get feature importance values from the fitted model."""
        pass


class BaseAutoTuner(ABC, ProgressTrackingMixin, ParameterValidationMixin, DeviceMixin):
    """
    The blueprint for all SimpleMLR automatic hyperparameter tuners.
    
    This is the foundation that XGBAutoTuner, LGBMAutoTuner, and GBMAutoTuner
    build upon. It handles the optimization process (trying different settings,
    tracking progress, finding the best combination) while letting each algorithm
    define its specific parameter ranges and constraints.
    
    Users typically don't create this class directly - use XGBAutoTuner,
    LGBMAutoTuner, or GBMAutoTuner instead.
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
                 override_params: Optional[Dict[str, Union[Tuple[float, float], float, int, List[str], str]]] = None,
                 multi_fidelity: bool = False,
                 fidelity_stages: Optional[List] = None,
                 fidelity_percentile: float = 0.7,
                 **kwargs):
        """
        Initialize auto-tuner with enhanced features.
        
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
            validation_split: Use hold-out validation instead of cross-validation
            early_stopping_rounds: Enable early stopping with specified patience rounds
            override_params: Dictionary of parameter overrides
            multi_fidelity: Enable multi-fidelity early stopping evaluation
            fidelity_stages: List of budget stages for multi-fidelity
            fidelity_percentile: Percentile threshold for advancing to next fidelity stage
            **kwargs: Algorithm-specific parameters
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scale_features = scale_features
        self.handle_categorical = handle_categorical
        self.verbose = verbose
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.validation_split = validation_split
        self.early_stopping_rounds = early_stopping_rounds
        self.override_params = override_params
        self.multi_fidelity = multi_fidelity
        self.fidelity_stages = fidelity_stages or self._get_default_fidelity_stages()
        self.fidelity_percentile = fidelity_percentile
        self.extra_params = kwargs
        
        # Validate parameters
        self._validate_initialization_params()
        
        # Merge strategy ranges with overrides - Enhanced for categorical support
        self._merged_config = self._merge_strategy_with_overrides(self.strategy, self.override_params)
        self._merged_ranges = self._merged_config['ranges']
        self._fixed_params = self._merged_config['fixed_params']
        self._categorical_params = self._merged_config.get('categorical_params', {})
        
        # Initialize state
        self.study_ = None
        self.best_model_ = None
        self.preprocessing_artifacts_ = None
        
        # Initialize progress tracking using the proven mixin logic
        self._initialize_progress_tracking()
    
    def _validate_initialization_params(self):
        """Validate initialization parameters."""
        if self.override_params is not None:
            self._validate_override_params(self.override_params)
            
        if self.validation_split is not None:
            if not (0.0 < self.validation_split < 1.0):
                raise ValueError("validation_split must be between 0.0 and 1.0")
                
        if self.early_stopping_rounds is not None:
            if self.early_stopping_rounds <= 0:
                raise ValueError("early_stopping_rounds must be positive")
        
        # Validate strategy
        available_strategies = self._get_available_strategies()
        if self.strategy not in available_strategies:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Available strategies: {available_strategies}")
        
    
    @abstractmethod
    def _create_model_instance(self, **params):
        """Create model instance for optimization."""
        pass
    
    @abstractmethod
    def _get_algorithm_name(self) -> str:
        """Return algorithm name for display."""
        pass
    
    @abstractmethod
    def _get_available_strategies(self) -> List[str]:
        """Get list of available strategies for this algorithm."""
        pass
    
    
    @abstractmethod
    def _get_default_fidelity_stages(self) -> List:
        """Get default fidelity stages for multi-fidelity optimization."""
        pass
    
    @abstractmethod
    def _check_algorithm_constraints(self, params: dict, n_estimators: int = None,
                                   optimization_context: dict = None) -> bool:
        """
        Check algorithm-specific parameter constraints.
        
        Args:
            params: Parameter dictionary to validate
            n_estimators: Specific n_estimators value (for multi-fidelity)
            optimization_context: Dict with keys like 'mode', 'stage_info', 'algorithm'
        
        Returns:
            bool: True if constraints are satisfied for this algorithm
        """
        pass
    
    def _get_optimization_context(self, stage_info: dict = None) -> dict:
        """Create optimization context for constraint checking."""
        mode = 'traditional'
        if hasattr(self, 'multi_fidelity') and self.multi_fidelity:
            mode = 'multi_fidelity'
        
        context = {
            'mode': mode,
            'algorithm': self._get_algorithm_name(),
            'fidelity_stages': getattr(self, 'fidelity_stages', None),
            'current_trial': getattr(self, '_trial_count', 0)
        }
        
        if stage_info:
            context.update(stage_info)
            
        return context
    
    # The optimize() method and all optimization logic would be implemented here
    # using the same patterns as the current XGBAutoTuner but calling abstract methods
    # for algorithm-specific functionality. This preserves all existing functionality
    # while making it algorithm-agnostic.
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]):
        """
        Algorithm-agnostic hyperparameter optimization using mixins.
        This uses the abstract methods to work with any boosting algorithm.
        """
        import optuna
        import warnings
        import logging
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import mean_squared_error
        import numpy as np
        
        # Initialize progress tracking (from ProgressTrackingMixin)
        self._initialize_progress_tracking()
        
        # Validate and preprocess inputs
        X_validated, y_validated = DataValidator.validate_inputs(X, y)
        X_processed, self.preprocessing_artifacts_ = DataValidator.prepare_features(
            X_validated,
            scale_features=self.scale_features,
            handle_categorical=self.handle_categorical
        )
        
        # Merge strategy configuration with overrides - Enhanced for categorical support
        merged_config = self._merge_strategy_with_overrides(self.strategy, self.override_params)
        self._merged_ranges = merged_config['ranges']
        self._fixed_params = merged_config['fixed_params']
        self._categorical_params = merged_config.get('categorical_params', {})
        
        # Show optimization header (from ProgressTrackingMixin)
        self._show_optimization_header()
        
        # Set up Optuna study
        if self.verbose <= 0:
            logging.getLogger('optuna').setLevel(logging.ERROR)
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        self.study_ = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Test GPU availability if requested
        if self.use_gpu:
            gpu_available = self._test_gpu_availability()
            if not gpu_available:
                if self.verbose >= 0:
                    print("WARNING: GPU requested but not available, switching to optimized CPU mode")
                self.use_gpu = False
        
        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.study_.optimize(
                lambda trial: self._objective(trial, X_processed, y_validated),
                n_trials=self.n_trials,
                callbacks=None,
                show_progress_bar=False
            )
        
        # Clear progress line and add newline for clean final display
        self._clear_progress_line()
        if self._trial_count > 0:  # Only add newline if we actually showed progress
            print()  # Move to next line for clean final summary
        
        # Create final model
        best_params = self.study_.best_params
        final_model = self._create_final_model(X_validated, y_validated, X_processed, best_params)
        
        # Show final summary (from ProgressTrackingMixin)
        total_time = time.time() - self._start_time
        self._show_final_summary(best_params, self._overall_best_rmse, total_time)
        
        return final_model
    
    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Algorithm-agnostic objective function using abstract methods - Enhanced for categorical support."""
        # Record trial start time for accurate ETA calculation
        trial_start_time = time.time()
        
        # Increment trial counter
        self._trial_count += 1
        
        # Build parameters using merged ranges, categorical params, and fixed params
        params = {}
        params.update(self._fixed_params)
        
        # Add optimizable numerical parameters
        for param_name, param_range in self._merged_ranges.items():
            params[param_name] = self._suggest_parameter(trial, param_name, param_range)
        
        # Add optimizable categorical parameters
        for param_name, choices in self._categorical_params.items():
            params[param_name] = trial.suggest_categorical(param_name, choices)
        
        # Add device-specific parameters
        device_params = self._get_device_params()
        params.update(device_params)
        
        # Create model using algorithm-specific method
        model = self._create_model_instance(**params)
        
        # Evaluate model
        if hasattr(self, 'validation_split') and self.validation_split is not None:
            rmse = self._evaluate_with_validation_split(model, X, y)
        else:
            rmse = self._evaluate_with_cv(model, X, y)
        
        # Update progress display
        self._update_standard_progress(trial_start_time, rmse, params)
        
        return rmse
    
    def _evaluate_with_cv(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model using cross-validation."""
        try:
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
    
    def _evaluate_with_validation_split(self, model, X: pd.DataFrame, y: pd.Series) -> float:
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
    
    def _create_final_model(self, X: pd.DataFrame, y: pd.Series, 
                          X_processed: pd.DataFrame, best_params: dict):
        """Create the final optimized model using algorithm-specific methods."""
        # Create best model using algorithm-specific implementation
        final_model = self._create_model(**best_params)
        
        # Store training data for auto-graphing
        final_model.training_X_ = X.copy()
        final_model.training_y_ = y.copy()
        
        # Set preprocessing artifacts to avoid double preprocessing
        final_model.preprocessing_artifacts_ = self.preprocessing_artifacts_
        
        # Fit final model on full dataset
        model_instance = self._create_model_instance(**best_params)
        model_instance.fit(X_processed, y)
        final_model.model_ = model_instance
        
        # Calculate metrics and feature importance
        y_pred = model_instance.predict(X_processed)
        final_model.training_metrics_ = ModelEvaluator.calculate_metrics(y.values, y_pred)
        
        # Get feature importance using algorithm-specific method
        if hasattr(model_instance, 'feature_importances_') and model_instance.feature_importances_ is not None:
            final_model.feature_importance_ = pd.DataFrame({
                'feature': X_processed.columns,
                'importance': model_instance.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            final_model.feature_importance_ = None
        
        return final_model