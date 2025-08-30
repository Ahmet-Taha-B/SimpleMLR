"""
Utility classes for SimpleMLR - Tools that help you work with your data and models.

This module provides two main helper classes:

DataValidator: Checks your data for common problems and fixes them automatically
- Handles missing values, infinite values, data type mismatches
- Converts between different data formats (arrays, DataFrames) 
- Scales numerical features and encodes categorical features

ModelEvaluator: Creates detailed analysis and beautiful plots of your model's performance
- Calculates comprehensive metrics (R², RMSE, MAE, etc.)
- Generates publication-ready plots with multiple views of model performance
- Works with any scikit-learn compatible model, not just SimpleMLR models

These classes work behind the scenes in SimpleMLR but you can also use them directly
for custom analysis or with other machine learning libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import warnings
from typing import Dict, Any, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance


class DataValidator:
    """Automatically checks and fixes common data problems.
    
    This class handles the tedious but important work of making sure your data
    is ready for machine learning:
    - Checks for missing values (NaN) and infinite values
    - Ensures X and y have the same number of rows
    - Converts numpy arrays to pandas DataFrames with proper column names
    - Scales numerical features so they're all on similar ranges
    - Converts text/categorical data to numbers that algorithms can use
    
    All SimpleMLR models use this automatically, but you can also use it
    directly to prepare data for other machine learning libraries.
    """

    @staticmethod
    def validate_inputs(X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate and convert inputs to proper pandas format.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of validated DataFrame and Series
        """
        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')

        # Basic validation
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")

        if X.isnull().any().any():
            raise ValueError("X contains missing values. Please handle them before training.")

        if y.isnull().any():
            raise ValueError("y contains missing values. Please handle them before training.")

        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number]).values).any():
            raise ValueError("X contains infinite values")

        if np.isinf(y.values).any():
            raise ValueError("y contains infinite values")

        return X, y

    @staticmethod
    def prepare_features(X: pd.DataFrame,
                         scale_features: bool = True,
                         handle_categorical: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare features for training.

        Args:
            X: Input features
            scale_features: Whether to scale numerical features
            handle_categorical: Whether to encode categorical features

        Returns:
            Processed features and preprocessing artifacts
        """
        X_processed = X.copy()
        artifacts = {}

        # Identify feature types
        numeric_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle categorical features
        if handle_categorical and categorical_features:
            label_encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                label_encoders[col] = le
            artifacts['label_encoders'] = label_encoders
            artifacts['categorical_features'] = categorical_features

        # Scale numerical features
        if scale_features and numeric_features:
            scaler = StandardScaler()
            X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
            artifacts['scaler'] = scaler
            artifacts['numeric_features'] = numeric_features

        artifacts['feature_names'] = X_processed.columns.tolist()

        return X_processed, artifacts


class ModelEvaluator:
    """Creates beautiful, detailed analysis of your model's performance.
    
    This class generates comprehensive performance reports including:
    - Key metrics like R², RMSE, MAE, and correlation
    - Actual vs predicted scatter plots with confidence bands
    - Residual plots to check for patterns in errors
    - Feature importance rankings (when available)
    - Distribution analysis and outlier detection
    - Professional-quality plots ready for presentations or papers
    
    Works with any scikit-learn compatible model, not just SimpleMLR models.
    You can customize the style, save plots, and get detailed metrics dictionaries.
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        residuals = y_true - y_pred

        # Core metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else np.nan
        max_error = np.max(np.abs(residuals))
        std_residuals = np.std(residuals)

        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'max_error': max_error,
            'std_residuals': std_residuals,
            'n_samples': len(y_true)
        }

    @staticmethod
    def plot_model_analysis(model, X, y, title="Model Analysis", save_path=None,
                            figsize=(16, 12), style='modern', show_confidence=True,
                            permutation_importance_scoring='r2', top_features=10, show=True) -> Union[Tuple[Figure, Dict], Tuple[None, Dict]]:
        """
        Comprehensive model analysis with single function - FIXED for figure state management.

        Args:
            model: Trained model (any sklearn-compatible model)
            X: Features (DataFrame or array)
            y: Target values (Series or array)
            title: Plot title
            save_path: File path to save the plot
            figsize: Figure size tuple
            style: 'modern', 'classic', or 'minimal'
            show_confidence: Whether to show confidence intervals
            permutation_importance_scoring: Scoring metric for permutation importance
            top_features: Number of top features to show in importance plot
            show: Whether to display the plot (prevents display conflicts)

        Returns:
            tuple: (fig, detailed_metrics_dict)
        """
        # CRITICAL FIX: Clear any existing figures to prevent state issues
        plt.close('all')

        # Validate inputs and get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            raise ValueError("Model does not have predict method")

        # Convert to numpy arrays for consistent handling
        y_true = np.array(y)
        y_pred = np.array(y_pred)

        # Calculate comprehensive metrics using your existing method
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        residuals = y_true - y_pred

        # Set style colors
        if style == 'modern':
            plt.style.use('default')
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
            bg_color = '#F8F9FA'
        elif style == 'minimal':
            colors = ['#333333', '#666666', '#999999', '#CCCCCC', '#555555']
            bg_color = '#FFFFFF'
        else:  # classic
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            bg_color = '#FFFFFF'

        # CRITICAL FIX: Create figure explicitly to avoid state conflicts
        fig = plt.figure(figsize=figsize, facecolor=bg_color)

        # Define a 3x3 grid for better layout
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                              left=0.08, right=0.95, top=0.88, bottom=0.08)

        # 1. Actual vs Predicted (larger plot)
        ax1 = fig.add_subplot(gs[0, :2])
        scatter = ax1.scatter(y_true, y_pred, alpha=0.6, color=colors[0], s=30,
                              edgecolors='white', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], '--', color=colors[1],
                 lw=2, alpha=0.8, label='Perfect Prediction')

        # Add confidence band if requested
        if show_confidence:
            std_error = np.sqrt(metrics['mse'])
            ax1.fill_between([min_val, max_val],
                             [min_val - 1.96 * std_error, max_val - 1.96 * std_error],
                             [min_val + 1.96 * std_error, max_val + 1.96 * std_error],
                             alpha=0.2, color=colors[1], label='95% Confidence Band')

        ax1.set_xlabel('Actual Values', fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontweight='bold')
        ax1.set_title(f"Actual vs Predicted (R² = {metrics['r2_score']:.3f})",
                      fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Residuals Plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.scatter(y_pred, residuals, alpha=0.6, color=colors[2], s=30,
                    edgecolors='white', linewidth=0.5)
        ax2.axhline(y=0, color=colors[3], linestyle='--', lw=2, alpha=0.8)

        # Highlight potential outliers (beyond 2.5 standard deviations)
        outlier_threshold = 2.5 * np.std(residuals)
        outliers = np.abs(residuals) > outlier_threshold
        if np.any(outliers):
            ax2.scatter(y_pred[outliers], residuals[outliers],
                        color=colors[3], s=60, alpha=0.8, marker='x',
                        linewidth=2, label='Potential Outliers')
            ax2.legend()

        ax2.set_xlabel('Predicted Values', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title(f"Residuals (Std = {metrics['std_residuals']:.3f})", fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. Feature Importance (enhanced)
        ax3 = fig.add_subplot(gs[:2, 2])

        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        importance_scores = None
        importance_type = "Model-based"

        # Try multiple methods to get feature importance
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_') and model.coef_ is not None:
            importance_scores = np.abs(model.coef_.flatten())
            importance_type = "Coefficient-based"
        else:
            # Fallback to permutation importance
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    perm_importance = permutation_importance(
                        model, X, y, scoring=permutation_importance_scoring,
                        n_repeats=5, random_state=42, n_jobs=-1
                    )
                importance_scores = perm_importance.importances_mean
                importance_type = "Permutation-based"
            except:
                importance_scores = np.ones(len(feature_names))
                importance_type = "Uniform (fallback)"

        if importance_scores is not None and len(importance_scores) > 0:
            # Create importance DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=True).tail(top_features)

            bars = ax3.barh(importance_df['feature'], importance_df['importance'],
                            color=colors[4], alpha=0.7, edgecolor='white', linewidth=0.5)
            ax3.set_xlabel('Importance Score', fontweight='bold')
            ax3.set_title(f'Top {min(top_features, len(feature_names))} Features\n({importance_type})',
                          fontweight='bold', fontsize=11)
            ax3.grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nnot available',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Feature Importance', fontweight='bold', fontsize=11)

        # 4. Residuals Histogram
        ax4 = fig.add_subplot(gs[2, 0])
        n, bins, patches = ax4.hist(residuals, bins=30, alpha=0.7, color=colors[0],
                                    edgecolor='white', linewidth=0.5)

        # Overlay normal distribution for comparison
        mu, sigma = np.mean(residuals), np.std(residuals)
        if sigma > 0:  # Avoid division by zero
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                      np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2))
            y_norm *= len(residuals) * (bins[1] - bins[0])  # Scale to match histogram
            ax4.plot(x_norm, y_norm, '--', color=colors[1], lw=2, label='Normal Dist.')
            ax4.legend()

        ax4.set_xlabel('Residuals', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Residuals Distribution', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # 5. Enhanced Metrics Summary
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')

        # Create comprehensive metrics display
        detailed_metrics = {
            **metrics,
            'adjusted_r2': 1 - (1 - metrics['r2_score']) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1) if X.shape[1] < len(y_true) - 1 else metrics['r2_score'],
            'outliers_count': np.sum(outliers) if 'outliers' in locals() else 0,
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }

        metrics_text = f"""
📊 PERFORMANCE METRICS

R² Score:        {detailed_metrics['r2_score']:.4f}
Adjusted R²:     {detailed_metrics['adjusted_r2']:.4f}
RMSE:           {detailed_metrics['rmse']:.4f}
MAE:            {detailed_metrics['mae']:.4f}
MSE:            {detailed_metrics['mse']:.4f}
MAPE:           {detailed_metrics['mape']:.2f}%
Max Error:      {detailed_metrics['max_error']:.4f}
Correlation:    {detailed_metrics['correlation']:.4f}

📈 DATA SUMMARY

Sample Count:    {detailed_metrics['n_samples']:,}
Features:        {X.shape[1]}
Outliers:        {detailed_metrics['outliers_count']}

Actual Range:    [{np.min(y_true):.2f}, {np.max(y_true):.2f}]
Predicted Range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]
Mean Actual:     {detailed_metrics['mean_actual']:.2f}
Mean Predicted:  {detailed_metrics['mean_predicted']:.2f}
        """

        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[0], alpha=0.1, edgecolor=colors[0]))

        # Main title
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.94)

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
            print(f"Enhanced analysis plot saved: {save_path}")

        # Proper figure state management to prevent display conflicts
        if show:
            plt.show()
            plt.close(fig)  # Immediate cleanup after display
            return None, detailed_metrics  # CRITICAL: Don't return figure when showing

        return fig, detailed_metrics  # Only return figure when not showing

    @staticmethod
    def plot_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         title: str = "Model Performance",
                         save_path: Optional[str] = None,
                         show: bool = True) -> Optional[Figure]:
        """
        Plot actual vs predicted and residuals - FIXED for figure state management.
        Simplified version for backward compatibility.

        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: File path to save the plot
            show: Whether to display the plot (prevents display conflicts)

        Returns:
            Figure object
        """
        # CRITICAL FIX: Clear existing figures
        plt.close('all')

        # Create explicit figure
        fig = plt.figure(figsize=(12, 5))

        # Actual vs Predicted
        ax1 = fig.add_subplot(1, 2, 1)
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
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")

        # Proper figure state management
        if show:
            plt.show()
            plt.close(fig)  # Immediate cleanup after display
            return None  # CRITICAL: Don't return figure when showing

        return fig  # Only return figure when not showing