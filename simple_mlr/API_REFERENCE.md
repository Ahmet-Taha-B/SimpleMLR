# SimpleMLR API Reference

This document provides detailed information about all classes, functions, and parameters in SimpleMLR.

## Table of Contents

- [Core Classes](#core-classes)
- [Convenience Functions](#convenience-functions)
- [Auto-Optimization Functions](#auto-optimization-functions)
- [Utility Classes](#utility-classes)
- [Strategy System](#strategy-system)
- [Parameter Reference](#parameter-reference)

## Core Classes

### XGBRegressor

XGBoost regression model with SimpleMLR interface.

```python
class XGBRegressor(BaseBoostingRegressor):
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
                 **kwargs)
```

**Parameters:**
- `n_estimators` (int): Number of boosting rounds
- `max_depth` (int): Maximum tree depth
- `learning_rate` (float): Learning rate (eta)
- `subsample` (float): Subsample ratio of training instances
- `colsample_bytree` (float): Subsample ratio of features
- `reg_alpha` (float): L1 regularization term
- `reg_lambda` (float): L2 regularization term
- `gamma` (float): Minimum loss reduction for splits
- `min_child_weight` (float): Minimum sum of instance weight in child
- `random_state` (int): Random seed
- `scale_features` (bool): Automatically scale numerical features
- `handle_categorical` (bool): Automatically encode categorical features
- `verbose` (bool): Print training progress
- `n_jobs` (int): Number of parallel threads (-1 for all cores)
- `use_gpu` (bool): Use GPU acceleration
- `gpu_id` (int): GPU device ID
- `**kwargs`: Additional XGBoost parameters

**Methods:**

#### fit(X, y)
Train the XGBoost model.

```python
model.fit(X_train, y_train)
```

**Parameters:**
- `X` (DataFrame or array): Training features
- `y` (Series or array): Training targets

**Returns:** Self for method chaining

#### predict(X)
Make predictions using the trained model.

```python
predictions = model.predict(X_test)
```

**Parameters:**
- `X` (DataFrame or array): Features to predict

**Returns:** numpy array of predictions

#### quick_graph(X=None, y=None)
Create a simple 2-panel performance plot.

```python
model.quick_graph()  # Uses training data
model.quick_graph(X_test, y_test)  # Uses test data
```

**Parameters:**
- `X` (DataFrame or array, optional): Features for evaluation
- `y` (Series or array, optional): True values for evaluation

**Returns:** None (displays plot)

#### plot_analysis(X, y, title=None, save_path=None, style='modern')
Create comprehensive model analysis plot.

```python
fig, metrics = model.plot_analysis(X_test, y_test, 
                                   title="My Analysis",
                                   save_path="analysis.png",
                                   style='modern')
```

**Parameters:**
- `X` (DataFrame or array): Features for evaluation
- `y` (Series or array): True values for evaluation
- `title` (str, optional): Plot title
- `save_path` (str, optional): File path to save plot
- `style` (str): Plot style ('modern', 'classic', 'minimal')

**Returns:** Tuple of (None, detailed_metrics_dict)

#### get_feature_importance(top_n=10)
Get feature importance rankings.

```python
importance_df = model.get_feature_importance(top_n=15)
```

**Parameters:**
- `top_n` (int): Number of top features to return

**Returns:** DataFrame with 'feature' and 'importance' columns

### LGBMRegressor

LightGBM regression model with SimpleMLR interface.

```python
class LGBMRegressor(BaseBoostingRegressor):
    def __init__(self,
                 num_boost_round: int = 100,
                 num_leaves: int = 31,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 feature_fraction: float = 1.0,
                 bagging_fraction: float = 1.0,
                 bagging_freq: int = 0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0,
                 min_child_weight: float = 1e-3,
                 min_child_samples: int = 20,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 gpu_device_id: int = 0,
                 **kwargs)
```

**Parameters:**
- `num_boost_round` (int): Number of boosting rounds
- `num_leaves` (int): Maximum number of leaves in one tree
- `max_depth` (int): Maximum tree depth (-1 for no limit)
- `learning_rate` (float): Learning rate
- `feature_fraction` (float): Fraction of features for each tree
- `bagging_fraction` (float): Fraction of data for each tree
- `bagging_freq` (int): Frequency for bagging (0 = disabled)
- `reg_alpha` (float): L1 regularization
- `reg_lambda` (float): L2 regularization
- `min_child_weight` (float): Minimum sum of hessian in leaf
- `min_child_samples` (int): Minimum number of samples in leaf
- `random_state` (int): Random seed
- `scale_features` (bool): Automatically scale numerical features
- `handle_categorical` (bool): Automatically encode categorical features
- `verbose` (bool): Print training progress
- `n_jobs` (int): Number of parallel threads (-1 for all cores)
- `use_gpu` (bool): Use GPU acceleration
- `gpu_device_id` (int): GPU device ID
- `**kwargs`: Additional LightGBM parameters

**Methods:** Same as XGBRegressor (fit, predict, quick_graph, plot_analysis, get_feature_importance)

### GBMRegressor

Sklearn Gradient Boosting regression model with SimpleMLR interface.

```python
class GBMRegressor(BaseBoostingRegressor):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 max_features: Union[str, int, float] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 alpha: float = 0.9,
                 random_state: int = 42,
                 scale_features: bool = True,
                 handle_categorical: bool = True,
                 verbose: bool = True,
                 **kwargs)
```

**Parameters:**
- `n_estimators` (int): Number of boosting stages
- `max_depth` (int): Maximum depth of trees
- `learning_rate` (float): Learning rate shrinks contribution of each tree
- `subsample` (float): Fraction of samples for each tree
- `max_features` (str, int, float): Number of features for best split
- `min_samples_split` (int): Minimum samples required to split node
- `min_samples_leaf` (int): Minimum samples required at leaf node
- `alpha` (float): Alpha-quantile of huber loss function
- `random_state` (int): Random seed
- `scale_features` (bool): Automatically scale numerical features
- `handle_categorical` (bool): Automatically encode categorical features
- `verbose` (bool): Print training progress
- `**kwargs`: Additional sklearn GBM parameters

**Methods:** Same as XGBRegressor (fit, predict, quick_graph, plot_analysis, get_feature_importance)

## Convenience Functions

### xgb_regressor(X, y, **kwargs)
One-line XGBoost model training.

```python
model = xgb_regressor(X_train, y_train, n_estimators=200, verbose=True)
```

**Parameters:**
- `X` (DataFrame or array): Training features
- `y` (Series or array): Training targets
- `**kwargs`: Parameters passed to XGBRegressor

**Returns:** Trained XGBRegressor instance

### lgbm_regressor(X, y, **kwargs)
One-line LightGBM model training.

```python
model = lgbm_regressor(X_train, y_train, num_boost_round=200, verbose=True)
```

**Parameters:**
- `X` (DataFrame or array): Training features
- `y` (Series or array): Training targets
- `**kwargs`: Parameters passed to LGBMRegressor

**Returns:** Trained LGBMRegressor instance

### gbm_regressor(X, y, **kwargs)
One-line sklearn GBM model training.

```python
model = gbm_regressor(X_train, y_train, n_estimators=200, verbose=True)
```

**Parameters:**
- `X` (DataFrame or array): Training features
- `y` (Series or array): Training targets
- `**kwargs`: Parameters passed to GBMRegressor

**Returns:** Trained GBMRegressor instance

## Auto-Optimization Functions

### xgb_auto(X, y, **kwargs)
Automatic XGBoost hyperparameter optimization.

```python
model = xgb_auto(X_train, y_train,
                 strategy='fast',
                 n_trials=50,
                 verbose=1)
```

**Parameters:**
- `X` (DataFrame or array): Training features
- `y` (Series or array): Training targets
- `strategy` (str): Optimization strategy ('default', 'fast', 'stable', 'aggressive', 'balanced')
- `n_trials` (int): Number of optimization trials
- `cv_folds` (int): Number of cross-validation folds
- `validation_split` (float, optional): Use holdout validation instead of CV
- `verbose` (int): Verbosity level (0=silent, 1=progress, 2=detailed)
- `use_gpu` (bool): Use GPU acceleration
- `override_params` (dict, optional): Parameter overrides
- `multi_fidelity` (bool): Enable multi-fidelity optimization
- `early_stopping_rounds` (int, optional): Enable early stopping
- `**kwargs`: Additional parameters

**Returns:** Optimized XGBRegressor instance

### lgbm_auto(X, y, **kwargs)
Automatic LightGBM hyperparameter optimization.

```python
model = lgbm_auto(X_train, y_train,
                  strategy='stable',
                  n_trials=100,
                  use_gpu=True)
```

**Parameters:** Same as xgb_auto

**Returns:** Optimized LGBMRegressor instance

### gbm_auto(X, y, **kwargs)
Automatic sklearn GBM hyperparameter optimization.

```python
model = gbm_auto(X_train, y_train,
                 strategy='balanced',
                 n_trials=75)
```

**Parameters:** Same as xgb_auto (except use_gpu not available)

**Returns:** Optimized GBMRegressor instance

## Utility Classes

### DataValidator

Handles data validation and preprocessing.

```python
class DataValidator:
    @staticmethod
    def validate_inputs(X, y):
        """Validate and convert inputs to proper pandas format."""
        
    @staticmethod
    def prepare_features(X, scale_features=True, handle_categorical=True):
        """Prepare features for training."""
```

**Methods:**

#### validate_inputs(X, y)
Validate input data format and consistency.

```python
X_validated, y_validated = DataValidator.validate_inputs(X, y)
```

**Parameters:**
- `X` (DataFrame or array): Features
- `y` (Series or array): Targets

**Returns:** Tuple of (validated_X_DataFrame, validated_y_Series)

#### prepare_features(X, scale_features=True, handle_categorical=True)
Preprocess features for machine learning.

```python
X_processed, artifacts = DataValidator.prepare_features(
    X, 
    scale_features=True, 
    handle_categorical=True
)
```

**Parameters:**
- `X` (DataFrame): Input features
- `scale_features` (bool): Apply StandardScaler to numerical features
- `handle_categorical` (bool): Apply LabelEncoder to categorical features

**Returns:** Tuple of (processed_DataFrame, preprocessing_artifacts_dict)

### ModelEvaluator

Creates performance analysis and visualizations.

```python
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive regression metrics."""
        
    @staticmethod
    def plot_model_analysis(model, X, y, **kwargs):
        """Create comprehensive model analysis plot."""
```

**Methods:**

#### calculate_metrics(y_true, y_pred)
Calculate comprehensive regression metrics.

```python
metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
```

**Parameters:**
- `y_true` (array): True values
- `y_pred` (array): Predicted values

**Returns:** Dictionary with metrics:
- `r2_score`: R-squared score
- `rmse`: Root Mean Square Error
- `mae`: Mean Absolute Error
- `mse`: Mean Square Error
- `mape`: Mean Absolute Percentage Error
- `max_error`: Maximum absolute error
- `std_residuals`: Standard deviation of residuals
- `n_samples`: Number of samples

#### plot_model_analysis(model, X, y, **kwargs)
Create comprehensive model analysis visualization.

```python
fig, metrics = ModelEvaluator.plot_model_analysis(
    model, X_test, y_test,
    title="Model Analysis",
    style='modern',
    save_path='analysis.png'
)
```

**Parameters:**
- `model`: Trained sklearn-compatible model
- `X` (DataFrame or array): Features for evaluation
- `y` (Series or array): True values
- `title` (str): Plot title
- `style` (str): Plot style ('modern', 'classic', 'minimal')
- `save_path` (str, optional): File path to save plot
- `show` (bool): Whether to display plot
- `figsize` (tuple): Figure size
- `top_features` (int): Number of top features to show

**Returns:** Tuple of (figure_or_None, detailed_metrics_dict)

## Strategy System

### Available Strategies

All algorithms support these optimization strategies:

| Strategy | Description | Use Case | Speed | Performance |
|----------|-------------|----------|--------|-------------|
| `'default'` | Balanced optimization | General-purpose | Medium | Good |
| `'fast'` | Quick optimization | Prototyping, experimentation | Fast | Good |
| `'stable'` | Conservative settings | Production deployment | Slow | Reliable |
| `'aggressive'` | Maximum performance | Competition, research | Slow | Best |
| `'balanced'` | Well-rounded approach | Education, baseline | Medium | Good |

### Strategy Usage

```python
# Choose strategy based on your needs
model = xgb_auto(X, y, strategy='fast')      # Quick results
model = lgbm_auto(X, y, strategy='stable')   # Production ready
model = gbm_auto(X, y, strategy='aggressive') # Maximum performance
```

### Custom Strategy with Overrides

```python
model = xgb_auto(X, y, 
                 strategy='fast',
                 override_params={
                     'max_depth': 6,              # Fix specific parameter
                     'learning_rate': (0.05, 0.2), # Custom range
                     'n_estimators': (50, 200)    # Custom range
                 })
```

## Parameter Reference

### Override Parameters Format

When using `override_params`, use these formats:

```python
override_params = {
    # Fixed value
    'max_depth': 6,
    
    # Numerical range (min, max)
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 500),
    
    # Categorical choices
    'booster': ['gbtree', 'gblinear'],
    
    # Fixed categorical
    'objective': 'reg:squarederror'
}
```

### Common Parameter Mappings

| Concept | XGBoost | LightGBM | Sklearn GBM |
|---------|---------|----------|-------------|
| Number of trees | `n_estimators` | `num_boost_round` | `n_estimators` |
| Tree depth | `max_depth` | `max_depth` | `max_depth` |
| Learning rate | `learning_rate` | `learning_rate` | `learning_rate` |
| Feature sampling | `colsample_bytree` | `feature_fraction` | `max_features` |
| Sample sampling | `subsample` | `bagging_fraction` | `subsample` |
| L1 regularization | `reg_alpha` | `reg_alpha` | N/A |
| L2 regularization | `reg_lambda` | `reg_lambda` | N/A |

### Verbose Levels

- `verbose=-1`: Completely silent
- `verbose=0`: Minimal output (start/finish messages)
- `verbose=1`: Progress bar with key metrics
- `verbose=2`: Detailed trial-by-trial information

### GPU Parameters

```python
# XGBoost GPU
model = xgb_auto(X, y, use_gpu=True, gpu_id=0)

# LightGBM GPU  
model = lgbm_auto(X, y, use_gpu=True, gpu_device_id=0)

# Check GPU availability
import xgboost as xgb
print(f"XGBoost GPU available: {xgb.gpu.is_available()}")
```

### Performance Tuning Parameters

```python
# For large datasets
model = xgb_auto(X, y,
                 validation_split=0.2,    # Faster than CV
                 n_trials=30,             # Reduce trials
                 early_stopping_rounds=10) # Stop early

# For maximum performance
model = xgb_auto(X, y,
                 strategy='aggressive',
                 n_trials=200,            # More exploration
                 cv_folds=10)             # More robust validation
```

This completes the comprehensive API reference for SimpleMLR. Each function and class provides the flexibility needed for both simple use cases and advanced machine learning workflows.