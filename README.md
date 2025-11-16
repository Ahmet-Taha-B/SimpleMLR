# SimpleMLR Documentation

**Machine Learning Made Simple**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SimpleMLR is a user-friendly machine learning library that makes powerful boosting algorithms easy to use. It provides XGBoost, LightGBM, and sklearn GBM through a simple, consistent interface with automatic hyperparameter optimization and beautiful visualizations.

## 🚀 Quick Start

```python
# Install
pip install -e .

# Import and use in 3 lines
from simple_mlr import xgb_auto
model = xgb_auto(X_train, y_train, strategy='fast')
model.quick_graph()  # Instant performance analysis!
```

## 📖 Table of Contents

- [💾 Installation](#-installation)
- [⭐ Core Features](#-core-features)
- [🎯 Quick Start Guide](#-quick-start-guide)
- [📚 API Reference](#-api-reference)
- [💡 Usage Examples](#-usage-examples)
- [🚀 Working Examples](#-working-examples)
- [🧠 Strategy System](#-strategy-system)
- [🎛️ Advanced Features](#️-advanced-features)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

## 💾 Installation

> **Note**: SimpleMLR is currently in development and not yet published to PyPI. Use the local installation method below.

### Install from GitHub
```python
import subprocess
import sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "git+https://github.com/Ahmet-Taha-B/SimpleMLR.git"
])
```

### Import Components
```python
from simple_mlr import (
    # Basic regressors - simple fit/predict interface
    XGBRegressor, LGBMRegressor, GBMRegressor,
    
    # Auto-tuners - automatically find best parameters
    xgb_auto, lgbm_auto, gbm_auto,
    
    # Convenience functions
    xgb_regressor, lgbm_regressor, gbm_regressor
)
```
### Dependencies
SimpleMLR automatically installs all required dependencies:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- **xgboost** (automatically installed)
- **lightgbm** (automatically installed)
- optuna
- matplotlib
- seaborn

All dependencies are installed automatically when you install SimpleMLR - no additional steps needed!

## ⭐ Core Features

- **🎯 One-line model training** - Build powerful models with minimal code
- **🚀 Three algorithms, one interface** - XGBoost, LightGBM, sklearn GBM
- **🤖 Automatic hyperparameter optimization** - Smart strategies find best parameters
- **🎨 Beautiful visualizations** - Instant model analysis and performance plots
- **⚡ GPU acceleration** - Faster training when available
- **🔧 Flexible customization** - Override parameters while keeping simplicity
- **📊 Comprehensive metrics** - Detailed performance analysis
- **🛡️ Production ready** - Robust preprocessing and validation

## 🎯 Quick Start Guide

### Method 1: One-Line Training (Fastest)

```python
from simple_mlr import xgb_regressor, lgbm_regressor, gbm_regressor

# Train models with one line each
xgb_model = xgb_regressor(X_train, y_train)
lgbm_model = lgbm_regressor(X_train, y_train)
gbm_model = gbm_regressor(X_train, y_train)

# Instant performance analysis
xgb_model.quick_graph()
```

### Method 2: Class-Based Approach (More Control)

```python
from simple_mlr import XGBRegressor, LGBMRegressor, GBMRegressor

# Create and train models
model = XGBRegressor(n_estimators=200, max_depth=4)
model.fit(X_train, y_train)

# Comprehensive analysis
model.plot_analysis(X_test, y_test, style='modern')
```

### Method 3: Automatic Optimization (Best Performance)

```python
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto

# Automatically find best parameters
model = xgb_auto(X_train, y_train, 
                 strategy='fast',    # Optimization strategy
                 n_trials=50,        # Number of trials
                 verbose=1)          # Show progress

# Use the optimized model
predictions = model.predict(X_test)
model.quick_graph()
```

## 📚 API Reference

### Main Regressor Classes

All regressor classes share the same interface and methods while using different underlying algorithms.

#### XGBRegressor

XGBoost gradient boosting regressor with SimpleMLR interface.

```python
model = XGBRegressor(
    n_estimators=100,         # Number of boosting rounds
    max_depth=6,              # Maximum tree depth
    learning_rate=0.1,        # Learning rate (eta)
    gamma=0,                  # Minimum loss reduction for further partition
    min_child_weight=1,       # Minimum sum of instance weight in child
    random_state=42,          # Random seed
    scale_features=True,      # Automatic feature scaling
    handle_categorical=True,  # Automatic categorical encoding
    verbose=True,             # Print progress
    use_gpu=False,            # Use GPU acceleration
    gpu_id=0                  # GPU device ID
)
```

#### LGBMRegressor

LightGBM gradient boosting regressor with SimpleMLR interface.

```python
model = LGBMRegressor(
    num_boost_round=100,      # Number of boosting rounds
    num_leaves=31,            # Number of leaves in tree
    max_depth=-1,             # Maximum tree depth (-1 for no limit)
    learning_rate=0.1,        # Learning rate
    feature_fraction=1.0,     # Fraction of features for training
    bagging_fraction=1.0,     # Fraction of data for training
    lambda_l1=0,              # L1 regularization term
    lambda_l2=0,              # L2 regularization term
    min_data_in_leaf=20,      # Minimum data points in leaf
    min_sum_hessian_in_leaf=1e-3,  # Minimum sum of Hessian in leaf
    random_state=42,          # Random seed
    scale_features=True,      # Automatic feature scaling
    handle_categorical=True,  # Automatic categorical encoding
    verbose=True,             # Print progress
    use_gpu=False,            # Use GPU acceleration
    gpu_id=0                  # GPU device ID
)
```

#### GBMRegressor

Scikit-learn gradient boosting regressor with SimpleMLR interface.

```python
model = GBMRegressor(
    n_estimators=100,         # Number of boosting stages
    max_depth=3,              # Maximum tree depth
    learning_rate=0.1,        # Learning rate (shrinkage)
    subsample=1.0,            # Fraction of samples for training
    max_features=None,        # Number of features for best split
    min_samples_split=2,      # Minimum samples to split node
    min_samples_leaf=1,       # Minimum samples in leaf
    min_weight_fraction_leaf=0, # Minimum weighted fraction in leaf
    alpha=0.9,                # Alpha-quantile for Huber and quantile loss
    random_state=42,          # Random seed
    scale_features=True,      # Automatic feature scaling
    handle_categorical=True,  # Automatic categorical encoding
    verbose=True              # Print progress
)
```

**Common Methods (All Classes):**
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `quick_graph()` - Simple 2-panel performance plot (also available as `graph()`)
- `plot_analysis(X, y)` - Comprehensive 6-panel analysis
- `get_feature_importance()` - Get feature importance DataFrame

### Convenience Functions

#### One-Line Training Functions

Train and return a fitted model in a single function call:

```python
# Quick model training - handles everything automatically
xgb_model = xgb_regressor(X_train, y_train, **kwargs)    # XGBoost
lgbm_model = lgbm_regressor(X_train, y_train, **kwargs)  # LightGBM  
gbm_model = gbm_regressor(X_train, y_train, **kwargs)    # Sklearn GBM
```

#### Auto-Optimization Functions

Automatic hyperparameter optimization with intelligent strategies:

**XGBoost Auto-Tuning:**
```python
xgb_auto(X_train, y_train,
         strategy='default',           # Optimization strategy
         n_trials=50,                 # Number of optimization trials
         cv_folds=5,                  # Cross-validation folds
         validation_split=None,       # Use holdout validation instead
         verbose=1,                   # Verbosity level (0-2)
         use_gpu=False,              # GPU acceleration
         gpu_id=0,                   # GPU device ID
         override_params=None)        # Parameter overrides
```

**LightGBM Auto-Tuning:**
```python
lgbm_auto(X_train, y_train,
          strategy='default',          # Optimization strategy
          n_trials=50,                # Number of optimization trials
          cv_folds=5,                 # Cross-validation folds
          validation_split=None,      # Use holdout validation instead
          verbose=1,                  # Verbosity level (0-2)
          use_gpu=False,             # GPU acceleration
          gpu_id=0,                  # GPU device ID
          override_params=None)       # Parameter overrides
```

**Sklearn GBM Auto-Tuning:**
```python
gbm_auto(X_train, y_train,
         strategy='default',           # Optimization strategy
         n_trials=50,                 # Number of optimization trials
         cv_folds=5,                  # Cross-validation folds
         validation_split=None,       # Use holdout validation instead
         verbose=1,                   # Verbosity level (0-2)
         override_params=None)        # Parameter overrides
```

### Parameter Override System

Override specific parameters while keeping strategy defaults. Supports multiple override types:

```python
model = xgb_auto(X, y, 
                 strategy='fast',
                 override_params={
                     'max_depth': 6,                    # Fix to specific value
                     'learning_rate': (0.05, 0.2),     # Custom numerical range
                     'n_estimators': (100, 300),       # Custom numerical range
                     'tree_method': ['hist', 'approx'], # Categorical choices
                     'booster': 'gbtree'               # Fix categorical parameter
                 })
```

**Override Types:**
- **Fixed value**: `'max_depth': 6` - Use specific value instead of optimization
- **Numerical range**: `'learning_rate': (0.05, 0.2)` - Custom min/max range
- **Categorical choices**: `'tree_method': ['hist', 'approx']` - List of options to choose from
- **Fixed categorical**: `'booster': 'gbtree'` - Use specific categorical value

This works with all auto-tuning functions (xgb_auto, lgbm_auto, gbm_auto).

## 🧠 Strategy System

SimpleMLR provides intelligent optimization strategies that automatically configure parameter ranges for different use cases. Instead of manually setting hundreds of hyperparameters, you simply choose a strategy that matches your goals.

### Available Strategies

| Strategy | Description | Best For | Speed | Performance |
|----------|-------------|----------|-------|-------------|
| `'default'` | Balanced optimization with proven parameter ranges | General use, unknown data characteristics | Medium | Balanced |
| `'fast'` | Speed-optimized ranges for rapid experimentation | Quick prototyping, time-constrained optimization | Fast | Good |
| `'stable'` | Conservative, production-ready settings | Production deployment, ensemble base models | Medium | Reliable |
| `'balanced'` | Well-rounded approach for most datasets | Educational use, baseline establishment | Medium | Good |
| `'aggressive'` | High-variance, complex pattern detection | Competition settings, maximum accuracy needed | Slow | High |
| `'conservative'` | Maximum stability with heavy regularization | Risk-averse applications, noisy data | Medium | Stable |
| `'regularized'` | Strong overfitting prevention focus | Small datasets, overfitting-prone scenarios | Medium | Robust |
| `'diversity'` | Maximum parameter variation for ensembles | Ensemble building, model diversity | Medium | Varied |
| `'competition'` | State-of-the-art configuration | Kaggle competitions, research | Slow | Maximum |

### Using Strategies

```python
# Basic strategy usage
xgb_model = xgb_auto(X, y, strategy='fast')         # Quick experiments
lgbm_model = lgbm_auto(X, y, strategy='stable')     # Production ready
gbm_model = gbm_auto(X, y, strategy='aggressive')   # Maximum performance
```

### Strategy Parameter Ranges

Each strategy defines optimized parameter ranges. For example:

```python
# Fast strategy focuses on speed
fast_model = xgb_auto(X, y, strategy='fast')
# - Uses smaller n_estimators ranges (50-200)
# - Limits max_depth (3-6) 
# - Higher learning rates (0.1-0.3)

# Aggressive strategy pushes boundaries
aggressive_model = xgb_auto(X, y, strategy='aggressive') 
# - Wider n_estimators ranges (100-1000)
# - Deeper trees allowed (4-12)
# - Lower learning rates (0.01-0.2)
```

### Parameter Overrides

Customize any strategy by overriding specific parameters:

```python
# Start with 'fast' strategy but customize key parameters
model = xgb_auto(X, y, 
                 strategy='fast',
                 override_params={
                     'max_depth': 8,                    # Fix to specific value
                     'learning_rate': (0.05, 0.15),   # Custom range
                     'n_estimators': (200, 500),      # Custom range
                     'subsample': [0.8, 0.9, 1.0]     # Categorical choices
                 })
```

### Cross-Algorithm Consistency

The same strategy name works across all algorithms:

```python
# All use same conceptual approach with algorithm-specific parameters
xgb_model = xgb_auto(X, y, strategy='stable')     # XGBoost parameters
lgbm_model = lgbm_auto(X, y, strategy='stable')   # LightGBM parameters  
gbm_model = gbm_auto(X, y, strategy='stable')     # sklearn GBM parameters
```

### Strategy Selection Guide

**Choose `'fast'` when:**
- Prototyping and experimentation
- Time constraints (< 5 minutes)
- Initial model exploration

**Choose `'stable'` when:**
- Production deployment
- Consistent performance needed
- Building ensemble base models

**Choose `'aggressive'` when:**
- Maximum accuracy required
- Kaggle competitions
- Complex relationship detection

**Choose `'balanced'` when:**
- Unsure about data characteristics
- Educational purposes
- General-purpose modeling

## 💡 Usage Examples

### Basic Regression Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from simple_mlr import xgb_auto

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train optimized model
model = xgb_auto(X_train, y_train, strategy='balanced', n_trials=50, verbose=1)

# Analyze performance
model.quick_graph()  # Quick analysis
fig, metrics = model.plot_analysis(X_test, y_test, save_path='analysis.png')  # Detailed report

# Make predictions
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()
print(importance.head())
```

### Comparing Multiple Algorithms

```python
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Train all three algorithms
models = {
    'XGBoost': xgb_auto(X_train, y_train, strategy='fast', n_trials=30),
    'LightGBM': lgbm_auto(X_train, y_train, strategy='fast', n_trials=30),
    'Sklearn GBM': gbm_auto(X_train, y_train, strategy='stable', n_trials=30)
}

# Compare performance
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({'Model': name, 'R²': r2, 'RMSE': rmse})

comparison_df = pd.DataFrame(results)
print(comparison_df)
```

### GPU Acceleration

```python
# Enable GPU acceleration (if available)
model = xgb_auto(X_train, y_train, 
                 use_gpu=True,        # Enable GPU
                 strategy='fast',
                 n_trials=50)

# Check if GPU was used
if hasattr(model, 'model_') and hasattr(model.model_, 'get_params'):
    params = model.model_.get_params()
    if 'tree_method' in params:
        print(f"Training method: {params['tree_method']}")
    elif 'device' in params:
        print(f"Training device: {params['device']}")
```

### Custom Preprocessing

```python
# Disable automatic preprocessing if you want manual control
model = XGBRegressor(
    scale_features=False,      # Don't scale features
    handle_categorical=False   # Don't encode categoricals
)

# Do your own preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model.fit(X_scaled, y_train)
```

## 🚀 Working Examples

This section provides complete, self-contained examples that you can copy and run immediately.

### Complete Workflow with Sample Data

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto

# Create sample dataset
print("Creating sample dataset...")
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
feature_names = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], 
    test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train model with automatic optimization
print("\nTraining XGBoost model...")
model = xgb_auto(X_train, y_train, 
                 strategy='fast', 
                 n_trials=30, 
                 verbose=1)

# Evaluate and visualize
print("\nGenerating analysis...")
model.quick_graph()  # Quick 2-panel analysis

# Get detailed performance metrics
fig, metrics = model.plot_analysis(X_test, y_test, 
                                 title="Sample Data Analysis",
                                 save_path="model_analysis.png")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# Feature importance
importance = model.get_feature_importance()
print(f"\nTop 5 features:")
print(importance.head())
```

### Multi-Algorithm Comparison

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto

# Load real dataset
print("Loading diabetes dataset...")
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all three algorithms
algorithms = {
    'XGBoost': lambda: xgb_auto(X_train, y_train, strategy='fast', n_trials=20, verbose=0),
    'LightGBM': lambda: lgbm_auto(X_train, y_train, strategy='fast', n_trials=20, verbose=0),
    'Sklearn GBM': lambda: gbm_auto(X_train, y_train, strategy='fast', n_trials=20, verbose=0)
}

results = []
models = {}

for name, train_func in algorithms.items():
    print(f"\nTraining {name}...")
    
    try:
        # Train model
        model = train_func()
        models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'Algorithm': name,
            'R² Score': r2,
            'RMSE': rmse,
            'Status': 'Success'
        })
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        results.append({
            'Algorithm': name,
            'R² Score': np.nan,
            'RMSE': np.nan,
            'Status': f'Failed: {str(e)[:50]}'
        })

# Display comparison
comparison_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("ALGORITHM COMPARISON RESULTS")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))

# Show best model analysis
successful_models = [name for name in models.keys()]
if successful_models:
    best_model_name = successful_models[0]  # You could choose based on metrics
    print(f"\nShowing analysis for {best_model_name}...")
    models[best_model_name].quick_graph()
```

### Custom Parameter Override Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from simple_mlr import xgb_auto

# Create dataset with more samples for demonstration
X, y = make_regression(n_samples=5000, n_features=15, 
                      noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training with custom parameter overrides...")

# Example 1: Fix some parameters, customize others
model1 = xgb_auto(X_train, y_train,
                  strategy='balanced',  # Start with balanced strategy
                  n_trials=30,
                  override_params={
                      'max_depth': 8,                    # Fix max_depth
                      'learning_rate': (0.05, 0.15),   # Custom range
                      'n_estimators': (200, 800),      # Custom range
                      'tree_method': ['hist', 'approx'], # Categorical choices
                  },
                  verbose=1)

print("Model 1 trained successfully!")

# Example 2: GPU configuration (if available)
try:
    model2 = xgb_auto(X_train, y_train,
                      strategy='fast',
                      n_trials=20,
                      use_gpu=True,
                      gpu_id=0,
                      override_params={
                          'tree_method': 'gpu_hist',  # Force GPU method
                          'max_depth': 6,
                          'learning_rate': (0.1, 0.3)
                      },
                      verbose=1)
    print("GPU model trained successfully!")
    
    # Compare GPU vs CPU performance
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    
    from sklearn.metrics import r2_score
    print(f"CPU Model R²: {r2_score(y_test, y_pred1):.4f}")
    print(f"GPU Model R²: {r2_score(y_test, y_pred2):.4f}")
    
except Exception as e:
    print(f"GPU training failed (expected if no GPU available): {e}")

# Show analysis for the first model
model1.plot_analysis(X_test, y_test, title="Custom Parameters Model")
```

### Save and Load Model Example

```python
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from simple_mlr import lgbm_auto

# Create and train model
print("Creating dataset and training model...")
X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = lgbm_auto(X_train, y_train, 
                  strategy='stable', 
                  n_trials=25, 
                  verbose=1)

# Test original model
original_predictions = model.predict(X_test)
print(f"Original model predictions sample: {original_predictions[:5]}")

# Save model
model_file = "trained_lgbm_model.pkl"
print(f"\nSaving model to {model_file}...")
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully!")

# Load model back
print(f"\nLoading model from {model_file}...")
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully!")

# Test loaded model
loaded_predictions = loaded_model.predict(X_test)
print(f"Loaded model predictions sample: {loaded_predictions[:5]}")

# Verify predictions are identical
import numpy as np
predictions_match = np.allclose(original_predictions, loaded_predictions)
print(f"\nPredictions match: {predictions_match}")

# Use loaded model for analysis
print("\nGenerating analysis with loaded model...")
loaded_model.quick_graph()

# Get feature importance from loaded model
importance = loaded_model.get_feature_importance()
print(f"\nFeature importance from loaded model:")
print(importance)

# Clean up
import os
os.remove(model_file)
print(f"\nCleaned up {model_file}")
```

### Integration with Pandas DataFrames

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from simple_mlr import gbm_auto

# Create a realistic dataset with mixed data types
print("Creating realistic mixed-type dataset...")
np.random.seed(42)

# Generate data
n_samples = 2000
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'education_years': np.random.randint(8, 20, n_samples),
    'experience': np.random.exponential(5, n_samples),
    'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
    'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
    'has_degree': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
}

# Create target variable with realistic relationships
df = pd.DataFrame(data)
df['salary'] = (
    df['age'] * 800 +
    df['income'] * 0.1 +
    df['education_years'] * 2000 +
    df['experience'] * 1500 +
    (df['city_size'] == 'Large') * 10000 +
    (df['department'] == 'Engineering') * 15000 +
    df['has_degree'] * 8000 +
    np.random.normal(0, 5000, n_samples)
)

print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())

# Prepare features and target
feature_columns = ['age', 'income', 'education_years', 'experience', 
                   'city_size', 'department', 'has_degree', 'region']
X = df[feature_columns]
y = df['salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['department']
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train model - SimpleMLR automatically handles categorical features
print(f"\nTraining model with automatic categorical handling...")
model = gbm_auto(X_train, y_train,
                 strategy='balanced',
                 n_trials=40,
                 verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\nModel Performance:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")

# Analysis with DataFrames
print(f"\nGenerating comprehensive analysis...")
fig, metrics = model.plot_analysis(X_test, y_test,
                                 title="Salary Prediction Model",
                                 save_path="salary_model_analysis.png")

# Feature importance analysis
importance_df = model.get_feature_importance()
print(f"\nFeature Importance Rankings:")
print(importance_df)

# Predictions analysis
results_df = pd.DataFrame({
    'actual_salary': y_test,
    'predicted_salary': predictions,
    'error': y_test - predictions,
    'abs_error': np.abs(y_test - predictions),
    'department': X_test['department']
})

print(f"\nPrediction Results Summary:")
print(results_df.describe())

print(f"\nAverage Error by Department:")
dept_errors = results_df.groupby('department')['abs_error'].mean().sort_values(ascending=False)
print(dept_errors)

# Create a sample prediction for new data
new_employee = pd.DataFrame({
    'age': [35],
    'income': [75000],
    'education_years': [16],
    'experience': [8],
    'city_size': ['Large'],
    'department': ['Engineering'],
    'has_degree': [True],
    'region': ['West']
})

predicted_salary = model.predict(new_employee)[0]
print(f"\nSample prediction for new employee:")
print(f"Predicted salary: ${predicted_salary:,.2f}")
```

## 🎛️ Advanced Features

### Custom Validation

```python
# Use holdout validation instead of cross-validation (faster)
model = xgb_auto(X_train, y_train,
                 validation_split=0.2,  # 20% holdout validation
                 n_trials=100)
```

### Saving and Loading Models

```python
import pickle

# Save trained model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load saved model
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use loaded model
predictions = loaded_model.predict(X_new)
```

## 📊 Visualization Guide

### Quick Analysis
```python
# Simple 2-panel plot
model.quick_graph()  # Uses training data automatically
model.quick_graph(X_test, y_test)  # Specify test data
```

### Comprehensive Analysis
```python
# Detailed 6-panel analysis
fig, metrics = model.plot_analysis(
    X_test, y_test,
    title="My Model Analysis",
    style='modern',           # 'modern', 'classic', or 'minimal'
    save_path='analysis.png', # Save to file
    top_features=10           # Number of top features to show
)

# Access detailed metrics
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Custom Plotting
```python
# Use ModelEvaluator directly for custom analysis
from simple_mlr import ModelEvaluator

# Works with any sklearn-compatible model
fig, metrics = ModelEvaluator.plot_model_analysis(
    model=your_model,
    X=X_test,
    y=y_test,
    title="Custom Analysis"
)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### ImportError: No module named 'xgboost' or 'lightgbm'
This should not occur as both XGBoost and LightGBM are automatically installed with SimpleMLR. If you encounter this error:

```bash
# Install SimpleMLR to ensure all dependencies are installed
pip install -e .
```

#### GPU not detected
```python
# Check GPU availability
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
print(f"GPU available: {xgb.gpu.is_available()}")

# For LightGBM GPU
import lightgbm as lgb
print(f"LightGBM version: {lgb.__version__}")
```

#### Memory issues with large datasets
```python
# Use validation split instead of cross-validation
model = xgb_auto(X, y, 
                 validation_split=0.2,  # Faster than CV
                 n_trials=30)           # Reduce trials

# Use memory efficient tree method
model = XGBRegressor(tree_method='hist')  # Memory efficient
```

#### Categorical features not handled properly
```python
# Ensure categorical columns are properly typed
df['category_column'] = df['category_column'].astype('category')

# Or convert to string
df['category_column'] = df['category_column'].astype(str)
```

### Performance Tips

1. **Use validation split for large datasets**:
   ```python
   model = xgb_auto(X, y, validation_split=0.2)  # Faster than CV
   ```

2. **Enable GPU for large datasets**:
   ```python
   model = xgb_auto(X, y, use_gpu=True)
   ```

3. **Reduce trials for quick experiments**:
   ```python
   model = xgb_auto(X, y, strategy='fast', n_trials=20)
   ```

4. **Use appropriate strategy**:
   - Development: `strategy='fast'`
   - Production: `strategy='stable'`
   - Competition: `strategy='aggressive'`

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup
```bash
# Clone repository
git clone https://github.com/Ahmet-Taha-B/SimpleMLR.git
cd SimpleMLR

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black isort flake8
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
# Format code
black simple_mlr/
isort simple_mlr/

# Check style
flake8 simple_mlr/
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- Microsoft for LightGBM
- Scikit-learn team for the foundational ML library
- Optuna team for hyperparameter optimization framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Ahmet-Taha-B/SimpleMLR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ahmet-Taha-B/SimpleMLR/discussions)
- **Documentation**: This README and docstrings

---

**Happy Machine Learning! 🚀**

*SimpleMLR - Making machine learning accessible to everyone.*
