# SimpleMLR User Guide

A comprehensive guide to using SimpleMLR for machine learning projects.

## Table of Contents

- [Getting Started](#getting-started)
- [Understanding the Three Approaches](#understanding-the-three-approaches)
- [Choosing the Right Algorithm](#choosing-the-right-algorithm)
- [Data Preprocessing](#data-preprocessing)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Model Evaluation](#model-evaluation)
- [Production Deployment](#production-deployment)
- [Best Practices](#best-practices)
- [Common Workflows](#common-workflows)
- [Performance Tips](#performance-tips)

## Getting Started

### Your First SimpleMLR Model

Let's start with the simplest possible example:

```python
import pandas as pd
from simple_mlr import xgb_regressor

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target_column', axis=1)  # Features
y = df['target_column']               # What you want to predict

# Train a model in one line
model = xgb_regressor(X, y)

# See how well it performed
model.quick_graph()
```

That's it! SimpleMLR handled:
- ✅ Data validation and preprocessing
- ✅ Feature scaling and categorical encoding  
- ✅ Model training with good default parameters
- ✅ Performance visualization

## Understanding the Three Approaches

SimpleMLR offers three ways to build models, each with different levels of control:

### 1. Convenience Functions (Fastest)
**Best for: Quick experiments, learning, prototyping**

```python
from simple_mlr import xgb_regressor, lgbm_regressor, gbm_regressor

# One line per algorithm
xgb_model = xgb_regressor(X_train, y_train)
lgbm_model = lgbm_regressor(X_train, y_train) 
gbm_model = gbm_regressor(X_train, y_train)

# All have the same interface
for name, model in [('XGBoost', xgb_model), ('LightGBM', lgbm_model), ('GBM', gbm_model)]:
    print(f"{name} R²: {model.training_metrics_['r2_score']:.4f}")
```

### 2. Class-Based Approach (More Control)
**Best for: Custom parameters, understanding what's happening**

```python
from simple_mlr import XGBRegressor, LGBMRegressor, GBMRegressor

# Create model with custom parameters
model = XGBRegressor(
    n_estimators=200,      # More trees
    max_depth=4,           # Shallower trees  
    learning_rate=0.05,    # Slower learning
    scale_features=True,   # Enable preprocessing
    verbose=True           # Show training progress
)

# Train the model
model.fit(X_train, y_train)

# Analyze performance
model.plot_analysis(X_test, y_test, style='modern')
```

### 3. Auto-Optimization (Best Performance)
**Best for: Production models, competitions, maximum performance**

```python
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto

# Automatically find the best parameters
model = xgb_auto(
    X_train, y_train,
    strategy='balanced',    # Optimization strategy
    n_trials=50,           # How many combinations to try
    verbose=1              # Show progress
)

# This model is optimized for your specific data!
```

## Choosing the Right Algorithm

Each algorithm has its strengths. Here's when to use each:

### XGBoost 🚀
**Best for: General-purpose, proven performance**

```python
from simple_mlr import xgb_auto

model = xgb_auto(X, y, strategy='balanced')
```

**Pros:**
- Excellent performance on most datasets
- Handles missing values well
- Great for competitions
- Extensive community and resources

**Cons:**
- Can be memory intensive
- Slower than LightGBM on large datasets

**Use when:** You want reliable, high-performance results and aren't sure which algorithm to try first.

### LightGBM ⚡
**Best for: Large datasets, speed is important**

```python
from simple_mlr import lgbm_auto

model = lgbm_auto(X, y, strategy='fast', use_gpu=True)
```

**Pros:**
- Very fast training
- Low memory usage
- Excellent GPU support
- Built-in categorical feature handling

**Cons:**
- Can overfit on small datasets (<10k samples)
- Less stable than XGBoost on some datasets

**Use when:** You have large datasets (>100k samples), need fast training, or are memory-constrained.

### Sklearn GBM 🛡️
**Best for: Reliability, simplicity, no external dependencies**

```python
from simple_mlr import gbm_auto

model = gbm_auto(X, y, strategy='stable')
```

**Pros:**
- Rock-solid reliability
- Part of scikit-learn (no extra dependencies)
- Very predictable behavior
- Extensive documentation

**Cons:**
- Slower than XGBoost/LightGBM
- Fewer advanced features
- No GPU support

**Use when:** You need maximum reliability, are in a restricted environment, or want the simplest possible setup.

### Quick Comparison

| Algorithm | Speed | Memory | Performance | Reliability | GPU |
|-----------|-------|--------|-------------|-------------|-----|
| XGBoost | Medium | High | Excellent | High | Yes |
| LightGBM | Fast | Low | Excellent | Medium | Yes |
| Sklearn GBM | Slow | Medium | Good | Very High | No |

## Data Preprocessing

SimpleMLR handles preprocessing automatically, but understanding what happens helps you use it effectively.

### Automatic Preprocessing

```python
# SimpleMLR automatically:
# 1. Validates data (checks for missing values, infinite values)
# 2. Scales numerical features using StandardScaler
# 3. Encodes categorical features using LabelEncoder
# 4. Converts data types as needed

model = xgb_regressor(X, y)  # All preprocessing happens automatically
```

### Manual Preprocessing Control

```python
# Disable automatic preprocessing if you want control
model = XGBRegressor(
    scale_features=False,      # Don't scale features
    handle_categorical=False   # Don't encode categoricals
)

# Do your own preprocessing
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

scaler = MinMaxScaler()
encoder = OrdinalEncoder()

X_processed = X.copy()
X_processed[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_processed[categorical_cols] = encoder.fit_transform(X[categorical_cols])

model.fit(X_processed, y)
```

### Handling Different Data Types

```python
import pandas as pd

# Ensure categorical columns are properly identified
df['category_column'] = df['category_column'].astype('category')
df['text_column'] = df['text_column'].astype('string')
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')

# SimpleMLR will handle these appropriately
model = xgb_regressor(df.drop('target', axis=1), df['target'])
```

### Dealing with Missing Values

```python
# SimpleMLR requires no missing values, so handle them first:

# Option 1: Drop rows with missing values
df_clean = df.dropna()

# Option 2: Fill missing values
df_filled = df.fillna({
    'numerical_col': df['numerical_col'].median(),
    'categorical_col': df['categorical_col'].mode()[0]
})

# Option 3: Use sklearn imputers
from sklearn.impute import SimpleImputer, KNNImputer

# For numerical columns
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# For categorical columns  
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Now use with SimpleMLR
model = xgb_regressor(X, y)
```

## Hyperparameter Optimization

This is where SimpleMLR really shines. Understanding strategies and optimization can dramatically improve your results.

### Understanding Strategies

Strategies are pre-configured parameter ranges optimized for different situations:

```python
# Quick comparison of strategies
strategies = ['fast', 'balanced', 'stable', 'aggressive']
results = {}

for strategy in strategies:
    print(f"\nTesting {strategy} strategy...")
    model = xgb_auto(X_train, y_train, 
                     strategy=strategy, 
                     n_trials=25,  # Keep it quick for comparison
                     verbose=0)    # Quiet output
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    results[strategy] = r2
    print(f"{strategy}: R² = {r2:.4f}")

# Find best strategy for your data
best_strategy = max(results, key=results.get)
print(f"\nBest strategy for your data: {best_strategy}")
```

### Strategy Details

#### 'fast' Strategy
```python
# Best for: Quick experiments, prototyping
model = xgb_auto(X, y, strategy='fast', n_trials=25)
```
- Smaller parameter ranges for speed
- Fewer trees, higher learning rates
- 2-3x faster than 'default'
- Good for getting quick results

#### 'balanced' Strategy  
```python
# Best for: Most use cases, learning
model = xgb_auto(X, y, strategy='balanced', n_trials=50)
```
- Well-tested parameter ranges
- Good balance of speed and performance
- Safe choice when unsure
- Educational value - see typical ranges

#### 'stable' Strategy
```python
# Best for: Production deployment
model = gbm_auto(X, y, strategy='stable', n_trials=75)
```
- Conservative parameter ranges
- Reliable, reproducible results
- Less likely to overfit
- Great for production systems

#### 'aggressive' Strategy
```python
# Best for: Maximum performance, competitions
model = xgb_auto(X, y, strategy='aggressive', n_trials=100)
```
- Wide parameter ranges
- Explores extreme values
- Can achieve best performance
- May overfit on small datasets

### Parameter Override System

Fine-tune specific parameters while keeping strategy defaults:

```python
model = xgb_auto(X, y, 
                 strategy='balanced',
                 override_params={
                     # Fix specific values
                     'max_depth': 6,
                     'random_state': 42,
                     
                     # Custom ranges  
                     'learning_rate': (0.01, 0.2),
                     'n_estimators': (100, 300),
                     
                     # Categorical choices
                     'booster': ['gbtree'],  # Only tree booster
                 })
```

### Advanced Optimization Options

```python
# GPU acceleration
model = xgb_auto(X, y, use_gpu=True, strategy='fast')

# Validation split (faster than cross-validation)
model = xgb_auto(X, y, validation_split=0.2, n_trials=100)

# Early stopping for efficiency
model = xgb_auto(X, y, early_stopping_rounds=10, validation_split=0.2)

# Multi-fidelity optimization (advanced)
model = xgb_auto(X, y, multi_fidelity=True, fidelity_percentile=0.7)
```

## Model Evaluation

SimpleMLR provides comprehensive tools for understanding your model's performance.

### Quick Analysis

```python
# Simple 2-panel plot
model.quick_graph()  # Uses training data
model.quick_graph(X_test, y_test)  # Uses test data
```

### Comprehensive Analysis

```python
# Detailed 6-panel analysis
fig, metrics = model.plot_analysis(
    X_test, y_test,
    title="My Model Performance Analysis",
    style='modern',           # Professional appearance
    save_path='analysis.png'  # Save for reports
)

# Access detailed metrics
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"Outliers detected: {metrics['outliers_count']}")
```

### Feature Importance

```python
# Get feature importance rankings
importance_df = model.get_feature_importance(top_n=15)
print(importance_df)

# Create custom importance plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance Score')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Model Comparison

```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Compare multiple models
models = {
    'XGBoost Fast': xgb_auto(X_train, y_train, strategy='fast', n_trials=25, verbose=0),
    'LightGBM Fast': lgbm_auto(X_train, y_train, strategy='fast', n_trials=25, verbose=0),
    'XGBoost Balanced': xgb_auto(X_train, y_train, strategy='balanced', n_trials=50, verbose=0),
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Training Time': f"{model._training_time:.1f}s" if hasattr(model, '_training_time') else 'N/A'
    })

comparison_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
print(comparison_df)
```

### Cross-Validation Analysis

```python
from sklearn.model_selection import cross_val_score

# Test model stability with cross-validation
cv_scores = cross_val_score(model.model_, X, y, cv=5, scoring='r2')

print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Check if model is stable
if cv_scores.std() < 0.1:
    print("✅ Model is stable across different data splits")
else:
    print("⚠️ Model performance varies significantly - consider more regularization")
```

## Production Deployment

### Saving and Loading Models

```python
import pickle
import joblib

# Method 1: Using pickle (standard)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load saved model
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Method 2: Using joblib (better for sklearn models)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

# Verify loaded model works
test_predictions = loaded_model.predict(X_test[:5])
print(f"Test predictions: {test_predictions}")
```

### Model Versioning

```python
import datetime
import json

# Save model with metadata
model_info = {
    'model_type': 'XGBRegressor',
    'training_date': datetime.datetime.now().isoformat(),
    'training_samples': len(X_train),
    'features': list(X_train.columns),
    'performance': {
        'r2_score': float(model.training_metrics_['r2_score']),
        'rmse': float(model.training_metrics_['rmse'])
    },
    'parameters': model.model_.get_params() if hasattr(model.model_, 'get_params') else {}
}

# Save model and metadata
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'model_{timestamp}.pkl'
metadata_filename = f'metadata_{timestamp}.json'

with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

with open(metadata_filename, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"Model saved: {model_filename}")
print(f"Metadata saved: {metadata_filename}")
```

### Batch Prediction Pipeline

```python
def predict_batch(model, input_file, output_file, batch_size=1000):
    """Process large datasets in batches to avoid memory issues."""
    
    # Read data in chunks
    chunk_predictions = []
    
    for chunk in pd.read_csv(input_file, chunksize=batch_size):
        # Preprocess chunk (if needed)
        X_chunk = chunk.drop('id', axis=1)  # Assuming 'id' column exists
        
        # Make predictions
        predictions = model.predict(X_chunk)
        
        # Store results with IDs
        chunk_results = pd.DataFrame({
            'id': chunk['id'],
            'prediction': predictions
        })
        chunk_predictions.append(chunk_results)
    
    # Combine all results
    final_results = pd.concat(chunk_predictions, ignore_index=True)
    final_results.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    return final_results

# Use the pipeline
results = predict_batch(model, 'large_dataset.csv', 'predictions.csv')
```

### API Deployment Example

```python
# Simple Flask API example
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load model once at startup
MODEL = joblib.load('best_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = MODEL.predict(df)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

## Best Practices

### Data Preparation

1. **Always examine your data first:**
   ```python
   # Understand your dataset
   print(f"Dataset shape: {df.shape}")
   print(f"Missing values:\n{df.isnull().sum()}")
   print(f"Data types:\n{df.dtypes}")
   print(f"Target distribution:\n{df['target'].describe()}")
   ```

2. **Handle missing values explicitly:**
   ```python
   # Don't let SimpleMLR fail due to missing values
   df = df.dropna()  # Or use proper imputation
   ```

3. **Consider feature engineering:**
   ```python
   # Create meaningful features before training
   df['price_per_sqft'] = df['price'] / df['sqft']
   df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
   ```

### Model Selection

1. **Start with 'fast' strategy for exploration:**
   ```python
   # Quick model to understand baseline performance
   quick_model = xgb_auto(X, y, strategy='fast', n_trials=20, verbose=0)
   print(f"Baseline R²: {quick_model.training_metrics_['r2_score']:.4f}")
   ```

2. **Use appropriate strategy for your goal:**
   ```python
   # Development: fast
   # Production: stable  
   # Competition: aggressive
   ```

3. **Compare multiple algorithms:**
   ```python
   # Don't assume one algorithm is best
   algorithms = ['xgb_auto', 'lgbm_auto', 'gbm_auto']
   for alg_name in algorithms:
       alg_func = globals()[alg_name]  # Get function by name
       model = alg_func(X, y, strategy='fast', n_trials=20, verbose=0)
       print(f"{alg_name}: {model.training_metrics_['r2_score']:.4f}")
   ```

### Performance Optimization

1. **Use validation split for large datasets:**
   ```python
   # Faster than cross-validation
   model = xgb_auto(X, y, validation_split=0.2)
   ```

2. **Enable GPU when available:**
   ```python
   # Much faster for large datasets
   model = xgb_auto(X, y, use_gpu=True)
   ```

3. **Adjust trials based on dataset size:**
   ```python
   # More trials for important models, fewer for exploration
   n_trials = min(100, max(20, len(X_train) // 1000))
   model = xgb_auto(X, y, n_trials=n_trials)
   ```

### Model Validation

1. **Always validate on unseen data:**
   ```python
   # Don't evaluate on training data
   model.quick_graph(X_test, y_test)  # Not X_train, y_train
   ```

2. **Check for overfitting:**
   ```python
   train_pred = model.predict(X_train)
   test_pred = model.predict(X_test)
   
   train_r2 = r2_score(y_train, train_pred)
   test_r2 = r2_score(y_test, test_pred)
   
   print(f"Training R²: {train_r2:.4f}")
   print(f"Test R²: {test_r2:.4f}")
   
   if train_r2 - test_r2 > 0.1:
       print("⚠️ Possible overfitting - consider more regularization")
   ```

3. **Analyze residuals:**
   ```python
   # Look for patterns in residuals
   model.plot_analysis(X_test, y_test)
   # Check the residual plot for patterns
   ```

## Common Workflows

### Workflow 1: Quick Experiment

```python
# Goal: Quickly see if ML can solve your problem
from simple_mlr import xgb_regressor

# 1. Load and prepare data
df = pd.read_csv('data.csv')
X, y = df.drop('target', axis=1), df['target']

# 2. Quick model
model = xgb_regressor(X, y)

# 3. Check results
model.quick_graph()
print(f"R² Score: {model.training_metrics_['r2_score']:.4f}")

# Decision: If R² > 0.5, worth pursuing further
```

### Workflow 2: Model Development

```python
# Goal: Develop the best possible model
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto
from sklearn.model_selection import train_test_split

# 1. Proper train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Try multiple algorithms with different strategies
candidates = [
    ('XGBoost Balanced', lambda: xgb_auto(X_train, y_train, strategy='balanced', n_trials=50)),
    ('LightGBM Fast', lambda: lgbm_auto(X_train, y_train, strategy='fast', n_trials=50)),
    ('XGBoost Aggressive', lambda: xgb_auto(X_train, y_train, strategy='aggressive', n_trials=75)),
]

results = []
models = {}

for name, model_func in candidates:
    print(f"\nTraining {name}...")
    model = model_func()
    
    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    
    results.append({'Model': name, 'Test R²': test_r2})
    models[name] = model
    
    print(f"{name} Test R²: {test_r2:.4f}")

# 3. Select best model
best_result = max(results, key=lambda x: x['Test R²'])
best_model = models[best_result['Model']]

print(f"\nBest model: {best_result['Model']}")

# 4. Final analysis
best_model.plot_analysis(X_test, y_test, 
                        title=f"Final Model: {best_result['Model']}")

# 5. Save best model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```

### Workflow 3: Production Deployment

```python
# Goal: Deploy a reliable model to production
from simple_mlr import xgb_auto

# 1. Use stable strategy
model = xgb_auto(X_train, y_train, 
                 strategy='stable',      # Reliable parameters
                 n_trials=100,          # Thorough search
                 validation_split=0.2,  # Consistent validation
                 random_state=42)       # Reproducible

# 2. Comprehensive validation
test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, test_pred)

# Cross-validation for stability check
cv_scores = cross_val_score(model.model_, X, y, cv=5, scoring='r2')

print(f"Test R²: {test_r2:.4f}")
print(f"CV R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 3. Model documentation
model_doc = {
    'algorithm': 'XGBoost',
    'strategy': 'stable',
    'test_r2': float(test_r2),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'feature_count': len(X.columns),
    'training_samples': len(X_train)
}

# 4. Save with version info
import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

with open(f'production_model_{timestamp}.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(f'model_info_{timestamp}.json', 'w') as f:
    json.dump(model_doc, f, indent=2)
```

## Performance Tips

### Speed Optimization

```python
# 1. Use validation split instead of cross-validation
model = xgb_auto(X, y, validation_split=0.2)  # ~5x faster

# 2. Reduce number of trials for quick iterations  
model = xgb_auto(X, y, n_trials=25)  # vs default 50

# 3. Use 'fast' strategy for experimentation
model = xgb_auto(X, y, strategy='fast')

# 4. Enable GPU acceleration
model = xgb_auto(X, y, use_gpu=True)

# 5. Use fewer CV folds
model = xgb_auto(X, y, cv_folds=3)  # vs default 5
```

### Memory Optimization

```python
# 1. Process data in chunks for large datasets
def train_on_large_data(file_path, target_col, model_func):
    # Sample data first to get feature info
    sample = pd.read_csv(file_path, nrows=1000)
    features = [col for col in sample.columns if col != target_col]
    
    # Train on sample
    X_sample = sample[features]
    y_sample = sample[target_col]
    
    model = model_func(X_sample, y_sample)
    return model

# 2. Use LightGBM for memory efficiency
model = lgbm_auto(X, y)  # More memory efficient than XGBoost

# 3. Reduce precision if acceptable
X = X.astype('float32')  # vs float64
```

### Quality Optimization

```python
# 1. Use more trials for important models
model = xgb_auto(X, y, n_trials=200)

# 2. Use aggressive strategy for maximum performance
model = xgb_auto(X, y, strategy='aggressive')

# 3. Use more CV folds for small datasets
model = xgb_auto(X, y, cv_folds=10)

# 4. Enable early stopping to avoid overfitting
model = xgb_auto(X, y, early_stopping_rounds=20, validation_split=0.2)
```

This user guide covers the essential knowledge needed to effectively use SimpleMLR in real-world projects. Remember: start simple, iterate quickly, and let SimpleMLR handle the complexity while you focus on solving your business problem!