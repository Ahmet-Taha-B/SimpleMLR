# SimpleMLR Quick Reference

A condensed cheat sheet for SimpleMLR usage.

## 🚀 Installation
```bash
pip install -e .
pip install xgboost lightgbm  # Optional dependencies
```

## 💨 Quick Start (3 Lines!)
```python
from simple_mlr import xgb_regressor
model = xgb_regressor(X_train, y_train)
model.quick_graph()  # Instant analysis!
```

## 📊 Three Usage Methods

### 1️⃣ One-Line Functions (Fastest)
```python
from simple_mlr import xgb_regressor, lgbm_regressor, gbm_regressor

xgb_model = xgb_regressor(X, y)
lgbm_model = lgbm_regressor(X, y)
gbm_model = gbm_regressor(X, y)
```

### 2️⃣ Class-Based (More Control)
```python
from simple_mlr import XGBRegressor, LGBMRegressor, GBMRegressor

model = XGBRegressor(n_estimators=200, max_depth=4)
model.fit(X, y)
model.plot_analysis(X_test, y_test)
```

### 3️⃣ Auto-Optimization (Best Performance)
```python
from simple_mlr import xgb_auto, lgbm_auto, gbm_auto

model = xgb_auto(X, y, strategy='fast', n_trials=50, verbose=1)
```

## 🎯 Algorithm Selection

| Algorithm | Best For | Strengths |
|-----------|----------|-----------|
| **XGBoost** | General use | Proven performance, handles missing values |
| **LightGBM** | Large datasets | Fast training, low memory |
| **Sklearn GBM** | Reliability | Stable, no dependencies |

```python
# Quick comparison
xgb_model = xgb_auto(X, y, strategy='fast', n_trials=25, verbose=0)
lgbm_model = lgbm_auto(X, y, strategy='fast', n_trials=25, verbose=0) 
gbm_model = gbm_auto(X, y, strategy='fast', n_trials=25, verbose=0)
```

## 🎛️ Strategy System

| Strategy | Speed | Performance | Best For |
|----------|-------|-------------|----------|
| `'fast'` | ⚡⚡⚡ | ⭐⭐ | Prototyping |
| `'balanced'` | ⚡⚡ | ⭐⭐⭐ | Most use cases |
| `'stable'` | ⚡ | ⭐⭐⭐ | Production |
| `'aggressive'` | ⚡ | ⭐⭐⭐⭐ | Competition |

```python
model = xgb_auto(X, y, strategy='balanced')  # Recommended default
```

## 🔧 Common Parameters

### Auto-Optimization
```python
model = xgb_auto(
    X, y,
    strategy='balanced',        # Optimization strategy
    n_trials=50,               # Number of trials
    verbose=1,                 # Progress display (0-2)
    use_gpu=False,             # GPU acceleration
    validation_split=0.2,      # Holdout validation (faster)
    early_stopping_rounds=10   # Stop early if no improvement
)
```

### Parameter Overrides
```python
model = xgb_auto(X, y, 
                 strategy='fast',
                 override_params={
                     'max_depth': 6,              # Fixed value
                     'learning_rate': (0.01, 0.2), # Custom range
                     'n_estimators': (100, 300)   # Custom range
                 })
```

### Manual Settings
```python
model = XGBRegressor(
    n_estimators=200,          # Number of trees
    max_depth=4,               # Tree depth
    learning_rate=0.1,         # Learning rate
    scale_features=True,       # Auto-scaling
    handle_categorical=True,   # Auto-encoding
    use_gpu=False             # GPU usage
)
```

## 📈 Visualization

### Quick Analysis
```python
model.quick_graph()                    # Training data
model.quick_graph(X_test, y_test)      # Test data
```

### Comprehensive Analysis
```python
fig, metrics = model.plot_analysis(
    X_test, y_test,
    title="My Model Analysis",
    style='modern',                    # 'modern', 'classic', 'minimal'
    save_path='analysis.png'
)
```

### Feature Importance
```python
importance = model.get_feature_importance(top_n=10)
print(importance)
```

## 🎯 Model Evaluation

### Basic Metrics
```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model.model_, X, y, cv=5, scoring='r2')
print(f"CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 💾 Save/Load Models

### Save Model
```python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Load Model
```python
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

predictions = loaded_model.predict(X_new)
```

## 🔥 Performance Tips

### Speed Optimization
```python
# Use validation split (5x faster than CV)
model = xgb_auto(X, y, validation_split=0.2)

# Reduce trials for quick tests
model = xgb_auto(X, y, n_trials=25)

# Use fast strategy
model = xgb_auto(X, y, strategy='fast')

# Enable GPU
model = xgb_auto(X, y, use_gpu=True)
```

### Memory Optimization
```python
# Use LightGBM for large datasets
model = lgbm_auto(X, y)

# Reduce precision
X = X.astype('float32')
```

### Quality Optimization
```python
# More trials for important models
model = xgb_auto(X, y, n_trials=200)

# Aggressive strategy for competitions
model = xgb_auto(X, y, strategy='aggressive')
```

## 🛠️ Troubleshooting

### Common Issues

**ImportError: No module named 'xgboost'**
```bash
pip install xgboost
```

**ImportError: No module named 'lightgbm'**
```bash
pip install lightgbm
```

**Memory errors**
```python
# Use validation split instead of CV
model = xgb_auto(X, y, validation_split=0.2)

# Use LightGBM
model = lgbm_auto(X, y)
```

**GPU not working**
```python
import xgboost as xgb
print(f"GPU available: {xgb.gpu.is_available()}")
```

## 📋 Complete Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from simple_mlr import xgb_auto

# 1. Load data
df = pd.read_csv('data.csv')
X, y = df.drop('target', axis=1), df['target']

# 2. Split data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train optimized model
model = xgb_auto(X_train, y_train, strategy='balanced', n_trials=50, verbose=1)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")

# 5. Analyze
model.plot_analysis(X_test, y_test, save_path='analysis.png')

# 6. Save model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## 🚀 Production Workflow

```python
# 1. Development (fast iteration)
dev_model = xgb_auto(X, y, strategy='fast', n_trials=25, verbose=0)

# 2. Testing (thorough evaluation)
test_model = xgb_auto(X, y, strategy='balanced', n_trials=75, verbose=1)

# 3. Production (maximum reliability)
prod_model = xgb_auto(X, y, strategy='stable', n_trials=100, 
                      validation_split=0.2, random_state=42)

# 4. Deploy
with open('production_model.pkl', 'wb') as f:
    pickle.dump(prod_model, f)
```

## 🎓 Learning Path

### Beginner
1. Start with `xgb_regressor(X, y)`
2. Use `quick_graph()` for analysis
3. Try different algorithms: `lgbm_regressor()`, `gbm_regressor()`

### Intermediate  
1. Use auto-optimization: `xgb_auto(X, y, strategy='balanced')`
2. Try different strategies: `'fast'`, `'stable'`, `'aggressive'`
3. Use comprehensive analysis: `plot_analysis()`

### Advanced
1. Parameter overrides with `override_params`
2. GPU acceleration with `use_gpu=True`  
3. Production deployment with model versioning

---

**Remember**: Start simple, iterate quickly, let SimpleMLR handle the complexity! 🚀