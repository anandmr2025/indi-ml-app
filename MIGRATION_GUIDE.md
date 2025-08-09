# Migration Guide: Integrating Enhanced ML Models

This guide explains how to integrate the enhanced ML models into your existing indi-ml project and migrate from the original models.

## Overview of Changes

The enhanced ML models have been added to your project with the following new files:

### New Files Created:
1. **Enhanced Models**:
   - `indi_ml/models/enhanced_ensemble.py` - Multi-algorithm ensemble with hyperparameter tuning
   - `indi_ml/models/enhanced_arima.py` - Seasonal ARIMA with comprehensive diagnostics
   - `indi_ml/models/enhanced_lstm.py` - Advanced LSTM architectures

2. **Enhanced Pipeline**:
   - `indi_ml/enhanced_pipeline.py` - Enhanced pipeline with model caching and performance tracking

3. **Enhanced Application**:
   - `enhanced_app.py` - Enhanced Streamlit app with support for both original and enhanced models

4. **Configuration and Documentation**:
   - `config.py` - Comprehensive configuration management
   - `indi_ml/__init__.py` - Updated module initialization
   - `indi_ml/models/__init__.py` - Models module initialization
   - `ML_MODELS_ENHANCEMENT_SUMMARY.md` - Detailed enhancement summary

## Migration Options

You have three migration options:

### Option 1: Gradual Migration (Recommended)
Keep both original and enhanced models, gradually transition:

```python
# Use enhanced app that supports both
streamlit run enhanced_app.py

# Or use enhanced pipeline alongside original
from indi_ml.enhanced_pipeline import run_enhanced_pipeline
from indi_ml.pipeline import run_pipeline

# Compare results
enhanced_results = run_enhanced_pipeline("RELIANCE", "1y")
original_results = run_pipeline("RELIANCE", "1y")
```

### Option 2: Direct Replacement
Replace original models with enhanced versions:

1. **Update app.py imports**:
```python
# Replace this:
from indi_ml.pipeline import run_pipeline

# With this:
from indi_ml.enhanced_pipeline import run_enhanced_pipeline as run_pipeline
```

2. **Update any custom scripts** that use the models directly

### Option 3: Hybrid Approach
Use enhanced models for production, keep original for comparison:

```python
# In your main application
import config
if config.ENVIRONMENT == 'production':
    from indi_ml.enhanced_pipeline import run_enhanced_pipeline as run_pipeline
else:
    from indi_ml.pipeline import run_pipeline
```

## Step-by-Step Migration

### Step 1: Test Enhanced Models
First, test the enhanced models to ensure they work in your environment:

```bash
# Test enhanced ensemble
python -m indi_ml.models.enhanced_ensemble

# Test enhanced ARIMA
python -m indi_ml.models.enhanced_arima

# Test enhanced LSTM
python -m indi_ml.models.enhanced_lstm

# Test enhanced pipeline
python -m indi_ml.enhanced_pipeline
```

### Step 2: Update Your Application

#### Option A: Use Enhanced App (Easiest)
```bash
# Run the enhanced Streamlit app
streamlit run enhanced_app.py
```

#### Option B: Update Existing App
Add enhanced model support to your existing `app.py`:

```python
# Add at the top of app.py
import streamlit as st
from indi_ml.pipeline import run_pipeline
from indi_ml.enhanced_pipeline import run_enhanced_pipeline  # Add this

# Add pipeline selection in sidebar
pipeline_type = st.sidebar.selectbox("Pipeline Type", ["Original", "Enhanced", "Compare"])

# Update the analysis button logic
if st.sidebar.button("Run / Refresh Analysis"):
    if pipeline_type == "Enhanced":
        state = run_enhanced_pipeline(symbol, period=period)
    elif pipeline_type == "Compare":
        # Run both and compare
        original_state = run_pipeline(symbol, period=period)
        enhanced_state = run_enhanced_pipeline(symbol, period=period)
        # Add comparison logic
    else:
        state = run_pipeline(symbol, period=period)
```

### Step 3: Configuration Management

Use the new configuration system:

```python
import config

# Get model configuration
ensemble_config = config.get_model_config('enhanced_ensemble')
pipeline_config = config.get_pipeline_config()

# Update configuration if needed
config.update_config('model', 'enhanced_lstm', {'epochs': 100})
```

### Step 4: Performance Monitoring

The enhanced pipeline includes performance tracking:

```python
from indi_ml.enhanced_pipeline import run_enhanced_pipeline

results = run_enhanced_pipeline("RELIANCE", "1y")

# Access performance metrics
execution_time = results['execution_time']
best_model = results['best_model']
performance_tracker = results['performance_tracker']

# Get performance summary
summary = performance_tracker.get_performance_summary()
print(summary)
```

## Key Differences to Be Aware Of

### 1. Return Value Changes
Enhanced pipeline returns additional fields:

```python
# Original pipeline returns:
{
    'price_df': df,
    'arima_rmse': float,
    'lstm_rmse': float,
    'rf_rmse': float,
    'sentiment': float,
    'risk_score': float,
    # ... other fields
}

# Enhanced pipeline adds:
{
    # All original fields plus:
    'enhanced_models': dict,
    'best_model': dict,
    'performance_tracker': object,
    'execution_time': float,
    'feature_count': int,
    'data_points': int,
    'pipeline_version': str
}
```

### 2. Model Caching
Enhanced models use caching by default:

```python
# Disable caching if needed
results = run_enhanced_pipeline("RELIANCE", "1y", use_cache=False)

# Clear cache manually
import shutil
shutil.rmtree("data/model_cache", ignore_errors=True)
```

### 3. Configuration Dependencies
Enhanced models use the config.py file. Ensure it's properly set up:

```python
# Check configuration
import config
print(config.get_all_config())
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```python
# If you get import errors, ensure __init__.py files are present
# Check that all new files are in the correct directories
```

#### 2. Memory Issues
```python
# If enhanced models use too much memory, reduce parameters:
config.update_config('model', 'enhanced_ensemble', {'cv_folds': 3, 'n_iter': 10})
config.update_config('model', 'enhanced_lstm', {'epochs': 20, 'batch_size': 16})
```

#### 3. Slow Performance
```python
# Enable caching and reduce model complexity:
config.update_config('pipeline', 'use_cache', True)
config.update_config('model', 'enhanced_ensemble', {'use_randomized_search': True})
```

#### 4. Model Training Failures
```python
# Check data requirements:
# - Enhanced ensemble needs at least 50 data points
# - Enhanced ARIMA needs at least 50 data points
# - Enhanced LSTM needs at least lookback_window + 20 data points

# Use longer periods if needed:
results = run_enhanced_pipeline("RELIANCE", "2y")  # Instead of "1y"
```

## Testing Your Migration

### 1. Functionality Test
```python
# Test basic functionality
from indi_ml.enhanced_pipeline import run_enhanced_pipeline

try:
    results = run_enhanced_pipeline("TCS", "1y")
    print("✅ Enhanced pipeline working correctly")
    print(f"Best model: {results.get('best_model', {}).get('model', 'None')}")
except Exception as e:
    print(f"❌ Error: {e}")
```

### 2. Performance Test
```python
# Compare performance
from indi_ml.enhanced_pipeline import compare_pipelines

comparison = compare_pipelines("RELIANCE", "1y")
print("Performance comparison:")
print(f"Original execution time: {comparison['original']['execution_time']:.2f}s")
print(f"Enhanced execution time: {comparison['enhanced']['execution_time']:.2f}s")
```

### 3. Accuracy Test
```python
# Check if enhanced models perform better
enhanced_results = run_enhanced_pipeline("RELIANCE", "1y")
original_results = run_pipeline("RELIANCE", "1y")

print("RMSE Comparison:")
print(f"Original ARIMA: {original_results['arima_rmse']:.2f}")
print(f"Enhanced ARIMA: {enhanced_results['arima_rmse']:.2f}")
print(f"Original LSTM: {original_results['lstm_rmse']:.2f}")
print(f"Enhanced LSTM: {enhanced_results['lstm_rmse']:.2f}")
```

## Rollback Plan

If you need to rollback to original models:

### 1. Quick Rollback
```python
# Simply use the original app and pipeline
streamlit run app.py  # Instead of enhanced_app.py
```

### 2. Remove Enhanced Files (if needed)
```bash
# Remove enhanced model files
rm indi_ml/models/enhanced_*.py
rm indi_ml/enhanced_pipeline.py
rm enhanced_app.py

# Restore original config.py if needed
echo " " > config.py
```

### 3. Restore Original __init__.py
```python
# Restore original indi_ml/__init__.py
echo " " > indi_ml/__init__.py
```

## Best Practices

### 1. Start with Enhanced App
- Use `enhanced_app.py` to test all features
- Compare original vs enhanced models side by side

### 2. Monitor Performance
- Check execution times and memory usage
- Use caching for production environments

### 3. Gradual Adoption
- Start with one enhanced model at a time
- Monitor accuracy improvements
- Gradually increase usage

### 4. Configuration Management
- Use `config.py` for all settings
- Set environment-specific configurations
- Document any custom changes

## Support and Further Development

### Getting Help
1. Check the `ML_MODELS_ENHANCEMENT_SUMMARY.md` for detailed technical information
2. Review model-specific documentation in each enhanced model file
3. Use the built-in help functions:
   ```python
   import indi_ml
   print(indi_ml.get_model_info())
   print(indi_ml.list_available_models())
   ```

### Future Enhancements
The enhanced models are designed to be extensible. You can:
- Add new algorithms to the enhanced ensemble
- Implement custom LSTM architectures
- Add new validation methods
- Integrate additional data sources

### Contributing
If you make improvements to the enhanced models:
1. Update the relevant model files
2. Update configuration in `config.py`
3. Update documentation
4. Test thoroughly before deployment

---

This migration guide should help you successfully integrate the enhanced ML models into your existing workflow. Start with the enhanced app to see all features, then gradually migrate your existing code as needed.
