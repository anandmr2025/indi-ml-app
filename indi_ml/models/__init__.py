"""
Models Module for Indi-ML Platform

This module contains both original and enhanced ML models for stock price prediction:

Original Models:
- ensemble.py: Basic Random Forest regression
- arima.py: Simple ARIMA time series model
- lstm.py: Basic LSTM neural network

Enhanced Models:
- enhanced_ensemble.py: Multi-algorithm ensemble with hyperparameter tuning
- enhanced_arima.py: Seasonal ARIMA with comprehensive diagnostics
- enhanced_lstm.py: Advanced LSTM architectures with attention mechanisms

The enhanced models provide significant improvements in:
- Accuracy: Better prediction performance
- Robustness: Comprehensive error handling and validation
- Features: Advanced preprocessing and feature engineering
- Validation: Time series specific validation methods

Author: ML Team
Version: 2.0 (Enhanced)
"""

# Import original models
try:
    from . import ensemble
    from . import arima
    from . import lstm
except ImportError as e:
    print(f"Warning: Could not import original models: {e}")

# Import enhanced models
try:
    from . import enhanced_ensemble
    from . import enhanced_arima
    from . import enhanced_lstm
except ImportError as e:
    print(f"Warning: Could not import enhanced models: {e}")

# Convenience functions for model access
def get_original_models():
    """Get dictionary of original model training functions."""
    try:
        return {
            'random_forest': ensemble.train_rf,
            'arima': arima.train_arima_for_pipeline,
            'lstm': lstm.train_lstm_for_pipeline
        }
    except:
        return {}

def get_enhanced_models():
    """Get dictionary of enhanced model training functions."""
    try:
        return {
            'enhanced_ensemble': enhanced_ensemble.train_enhanced_ensemble,
            'enhanced_arima': enhanced_arima.train_enhanced_arima_for_pipeline,
            'enhanced_lstm': enhanced_lstm.train_enhanced_lstm_for_pipeline
        }
    except:
        return {}

def get_all_models():
    """Get dictionary of all available model training functions."""
    models = {}
    models.update(get_original_models())
    models.update(get_enhanced_models())
    return models

def list_model_features():
    """List features of each model type."""
    return {
        'original': {
            'random_forest': [
                'Basic Random Forest with 200 estimators',
                'Simple 80/20 train/test split',
                'No hyperparameter tuning',
                'Basic error handling'
            ],
            'arima': [
                'Simple ARIMA parameter selection',
                'Basic grid search (3x2x3)',
                'AIC-based model selection',
                'Limited error handling'
            ],
            'lstm': [
                '2-layer LSTM architecture',
                'Basic preprocessing with MinMaxScaler',
                'Fixed lookback window',
                'Simple regularization'
            ]
        },
        'enhanced': {
            'enhanced_ensemble': [
                'Multiple algorithms: RF, GBM, XGBoost, LightGBM',
                'Advanced hyperparameter tuning',
                'TimeSeriesSplit validation',
                'Feature scaling and selection',
                'Comprehensive error handling'
            ],
            'enhanced_arima': [
                'Seasonal ARIMA (SARIMA) support',
                'Comprehensive parameter search',
                'Stationarity testing (ADF + KPSS)',
                'Advanced diagnostics (Ljung-Box)',
                'Walk-forward validation'
            ],
            'enhanced_lstm': [
                'Multiple architectures (Bidirectional, CNN-LSTM, Attention)',
                'Advanced regularization (Dropout, BatchNorm, L1/L2)',
                'Learning rate scheduling and early stopping',
                'Ensemble methods',
                'Flexible preprocessing options'
            ]
        }
    }

def get_model_recommendations():
    """Get recommendations for when to use each model."""
    return {
        'enhanced_ensemble': {
            'best_for': ['High accuracy requirements', 'Multiple feature types', 'Robust predictions'],
            'use_when': 'You need the most accurate predictions and have sufficient computational resources'
        },
        'enhanced_arima': {
            'best_for': ['Time series patterns', 'Seasonal data', 'Interpretable models'],
            'use_when': 'Your data shows clear time series patterns and you need interpretable results'
        },
        'enhanced_lstm': {
            'best_for': ['Complex patterns', 'Long sequences', 'Non-linear relationships'],
            'use_when': 'Your data has complex non-linear patterns and you have sufficient training data'
        },
        'original_models': {
            'best_for': ['Quick prototyping', 'Limited resources', 'Simple baselines'],
            'use_when': 'You need quick results or are working with limited computational resources'
        }
    }

# Module metadata
__all__ = [
    # Original models
    'ensemble',
    'arima', 
    'lstm',
    # Enhanced models
    'enhanced_ensemble',
    'enhanced_arima',
    'enhanced_lstm',
    # Utility functions
    'get_original_models',
    'get_enhanced_models',
    'get_all_models',
    'list_model_features',
    'get_model_recommendations'
]
