"""
Indi-ML: Indian Stock Market Analysis Platform

A comprehensive platform for Indian stock market analysis combining:
- Traditional technical analysis
- Machine learning models (Random Forest, ARIMA, LSTM)
- Enhanced ML models with advanced features
- Sentiment analysis from news sources
- Risk assessment and fundamental analysis
- DCF valuation and momentum analysis

Modules:
- ingest: Data ingestion from Yahoo Finance
- features: Technical indicators and feature engineering
- models: ML models (original and enhanced versions)
- pipeline: Analysis pipelines
- sentiment: News sentiment analysis
- risk: Risk assessment
- fundamental: Fundamental analysis
- momentum: Momentum analysis
- dcf: DCF valuation

Author: ML Team
Version: 2.0 (Enhanced)
"""

# Version information
__version__ = "2.0.0"
__author__ = "ML Team"
__description__ = "Enhanced Indian Stock Market Analysis Platform"

# Import main modules for easy access
try:
    from . import ingest
    from . import features
    from . import pipeline
    from . import enhanced_pipeline
    from . import sentiment
    from . import risk
    from . import fundamental
    from . import momentum
    from . import dcf
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")

# Import model modules
try:
    from .models import ensemble, arima, lstm
    from .models import enhanced_ensemble, enhanced_arima, enhanced_lstm
except ImportError as e:
    print(f"Warning: Some model modules could not be imported: {e}")

# Convenience functions
def get_version():
    """Get the current version of indi-ml."""
    return __version__

def list_available_models():
    """List all available ML models."""
    original_models = ['Random Forest', 'ARIMA', 'LSTM']
    enhanced_models = ['Enhanced Ensemble', 'Enhanced ARIMA', 'Enhanced LSTM']
    
    return {
        'original': original_models,
        'enhanced': enhanced_models
    }

def get_model_info():
    """Get information about available models."""
    return {
        'original_models': {
            'Random Forest': 'Basic ensemble learning with fixed parameters',
            'ARIMA': 'Time series model with simple parameter selection',
            'LSTM': 'Basic deep learning model with 2-layer architecture'
        },
        'enhanced_models': {
            'Enhanced Ensemble': 'Multi-algorithm ensemble with hyperparameter tuning',
            'Enhanced ARIMA': 'Seasonal ARIMA with comprehensive diagnostics',
            'Enhanced LSTM': 'Advanced architectures with attention and regularization'
        }
    }

# Module metadata
__all__ = [
    'ingest',
    'features', 
    'pipeline',
    'enhanced_pipeline',
    'sentiment',
    'risk',
    'fundamental',
    'momentum',
    'dcf',
    'get_version',
    'list_available_models',
    'get_model_info'
]
