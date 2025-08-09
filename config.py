"""
Configuration file for Indi-ML Platform

This file contains configuration settings for:
- Model selection and parameters
- Pipeline execution settings
- Data processing options
- Performance and caching settings

Author: ML Team
Version: 2.0 (Enhanced)
"""

import os
from pathlib import Path

# Project settings
PROJECT_NAME = "Indi-ML Enhanced Platform"
VERSION = "2.0.0"
AUTHOR = "ML Team"

# Data settings
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
LOGS_DIR = Path("logs")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Enhanced Ensemble settings
    'enhanced_ensemble': {
        'use_scaling': True,
        'feature_selection': True,
        'n_features': None,  # None for automatic selection
        'cv_folds': 5,
        'scoring': 'neg_mean_squared_error',
        'use_randomized_search': True,
        'n_iter': 20,
        'test_size': 0.2
    },
    
    # Enhanced ARIMA settings
    'enhanced_arima': {
        'seasonal': False,
        'seasonal_periods': 12,
        'information_criterion': 'aic',
        'max_p': 5,
        'max_d': 2,
        'max_q': 5,
        'seasonal_max_P': 2,
        'seasonal_max_D': 1,
        'seasonal_max_Q': 2,
        'test_size': 0.2
    },
    
    # Enhanced LSTM settings
    'enhanced_lstm': {
        'lookback_window': 30,
        'prediction_horizon': 1,
        'architecture': 'bidirectional',  # 'simple', 'bidirectional', 'cnn_lstm'
        'use_attention': False,
        'scaler_type': 'robust',  # 'minmax', 'standard', 'robust'
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'test_size': 0.2
    },
    
    # Original model settings (for comparison)
    'original': {
        'random_forest': {
            'n_estimators': 200,
            'random_state': 42,
            'test_size': 0.2
        },
        'arima': {
            'max_p': 3,
            'max_d': 2,
            'max_q': 3,
            'test_size': 30
        },
        'lstm': {
            'epochs': 30,
            'lookback': 60,
            'test_size': 0.2
        }
    }
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'use_cache': True,
    'cache_expiry_hours': 24,
    'model_selection_strategy': 'best_cv',  # 'best_cv', 'best_rmse', 'ensemble_all'
    'enable_logging': True,
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'parallel_execution': False,
    'max_workers': 4
}

# Data processing configuration
DATA_CONFIG = {
    'default_period': '1y',
    'default_interval': '1d',
    'default_exchange': 'NSE',
    'cache_data': True,
    'force_refresh': False,
    'min_data_points': 50,
    'feature_engineering': {
        'add_technical_indicators': True,
        'add_lagged_features': True,
        'add_rolling_features': True,
        'rolling_windows': [5, 10, 20, 50]
    }
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'track_execution_time': True,
    'track_memory_usage': False,
    'save_performance_history': True,
    'performance_history_file': LOGS_DIR / 'performance_history.json',
    'model_comparison_enabled': True
}

# Visualization settings
VIZ_CONFIG = {
    'default_theme': 'plotly_white',
    'color_palette': ['#FF6B6B', '#9B59B6', '#4ECDC4', '#45B7D1', '#96CEB4'],
    'figure_width': 800,
    'figure_height': 400,
    'show_advanced_metrics': True,
    'show_model_details': False
}

# API and external service settings
API_CONFIG = {
    'yahoo_finance': {
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1
    },
    'news_sources': {
        'max_items_per_source': 10,
        'timeout': 15,
        'retry_attempts': 2
    }
}

# Environment-specific settings
ENVIRONMENT = os.getenv('INDI_ML_ENV', 'development')  # 'development', 'production', 'testing'

if ENVIRONMENT == 'production':
    # Production settings - more conservative
    MODEL_CONFIG['enhanced_ensemble']['cv_folds'] = 3
    MODEL_CONFIG['enhanced_ensemble']['n_iter'] = 10
    MODEL_CONFIG['enhanced_lstm']['epochs'] = 30
    PIPELINE_CONFIG['enable_logging'] = True
    PIPELINE_CONFIG['log_level'] = 'WARNING'
    
elif ENVIRONMENT == 'development':
    # Development settings - more verbose
    PIPELINE_CONFIG['enable_logging'] = True
    PIPELINE_CONFIG['log_level'] = 'DEBUG'
    VIZ_CONFIG['show_model_details'] = True
    
elif ENVIRONMENT == 'testing':
    # Testing settings - minimal resources
    MODEL_CONFIG['enhanced_ensemble']['cv_folds'] = 2
    MODEL_CONFIG['enhanced_ensemble']['n_iter'] = 5
    MODEL_CONFIG['enhanced_lstm']['epochs'] = 5
    PIPELINE_CONFIG['use_cache'] = False

# Utility functions
def get_model_config(model_name):
    """Get configuration for a specific model."""
    return MODEL_CONFIG.get(model_name, {})

def get_pipeline_config():
    """Get pipeline configuration."""
    return PIPELINE_CONFIG

def get_data_config():
    """Get data processing configuration."""
    return DATA_CONFIG

def update_config(section, key, value):
    """Update a configuration value."""
    config_sections = {
        'model': MODEL_CONFIG,
        'pipeline': PIPELINE_CONFIG,
        'data': DATA_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'viz': VIZ_CONFIG,
        'api': API_CONFIG
    }
    
    if section in config_sections:
        config_sections[section][key] = value
        return True
    return False

def get_all_config():
    """Get all configuration settings."""
    return {
        'project': {
            'name': PROJECT_NAME,
            'version': VERSION,
            'author': AUTHOR,
            'environment': ENVIRONMENT
        },
        'model': MODEL_CONFIG,
        'pipeline': PIPELINE_CONFIG,
        'data': DATA_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'visualization': VIZ_CONFIG,
        'api': API_CONFIG
    }

# Export main configurations
__all__ = [
    'MODEL_CONFIG',
    'PIPELINE_CONFIG', 
    'DATA_CONFIG',
    'PERFORMANCE_CONFIG',
    'VIZ_CONFIG',
    'API_CONFIG',
    'get_model_config',
    'get_pipeline_config',
    'get_data_config',
    'update_config',
    'get_all_config'
]
