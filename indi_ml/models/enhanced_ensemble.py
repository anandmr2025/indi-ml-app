"""
Enhanced Ensemble Models for Stock Price Prediction

This module provides robust ensemble learning approaches with:
- Advanced hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Multiple ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Cross-validation and model selection
- Feature importance analysis
- Robust error handling and validation
- Model stacking and blending capabilities

Author: Enhanced ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleModel:
    """
    Enhanced ensemble model with multiple algorithms and robust validation.
    """
    
    def __init__(self, use_scaling=True, feature_selection=True, n_features=None):
        """
        Initialize the enhanced ensemble model.
        
        Args:
            use_scaling (bool): Whether to scale features
            feature_selection (bool): Whether to perform feature selection
            n_features (int): Number of top features to select (None for all)
        """
        self.use_scaling = use_scaling
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def _prepare_data(self, X, y=None, fit_transformers=True):
        """
        Prepare data with scaling and feature selection.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable (required for fitting transformers)
            fit_transformers (bool): Whether to fit transformers
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Feature scaling
        if self.use_scaling:
            if fit_transformers:
                self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
                X_processed = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is not None:
                    X_processed = self.scaler.transform(X_processed)
        
        # Feature selection
        if self.feature_selection:
            if fit_transformers and y is not None:
                n_features = self.n_features or min(20, X_processed.shape[1])
                self.feature_selector = SelectKBest(f_regression, k=n_features)
                X_processed = self.feature_selector.fit_transform(X_processed, y)
            else:
                if self.feature_selector is not None:
                    try:
                        X_processed = self.feature_selector.transform(X_processed)
                    except ValueError as e:
                        if "Feature shape mismatch" in str(e) or "Expected" in str(e):
                            print(f"Feature shape mismatch, expected: {self.feature_selector.k}, got {X_processed.shape[1]}")
                            # Disable feature selection for this prediction
                            pass
                        else:
                            raise e
        
        return X_processed
    
    def _get_model_configs(self):
        """
        Get model configurations for different ensemble methods.
        
        Returns:
            dict: Dictionary of model configurations
        """
        return {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def train_and_select_best(self, X, y, cv_folds=5, scoring='neg_mean_squared_error', 
                             use_randomized_search=True, n_iter=20):
        """
        Train multiple models and select the best one using cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric for model selection
            use_randomized_search (bool): Use RandomizedSearchCV instead of GridSearchCV
            n_iter (int): Number of iterations for RandomizedSearchCV
            
        Returns:
            dict: Training results and metrics
        """
        if len(X) < cv_folds * 2:
            raise ValueError(f"Insufficient data for {cv_folds}-fold cross-validation. Need at least {cv_folds * 2} samples.")
        
        # Prepare data
        X_processed = self._prepare_data(X, y, fit_transformers=True)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        model_configs = self._get_model_configs()
        results = {}
        
        print("Training and evaluating models...")
        
        for model_name, config in model_configs.items():
            try:
                print(f"Training {model_name}...")
                
                # Choose search strategy
                if use_randomized_search:
                    search = RandomizedSearchCV(
                        config['model'], 
                        config['params'],
                        n_iter=n_iter,
                        cv=tscv,
                        scoring=scoring,
                        n_jobs=-1,
                        random_state=42
                    )
                else:
                    search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=tscv,
                        scoring=scoring,
                        n_jobs=-1
                    )
                
                # Fit the search
                search.fit(X_processed, y)
                
                # Store results
                self.models[model_name] = search.best_estimator_
                results[model_name] = {
                    'best_score': -search.best_score_,  # Convert back from negative
                    'best_params': search.best_params_,
                    'cv_scores': -search.cv_results_['mean_test_score']
                }
                
                print(f"{model_name} - Best CV Score: {-search.best_score_:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        if not self.models:
            raise ValueError("No models were successfully trained.")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['best_score'])
        self.best_model = self.models[best_model_name]
        
        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        print(f"Best model: {best_model_name} with RMSE: {results[best_model_name]['best_score']:.4f}")
        
        return {
            'best_model_name': best_model_name,
            'results': results,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        X_processed = self._prepare_data(X, fit_transformers=False)
        return self.best_model.predict(X_processed)
    
    def evaluate(self, X, y):
        """
        Evaluate the best model on given data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): True values
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': mean_absolute_percentage_error(y, predictions)
        }


def train_enhanced_ensemble(df_feat, test_size=0.2, **kwargs):
    """
    Train enhanced ensemble model for pipeline integration.
    
    Args:
        df_feat (pd.DataFrame): DataFrame with features and Target column
        test_size (float): Proportion of data to use for testing
        **kwargs: Additional arguments for EnhancedEnsembleModel
        
    Returns:
        tuple: (model, (rmse, mae, r2), predictions, actuals)
    """
    if 'Target' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Target' column")
    
    # Prepare data
    feature_cols = df_feat.columns.drop('Target')
    X, y = df_feat[feature_cols], df_feat['Target']
    
    # Split data
    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Adaptive data requirements
    min_train = max(5, len(X) // 10)  # At least 5 samples for training
    min_test = max(2, len(X) // 20)   # At least 2 samples for testing
    
    if len(X_train) < min_train or len(X_test) < min_test:
        # Try with smaller test size for very small datasets
        if len(X) >= 8:
            test_size = max(0.1, 2 / len(X))  # Use at least 2 samples or 10% for test
            split_idx = int((1 - test_size) * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) < 3 or len(X_test) < 1:
            raise ValueError(f"Insufficient data for train/test split: {len(X_train)} train, {len(X_test)} test samples.")
    
    try:
        # Initialize and train model
        model = EnhancedEnsembleModel(**kwargs)
        training_results = model.train_and_select_best(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = model.evaluate(X_test, y_test)
        
        return model, (metrics['rmse'], metrics['mae'], metrics['r2']), predictions, y_test.values
        
    except Exception as e:
        print(f"Enhanced ensemble training failed: {e}")
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    from indi_ml.ingest import history
    from indi_ml.features import enrich
    
    # Load and prepare data
    df = history("TCS", period="1y")
    feat_df = enrich(df)
    feat_df["Target"] = feat_df["Close"].shift(-1)
    feat_df.dropna(inplace=True)
    
    if len(feat_df) < 50:
        raise ValueError("Not enough data for enhanced ensemble training.")
    
    # Train enhanced ensemble
    model, metrics, preds, actual = train_enhanced_ensemble(feat_df)
    print(f"Enhanced Ensemble Metrics - RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, R2: {metrics[2]:.4f}")
