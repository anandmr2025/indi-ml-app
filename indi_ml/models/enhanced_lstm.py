"""
Enhanced LSTM Model for Stock Price Prediction

This module provides a robust LSTM implementation with:
- Advanced architecture with attention mechanisms
- Bidirectional LSTM layers
- Multiple regularization techniques (Dropout, L1/L2, Batch Normalization)
- Learning rate scheduling and early stopping
- Multiple sequence lengths and prediction horizons
- Ensemble of LSTM models
- Advanced preprocessing and feature engineering
- Robust validation and error handling

Author: Enhanced ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Bidirectional, Attention, Input, Concatenate,
                                   Conv1D, MaxPooling1D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging
tf.get_logger().setLevel('ERROR')

class EnhancedLSTM:
    """
    Enhanced LSTM model with advanced features and robust validation.
    """
    
    def __init__(self, lookback_window=60, prediction_horizon=1, architecture='bidirectional',
                 use_attention=False, scaler_type='minmax'):
        """
        Initialize the enhanced LSTM model.
        
        Args:
            lookback_window (int): Number of previous time steps to use
            prediction_horizon (int): Number of future steps to predict
            architecture (str): Model architecture ('simple', 'bidirectional', 'cnn_lstm', 'ensemble')
            use_attention (bool): Whether to use attention mechanism
            scaler_type (str): Type of scaler ('minmax', 'standard', 'robust')
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.architecture = architecture
        self.use_attention = use_attention
        self.scaler_type = scaler_type
        
        self.model = None
        self.scaler = None
        self.history = None
        self.best_model_path = None
        
        # Initialize scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:  # robust
            self.scaler = RobustScaler()
    
    def _create_sequences(self, data, include_features=False):
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.array): Input data
            include_features (bool): Whether to include additional features
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.lookback_window:i])
            if self.prediction_horizon == 1:
                y.append(data[i])
            else:
                y.append(data[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def _build_simple_lstm(self, input_shape):
        """Build simple LSTM architecture."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01, 0.01)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        return model
    
    def _build_bidirectional_lstm(self, input_shape):
        """Build bidirectional LSTM architecture."""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01, 0.01))),
            Dropout(0.3),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        return model
    
    def _build_cnn_lstm(self, input_shape):
        """Build CNN-LSTM hybrid architecture."""
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        return model
    
    def _build_attention_lstm(self, input_shape):
        """Build LSTM with attention mechanism."""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.3)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism (simplified)
        attention = Dense(1, activation='tanh')(lstm2)
        attention = tf.keras.layers.Softmax(axis=1)(attention)
        context = tf.keras.layers.Multiply()([lstm2, attention])
        context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Output layers
        dense1 = Dense(32, activation='relu')(context)
        dense1 = Dropout(0.2)(dense1)
        outputs = Dense(self.prediction_horizon)(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_model(self, input_shape):
        """
        Build the appropriate model architecture.
        
        Args:
            input_shape (tuple): Input shape for the model
            
        Returns:
            tf.keras.Model: Compiled model
        """
        if self.architecture == 'simple':
            model = self._build_simple_lstm(input_shape)
        elif self.architecture == 'bidirectional':
            model = self._build_bidirectional_lstm(input_shape)
        elif self.architecture == 'cnn_lstm':
            model = self._build_cnn_lstm(input_shape)
        elif self.use_attention:
            model = self._build_attention_lstm(input_shape)
        else:
            model = self._build_bidirectional_lstm(input_shape)  # Default
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _get_callbacks(self, validation_data=None):
        """
        Get training callbacks.
        
        Args:
            validation_data (tuple): Validation data for early stopping
            
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(lr_scheduler)
        
        # Early stopping
        if validation_data:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
        
        return callbacks
    
    def fit(self, data, epochs=100, batch_size=32, validation_split=0.2, verbose=0):
        """
        Fit the LSTM model.
        
        Args:
            data (pd.Series or np.array): Time series data
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_split (float): Validation split ratio
            verbose (int): Verbosity level
            
        Returns:
            self: Fitted model instance
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.values
        
        # Check data length
        if len(data) < self.lookback_window + 20:
            raise ValueError(f"Insufficient data. Need at least {self.lookback_window + 20} points, got {len(data)}")
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(data_scaled)
        
        if len(X) < 10:
            raise ValueError("Insufficient sequences created. Need more data or reduce lookback window.")
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2] if len(X.shape) > 2 else 1))
        
        # Reshape X if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Prepare validation data for callbacks
        if validation_split > 0:
            val_size = int(len(X) * validation_split)
            X_val, y_val = X[-val_size:], y[-val_size:]
            X_train, y_train = X[:-val_size], y[:-val_size]
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
            validation_data = None
        
        # Get callbacks
        callbacks = self._get_callbacks(validation_data)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, data, steps=None):
        """
        Make predictions.
        
        Args:
            data (pd.Series or np.array): Input data
            steps (int): Number of steps to predict (None for single prediction)
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.values
        
        # Scale data
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        if steps is None:
            # Single prediction on the last sequence
            if len(data_scaled) < self.lookback_window:
                raise ValueError(f"Need at least {self.lookback_window} data points for prediction.")
            
            X = data_scaled[-self.lookback_window:].reshape(1, self.lookback_window, 1)
            pred_scaled = self.model.predict(X, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            return pred.flatten()
        else:
            # Multi-step prediction
            predictions = []
            current_data = data_scaled.copy()
            
            for _ in range(steps):
                if len(current_data) < self.lookback_window:
                    break
                
                X = current_data[-self.lookback_window:].reshape(1, self.lookback_window, 1)
                pred_scaled = self.model.predict(X, verbose=0)
                predictions.append(pred_scaled[0, 0])
                
                # Update current_data with prediction
                current_data = np.append(current_data, pred_scaled[0, 0])
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions.flatten()
    
    def evaluate(self, data, test_size=0.2):
        """
        Evaluate model performance.
        
        Args:
            data (pd.Series or np.array): Time series data
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Evaluation metrics
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.values
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        if len(test_data) < self.lookback_window:
            raise ValueError("Test set too small for evaluation.")
        
        # Make predictions on test set
        predictions = []
        actuals = []
        
        for i in range(self.lookback_window, len(test_data)):
            # Use previous data for prediction
            input_data = np.concatenate([train_data, test_data[:i]])
            pred = self.predict(input_data[-self.lookback_window:])
            
            predictions.append(pred[0])
            actuals.append(test_data[i])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'mape': mean_absolute_percentage_error(actuals, predictions),
            'predictions': predictions,
            'actuals': actuals
        }


class EnsembleLSTM:
    """
    Ensemble of multiple LSTM models for improved robustness.
    """
    
    def __init__(self, n_models=3, **lstm_kwargs):
        """
        Initialize ensemble of LSTM models.
        
        Args:
            n_models (int): Number of models in ensemble
            **lstm_kwargs: Arguments for individual LSTM models
        """
        self.n_models = n_models
        self.lstm_kwargs = lstm_kwargs
        self.models = []
        self.weights = None
    
    def fit(self, data, **fit_kwargs):
        """
        Fit ensemble of LSTM models.
        
        Args:
            data: Training data
            **fit_kwargs: Arguments for model fitting
        """
        self.models = []
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}...")
            
            # Create model with slight variations
            model_kwargs = self.lstm_kwargs.copy()
            model_kwargs['lookback_window'] = model_kwargs.get('lookback_window', 60) + i * 5
            
            model = EnhancedLSTM(**model_kwargs)
            model.fit(data, **fit_kwargs)
            self.models.append(model)
        
        # Calculate weights based on validation performance (simplified)
        self.weights = np.ones(self.n_models) / self.n_models
        
        return self
    
    def predict(self, data, **predict_kwargs):
        """
        Make ensemble predictions.
        
        Args:
            data: Input data
            **predict_kwargs: Arguments for prediction
            
        Returns:
            np.array: Ensemble predictions
        """
        if not self.models:
            raise ValueError("Ensemble must be fitted before making predictions.")
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(data, **predict_kwargs)
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            raise ValueError("All models failed to make predictions.")
        
        # Weighted average
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
        
        return ensemble_pred


def train_enhanced_lstm_for_pipeline(df_feat, test_size=0.2, architecture='bidirectional', 
                                   epochs=50, lookback_window=30):
    """
    Train enhanced LSTM model for pipeline integration.
    
    Args:
        df_feat (pd.DataFrame): DataFrame with 'Close' column
        test_size (float): Proportion of data for testing
        architecture (str): Model architecture
        epochs (int): Number of training epochs
        lookback_window (int): Lookback window size
        
    Returns:
        tuple: (model, (rmse, mae, r2), predictions, actuals)
    """
    if 'Close' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    close_series = df_feat['Close'].dropna()
    
    # Adaptive requirements based on data size
    min_required = lookback_window + max(10, len(close_series) // 10)
    
    # Adjust lookback window for small datasets
    if len(close_series) < min_required:
        new_lookback = max(5, min(lookback_window, len(close_series) // 3))
        print(f"Enhanced LSTM: Limited data ({len(close_series)} points). Adjusting lookback window from {lookback_window} to {new_lookback}.")
        lookback_window = new_lookback
        min_required = lookback_window + 10
    
    if len(close_series) < min_required:
        print(f"Enhanced LSTM: Insufficient data ({len(close_series)} points). Need at least {min_required}.")
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])
    
    try:
        # Initialize model
        model = EnhancedLSTM(
            lookback_window=lookback_window,
            architecture=architecture,
            scaler_type='robust'
        )
        
        # Adaptive test size for small datasets
        if len(close_series) < 50:
            test_size = max(0.1, 3 / len(close_series))  # Use at least 3 points or 10% for test
        
        # Split data
        split_idx = int(len(close_series) * (1 - test_size))
        train_data = close_series[:split_idx]
        test_data = close_series[split_idx:]
        
        # Ensure minimum training data - be more flexible
        min_train_required = lookback_window + 3  # Reduced from +5 to +3
        if len(train_data) < min_train_required:
            # Try reducing lookback window further
            new_lookback = max(3, len(train_data) - 5)
            if new_lookback != lookback_window:
                print(f"Enhanced LSTM: Further reducing lookback window from {lookback_window} to {new_lookback}.")
                lookback_window = new_lookback
                model = EnhancedLSTM(
                    lookback_window=lookback_window,
                    architecture=architecture,
                    scaler_type='robust'
                )
                min_train_required = lookback_window + 3
            
            if len(train_data) < min_train_required:
                print(f"Enhanced LSTM: Insufficient training data ({len(train_data)} points). Need at least {min_train_required}.")
                return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])
        
        # Fit model
        model.fit(train_data, epochs=epochs, verbose=0)
        
        # Make predictions
        predictions = []
        actuals = test_data.values
        
        for i in range(len(test_data)):
            if i == 0:
                input_data = train_data
            else:
                input_data = pd.concat([train_data, test_data[:i]])
            
            pred = model.predict(input_data.values)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return model, (rmse, mae, r2), predictions, actuals
        
    except Exception as e:
        print(f"Enhanced LSTM training failed: {e}")
        return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    from indi_ml.ingest import history
    
    # Load data
    df = history("TCS", period="1y")
    close = df["Close"].dropna()
    
    if len(close) < 100:
        raise ValueError("Not enough data for enhanced LSTM testing.")
    
    # Train enhanced LSTM
    model, metrics, preds, actual = train_enhanced_lstm_for_pipeline(
        df, architecture='bidirectional', epochs=20, lookback_window=30
    )
    
    if model is not None:
        print(f"Enhanced LSTM Metrics - RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, R2: {metrics[2]:.4f}")
    else:
        print("Enhanced LSTM training failed.")
