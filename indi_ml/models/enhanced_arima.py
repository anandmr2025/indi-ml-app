"""
Enhanced ARIMA Model for Stock Price Prediction

This module provides a robust ARIMA implementation with:
- Advanced parameter selection using AIC, BIC, and HQIC criteria
- Seasonal ARIMA (SARIMA) support
- Automatic differencing and stationarity testing
- Multiple model selection strategies
- Robust error handling and validation
- Walk-forward validation for time series
- Residual analysis and diagnostics

Author: Enhanced ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from itertools import product
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class EnhancedARIMA:
    """
    Enhanced ARIMA model with advanced features and robust validation.
    """
    
    def __init__(self, seasonal=False, seasonal_periods=None, information_criterion='aic'):
        """
        Initialize the enhanced ARIMA model.
        
        Args:
            seasonal (bool): Whether to use seasonal ARIMA (SARIMA)
            seasonal_periods (int): Number of periods in seasonal cycle
            information_criterion (str): Criterion for model selection ('aic', 'bic', 'hqic')
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods or 12
        self.information_criterion = information_criterion
        self.model = None
        self.fitted_model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.diagnostics = {}
        
    def check_stationarity(self, series, significance_level=0.05):
        """
        Comprehensive stationarity testing using ADF and KPSS tests.
        
        Args:
            series (pd.Series): Time series data
            significance_level (float): Significance level for tests
            
        Returns:
            dict: Stationarity test results
        """
        series_clean = series.dropna()
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series_clean)
        adf_stationary = adf_result[1] <= significance_level
        
        # KPSS test
        try:
            kpss_result = kpss(series_clean, regression='c')
            kpss_stationary = kpss_result[1] > significance_level
        except:
            kpss_stationary = None
            kpss_result = None
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_stationary': adf_stationary,
            'kpss_statistic': kpss_result[0] if kpss_result else None,
            'kpss_pvalue': kpss_result[1] if kpss_result else None,
            'kpss_stationary': kpss_stationary,
            'is_stationary': adf_stationary and (kpss_stationary if kpss_stationary is not None else True)
        }
    
    def determine_differencing(self, series, max_d=2):
        """
        Determine optimal differencing order.
        
        Args:
            series (pd.Series): Time series data
            max_d (int): Maximum differencing order to test
            
        Returns:
            int: Optimal differencing order
        """
        for d in range(max_d + 1):
            if d == 0:
                diff_series = series
            else:
                diff_series = series.diff(d).dropna()
            
            if len(diff_series) < 10:
                continue
                
            stationarity = self.check_stationarity(diff_series)
            if stationarity['is_stationary']:
                return d
        
        return 1  # Default to first differencing
    
    def grid_search_arima(self, series, max_p=5, max_d=2, max_q=5, 
                         seasonal_max_P=2, seasonal_max_D=1, seasonal_max_Q=2):
        """
        Comprehensive grid search for optimal ARIMA parameters.
        
        Args:
            series (pd.Series): Time series data
            max_p, max_d, max_q (int): Maximum values for non-seasonal parameters
            seasonal_max_P, seasonal_max_D, seasonal_max_Q (int): Maximum values for seasonal parameters
            
        Returns:
            tuple: Best order and seasonal order
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 50:
            print("Warning: Limited data for comprehensive grid search. Using simplified search.")
            max_p, max_q = min(max_p, 2), min(max_q, 2)
        
        best_ic = float('inf')
        best_order = (1, 1, 0)
        best_seasonal_order = (0, 0, 0, 0)
        
        # Non-seasonal parameters
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        # Seasonal parameters (if seasonal)
        if self.seasonal:
            P_values = range(0, seasonal_max_P + 1)
            D_values = range(0, seasonal_max_D + 1)
            Q_values = range(0, seasonal_max_Q + 1)
            seasonal_combinations = list(product(P_values, D_values, Q_values))
        else:
            seasonal_combinations = [(0, 0, 0)]
        
        total_combinations = len(p_values) * len(d_values) * len(q_values) * len(seasonal_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        tested = 0
        for p, d, q in product(p_values, d_values, q_values):
            for P, D, Q in seasonal_combinations:
                tested += 1
                if tested % 50 == 0:
                    print(f"Progress: {tested}/{total_combinations}")
                
                try:
                    if self.seasonal:
                        model = SARIMAX(series_clean, 
                                      order=(p, d, q),
                                      seasonal_order=(P, D, Q, self.seasonal_periods))
                    else:
                        model = ARIMA(series_clean, order=(p, d, q))
                    
                    fitted = model.fit()
                    
                    # Get information criterion
                    if self.information_criterion == 'aic':
                        ic_value = fitted.aic
                    elif self.information_criterion == 'bic':
                        ic_value = fitted.bic
                    else:  # hqic
                        ic_value = fitted.hqic
                    
                    if ic_value < best_ic:
                        best_ic = ic_value
                        best_order = (p, d, q)
                        best_seasonal_order = (P, D, Q, self.seasonal_periods) if self.seasonal else (0, 0, 0, 0)
                        
                except:
                    continue
        
        print(f"Best order found: {best_order}, Seasonal: {best_seasonal_order}, {self.information_criterion.upper()}: {best_ic:.2f}")
        return best_order, best_seasonal_order
    
    def fit(self, series, auto_order=True, **grid_search_kwargs):
        """
        Fit the ARIMA model to the time series.
        
        Args:
            series (pd.Series): Time series data
            auto_order (bool): Whether to automatically select optimal parameters
            **grid_search_kwargs: Additional arguments for grid search
            
        Returns:
            self: Fitted model instance
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            raise ValueError("Insufficient data for ARIMA modeling. Need at least 20 observations.")
        
        # Determine parameters
        if auto_order:
            self.best_order, self.best_seasonal_order = self.grid_search_arima(
                series_clean, **grid_search_kwargs
            )
        else:
            self.best_order = (1, 1, 0)
            self.best_seasonal_order = (0, 0, 0, 0)
        
        # Fit the model
        try:
            if self.seasonal and any(x > 0 for x in self.best_seasonal_order[:3]):
                self.model = SARIMAX(series_clean,
                                   order=self.best_order,
                                   seasonal_order=self.best_seasonal_order)
            else:
                self.model = ARIMA(series_clean, order=self.best_order)
            
            self.fitted_model = self.model.fit()
            
            # Run diagnostics
            self._run_diagnostics()
            
        except Exception as e:
            print(f"Model fitting failed with optimal parameters: {e}")
            # Fallback to simple model
            try:
                self.model = ARIMA(series_clean, order=(1, 1, 0))
                self.fitted_model = self.model.fit()
                self.best_order = (1, 1, 0)
                self.best_seasonal_order = (0, 0, 0, 0)
            except Exception as e2:
                raise ValueError(f"Even fallback model failed: {e2}")
        
        return self
    
    def _run_diagnostics(self):
        """
        Run diagnostic tests on the fitted model.
        """
        if self.fitted_model is None:
            return
        
        try:
            residuals = self.fitted_model.resid
            
            # Ljung-Box test for autocorrelation in residuals
            lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
            
            self.diagnostics = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'residual_autocorr': lb_test['lb_pvalue'].iloc[-1] > 0.05  # True if no autocorr
            }
        except:
            self.diagnostics = {'error': 'Diagnostic tests failed'}
    
    def forecast(self, steps):
        """
        Generate forecasts.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            tuple: (forecast, confidence_intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        
        forecast_result = self.fitted_model.forecast(steps=steps)
        
        # Get confidence intervals if available
        try:
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            return forecast_result, conf_int
        except:
            return forecast_result, None
    
    def walk_forward_validation(self, series, test_size=30, step_size=1):
        """
        Perform walk-forward validation.
        
        Args:
            series (pd.Series): Time series data
            test_size (int): Size of test set
            step_size (int): Step size for walk-forward
            
        Returns:
            dict: Validation results
        """
        series_clean = series.dropna()
        
        if len(series_clean) < test_size + 50:
            raise ValueError("Insufficient data for walk-forward validation.")
        
        predictions = []
        actuals = []
        
        for i in range(0, test_size, step_size):
            train_end = len(series_clean) - test_size + i
            train_data = series_clean[:train_end]
            actual_value = series_clean.iloc[train_end]
            
            try:
                # Fit model on training data
                temp_model = EnhancedARIMA(seasonal=self.seasonal, 
                                         seasonal_periods=self.seasonal_periods)
                temp_model.fit(train_data, auto_order=False)
                temp_model.best_order = self.best_order
                temp_model.best_seasonal_order = self.best_seasonal_order
                
                # Make one-step forecast
                forecast, _ = temp_model.forecast(1)
                predictions.append(forecast[0])
                actuals.append(actual_value)
                
            except:
                continue
        
        if len(predictions) == 0:
            return {'error': 'Walk-forward validation failed'}
        
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


def train_enhanced_arima_for_pipeline(df_feat, test_size=0.2, seasonal=False):
    """
    Train enhanced ARIMA model for pipeline integration.
    
    Args:
        df_feat (pd.DataFrame): DataFrame with 'Close' column
        test_size (float): Proportion of data for testing
        seasonal (bool): Whether to use seasonal ARIMA
        
    Returns:
        tuple: (model, (rmse, mae, r2), predictions, actuals)
    """
    if 'Close' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    close_series = df_feat['Close'].dropna()
    
    # Adaptive minimum data requirement
    min_required = max(20, len(close_series) // 10)  # At least 20 points, but adapt to data size
    
    if len(close_series) < min_required:
        print(f"Enhanced ARIMA: Limited data ({len(close_series)} points). Using simplified approach.")
        if len(close_series) < 15:
            print(f"Enhanced ARIMA: Insufficient data ({len(close_series)} points). Need at least 15.")
            return None, (float('inf'), float('inf'), 0.0), np.array([]), np.array([])
    
    try:
        # Determine seasonal periods based on data frequency
        seasonal_periods = 12 if seasonal else None
        
        # Initialize and fit model
        model = EnhancedARIMA(seasonal=seasonal, seasonal_periods=seasonal_periods)
        model.fit(close_series)
        
        # Calculate test size
        test_steps = max(10, int(len(close_series) * test_size))
        
        # Split data
        train_data = close_series[:-test_steps]
        test_data = close_series[-test_steps:]
        
        # Refit on training data only
        model.fit(train_data)
        
        # Make predictions
        predictions, _ = model.forecast(test_steps)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        
        return model, (rmse, mae, r2), predictions, test_data.values
        
    except Exception as e:
        print(f"Enhanced ARIMA training failed: {e}")
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
        raise ValueError("Not enough data for enhanced ARIMA testing.")
    
    # Train enhanced ARIMA
    model, metrics, preds, actual = train_enhanced_arima_for_pipeline(df, seasonal=False)
    
    if model is not None:
        print(f"Enhanced ARIMA Metrics - RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, R2: {metrics[2]:.4f}")
        print(f"Model diagnostics: {model.diagnostics}")
    else:
        print("Enhanced ARIMA training failed.")
