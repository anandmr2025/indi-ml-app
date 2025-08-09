"""
Risk Assessment Module for Investment Analysis

This module provides comprehensive risk assessment capabilities for investment analysis,
including various risk metrics calculation, portfolio risk evaluation, and risk scoring.

Key Features:
- Volatility and variance calculations
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Maximum drawdown analysis
- Sharpe ratio and risk-adjusted returns
- Beta calculation for market risk
- Comprehensive risk scoring system

Usage:
    from indi_ml.risk import RiskAssessmentModule
    risk = RiskAssessmentModule()
    metrics = risk.comprehensive_risk_assessment(returns, predictions, actual)
    score = risk.risk_score_calculation(metrics)

Author: ML Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings


class RiskAssessmentModule:
    """
    Comprehensive risk assessment module for investment analysis.
    
    This class provides methods to calculate various risk metrics including
    volatility, VaR, drawdown, Sharpe ratio, and other risk-adjusted measures.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the risk assessment module.
        
        Args:
            confidence_level (float): Confidence level for VaR calculations (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = 0.02  # Default risk-free rate (2%)
        
    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility of returns.
        
        Args:
            returns (pd.Series): Daily returns series
            
        Returns:
            float: Annualized volatility
        """
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(252)  # Annualized (252 trading days)
    
    def calculate_var(self, returns: pd.Series, confidence_level: Optional[float] = None) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.
        
        Args:
            returns (pd.Series): Daily returns series
            confidence_level (float, optional): Confidence level (default: self.confidence_level)
            
        Returns:
            float: Value at Risk
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if returns.empty:
            return 0.0
            
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: Optional[float] = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) at specified confidence level.
        
        Args:
            returns (pd.Series): Daily returns series
            confidence_level (float, optional): Confidence level (default: self.confidence_level)
            
        Returns:
            float: Conditional Value at Risk
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if returns.empty:
            return 0.0
            
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            tuple: (max_drawdown, start_index, end_index)
        """
        if prices.empty:
            return 0.0, 0, 0
            
        # Calculate cumulative returns
        cumulative = (1 + prices.pct_change()).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        end_idx = drawdown.idxmin()
        start_idx = cumulative[:end_idx].idxmax()
        
        return max_drawdown, start_idx, end_idx
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Args:
            returns (pd.Series): Daily returns series
            risk_free_rate (float, optional): Risk-free rate (default: self.risk_free_rate)
            
        Returns:
            float: Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        if returns.empty:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (market risk measure).
        
        Args:
            returns (pd.Series): Asset returns
            market_returns (pd.Series): Market returns
            
        Returns:
            float: Beta coefficient
        """
        if returns.empty or market_returns.empty:
            return 1.0
            
        # Align the series
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        if aligned_data.empty:
            return 1.0
            
        asset_returns = aligned_data.iloc[:, 0]
        market_returns = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def comprehensive_risk_assessment(self, 
                                   prices: pd.Series, 
                                   predictions: np.ndarray, 
                                   actual: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive risk assessment on investment data.
        
        Args:
            prices (pd.Series): Historical price series
            predictions (np.ndarray): Model predictions
            actual (np.ndarray): Actual values
            
        Returns:
            Dict[str, float]: Dictionary containing all risk metrics
        """
        # Calculate returns from prices
        returns = prices.pct_change().dropna()
        
        # Calculate prediction errors
        prediction_errors = predictions - actual
        
        # Basic risk metrics
        volatility = self.calculate_volatility(returns)
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # Drawdown analysis
        max_dd, start_idx, end_idx = self.calculate_max_drawdown(prices)
        
        # Risk-adjusted returns
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        
        # Prediction accuracy metrics
        mse = np.mean(prediction_errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction_errors))
        
        # Additional risk metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe_ratio,
            'prediction_mse': mse,
            'prediction_rmse': rmse,
            'prediction_mae': mae,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'start_date': start_idx,
            'end_date': end_idx
        }
    
    def risk_score_calculation(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate comprehensive risk score based on various metrics.
        
        Args:
            metrics (Dict[str, float]): Risk metrics from comprehensive_risk_assessment
            
        Returns:
            Dict[str, float]: Risk scores and overall risk rating
        """
        scores = {}
        
        # Volatility score (0-100, higher is riskier)
        volatility_score = min(100, max(0, metrics['volatility'] * 100))
        scores['volatility_score'] = volatility_score
        
        # VaR score (0-100, higher is riskier)
        var_score = min(100, max(0, abs(metrics['var_95']) * 100))
        scores['var_score'] = var_score
        
        # Drawdown score (0-100, higher is riskier)
        drawdown_score = min(100, max(0, abs(metrics['max_drawdown']) * 100))
        scores['drawdown_score'] = drawdown_score
        
        # Sharpe ratio score (0-100, higher is better)
        sharpe_score = max(0, min(100, (metrics['sharpe_ratio'] + 2) * 25))
        scores['sharpe_score'] = sharpe_score
        
        # Prediction accuracy score (0-100, higher is better)
        accuracy_score = max(0, min(100, 100 - metrics['prediction_rmse']))
        scores['accuracy_score'] = accuracy_score
        
        # Overall risk score (weighted average)
        weights = {
            'volatility_score': 0.25,
            'var_score': 0.25,
            'drawdown_score': 0.20,
            'sharpe_score': 0.15,
            'accuracy_score': 0.15
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        scores['overall_risk_score'] = overall_score
        
        # Risk rating
        if overall_score < 20:
            risk_rating = "Low Risk"
        elif overall_score < 40:
            risk_rating = "Moderate Risk"
        elif overall_score < 60:
            risk_rating = "High Risk"
        else:
            risk_rating = "Very High Risk"
            
        scores['risk_rating'] = risk_rating
        
        return scores


# Example usage and testing
if __name__ == "__main__":
    # Import required modules
    from indi_ml.ingest import history
    from indi_ml.models.lstm import train_lstm
    import matplotlib.pyplot as plt
    
    print("Risk Assessment Module Test")
    print("=" * 40)
    
    try:
        # Load sample data
        print("Loading sample data...")
        df = history("TCS", period="6mo")
        close_prices = df["Close"].dropna()
        
        # Train LSTM model for predictions
        print("Training LSTM model...")
        model, rmse, preds, actual, scaler = train_lstm(close_prices, epochs=3, lookback=20)
        
        # Initialize risk assessment module
        print("Performing risk assessment...")
        risk = RiskAssessmentModule()
        
        # Perform comprehensive risk assessment
        metrics = risk.comprehensive_risk_assessment(close_prices, preds, actual)
        
        # Calculate risk scores
        scores = risk.risk_score_calculation(metrics)
        
        # Display results
        print("\nRisk Metrics:")
        print(f"Volatility: {metrics['volatility']:.4f}")
        print(f"VaR (95%): {metrics['var_95']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Prediction RMSE: {metrics['prediction_rmse']:.4f}")
        
        print("\nRisk Scores:")
        print(f"Overall Risk Score: {scores['overall_risk_score']:.2f}")
        print(f"Risk Rating: {scores['risk_rating']}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Please ensure all dependencies are installed and data is available.")


