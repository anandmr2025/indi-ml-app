"""
Momentum Analysis Module for Indian Stocks

This module provides comprehensive momentum analysis capabilities including:
- Technical momentum indicators
- Price momentum analysis
- Momentum-based trading signals
- Relative strength analysis
- Momentum scoring and ranking

Key Features:
- RSI, MACD, Stochastic oscillators
- Price momentum calculations
- Volume momentum analysis
- Momentum divergence detection
- Momentum-based scoring system

Usage:
    from indi_ml.momentum import MomentumAnalysis
    ma = MomentumAnalysis()
    analysis = ma.analyze_momentum(df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MomentumAnalysis:
    """
    Comprehensive momentum analysis for Indian stocks.
    """
    
    def __init__(self):
        """Initialize the momentum analysis module."""
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            
        Returns:
            tuple: (%K, %D)
        """
        lowest_low = low.rolling(window=self.stoch_k).min()
        highest_high = high.rolling(window=self.stoch_k).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.stoch_d).mean()
        return k_percent, d_percent
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all momentum indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Momentum indicators
        """
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            indicators = {}
            
            # RSI
            indicators['rsi'] = self.calculate_rsi(close, self.rsi_period)
            
            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(close)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = histogram
            
            # Stochastic
            k_percent, d_percent = self.calculate_stochastic(high, low, close)
            indicators['stoch_k'] = k_percent
            indicators['stoch_d'] = d_percent
            
            # Price momentum
            indicators['price_momentum_5d'] = close.pct_change(5)
            indicators['price_momentum_10d'] = close.pct_change(10)
            indicators['price_momentum_20d'] = close.pct_change(20)
            
            # Volume momentum
            indicators['volume_momentum'] = volume.pct_change(5)
            
            # Moving averages
            indicators['sma_20'] = close.rolling(window=20).mean()
            indicators['sma_50'] = close.rolling(window=50).mean()
            indicators['ema_12'] = close.ewm(span=12).mean()
            indicators['ema_26'] = close.ewm(span=26).mean()
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
            return {}
    
    def analyze_momentum_signals(self, indicators: Dict) -> Dict:
        """
        Analyze momentum signals and generate trading recommendations.
        
        Args:
            indicators (dict): Momentum indicators
            
        Returns:
            dict: Momentum signals and analysis
        """
        try:
            signals = {}
            
            # RSI signals
            rsi = indicators.get('rsi', pd.Series())
            if not rsi.empty:
                latest_rsi = rsi.iloc[-1]
                if latest_rsi > 70:
                    signals['rsi_signal'] = 'Overbought'
                    signals['rsi_strength'] = 'Strong Sell'
                elif latest_rsi > 60:
                    signals['rsi_signal'] = 'Bullish'
                    signals['rsi_strength'] = 'Weak Buy'
                elif latest_rsi < 30:
                    signals['rsi_signal'] = 'Oversold'
                    signals['rsi_strength'] = 'Strong Buy'
                elif latest_rsi < 40:
                    signals['rsi_signal'] = 'Bearish'
                    signals['rsi_strength'] = 'Weak Sell'
                else:
                    signals['rsi_signal'] = 'Neutral'
                    signals['rsi_strength'] = 'Hold'
                
                signals['rsi_value'] = latest_rsi
            
            # MACD signals
            macd_line = indicators.get('macd_line', pd.Series())
            macd_signal = indicators.get('macd_signal', pd.Series())
            histogram = indicators.get('macd_histogram', pd.Series())
            
            if not macd_line.empty and not macd_signal.empty:
                latest_macd = macd_line.iloc[-1]
                latest_signal = macd_signal.iloc[-1]
                latest_histogram = histogram.iloc[-1]
                
                if latest_macd > latest_signal and latest_histogram > 0:
                    signals['macd_signal'] = 'Bullish'
                    signals['macd_strength'] = 'Strong Buy'
                elif latest_macd > latest_signal:
                    signals['macd_signal'] = 'Bullish'
                    signals['macd_strength'] = 'Weak Buy'
                elif latest_macd < latest_signal and latest_histogram < 0:
                    signals['macd_signal'] = 'Bearish'
                    signals['macd_strength'] = 'Strong Sell'
                elif latest_macd < latest_signal:
                    signals['macd_signal'] = 'Bearish'
                    signals['macd_strength'] = 'Weak Sell'
                else:
                    signals['macd_signal'] = 'Neutral'
                    signals['macd_strength'] = 'Hold'
                
                signals['macd_value'] = latest_macd
                signals['macd_histogram'] = latest_histogram
            
            # Stochastic signals
            stoch_k = indicators.get('stoch_k', pd.Series())
            stoch_d = indicators.get('stoch_d', pd.Series())
            
            if not stoch_k.empty and not stoch_d.empty:
                latest_k = stoch_k.iloc[-1]
                latest_d = stoch_d.iloc[-1]
                
                if latest_k > 80 and latest_d > 80:
                    signals['stoch_signal'] = 'Overbought'
                    signals['stoch_strength'] = 'Strong Sell'
                elif latest_k > 60 and latest_d > 60:
                    signals['stoch_signal'] = 'Bullish'
                    signals['stoch_strength'] = 'Weak Buy'
                elif latest_k < 20 and latest_d < 20:
                    signals['stoch_signal'] = 'Oversold'
                    signals['stoch_strength'] = 'Strong Buy'
                elif latest_k < 40 and latest_d < 40:
                    signals['stoch_signal'] = 'Bearish'
                    signals['stoch_strength'] = 'Weak Sell'
                else:
                    signals['stoch_signal'] = 'Neutral'
                    signals['stoch_strength'] = 'Hold'
                
                signals['stoch_k'] = latest_k
                signals['stoch_d'] = latest_d
            
            # Price momentum signals
            momentum_5d = indicators.get('price_momentum_5d', pd.Series())
            momentum_10d = indicators.get('price_momentum_10d', pd.Series())
            momentum_20d = indicators.get('price_momentum_20d', pd.Series())
            
            if not momentum_5d.empty:
                latest_5d = momentum_5d.iloc[-1]
                latest_10d = momentum_10d.iloc[-1] if not momentum_10d.empty else 0
                latest_20d = momentum_20d.iloc[-1] if not momentum_20d.empty else 0
                
                # Momentum trend analysis
                if latest_5d > 0 and latest_10d > 0 and latest_20d > 0:
                    signals['momentum_trend'] = 'Strong Uptrend'
                    signals['momentum_strength'] = 'Strong Buy'
                elif latest_5d > 0 and latest_10d > 0:
                    signals['momentum_trend'] = 'Uptrend'
                    signals['momentum_strength'] = 'Buy'
                elif latest_5d < 0 and latest_10d < 0 and latest_20d < 0:
                    signals['momentum_trend'] = 'Strong Downtrend'
                    signals['momentum_strength'] = 'Strong Sell'
                elif latest_5d < 0 and latest_10d < 0:
                    signals['momentum_trend'] = 'Downtrend'
                    signals['momentum_strength'] = 'Sell'
                else:
                    signals['momentum_trend'] = 'Sideways'
                    signals['momentum_strength'] = 'Hold'
                
                signals['momentum_5d'] = latest_5d * 100
                signals['momentum_10d'] = latest_10d * 100
                signals['momentum_20d'] = latest_20d * 100
            
            return signals
            
        except Exception as e:
            print(f"Error analyzing momentum signals: {e}")
            return {}
    
    def calculate_momentum_score(self, signals: Dict) -> Dict:
        """
        Calculate overall momentum score based on all signals.
        
        Args:
            signals (dict): Momentum signals
            
        Returns:
            dict: Momentum scoring
        """
        try:
            score = 0
            max_score = 0
            details = {}
            
            # RSI scoring
            if 'rsi_strength' in signals:
                max_score += 25
                if signals['rsi_strength'] == 'Strong Buy':
                    score += 25
                elif signals['rsi_strength'] == 'Weak Buy':
                    score += 15
                elif signals['rsi_strength'] == 'Hold':
                    score += 12
                elif signals['rsi_strength'] == 'Weak Sell':
                    score += 10
                elif signals['rsi_strength'] == 'Strong Sell':
                    score += 0
                details['rsi_score'] = score
            
            # MACD scoring
            if 'macd_strength' in signals:
                max_score += 25
                if signals['macd_strength'] == 'Strong Buy':
                    score += 25
                elif signals['macd_strength'] == 'Weak Buy':
                    score += 15
                elif signals['macd_strength'] == 'Hold':
                    score += 12
                elif signals['macd_strength'] == 'Weak Sell':
                    score += 10
                elif signals['macd_strength'] == 'Strong Sell':
                    score += 0
                details['macd_score'] = score - details.get('rsi_score', 0)
            
            # Stochastic scoring
            if 'stoch_strength' in signals:
                max_score += 25
                if signals['stoch_strength'] == 'Strong Buy':
                    score += 25
                elif signals['stoch_strength'] == 'Weak Buy':
                    score += 15
                elif signals['stoch_strength'] == 'Hold':
                    score += 12
                elif signals['stoch_strength'] == 'Weak Sell':
                    score += 10
                elif signals['stoch_strength'] == 'Strong Sell':
                    score += 0
                details['stoch_score'] = score - (details.get('rsi_score', 0) + details.get('macd_score', 0))
            
            # Momentum trend scoring
            if 'momentum_strength' in signals:
                max_score += 25
                if signals['momentum_strength'] == 'Strong Buy':
                    score += 25
                elif signals['momentum_strength'] == 'Buy':
                    score += 20
                elif signals['momentum_strength'] == 'Hold':
                    score += 12
                elif signals['momentum_strength'] == 'Sell':
                    score += 5
                elif signals['momentum_strength'] == 'Strong Sell':
                    score += 0
                details['momentum_score'] = score - (details.get('rsi_score', 0) + details.get('macd_score', 0) + details.get('stoch_score', 0))
            
            # Overall score
            overall_score = (score / max_score * 100) if max_score > 0 else 50
            details['overall_score'] = overall_score
            
            # Momentum rating
            if overall_score >= 80:
                details['momentum_rating'] = 'Very Strong'
                details['recommendation'] = 'Strong Buy'
            elif overall_score >= 60:
                details['momentum_rating'] = 'Strong'
                details['recommendation'] = 'Buy'
            elif overall_score >= 40:
                details['momentum_rating'] = 'Moderate'
                details['recommendation'] = 'Hold'
            elif overall_score >= 20:
                details['momentum_rating'] = 'Weak'
                details['recommendation'] = 'Sell'
            else:
                details['momentum_rating'] = 'Very Weak'
                details['recommendation'] = 'Strong Sell'
            
            return details
            
        except Exception as e:
            print(f"Error calculating momentum score: {e}")
            return {'overall_score': 50, 'momentum_rating': 'Neutral', 'recommendation': 'Hold'}
    
    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive momentum analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Complete momentum analysis
        """
        try:
            print("Analyzing momentum indicators...")
            
            # Calculate momentum indicators
            indicators = self.calculate_momentum_indicators(df)
            
            # Analyze momentum signals
            signals = self.analyze_momentum_signals(indicators)
            
            # Calculate momentum score
            scoring = self.calculate_momentum_score(signals)
            
            # Compile analysis
            analysis = {
                'indicators': indicators,
                'signals': signals,
                'scoring': scoring
            }
            
            print("Momentum analysis completed")
            return analysis
            
        except Exception as e:
            print(f"Error in momentum analysis: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    from indi_ml.ingest import history
    
    # Get sample data
    df = history("RELIANCE", period="6mo")
    
    # Analyze momentum
    ma = MomentumAnalysis()
    analysis = ma.analyze_momentum(df)
    
    print("Momentum Analysis Results:")
    print(analysis) 