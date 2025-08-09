from indi_ml import ingest, features
from indi_ml.models import ensemble, arima, lstm
from indi_ml.sentiment import _headlines_multi_source, sentiment_scores, aggregate
from indi_ml.risk import RiskAssessmentModule
from indi_ml.fundamental import FundamentalAnalysis
from indi_ml.momentum import MomentumAnalysis
from indi_ml.dcf import DCFAnalysis
import pandas as pd
import numpy as np

def run_pipeline(symbol="RELIANCE", period="1y"):
    """
    Main pipeline function that runs the complete analysis for a given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., "RELIANCE", "TCS")
        period (str): Data period (e.g., "1y", "6mo")
        
    Returns:
        dict: Complete analysis results including price data, model metrics, sentiment, and risk
    """
    try:
        # 1. Data ingestion: get historical price data
        print(f"Fetching data for {symbol}...")
        df = ingest.history(symbol, period=period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # 2. Feature engineering: add technical indicators
        print("Adding technical indicators...")
        df_feat = features.enrich(df.copy())
        
        # 3. Create the target column: next day's closing price
        df_feat["Target"] = df_feat["Close"].shift(-1)
        
        # 4. Drop rows with any NaN values (introduced by shifts and rolling windows)
        df_feat.dropna(inplace=True)
        
        # 5. Train the random forest regression model
        print("Training Random Forest model...")
        try:
            rf_model, (rf_rmse, rf_mae, rf_r2), rf_predictions, rf_actuals = ensemble.train_rf(df_feat)
            print(f"Random Forest RMSE: {rf_rmse:.4f}")
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            rf_model, rf_rmse, rf_mae, rf_r2 = None, float('inf'), float('inf'), 0.0
            rf_predictions, rf_actuals = np.array([]), np.array([])
        
        # 6. Train the ARIMA model
        print("Training ARIMA model...")
        try:
            arima_model, (arima_rmse, arima_mae, arima_r2), arima_predictions, arima_actuals = arima.train_arima_for_pipeline(df_feat)
            print(f"ARIMA RMSE: {arima_rmse:.4f}")
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            arima_model, arima_rmse, arima_mae, arima_r2 = None, float('inf'), float('inf'), 0.0
            arima_predictions, arima_actuals = np.array([]), np.array([])
        
        # 7. Train the LSTM model
        print("Training LSTM model...")
        try:
            lstm_model, (lstm_rmse, lstm_mae, lstm_r2), lstm_predictions, lstm_actuals = lstm.train_lstm_for_pipeline(df_feat)
            print(f"LSTM RMSE: {lstm_rmse:.4f}")
        except Exception as e:
            print(f"LSTM training failed: {e}")
            lstm_model, lstm_rmse, lstm_mae, lstm_r2 = None, float('inf'), float('inf'), 0.0
            lstm_predictions, lstm_actuals = np.array([]), np.array([])
        
        # 8. Get sentiment analysis
        print("Analyzing news sentiment...")
        headlines_data = _headlines_multi_source(symbol, max_items_per_source=10)
        sentiment_df = sentiment_scores(headlines_data)
        sentiment_result = aggregate(sentiment_df)
        
        # 9. Fundamental analysis
        print("Performing fundamental analysis...")
        fundamental_analyzer = FundamentalAnalysis()
        fundamental_analysis = fundamental_analyzer.analyze_stock(symbol)
        
        # 10. Momentum analysis
        print("Performing momentum analysis...")
        momentum_analyzer = MomentumAnalysis()
        momentum_analysis = momentum_analyzer.analyze_momentum(df)
        
        # 11. DCF and Intrinsic Value analysis
        print("Performing DCF analysis...")
        dcf_analyzer = DCFAnalysis()
        dcf_analysis = dcf_analyzer.analyze_dcf(symbol)
        
        # 12. Risk assessment
        print("Calculating risk metrics...")
        risk_module = RiskAssessmentModule()
        
        # Calculate returns for risk analysis
        returns = df_feat["Close"].pct_change().dropna()
        
        # Get risk metrics (use RF predictions if available, otherwise use price data)
        try:
            if len(rf_predictions) > 0 and len(rf_actuals) > 0:
                risk_metrics = risk_module.comprehensive_risk_assessment(
                    prices=df_feat["Close"],
                    predictions=rf_predictions,
                    actual=rf_actuals
                )
            else:
                # Fallback: use price data only
                risk_metrics = risk_module.comprehensive_risk_assessment(
                    prices=df_feat["Close"],
                    predictions=df_feat["Close"].iloc[-len(df_feat)//5:].values,  # Use last 20% as "predictions"
                    actual=df_feat["Close"].iloc[-len(df_feat)//5:].values
                )
            
            # Calculate risk score
            risk_score_data = risk_module.risk_score_calculation(risk_metrics)
            risk_score = risk_score_data.get("overall_risk_score", 50.0)
        except Exception as e:
            print(f"Risk assessment failed: {e}")
            risk_metrics = {"volatility": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
            risk_score = 50.0
        
        # 8. Prepare results
        state = {
            "price_df": df,
            "arima_rmse": arima_rmse,
            "lstm_rmse": lstm_rmse,
            "rf_rmse": rf_rmse,
            "sentiment": sentiment_result.get("weighted_sentiment", 0.0),
            "risk_score": risk_score,
            "risk_metrics": risk_metrics,
            "sentiment_data": sentiment_result,
            "headlines_count": len(headlines_data),
            "fundamental_analysis": fundamental_analysis,
            "momentum_analysis": momentum_analysis,
            "dcf_analysis": dcf_analysis,
            "symbol": symbol
        }
        
        print(f"Pipeline completed for {symbol}")
        return state
        
    except Exception as e:
        print(f"Error in pipeline for {symbol}: {e}")
        # Return default state with error handling
        return {
            "price_df": pd.DataFrame(),
            "arima_rmse": float('inf'),
            "lstm_rmse": float('inf'),
            "rf_rmse": 0.0,
            "sentiment": 0.0,
            "risk_score": 50.0,
            "risk_metrics": {"error": str(e)},
            "sentiment_data": {},
            "headlines_count": 0,
            "fundamental_analysis": {"error": str(e)},
            "momentum_analysis": {"error": str(e)},
            "dcf_analysis": {"error": str(e)},
            "symbol": symbol
        }

def run_ensemble_training(symbol="RELIANCE", period="1y"):
    # 1. Data ingestion: get historical price data
    df = ingest.history(symbol, period=period)

    # 2. Feature engineering: add technical indicators
    df_feat = features.enrich(df.copy())

    # 3. Create the target column: next day's closing price
    df_feat["Target"] = df_feat["Close"].shift(-1)

    # 4. Drop rows with any NaN values (introduced by shifts and rolling windows)
    df_feat.dropna(inplace=True)

    # 5. Train the random forest regression model with features + target
    rf_model, (rf_rmse, rf_mae, rf_r2), rf_predictions, rf_actuals = ensemble.train_rf(df_feat)

    # 6. Train the ARIMA model
    arima_model, (arima_rmse, arima_mae, arima_r2), arima_predictions, arima_actuals = arima.train_arima_for_pipeline(df_feat)

    # 7. Train the LSTM model
    lstm_model, (lstm_rmse, lstm_mae, lstm_r2), lstm_predictions, lstm_actuals = lstm.train_lstm_for_pipeline(df_feat)

    # 8. Output training metrics
    print(f"Random Forest Training Metrics for {symbol}:")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print(f"R2 Score: {rf_r2:.4f}")
    
    print(f"\nARIMA Training Metrics for {symbol}:")
    print(f"RMSE: {arima_rmse:.4f}")
    print(f"MAE: {arima_mae:.4f}")
    print(f"R2 Score: {arima_r2:.4f}")
    
    print(f"\nLSTM Training Metrics for {symbol}:")
    print(f"RMSE: {lstm_rmse:.4f}")
    print(f"MAE: {lstm_mae:.4f}")
    print(f"R2 Score: {lstm_r2:.4f}")

    # Optionally: return models and results for further use
    return (rf_model, rf_predictions, rf_actuals), (arima_model, arima_predictions, arima_actuals), (lstm_model, lstm_predictions, lstm_actuals)


# Example usage:
if __name__ == "__main__":
    run_ensemble_training("TCS", "1y")
