"""
Enhanced Streamlit Application for Indian Stock ML Platform

This enhanced version provides:
- Support for both original and enhanced ML models
- Model comparison and selection
- Enhanced visualizations and metrics
- Performance tracking and monitoring
- Advanced configuration options

Author: Enhanced ML Team
Date: 2024
"""

import streamlit as st
from indi_ml.pipeline import run_pipeline
from indi_ml.enhanced_pipeline import run_enhanced_pipeline, compare_pipelines
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time

st.set_page_config(page_title="Enhanced Indian Stock ML Platform", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.enhanced-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.model-comparison {
    border: 2px solid #e1e5e9;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🚀 Enhanced Configuration")

# Model selection
pipeline_mode = st.sidebar.selectbox(
    "Pipeline Mode",
    ["Enhanced Pipeline", "Original Pipeline", "Compare Both"],
    index=0,
    help="Choose between enhanced ML models or original models"
)

symbol = st.sidebar.text_input("Symbol (NSE)", "RELIANCE").upper()
period = st.sidebar.selectbox("Data Period", ["1y", "6mo", "2y", "3y"], index=0)

# Enhanced model configuration
if pipeline_mode in ["Enhanced Pipeline", "Compare Both"]:
    st.sidebar.markdown("### 🔧 Enhanced Model Settings")
    
    use_cache = st.sidebar.checkbox("Use Model Caching", value=True, 
                                   help="Cache trained models to avoid retraining")
    
    model_selection = st.sidebar.selectbox(
        "Model Selection Strategy",
        ["best_cv", "best_rmse", "ensemble_all"],
        index=0,
        help="Strategy for selecting the best model"
    )
    
    show_advanced_metrics = st.sidebar.checkbox("Show Advanced Metrics", value=True)
    show_model_details = st.sidebar.checkbox("Show Model Details", value=False)

# Main application
if st.sidebar.button("🔄 Run / Refresh Analysis"):
    start_time = time.time()
    
    if pipeline_mode == "Enhanced Pipeline":
        # Run enhanced pipeline
        with st.spinner("🚀 Running enhanced analysis with advanced ML models..."):
            state = run_enhanced_pipeline(
                symbol, 
                period=period, 
                use_cache=use_cache,
                model_selection_strategy=model_selection
            )
        
        # Enhanced header
        st.markdown(f"""
        <div class="enhanced-header">
            <h2>🚀 {symbol} – Enhanced ML Forecast Dashboard</h2>
            <p>Analysis Period: {period} | Data Points: {state.get('data_points', 0)} | 
            Features: {state.get('feature_count', 0)} | 
            Execution Time: {state.get('execution_time', 0):.2f}s</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced model performance section
        st.markdown("### 🤖 Enhanced Model Performance")
        
        # Display enhanced model metrics
        enhanced_models = state.get('enhanced_models', {})
        
        if enhanced_models:
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            # Enhanced Ensemble
            ensemble_metrics = enhanced_models.get('enhanced_ensemble', {}).get('metrics', (float('inf'), float('inf'), 0.0))
            if ensemble_metrics[0] != float('inf'):
                col1.metric("Enhanced Ensemble RMSE", f"{ensemble_metrics[0]:.2f}", 
                          delta="Multi-Algorithm", delta_color="normal")
            else:
                col1.metric("Enhanced Ensemble RMSE", "Failed", delta="Error", delta_color="inverse")
            
            # Enhanced ARIMA
            arima_metrics = enhanced_models.get('enhanced_arima', {}).get('metrics', (float('inf'), float('inf'), 0.0))
            if arima_metrics[0] != float('inf'):
                col2.metric("Enhanced ARIMA RMSE", f"{arima_metrics[0]:.2f}", 
                          delta="Seasonal + Diagnostics", delta_color="normal")
            else:
                col2.metric("Enhanced ARIMA RMSE", "Failed", delta="Error", delta_color="inverse")
            
            # Enhanced LSTM
            lstm_metrics = enhanced_models.get('enhanced_lstm', {}).get('metrics', (float('inf'), float('inf'), 0.0))
            if lstm_metrics[0] != float('inf'):
                col3.metric("Enhanced LSTM RMSE", f"{lstm_metrics[0]:.2f}", 
                          delta="Advanced Architecture", delta_color="normal")
            else:
                col3.metric("Enhanced LSTM RMSE", "Failed", delta="Error", delta_color="inverse")
            
            # Best Model
            best_model = state.get('best_model')
            if best_model:
                col4.metric("🏆 Best Model", best_model.get('model', 'None'), 
                          delta=f"RMSE: {best_model.get('rmse', 0):.2f}", delta_color="normal")
            else:
                col4.metric("🏆 Best Model", "None Selected", delta="No valid models", delta_color="off")
            
            # Advanced metrics display
            if show_advanced_metrics:
                st.markdown("### 📊 Advanced Performance Metrics")
                
                metrics_data = []
                for model_name, model_info in enhanced_models.items():
                    if model_info['model'] is not None:
                        metrics = model_info['metrics']
                        metrics_data.append({
                            'Model': model_name.replace('enhanced_', '').title(),
                            'RMSE': f"{metrics[0]:.4f}" if metrics[0] != float('inf') else "Failed",
                            'MAE': f"{metrics[1]:.4f}" if metrics[1] != float('inf') else "Failed",
                            'R²': f"{metrics[2]:.4f}" if metrics[2] != 0.0 else "0.0000",
                            'Status': "✅ Success" if metrics[0] != float('inf') else "❌ Failed"
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Model comparison chart
                st.markdown("### 📈 Enhanced Model Comparison")
                valid_models = [(name.replace('enhanced_', '').title(), metrics[0]) 
                              for name, model_info in enhanced_models.items() 
                              if model_info['metrics'][0] != float('inf')]
                
                if len(valid_models) >= 2:
                    models, rmse_values = zip(*valid_models)
                    
                    fig = px.bar(
                        x=list(models), 
                        y=list(rmse_values),
                        title="Enhanced Model Performance Comparison (Lower RMSE = Better)",
                        labels={"x": "Model", "y": "RMSE"},
                        color=list(rmse_values),
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model highlight
                    best_model_name, best_rmse = min(valid_models, key=lambda x: x[1])
                    st.success(f"🏆 **Best Performing Model**: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        # Model details section
        if show_model_details and enhanced_models:
            st.markdown("### 🔍 Model Details")
            
            for model_name, model_info in enhanced_models.items():
                if model_info['model'] is not None:
                    with st.expander(f"{model_name.replace('enhanced_', '').title()} Model Details"):
                        st.write(f"**Model Type**: {model_name}")
                        st.write(f"**Training Status**: {'✅ Success' if model_info['metrics'][0] != float('inf') else '❌ Failed'}")
                        
                        if hasattr(model_info['model'], 'feature_importance') and model_info['model'].feature_importance is not None:
                            st.write("**Feature Importance Available**: Yes")
                        
                        if len(model_info['predictions']) > 0:
                            st.write(f"**Predictions Generated**: {len(model_info['predictions'])}")
                            
                            # Show prediction vs actual plot
                            if len(model_info['actuals']) > 0:
                                pred_df = pd.DataFrame({
                                    'Actual': model_info['actuals'][:50],  # Limit to 50 points for visualization
                                    'Predicted': model_info['predictions'][:50]
                                })
                                
                                fig = px.line(pred_df, title=f"{model_name.title()} - Predictions vs Actual")
                                fig.add_scatter(x=pred_df.index, y=pred_df['Actual'], name='Actual', line=dict(color='blue'))
                                fig.add_scatter(x=pred_df.index, y=pred_df['Predicted'], name='Predicted', line=dict(color='red'))
                                st.plotly_chart(fig, use_container_width=True)
    
    elif pipeline_mode == "Original Pipeline":
        # Run original pipeline
        with st.spinner("Running original analysis..."):
            state = run_pipeline(symbol, period=period)
        
        # Original header
        st.markdown(f"## {symbol} – Original ML Forecast Dashboard")
        st.markdown(f"*Analysis Period: {period} | Data Points: {len(state['price_df'])}*")
        
        # Original model metrics
        st.markdown("### 🤖 Original Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        # ARIMA metrics
        arima_rmse = state['arima_rmse']
        if arima_rmse == float('inf'):
            col1.metric("ARIMA RMSE", "Failed", delta="Error", delta_color="inverse")
        else:
            col1.metric("ARIMA RMSE", f"{arima_rmse:.2f}", 
                       delta="Time Series Model", delta_color="normal")
        
        # LSTM metrics
        lstm_rmse = state['lstm_rmse']
        if lstm_rmse == float('inf'):
            col2.metric("LSTM RMSE", "Failed", delta="Error", delta_color="inverse")
        else:
            col2.metric("LSTM RMSE", f"{lstm_rmse:.2f}", 
                       delta="Deep Learning", delta_color="normal")
        
        # Random Forest metrics
        rf_rmse = state['rf_rmse']
        col3.metric("Random Forest RMSE", f"{rf_rmse:.2f}", 
                   delta="ML Model", delta_color="normal")
        
        # Sentiment metrics
        sentiment = state['sentiment']
        sentiment_color = "normal" if sentiment > 0 else "inverse" if sentiment < 0 else "off"
        col4.metric("News Sentiment", f"{sentiment:+.2f}", 
                   delta="Sentiment Score", delta_color=sentiment_color)
    
    else:  # Compare Both
        # Run comparison
        with st.spinner("🔄 Running comprehensive comparison..."):
            comparison = compare_pipelines(symbol, period)
            enhanced_state = run_enhanced_pipeline(symbol, period, use_cache=use_cache)
            original_state = run_pipeline(symbol, period)
        
        # Comparison header
        st.markdown(f"""
        <div class="enhanced-header">
            <h2>⚖️ {symbol} – Pipeline Comparison Dashboard</h2>
            <p>Comparing Original vs Enhanced ML Models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison metrics
        st.markdown("### 📊 Pipeline Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Original Pipeline")
            st.metric("Execution Time", f"{comparison['original']['execution_time']:.2f}s")
            st.metric("ARIMA RMSE", f"{comparison['original']['arima_rmse']:.2f}" if comparison['original']['arima_rmse'] != float('inf') else "Failed")
            st.metric("LSTM RMSE", f"{comparison['original']['lstm_rmse']:.2f}" if comparison['original']['lstm_rmse'] != float('inf') else "Failed")
            st.metric("Random Forest RMSE", f"{comparison['original']['rf_rmse']:.2f}")
        
        with col2:
            st.markdown("#### 🚀 Enhanced Pipeline")
            st.metric("Execution Time", f"{comparison['enhanced']['execution_time']:.2f}s")
            st.metric("Enhanced ARIMA RMSE", f"{comparison['enhanced']['arima_rmse']:.2f}" if comparison['enhanced']['arima_rmse'] != float('inf') else "Failed")
            st.metric("Enhanced LSTM RMSE", f"{comparison['enhanced']['lstm_rmse']:.2f}" if comparison['enhanced']['lstm_rmse'] != float('inf') else "Failed")
            st.metric("Enhanced Ensemble RMSE", f"{comparison['enhanced']['rf_rmse']:.2f}")
            if comparison['enhanced']['best_model'] != 'none':
                st.success(f"🏆 Best Model: {comparison['enhanced']['best_model']}")
        
        # Performance improvement calculation
        st.markdown("### 📈 Performance Improvements")
        
        improvements = []
        if comparison['original']['arima_rmse'] != float('inf') and comparison['enhanced']['arima_rmse'] != float('inf'):
            arima_improvement = ((comparison['original']['arima_rmse'] - comparison['enhanced']['arima_rmse']) / comparison['original']['arima_rmse']) * 100
            improvements.append(('ARIMA', arima_improvement))
        
        if comparison['original']['lstm_rmse'] != float('inf') and comparison['enhanced']['lstm_rmse'] != float('inf'):
            lstm_improvement = ((comparison['original']['lstm_rmse'] - comparison['enhanced']['lstm_rmse']) / comparison['original']['lstm_rmse']) * 100
            improvements.append(('LSTM', lstm_improvement))
        
        if improvements:
            improvement_df = pd.DataFrame(improvements, columns=['Model', 'Improvement %'])
            
            fig = px.bar(improvement_df, x='Model', y='Improvement %', 
                        title="Performance Improvement (% reduction in RMSE)",
                        color='Improvement %',
                        color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        
        state = enhanced_state  # Use enhanced state for remaining sections
    
    # Common sections (Price chart, Risk, Sentiment, etc.)
    if not state.get('error'):
        # Price chart
        st.markdown("### 📈 Price Chart")
        if not state['price_df'].empty:
            fig = go.Figure(data=[go.Candlestick(
                x=state["price_df"].index,
                open=state["price_df"]["Open"],
                high=state["price_df"]["High"],
                low=state["price_df"]["Low"],
                close=state["price_df"]["Close"]
            )])
            fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk Assessment
        st.markdown("### ⚠️ Risk Assessment")
        risk_score = state['risk_score']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level indicator
            if risk_score < 30:
                st.success("🟢 **Low Risk** - Favorable conditions")
            elif risk_score < 70:
                st.warning("🟡 **Medium Risk** - Monitor closely")
            else:
                st.error("🔴 **High Risk** - Exercise caution")
            
            # Risk metrics
            risk_metrics = state.get('risk_metrics', {})
            if isinstance(risk_metrics, dict) and 'error' not in risk_metrics:
                st.write("**Risk Metrics:**")
                for key, value in list(risk_metrics.items())[:5]:  # Show top 5 metrics
                    if isinstance(value, (int, float)):
                        st.write(f"- {key.replace('_', ' ').title()}: {value:.4f}")
        
        # Sentiment Analysis
        st.markdown("### 📰 Sentiment Analysis")
        sentiment = state['sentiment']
        headlines_count = state['headlines_count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "News Sentiment"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "red"},
                        {'range': [-0.3, 0.3], 'color': "yellow"},
                        {'range': [0.3, 1], 'color': "green"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Headlines Analyzed", headlines_count)
            
            if sentiment > 0.3:
                st.success("🟢 **Positive Sentiment** - Bullish news")
            elif sentiment < -0.3:
                st.error("🔴 **Negative Sentiment** - Bearish news")
            else:
                st.info("🟡 **Neutral Sentiment** - Mixed signals")
        
        # Additional analysis sections (Fundamental, Momentum, DCF) can be added here
        # Following the same pattern as the original app.py
        
        # Performance summary
        execution_time = time.time() - start_time
        st.markdown("---")
        st.markdown(f"### ⏱️ Execution Summary")
        st.info(f"Total analysis completed in {execution_time:.2f} seconds using {pipeline_mode}")
        
        if pipeline_mode == "Enhanced Pipeline":
            st.success("✅ Enhanced ML models provide improved accuracy and robustness")
    
    else:
        st.error(f"❌ Analysis failed: {state.get('error', 'Unknown error')}")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About Enhanced Models")
st.sidebar.info("""
**Enhanced Features:**
- 🎯 Multiple ML algorithms
- 🔄 Advanced hyperparameter tuning
- 📊 Time series validation
- 🧠 Deep learning architectures
- 📈 Performance tracking
- 💾 Model caching
""")

st.sidebar.markdown("### 📊 Model Comparison")
st.sidebar.write("""
**Original Models:**
- Basic Random Forest
- Simple ARIMA
- Standard LSTM

**Enhanced Models:**
- Multi-algorithm ensemble
- Seasonal ARIMA with diagnostics
- Advanced LSTM architectures
""")

# Footer
st.markdown("---")
st.markdown("*🚀 Powered by Enhanced Indian Stock ML Platform - Next-generation ML models with advanced validation and robustness*")
