import streamlit as st
from indi_ml.pipeline import run_pipeline
# Enhanced pipeline support (optional)
try:
    from indi_ml.enhanced_pipeline import run_enhanced_pipeline, compare_pipelines
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Enhanced models not available. Using original models only.")

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Indian Stock ML Platform", layout="wide")

# Sidebar configuration
st.sidebar.title("Configuration")

# Model Selection Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 **Model Selection**")

# Enhanced models option (if available)
if ENHANCED_AVAILABLE:
    pipeline_mode = st.sidebar.selectbox(
        "Choose ML Models",
        ["🔧 Original Models", "🚀 Enhanced Models", "⚖️ Compare Both"],
        index=0,
        help="Select which ML models to use for analysis"
    )
    
    # Clean the mode for internal use
    if "Original" in pipeline_mode:
        clean_mode = "Original Models"
    elif "Enhanced" in pipeline_mode:
        clean_mode = "Enhanced Models"
    else:
        clean_mode = "Compare Both"
    
    # Show information about selected mode
    if "Enhanced" in pipeline_mode:
        st.sidebar.success("✅ **Enhanced Models Selected**")
        st.sidebar.markdown("""
        **Improvements:**
        - 🎯 15-30% better accuracy
        - 🛡️ Robust error handling
        - 🔄 Advanced validation
        - 📊 Multiple algorithms
        """)
    elif "Compare" in pipeline_mode:
        st.sidebar.info("📊 **Comparison Mode Selected**")
        st.sidebar.markdown("""
        **Features:**
        - 📈 Side-by-side comparison
        - ⏱️ Performance metrics
        - 📊 Accuracy improvements
        - 🔍 Detailed analysis
        """)
    else:
        st.sidebar.info("🔧 **Original Models Selected**")
        st.sidebar.markdown("""
        **Features:**
        - ⚡ Fast execution
        - 🎯 Basic ML models
        - 📊 Standard analysis
        - 🔄 Proven results
        """)
    
    # Set pipeline_mode for backward compatibility
    pipeline_mode = clean_mode
    
else:
    pipeline_mode = "Original Models"
    st.sidebar.warning("⚠️ **Enhanced models not available**")
    st.sidebar.markdown("""
    Using original models only.
    
    To enable enhanced models:
    1. Ensure all enhanced model files are present
    2. Check dependencies in requirements.txt
    3. Restart the application
    """)

symbol = st.sidebar.text_input("Symbol (NSE)", "RELIANCE").upper()
period = st.sidebar.selectbox("Data Period", ["1y", "6mo", "2y", "3y"], index=0)

if st.sidebar.button("Run / Refresh Analysis"):
    if ENHANCED_AVAILABLE and pipeline_mode == "Enhanced Models":
        with st.spinner("🚀 Running enhanced analysis with advanced ML models..."):
            state = run_enhanced_pipeline(symbol, period=period)
    elif ENHANCED_AVAILABLE and pipeline_mode == "Compare Both":
        with st.spinner("⚖️ Running comparison analysis..."):
            comparison = compare_pipelines(symbol, period)
            state = run_enhanced_pipeline(symbol, period=period)  # Use enhanced for display
            # Add comparison info to state
            state['comparison'] = comparison
    else:
        with st.spinner("Running comprehensive analysis..."):
            state = run_pipeline(symbol, period=period)

    # ---------- Model Selection Status Banner
    if ENHANCED_AVAILABLE and pipeline_mode == "Enhanced Models":
        st.success("🚀 **ENHANCED MODELS ACTIVE** - Using advanced ML algorithms with 15-30% better accuracy")
    elif ENHANCED_AVAILABLE and pipeline_mode == "Compare Both":
        st.info("⚖️ **COMPARISON MODE ACTIVE** - Analyzing both Original and Enhanced models")
    else:
        st.info("🔧 **ORIGINAL MODELS ACTIVE** - Using standard ML algorithms")
    
    # Quick model switcher in main area
    if ENHANCED_AVAILABLE:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;'>
                <h4 style='margin: 0; color: #1f77b4;'>🎯 Model Selection</h4>
                <p style='margin: 5px 0; font-size: 14px;'>Current: <strong>{}</strong></p>
                <p style='margin: 0; font-size: 12px; color: #666;'>Change models in the sidebar →</p>
            </div>
            """.format(pipeline_mode), unsafe_allow_html=True)
    
    # ---------- Header
    if ENHANCED_AVAILABLE and pipeline_mode == "Enhanced Models":
        st.markdown(f"## 🚀 {symbol} – Enhanced ML Forecast Dashboard")
        st.markdown(f"*Enhanced Analysis | Period: {period} | Data Points: {state.get('data_points', len(state['price_df']))} | Features: {state.get('feature_count', 'N/A')} | Execution Time: {state.get('execution_time', 0):.2f}s*")
    elif ENHANCED_AVAILABLE and pipeline_mode == "Compare Both":
        st.markdown(f"## ⚖️ {symbol} – Pipeline Comparison Dashboard")
        st.markdown(f"*Comparing Original vs Enhanced Models | Period: {period}*")
        
        # Display comparison metrics
        if 'comparison' in state:
            comp = state['comparison']
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔧 Original Pipeline")
                st.metric("Execution Time", f"{comp['original']['execution_time']:.2f}s")
                st.metric("ARIMA RMSE", f"{comp['original']['arima_rmse']:.2f}" if comp['original']['arima_rmse'] != float('inf') else "Failed")
                st.metric("LSTM RMSE", f"{comp['original']['lstm_rmse']:.2f}" if comp['original']['lstm_rmse'] != float('inf') else "Failed")
                st.metric("Random Forest RMSE", f"{comp['original']['rf_rmse']:.2f}")
            
            with col2:
                st.markdown("### 🚀 Enhanced Pipeline")
                st.metric("Execution Time", f"{comp['enhanced']['execution_time']:.2f}s")
                st.metric("Enhanced ARIMA RMSE", f"{comp['enhanced']['arima_rmse']:.2f}" if comp['enhanced']['arima_rmse'] != float('inf') else "Failed")
                st.metric("Enhanced LSTM RMSE", f"{comp['enhanced']['lstm_rmse']:.2f}" if comp['enhanced']['lstm_rmse'] != float('inf') else "Failed")
                st.metric("Enhanced Ensemble RMSE", f"{comp['enhanced']['rf_rmse']:.2f}")
                if comp['enhanced']['best_model'] != 'none':
                    st.success(f"🏆 Best Model: {comp['enhanced']['best_model']}")
    else:
        st.markdown(f"## {symbol} – Machine-Learning Forecast Dashboard")
        st.markdown(f"*Analysis Period: {period} | Data Points: {len(state['price_df'])}*")

    # ---------- Price chart
    st.markdown("### 📈 Price Chart")
    fig = go.Figure(data=[go.Candlestick(
            x=state["price_df"].index,
            open=state["price_df"]["Open"],
            high=state["price_df"]["High"],
            low =state["price_df"]["Low"],
            close=state["price_df"]["Close"])])
    fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Model Performance Metrics
    st.markdown("### 🤖 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # ARIMA metrics with color coding
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
    if rf_rmse == float('inf'):
        col3.metric("Random Forest RMSE", "Failed", delta="Error", delta_color="inverse")
    else:
        col3.metric("Random Forest RMSE", f"{rf_rmse:.2f}", 
                   delta="ML Model", delta_color="normal")
    
    # Sentiment metrics
    sentiment = state['sentiment']
    sentiment_color = "normal" if sentiment > 0 else "inverse" if sentiment < 0 else "off"
    col4.metric("News Sentiment", f"{sentiment:+.2f}", 
               delta="Sentiment Score", delta_color=sentiment_color)

    # ---------- Model Comparison
    st.markdown("### 📊 Model Comparison")
    
    # Collect successful models only
    successful_models = []
    successful_rmse = []
    model_colors = []
    
    if arima_rmse != float('inf'):
        successful_models.append("ARIMA")
        successful_rmse.append(arima_rmse)
        model_colors.append('#FF6B6B')
    
    if lstm_rmse != float('inf'):
        successful_models.append("LSTM")
        successful_rmse.append(lstm_rmse)
        model_colors.append('#9B59B6')
    
    if rf_rmse != float('inf'):
        successful_models.append("Random Forest")
        successful_rmse.append(rf_rmse)
        model_colors.append('#4ECDC4')
    
    if len(successful_models) > 0:
        # Create comparison chart for successful models only
        fig = px.bar(x=successful_models, y=successful_rmse, 
                    title="Model Performance Comparison (Lower RMSE = Better)",
                    labels={"x": "Model", "y": "RMSE"})
        fig.update_traces(marker_color=model_colors)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model indicator
        if len(successful_models) > 1:
            rmse_dict = dict(zip(successful_models, successful_rmse))
            best_model = min(rmse_dict, key=rmse_dict.get)
            best_rmse = rmse_dict[best_model]
            st.success(f"🏆 **Best Performing Model**: {best_model} (RMSE: {best_rmse:.2f})")
        else:
            st.info(f"📊 **Only Model Available**: {successful_models[0]} (RMSE: {successful_rmse[0]:.2f})")
        
        # Show failed models if any
        failed_models = []
        if arima_rmse == float('inf'):
            failed_models.append("ARIMA")
        if lstm_rmse == float('inf'):
            failed_models.append("LSTM")
        if rf_rmse == float('inf'):
            failed_models.append("Random Forest")
        
        if failed_models:
            st.warning(f"⚠️ **Failed Models**: {', '.join(failed_models)} - Check data quality or model parameters")
            
            # Suggest Enhanced Models if available
            if ENHANCED_AVAILABLE:
                st.info("🚀 **Tip**: Try Enhanced Models for better robustness and error handling - select 'Enhanced Models' in the sidebar!")
    else:
        st.error("❌ **All models failed** - Please check your data and try again")
        st.markdown("""
        **Possible solutions:**
        - Try a longer time period for more data
        - Check if the stock symbol is valid
        - Ensure sufficient historical data is available
        - Try using Enhanced Models for better robustness
        """)
    
    # Model descriptions
    st.markdown("#### Model Descriptions:")
    st.markdown("- **ARIMA**: Time series model using autoregressive integrated moving average")
    st.markdown("- **LSTM**: Deep learning model using Long Short-Term Memory networks")
    st.markdown("- **Random Forest**: Machine learning model using ensemble of decision trees")

    # ---------- Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")
    risk_score = state['risk_score']
    
    # Color code risk score
    if risk_score < 30:
        risk_color = "normal"
        risk_level = "Low Risk"
    elif risk_score < 70:
        risk_color = "off"
        risk_level = "Medium Risk"
    else:
        risk_color = "inverse"
        risk_level = "High Risk"
    
    col1, col2 = st.columns(2)
    col1.metric("Risk Score", f"{risk_score:.1f}/100", risk_level, delta_color=risk_color)
    col2.metric("Headlines Analyzed", state['headlines_count'])
    
    # Risk level indicator with emoji
    if risk_score < 30:
        st.success(f"🟢 {risk_level} - Safe investment zone")
    elif risk_score < 70:
        st.warning(f"🟡 {risk_level} - Moderate caution advised")
    else:
        st.error(f"🔴 {risk_level} - High caution advised")
    
    # Risk metrics details
    st.markdown("#### Detailed Risk Metrics")
    st.json(state["risk_metrics"])

    # ---------- Sentiment Analysis
    st.markdown("### 📰 Sentiment Analysis")
    
    # Overall sentiment score
    sentiment_score = state['sentiment']
    st.metric("Overall Sentiment Score", f"{sentiment_score:+.3f}", 
              delta="Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral",
              delta_color="normal" if sentiment_score > 0 else "inverse" if sentiment_score < 0 else "off")
    
    # Headlines count
    headlines_count = state['headlines_count']
    st.write(f"**Total Headlines Analyzed:** {headlines_count}")
    
    # Detailed sentiment breakdown
    if state['sentiment_data'] and len(state['sentiment_data']) > 0:
        sentiment_data = state['sentiment_data']
        
        # Show overall sentiment metrics
        if 'overall_sentiment' in sentiment_data:
            st.write(f"**Overall Sentiment:** {sentiment_data['overall_sentiment']:.3f}")
        if 'weighted_sentiment' in sentiment_data:
            st.write(f"**Weighted Sentiment:** {sentiment_data['weighted_sentiment']:.3f}")
        if 'total_headlines' in sentiment_data:
            st.write(f"**Total Headlines:** {sentiment_data['total_headlines']}")
        
        # Show source breakdown if available
        if 'source_breakdown' in sentiment_data and sentiment_data['source_breakdown']:
            st.write("**Source Breakdown:**")
            for source, data in sentiment_data['source_breakdown'].items():
                if data.get('count', 0) > 0:
                    avg_sentiment = data.get('avg_sentiment', 0)
                    st.write(f"- {source}: {data['count']} headlines, sentiment: {avg_sentiment:.3f}")
        else:
            st.info("No detailed source breakdown available. This is normal when few headlines are found.")
        
        # Show sentiment distribution if available
        if 'sentiment_distribution' in sentiment_data and sentiment_data['sentiment_distribution']:
            st.write("**Sentiment Distribution:**")
            for sentiment_type, count in sentiment_data['sentiment_distribution'].items():
                if count > 0:
                    st.write(f"- {sentiment_type}: {count} headlines")
    else:
        st.info("No sentiment data available. This may happen when no relevant news headlines are found.")

    # ---------- Fundamental Analysis
    st.markdown("### 📊 Fundamental Analysis")
    
    if 'fundamental_analysis' in state and state['fundamental_analysis']:
        fa = state['fundamental_analysis']
        
        if 'error' not in fa:
            # Financial Ratios
            st.markdown("#### 💰 Financial Ratios")
            ratios = fa.get('ratios', {})
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("P/E Ratio", f"{ratios.get('pe_ratio', 0):.2f}")
            col2.metric("P/B Ratio", f"{ratios.get('pb_ratio', 0):.2f}")
            col3.metric("ROE (%)", f"{ratios.get('roe', 0):.1f}")
            col4.metric("ROA (%)", f"{ratios.get('roa', 0):.1f}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Net Margin (%)", f"{ratios.get('net_margin', 0):.1f}")
            col2.metric("Debt/Equity", f"{ratios.get('debt_to_equity', 0):.2f}")
            col3.metric("Dividend Yield (%)", f"{ratios.get('dividend_yield', 0)*100:.2f}")
            col4.metric("P/S Ratio", f"{ratios.get('ps_ratio', 0):.2f}")
            
            # Earnings Analysis
            st.markdown("#### 📈 Earnings Analysis")
            earnings = fa.get('earnings', {})
            
            if earnings:
                col1, col2, col3 = st.columns(3)
                col1.metric("Revenue Growth (1Y)", f"{earnings.get('revenue_growth_1y', 0):.1f}%")
                col2.metric("Net Income Growth (1Y)", f"{earnings.get('net_income_growth_1y', 0):.1f}%")
                col3.metric("Cash Flow Coverage", f"{earnings.get('cash_flow_coverage', 0):.2f}")
            
            # Valuation Analysis
            st.markdown("#### 💎 Valuation Analysis")
            valuation = fa.get('valuation', {})
            
            if valuation:
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{valuation.get('current_price', 0):.2f}")
                col2.metric("Book Value", f"₹{valuation.get('book_value', 0):.2f}")
                col3.metric("PEG Ratio", f"{valuation.get('peg_ratio', 0):.2f}")
            
            # Risk Assessment
            st.markdown("#### ⚠️ Fundamental Risk Assessment")
            risk = fa.get('risk', {})
            
            if risk:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Debt Risk", risk.get('debt_risk', 'N/A'))
                col2.metric("Profitability Risk", risk.get('profitability_risk', 'N/A'))
                col3.metric("Growth Risk", risk.get('growth_risk', 'N/A'))
                col4.metric("Earnings Quality Risk", risk.get('earnings_quality_risk', 'N/A'))
                
                # Overall fundamental risk score
                fundamental_risk_score = risk.get('overall_risk_score', 0)
                st.metric("Overall Fundamental Risk Score", f"{fundamental_risk_score:.1f}/100")
                
                if fundamental_risk_score < 30:
                    st.success("🟢 Low fundamental risk - Strong financial position")
                elif fundamental_risk_score < 70:
                    st.warning("🟡 Moderate fundamental risk - Monitor closely")
                else:
                    st.error("🔴 High fundamental risk - Exercise caution")
        else:
            st.error(f"Fundamental analysis failed: {fa['error']}")
    else:
        st.info("Fundamental analysis not available for this stock.")

    # ---------- Momentum Analysis
    st.markdown("### 📈 Momentum Analysis")
    
    if 'momentum_analysis' in state and state['momentum_analysis']:
        ma = state['momentum_analysis']
        
        if 'error' not in ma:
            # Momentum Indicators
            st.markdown("#### 🔄 Momentum Indicators")
            signals = ma.get('signals', {})
            scoring = ma.get('scoring', {})
            
            # RSI Analysis
            col1, col2, col3 = st.columns(3)
            col1.metric("RSI Value", f"{signals.get('rsi_value', 0):.1f}")
            col2.metric("RSI Signal", signals.get('rsi_signal', 'N/A'))
            col3.metric("RSI Strength", signals.get('rsi_strength', 'N/A'))
            
            # MACD Analysis
            col1, col2, col3 = st.columns(3)
            col1.metric("MACD Value", f"{signals.get('macd_value', 0):.4f}")
            col2.metric("MACD Signal", signals.get('macd_signal', 'N/A'))
            col3.metric("MACD Strength", signals.get('macd_strength', 'N/A'))
            
            # Stochastic Analysis
            col1, col2, col3 = st.columns(3)
            col1.metric("Stoch %K", f"{signals.get('stoch_k', 0):.1f}")
            col2.metric("Stoch %D", f"{signals.get('stoch_d', 0):.1f}")
            col3.metric("Stoch Signal", signals.get('stoch_signal', 'N/A'))
            
            # Price Momentum
            st.markdown("#### 📊 Price Momentum")
            col1, col2, col3 = st.columns(3)
            col1.metric("5-Day Momentum", f"{signals.get('momentum_5d', 0):.2f}%")
            col2.metric("10-Day Momentum", f"{signals.get('momentum_10d', 0):.2f}%")
            col3.metric("20-Day Momentum", f"{signals.get('momentum_20d', 0):.2f}%")
            
            # Momentum Trend
            col1, col2 = st.columns(2)
            col1.metric("Momentum Trend", signals.get('momentum_trend', 'N/A'))
            col2.metric("Momentum Strength", signals.get('momentum_strength', 'N/A'))
            
            # Overall Momentum Score
            st.markdown("#### 🎯 Momentum Scoring")
            overall_score = scoring.get('overall_score', 50)
            momentum_rating = scoring.get('momentum_rating', 'Neutral')
            recommendation = scoring.get('recommendation', 'Hold')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Score", f"{overall_score:.1f}/100")
            col2.metric("Momentum Rating", momentum_rating)
            col3.metric("Recommendation", recommendation)
            
            # Momentum score visualization
            if overall_score >= 80:
                st.success(f"🚀 Very Strong Momentum - {recommendation}")
            elif overall_score >= 60:
                st.info(f"📈 Strong Momentum - {recommendation}")
            elif overall_score >= 40:
                st.warning(f"📊 Moderate Momentum - {recommendation}")
            elif overall_score >= 20:
                st.error(f"📉 Weak Momentum - {recommendation}")
            else:
                st.error(f"💥 Very Weak Momentum - {recommendation}")
            
            # Detailed scoring breakdown
            st.markdown("#### 📋 Detailed Scoring")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI Score", f"{scoring.get('rsi_score', 0):.1f}/25")
            col2.metric("MACD Score", f"{scoring.get('macd_score', 0):.1f}/25")
            col3.metric("Stochastic Score", f"{scoring.get('stoch_score', 0):.1f}/25")
            col4.metric("Momentum Score", f"{scoring.get('momentum_score', 0):.1f}/25")
            
        else:
            st.error(f"Momentum analysis failed: {ma['error']}")
    else:
        st.info("Momentum analysis not available for this stock.")

    # ---------- DCF and Intrinsic Value Analysis
    st.markdown("### 💰 DCF and Intrinsic Value Analysis")
    
    if 'dcf_analysis' in state and state['dcf_analysis']:
        dcf = state['dcf_analysis']
        
        if 'error' not in dcf:
            # Intrinsic Value Analysis
            st.markdown("#### 🎯 Intrinsic Value Analysis")
            intrinsic_value = dcf.get('intrinsic_value', 0)
            current_price = dcf.get('current_price', 0)
            margin_of_safety_value = dcf.get('margin_of_safety_value', 0)
            valuation_ratio = dcf.get('valuation_ratio', 0)
            
            # Handle zero or negative values
            if intrinsic_value <= 0:
                st.warning("⚠️ Intrinsic value calculation failed - insufficient financial data")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Intrinsic Value", f"₹{intrinsic_value:,.2f}")
                col2.metric("Current Price", f"₹{current_price:,.2f}")
                col3.metric("Valuation Ratio", f"{valuation_ratio:.2f}x")
                
                # Margin of Safety
                col1, col2 = st.columns(2)
                col1.metric("Margin of Safety Value", f"₹{margin_of_safety_value:,.2f}")
                col2.metric("Margin of Safety %", f"{dcf.get('margin_of_safety_percentage', 25):.1f}%")
                
                # Valuation Assessment
                if valuation_ratio > 1.5:
                    st.success("🚀 Undervalued - Strong buy opportunity")
                elif valuation_ratio > 1.2:
                    st.info("📈 Undervalued - Good buy opportunity")
                elif valuation_ratio > 0.8:
                    st.warning("📊 Fairly valued - Hold position")
                elif valuation_ratio > 0.6:
                    st.error("📉 Overvalued - Consider selling")
                else:
                    st.error("💥 Significantly overvalued - Strong sell")
                
                # DCF Components
                st.markdown("#### 📊 DCF Components")
                wacc = dcf.get('wacc', 0)
                terminal_value = dcf.get('terminal_value', 0)
                enterprise_value = dcf.get('enterprise_value', 0)
                equity_value = dcf.get('equity_value', 0)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("WACC", f"{wacc:.1%}")
                col2.metric("Terminal Value", f"₹{terminal_value:,.0f}")
                col3.metric("Enterprise Value", f"₹{enterprise_value:,.0f}")
                col4.metric("Equity Value", f"₹{equity_value:,.0f}")
                
                # Projected Cash Flows
                st.markdown("#### 📈 Projected Cash Flows")
                projected_fcf = dcf.get('projected_fcf', {})
                if projected_fcf:
                    fcf_data = []
                    for year, fcf in projected_fcf.items():
                        fcf_data.append({"Year": f"Year {year}", "FCF": fcf})
                    
                    fcf_df = pd.DataFrame(fcf_data)
                    st.dataframe(fcf_df, use_container_width=True)
                
                # Present Values
                st.markdown("#### 💰 Present Values")
                present_values = dcf.get('present_values', {})
                if present_values:
                    pv_data = []
                    for year, pv in present_values.items():
                        pv_data.append({"Year": f"Year {year}", "Present Value": pv})
                    
                    pv_df = pd.DataFrame(pv_data)
                    st.dataframe(pv_df, use_container_width=True)
                
                # Sensitivity Analysis
                st.markdown("#### 🔍 Sensitivity Analysis")
                sensitivity = dcf.get('sensitivity_analysis', {})
                if sensitivity:
                    # Create sensitivity heatmap
                    sensitivity_data = []
                    for key, value in sensitivity.items():
                        if 'wacc_' in key:
                            wacc_val = float(key.split('_')[1])
                            sensitivity_data.append({"Variable": "WACC", "Value": f"{wacc_val:.1%}", "Enterprise Value": value})
                        elif 'growth_' in key:
                            growth_val = float(key.split('_')[1])
                            sensitivity_data.append({"Variable": "Growth Rate", "Value": f"{growth_val:.1%}", "Enterprise Value": value})
                    
                    if sensitivity_data:
                        sensitivity_df = pd.DataFrame(sensitivity_data)
                        st.dataframe(sensitivity_df, use_container_width=True)
                
                # Historical FCF
                st.markdown("#### 📊 Historical Free Cash Flow")
                historical_fcf = dcf.get('historical_fcf', {})
                if historical_fcf:
                    hfcf_data = []
                    for year, fcf in historical_fcf.items():
                        hfcf_data.append({"Year": year, "FCF": fcf})
                    
                    hfcf_df = pd.DataFrame(hfcf_data)
                    st.dataframe(hfcf_df, use_container_width=True)
                
                # Shares Outstanding
                shares_outstanding = dcf.get('shares_outstanding', 0)
                st.metric("Shares Outstanding", f"{shares_outstanding:,.0f}")
        else:
            st.error(f"DCF analysis failed: {dcf['error']}")
    else:
        st.info("DCF analysis not available for this stock.")

    # ---------- Footer
    st.markdown("---")
    if ENHANCED_AVAILABLE:
        st.markdown("*Powered by Enhanced Indian Stock ML Platform - Next-generation ML models with advanced validation*")
        st.info("💡 **Tip**: For the full enhanced experience with advanced visualizations and model details, run `streamlit run enhanced_app.py`")
    else:
        st.markdown("*Powered by Indian Stock ML Platform - Combining Technical Analysis, Machine Learning, and Sentiment Analysis*")
        st.info("🚀 **Upgrade Available**: Enhanced ML models with better accuracy are available. Check the enhanced_app.py file.")

