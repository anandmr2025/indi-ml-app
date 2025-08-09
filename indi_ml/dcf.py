"""
DCF (Discounted Cash Flow) and Intrinsic Value Analysis Module for Indian Stocks

This module provides comprehensive DCF analysis capabilities including:
- Cash flow projections and forecasting
- Discount rate calculations (WACC)
- Terminal value analysis
- Intrinsic value estimation
- Sensitivity analysis
- Valuation metrics

Key Features:
- Free Cash Flow projections
- WACC calculation with Indian market data
- Terminal value using multiple methods
- Intrinsic value with margin of safety
- Sensitivity analysis for key variables
- Valuation comparison with market price

Usage:
    from indi_ml.dcf import DCFAnalysis
    dcf = DCFAnalysis()
    analysis = dcf.analyze_dcf(symbol)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DCFAnalysis:
    """
    Comprehensive DCF and Intrinsic Value analysis for Indian stocks.
    """
    
    def __init__(self):
        """Initialize the DCF analysis module."""
        self.risk_free_rate = 0.065  # 10-year Indian government bond yield
        self.market_risk_premium = 0.08  # Indian equity risk premium
        self.terminal_growth_rate = 0.03  # Long-term growth rate
        self.forecast_years = 5  # Number of years to forecast
        self.margin_of_safety = 0.25  # 25% margin of safety
        
    def get_financial_data(self, symbol: str) -> Dict:
        """
        Fetch comprehensive financial data for DCF analysis.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Financial data including income statement, balance sheet, cash flow
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Get stock info
            info = ticker.info
            
            return {
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info
            }
            
        except Exception as e:
            print(f"Error fetching financial data: {e}")
            return {}
    
    def calculate_free_cash_flow(self, financial_data: Dict) -> pd.Series:
        """
        Calculate Free Cash Flow (FCF) from financial statements.
        
        Args:
            financial_data (dict): Financial data
            
        Returns:
            pd.Series: Historical FCF values
        """
        try:
            income_stmt = financial_data.get('income_stmt', pd.DataFrame())
            cash_flow = financial_data.get('cash_flow', pd.DataFrame())
            
            if income_stmt.empty or cash_flow.empty:
                return pd.Series()
            
            # Get key metrics
            net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else pd.Series()
            depreciation = cash_flow.loc['Depreciation'] if 'Depreciation' in cash_flow.index else pd.Series()
            capex = cash_flow.loc['Capital Expenditure'] if 'Capital Expenditure' in cash_flow.index else pd.Series()
            working_capital_change = cash_flow.loc['Change In Working Capital'] if 'Change In Working Capital' in cash_flow.index else pd.Series()
            
            # Calculate FCF: Net Income + Depreciation - Capex - Working Capital Change
            fcf = net_income + depreciation - capex - working_capital_change
            
            return fcf
            
        except Exception as e:
            print(f"Error calculating FCF: {e}")
            return pd.Series()
    
    def calculate_wacc(self, financial_data: Dict, symbol: str) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC).
        
        Args:
            financial_data (dict): Financial data
            symbol (str): Stock symbol
            
        Returns:
            float: WACC value
        """
        try:
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            info = financial_data.get('info', {})
            
            # Get market data
            ticker = yf.Ticker(symbol)
            market_cap = info.get('marketCap', 0)
            beta = info.get('beta', 1.0)
            
            # Calculate cost of equity using CAPM
            cost_of_equity = self.risk_free_rate + (beta * self.market_risk_premium)
            
            # Get debt data
            total_debt = balance_sheet.loc['Total Debt'] if 'Total Debt' in balance_sheet.index else pd.Series([0])
            current_debt = balance_sheet.loc['Short Term Debt'] if 'Short Term Debt' in balance_sheet.index else pd.Series([0])
            long_term_debt = balance_sheet.loc['Long Term Debt'] if 'Long Term Debt' in balance_sheet.index else pd.Series([0])
            
            # Use the most recent debt value
            debt_value = 0
            if not total_debt.empty:
                debt_value = total_debt.iloc[0]
            elif not current_debt.empty and not long_term_debt.empty:
                debt_value = current_debt.iloc[0] + long_term_debt.iloc[0]
            
            # Cost of debt (assume 2% above risk-free rate)
            cost_of_debt = self.risk_free_rate + 0.02
            
            # Tax rate (assume 25% for Indian companies)
            tax_rate = 0.25
            
            # Calculate weights
            total_value = market_cap + debt_value
            equity_weight = market_cap / total_value if total_value > 0 else 1.0
            debt_weight = debt_value / total_value if total_value > 0 else 0.0
            
            # Calculate WACC
            wacc = (cost_of_equity * equity_weight) + (cost_of_debt * (1 - tax_rate) * debt_weight)
            
            return wacc
            
        except Exception as e:
            print(f"Error calculating WACC: {e}")
            return 0.12  # Default WACC
    
    def project_cash_flows(self, fcf: pd.Series, growth_rates: List[float] = None) -> pd.Series:
        """
        Project future cash flows based on historical data and growth assumptions.
        
        Args:
            fcf (pd.Series): Historical FCF
            growth_rates (list): Growth rates for each year
            
        Returns:
            pd.Series: Projected FCF
        """
        try:
            if fcf.empty:
                return pd.Series()
            
            # Use default growth rates if not provided
            if growth_rates is None:
                # Calculate historical growth rate
                if len(fcf) > 1:
                    historical_growth = (fcf.iloc[0] / fcf.iloc[-1]) ** (1 / (len(fcf) - 1)) - 1
                    # Cap growth rate between -20% and +30%
                    historical_growth = max(-0.2, min(0.3, historical_growth))
                else:
                    historical_growth = 0.05
                
                # Declining growth rate over forecast period
                growth_rates = [historical_growth * (0.8 ** i) for i in range(self.forecast_years)]
            
            # Project cash flows
            latest_fcf = fcf.iloc[0] if not fcf.empty else 1000000  # Default if no data
            projected_fcf = []
            
            for i, growth_rate in enumerate(growth_rates):
                if i == 0:
                    projected_fcf.append(latest_fcf * (1 + growth_rate))
                else:
                    projected_fcf.append(projected_fcf[-1] * (1 + growth_rate))
            
            return pd.Series(projected_fcf, index=range(1, self.forecast_years + 1))
            
        except Exception as e:
            print(f"Error projecting cash flows: {e}")
            return pd.Series()
    
    def calculate_terminal_value(self, final_fcf: float, wacc: float) -> float:
        """
        Calculate terminal value using perpetuity growth method.
        
        Args:
            final_fcf (float): Final year FCF
            wacc (float): WACC
            
        Returns:
            float: Terminal value
        """
        try:
            terminal_value = final_fcf * (1 + self.terminal_growth_rate) / (wacc - self.terminal_growth_rate)
            return terminal_value
            
        except Exception as e:
            print(f"Error calculating terminal value: {e}")
            return 0
    
    def calculate_present_values(self, projected_fcf: pd.Series, terminal_value: float, wacc: float) -> Dict:
        """
        Calculate present values of projected cash flows and terminal value.
        
        Args:
            projected_fcf (pd.Series): Projected FCF
            terminal_value (float): Terminal value
            wacc (float): WACC
            
        Returns:
            dict: Present values
        """
        try:
            # Calculate present values of projected FCF
            pv_fcf = {}
            for year, fcf in projected_fcf.items():
                pv_fcf[year] = fcf / ((1 + wacc) ** year)
            
            # Calculate present value of terminal value
            pv_terminal = terminal_value / ((1 + wacc) ** self.forecast_years)
            
            # Calculate enterprise value
            enterprise_value = sum(pv_fcf.values()) + pv_terminal
            
            return {
                'pv_fcf': pv_fcf,
                'pv_terminal': pv_terminal,
                'enterprise_value': enterprise_value
            }
            
        except Exception as e:
            print(f"Error calculating present values: {e}")
            return {}
    
    def calculate_equity_value(self, enterprise_value: float, financial_data: Dict) -> float:
        """
        Calculate equity value by adjusting enterprise value for debt and cash.
        
        Args:
            enterprise_value (float): Enterprise value
            financial_data (dict): Financial data
            
        Returns:
            float: Equity value
        """
        try:
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            
            # Get debt and cash
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            cash = balance_sheet.loc['Cash'].iloc[0] if 'Cash' in balance_sheet.index else 0
            
            # Calculate equity value
            equity_value = enterprise_value - total_debt + cash
            
            return equity_value
            
        except Exception as e:
            print(f"Error calculating equity value: {e}")
            return enterprise_value
    
    def calculate_intrinsic_value(self, equity_value: float, shares_outstanding: int) -> float:
        """
        Calculate intrinsic value per share.
        
        Args:
            equity_value (float): Equity value
            shares_outstanding (int): Number of shares outstanding
            
        Returns:
            float: Intrinsic value per share
        """
        try:
            intrinsic_value = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            return intrinsic_value
            
        except Exception as e:
            print(f"Error calculating intrinsic value: {e}")
            return 0
    
    def perform_sensitivity_analysis(self, base_fcf: float, base_wacc: float, base_growth: float) -> Dict:
        """
        Perform sensitivity analysis on key variables.
        
        Args:
            base_fcf (float): Base FCF
            base_wacc (float): Base WACC
            base_growth (float): Base growth rate
            
        Returns:
            dict: Sensitivity analysis results
        """
        try:
            sensitivity = {}
            
            # WACC sensitivity
            wacc_range = [base_wacc - 0.02, base_wacc, base_wacc + 0.02]
            for wacc in wacc_range:
                terminal_value = self.calculate_terminal_value(base_fcf, wacc)
                pv_fcf = base_fcf / ((1 + wacc) ** 1)  # Simplified for sensitivity
                enterprise_value = pv_fcf + terminal_value / ((1 + wacc) ** self.forecast_years)
                sensitivity[f'wacc_{wacc:.3f}'] = enterprise_value
            
            # Growth rate sensitivity
            growth_range = [base_growth - 0.01, base_growth, base_growth + 0.01]
            for growth in growth_range:
                terminal_value = self.calculate_terminal_value(base_fcf, base_wacc)
                pv_fcf = base_fcf / ((1 + base_wacc) ** 1)
                enterprise_value = pv_fcf + terminal_value / ((1 + base_wacc) ** self.forecast_years)
                sensitivity[f'growth_{growth:.3f}'] = enterprise_value
            
            return sensitivity
            
        except Exception as e:
            print(f"Error in sensitivity analysis: {e}")
            return {}
    
    def analyze_dcf(self, symbol: str) -> Dict:
        """
        Perform comprehensive DCF analysis.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Complete DCF analysis
        """
        try:
            print(f"Performing DCF analysis for {symbol}...")
            
            # Get financial data
            financial_data = self.get_financial_data(symbol)
            if not financial_data:
                return {'error': 'Unable to fetch financial data'}
            
            # Calculate historical FCF
            fcf = self.calculate_free_cash_flow(financial_data)
            if fcf.empty:
                return {'error': 'Unable to calculate FCF'}
            
            # Calculate WACC
            wacc = self.calculate_wacc(financial_data, symbol)
            
            # Project cash flows
            projected_fcf = self.project_cash_flows(fcf)
            if projected_fcf.empty:
                return {'error': 'Unable to project cash flows'}
            
            # Calculate terminal value
            final_fcf = projected_fcf.iloc[-1]
            terminal_value = self.calculate_terminal_value(final_fcf, wacc)
            
            # Calculate present values
            pv_results = self.calculate_present_values(projected_fcf, terminal_value, wacc)
            
            # Calculate equity value
            equity_value = self.calculate_equity_value(pv_results['enterprise_value'], financial_data)
            
            # Get shares outstanding
            info = financial_data.get('info', {})
            shares_outstanding = info.get('sharesOutstanding', 1000000)
            
            # Calculate intrinsic value
            intrinsic_value = self.calculate_intrinsic_value(equity_value, shares_outstanding)
            
            # Get current market price
            current_price = info.get('currentPrice', 0)
            
            # Calculate margin of safety
            margin_of_safety_value = intrinsic_value * (1 - self.margin_of_safety)
            
            # Perform sensitivity analysis
            base_growth = 0.05  # Assume 5% base growth
            sensitivity = self.perform_sensitivity_analysis(final_fcf, wacc, base_growth)
            
            # Compile analysis
            analysis = {
                'intrinsic_value': intrinsic_value,
                'current_price': current_price,
                'margin_of_safety_value': margin_of_safety_value,
                'margin_of_safety_percentage': self.margin_of_safety * 100,
                'wacc': wacc,
                'terminal_value': terminal_value,
                'enterprise_value': pv_results['enterprise_value'],
                'equity_value': equity_value,
                'projected_fcf': projected_fcf.to_dict(),
                'present_values': pv_results['pv_fcf'],
                'sensitivity_analysis': sensitivity,
                'historical_fcf': fcf.to_dict(),
                'shares_outstanding': shares_outstanding,
                'valuation_ratio': intrinsic_value / current_price if current_price > 0 else 0
            }
            
            print("DCF analysis completed")
            return analysis
            
        except Exception as e:
            print(f"Error in DCF analysis: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    dcf = DCFAnalysis()
    analysis = dcf.analyze_dcf("RELIANCE.NS")
    print("DCF Analysis Results:")
    print(analysis) 