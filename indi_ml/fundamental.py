"""
Fundamental Analysis Module for Indian Stocks

This module provides comprehensive fundamental analysis capabilities including:
- Financial ratios calculation
- Earnings analysis
- Valuation metrics
- Growth indicators
- Risk assessment from fundamental perspective

Key Features:
- P/E, P/B, ROE, ROA ratios
- Debt-to-Equity analysis
- Earnings growth trends
- Dividend analysis
- Sector comparison
- Valuation assessment

Usage:
    from indi_ml.fundamental import FundamentalAnalysis
    fa = FundamentalAnalysis()
    analysis = fa.analyze_stock('RELIANCE')
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FundamentalAnalysis:
    """
    Comprehensive fundamental analysis for Indian stocks.
    """
    
    def __init__(self):
        """Initialize the fundamental analysis module."""
        self.sector_averages = {
            'Technology': {'pe': 25, 'pb': 3.5, 'roe': 15, 'debt_to_equity': 0.3},
            'Banking': {'pe': 15, 'pb': 1.2, 'roe': 12, 'debt_to_equity': 0.8},
            'Oil & Gas': {'pe': 12, 'pb': 1.5, 'roe': 8, 'debt_to_equity': 0.4},
            'Pharmaceuticals': {'pe': 30, 'pb': 4.0, 'roe': 18, 'debt_to_equity': 0.2},
            'Automobile': {'pe': 20, 'pb': 2.5, 'roe': 12, 'debt_to_equity': 0.5},
            'Consumer Goods': {'pe': 35, 'pb': 5.0, 'roe': 20, 'debt_to_equity': 0.3},
            'Default': {'pe': 20, 'pb': 2.5, 'roe': 12, 'debt_to_equity': 0.4}
        }
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get basic stock information and financial data.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock information and financial data
        """
        try:
            # Add .NS suffix for NSE stocks
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get basic info
            info = ticker.info
            
            # Get financial statements
            balance_sheet = ticker.balance_sheet
            income_stmt = ticker.income_stmt
            cash_flow = ticker.cashflow
            
            return {
                'info': info,
                'balance_sheet': balance_sheet,
                'income_stmt': income_stmt,
                'cash_flow': cash_flow,
                'ticker': ticker
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return {}
    
    def calculate_ratios(self, financial_data: Dict) -> Dict:
        """
        Calculate key financial ratios.
        
        Args:
            financial_data (dict): Financial data from get_stock_info
            
        Returns:
            dict: Calculated ratios
        """
        try:
            info = financial_data.get('info', {})
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            income_stmt = financial_data.get('income_stmt', pd.DataFrame())
            
            ratios = {}
            
            # Market ratios
            ratios['pe_ratio'] = info.get('trailingPE', 0)
            ratios['pb_ratio'] = info.get('priceToBook', 0)
            ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0)
            ratios['dividend_yield'] = info.get('dividendYield', 0)
            
            # Profitability ratios
            if not income_stmt.empty:
                latest_revenue = income_stmt.iloc[0].get('Total Revenue', 0)
                latest_net_income = income_stmt.iloc[0].get('Net Income', 0)
                
                if latest_revenue and latest_revenue > 0:
                    ratios['net_margin'] = (latest_net_income / latest_revenue) * 100
                else:
                    ratios['net_margin'] = 0
            else:
                ratios['net_margin'] = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
            
            # Return ratios
            ratios['roe'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            ratios['roa'] = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
            
            # Debt ratios
            if not balance_sheet.empty:
                latest_assets = balance_sheet.iloc[0].get('Total Assets', 0)
                latest_liabilities = balance_sheet.iloc[0].get('Total Liabilities Net Minority Interest', 0)
                latest_equity = balance_sheet.iloc[0].get('Total Equity Gross Minority Interest', 0)
                
                if latest_equity and latest_equity > 0:
                    ratios['debt_to_equity'] = (latest_liabilities / latest_equity) if latest_liabilities else 0
                else:
                    ratios['debt_to_equity'] = 0
                    
                if latest_assets and latest_assets > 0:
                    ratios['debt_to_assets'] = (latest_liabilities / latest_assets) if latest_liabilities else 0
                else:
                    ratios['debt_to_assets'] = 0
            else:
                ratios['debt_to_equity'] = 0
                ratios['debt_to_assets'] = 0
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return {}
    
    def analyze_earnings(self, financial_data: Dict) -> Dict:
        """
        Analyze earnings trends and quality.
        
        Args:
            financial_data (dict): Financial data
            
        Returns:
            dict: Earnings analysis
        """
        try:
            income_stmt = financial_data.get('income_stmt', pd.DataFrame())
            cash_flow = financial_data.get('cash_flow', pd.DataFrame())
            
            analysis = {}
            
            if not income_stmt.empty and len(income_stmt) >= 4:
                # Get last 4 years of data
                revenues = income_stmt.iloc[:4]['Total Revenue']
                net_incomes = income_stmt.iloc[:4]['Net Income']
                
                if len(revenues) >= 2:
                    # Revenue growth
                    revenue_growth = ((revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]) * 100
                    analysis['revenue_growth_1y'] = revenue_growth
                
                if len(revenues) >= 4:
                    # 3-year revenue growth
                    revenue_growth_3y = ((revenues.iloc[0] - revenues.iloc[3]) / revenues.iloc[3]) * 100
                    analysis['revenue_growth_3y'] = revenue_growth_3y
                
                if len(net_incomes) >= 2:
                    # Net income growth
                    net_income_growth = ((net_incomes.iloc[0] - net_incomes.iloc[1]) / net_incomes.iloc[1]) * 100
                    analysis['net_income_growth_1y'] = net_income_growth
                
                if len(net_incomes) >= 4:
                    # 3-year net income growth
                    net_income_growth_3y = ((net_incomes.iloc[0] - net_incomes.iloc[3]) / net_incomes.iloc[3]) * 100
                    analysis['net_income_growth_3y'] = net_income_growth_3y
                
                # Earnings quality (cash flow vs net income)
                if not cash_flow.empty:
                    latest_operating_cf = cash_flow.iloc[0].get('Operating Cash Flow', 0)
                    latest_net_income = net_incomes.iloc[0] if len(net_incomes) > 0 else 0
                    
                    if latest_net_income and latest_net_income > 0:
                        analysis['cash_flow_coverage'] = latest_operating_cf / latest_net_income
                    else:
                        analysis['cash_flow_coverage'] = 0
                else:
                    analysis['cash_flow_coverage'] = 0
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing earnings: {e}")
            return {}
    
    def calculate_valuation(self, financial_data: Dict, ratios: Dict) -> Dict:
        """
        Calculate valuation metrics and assessment.
        
        Args:
            financial_data (dict): Financial data
            ratios (dict): Financial ratios
            
        Returns:
            dict: Valuation analysis
        """
        try:
            info = financial_data.get('info', {})
            
            valuation = {}
            
            # Current market price
            current_price = info.get('currentPrice', 0)
            valuation['current_price'] = current_price
            
            # Intrinsic value estimates
            if ratios.get('pe_ratio') and ratios.get('pe_ratio') > 0:
                # Simple P/E based valuation
                sector_pe = self.sector_averages['Default']['pe']
                valuation['pe_based_value'] = current_price * (sector_pe / ratios['pe_ratio'])
            
            # Book value analysis
            if ratios.get('pb_ratio') and ratios.get('pb_ratio') > 0:
                book_value = current_price / ratios['pb_ratio']
                valuation['book_value'] = book_value
                valuation['price_to_book'] = ratios['pb_ratio']
            
            # Dividend analysis
            dividend_yield = ratios.get('dividend_yield', 0)
            if dividend_yield:
                valuation['dividend_yield'] = dividend_yield * 100
                valuation['dividend_payout'] = dividend_yield * ratios.get('pe_ratio', 0) * 100
            
            # Growth-adjusted valuation
            if ratios.get('pe_ratio') and ratios.get('pe_ratio') > 0:
                peg_ratio = ratios['pe_ratio'] / 10  # Assuming 10% growth
                valuation['peg_ratio'] = peg_ratio
            
            return valuation
            
        except Exception as e:
            print(f"Error calculating valuation: {e}")
            return {}
    
    def assess_risk(self, ratios: Dict, earnings: Dict) -> Dict:
        """
        Assess fundamental risk factors.
        
        Args:
            ratios (dict): Financial ratios
            earnings (dict): Earnings analysis
            
        Returns:
            dict: Risk assessment
        """
        try:
            risk = {}
            
            # Debt risk
            debt_to_equity = ratios.get('debt_to_equity', 0)
            if debt_to_equity > 1.0:
                risk['debt_risk'] = 'High'
            elif debt_to_equity > 0.5:
                risk['debt_risk'] = 'Medium'
            else:
                risk['debt_risk'] = 'Low'
            
            # Profitability risk
            roe = ratios.get('roe', 0)
            if roe < 5:
                risk['profitability_risk'] = 'High'
            elif roe < 10:
                risk['profitability_risk'] = 'Medium'
            else:
                risk['profitability_risk'] = 'Low'
            
            # Growth risk
            revenue_growth = earnings.get('revenue_growth_1y', 0)
            if revenue_growth < -10:
                risk['growth_risk'] = 'High'
            elif revenue_growth < 0:
                risk['growth_risk'] = 'Medium'
            else:
                risk['growth_risk'] = 'Low'
            
            # Earnings quality risk
            cash_flow_coverage = earnings.get('cash_flow_coverage', 0)
            if cash_flow_coverage < 0.8:
                risk['earnings_quality_risk'] = 'High'
            elif cash_flow_coverage < 1.0:
                risk['earnings_quality_risk'] = 'Medium'
            else:
                risk['earnings_quality_risk'] = 'Low'
            
            # Overall risk score
            risk_scores = {
                'High': 3, 'Medium': 2, 'Low': 1
            }
            
            total_risk = sum(risk_scores.get(risk_level, 0) for risk_level in risk.values())
            max_risk = len(risk) * 3
            
            risk['overall_risk_score'] = (total_risk / max_risk) * 100
            
            return risk
            
        except Exception as e:
            print(f"Error assessing risk: {e}")
            return {}
    
    def analyze_stock(self, symbol: str) -> Dict:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Complete fundamental analysis
        """
        try:
            print(f"Analyzing fundamentals for {symbol}...")
            
            # Get financial data
            financial_data = self.get_stock_info(symbol)
            if not financial_data:
                return {'error': f'Could not fetch data for {symbol}'}
            
            # Calculate ratios
            ratios = self.calculate_ratios(financial_data)
            
            # Analyze earnings
            earnings = self.analyze_earnings(financial_data)
            
            # Calculate valuation
            valuation = self.calculate_valuation(financial_data, ratios)
            
            # Assess risk
            risk = self.assess_risk(ratios, earnings)
            
            # Compile analysis
            analysis = {
                'symbol': symbol,
                'ratios': ratios,
                'earnings': earnings,
                'valuation': valuation,
                'risk': risk,
                'info': financial_data.get('info', {})
            }
            
            print(f"Fundamental analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            print(f"Error in fundamental analysis for {symbol}: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    fa = FundamentalAnalysis()
    analysis = fa.analyze_stock('RELIANCE')
    print("Fundamental Analysis Results:")
    print(analysis) 