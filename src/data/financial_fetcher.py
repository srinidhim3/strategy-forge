"""
Financial Data Fetcher Module

This module handles fetching quarterly and annual financial statements from Yahoo Finance
using the yfinance library. It provides functionality to download income statements,
balance sheets, and cash flow statements with proper error handling and data validation.

Classes:
    FinancialFetcher: Main class for fetching and processing financial statement data

Functions:
    fetch_financial_statements: Convenience function for complete financial data fetching
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Tuple
import logging
import warnings

# Suppress yfinance warnings about future deprecations
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialFetcher:
    """
    A class to fetch quarterly and annual financial statements from Yahoo Finance.
    
    This class provides methods to download financial statement data including
    income statements, balance sheets, and cash flow statements with proper
    error handling, data validation, and standardized output format.
    """
    
    def __init__(self):
        """Initialize the FinancialFetcher."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_income_statement(
        self, 
        symbol: str, 
        period: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Fetch income statement data for a stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            period: Data period - 'quarterly' or 'annual'. Default: 'quarterly'
            
        Returns:
            pandas.DataFrame: Income statement with financial metrics as rows
                             and reporting periods as columns
                             
        Raises:
            ValueError: If symbol is invalid or period is not supported
            ConnectionError: If unable to connect to Yahoo Finance
            Exception: For other data fetching errors
            
        Example:
            >>> fetcher = FinancialFetcher()
            >>> income = fetcher.fetch_income_statement("AAPL", "quarterly")
            >>> print(income.head())
        """
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            period = self._validate_period(period)
            
            self.logger.info(f"Fetching {period} income statement for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch income statement data
            if period == "quarterly":
                income_stmt = ticker.quarterly_income_stmt
            else:  # annual
                income_stmt = ticker.income_stmt
            
            # Validate returned data
            if income_stmt.empty:
                raise ValueError(f"No income statement data found for symbol {symbol}")
            
            # Clean and standardize the data
            income_stmt = self._clean_financial_data(income_stmt, symbol, "income_statement")
            
            self.logger.info(f"Successfully fetched {period} income statement for {symbol}: {income_stmt.shape}")
            return income_stmt
            
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {symbol}: {str(e)}")
            raise
    
    def fetch_balance_sheet(
        self, 
        symbol: str, 
        period: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Fetch balance sheet data for a stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            period: Data period - 'quarterly' or 'annual'. Default: 'quarterly'
            
        Returns:
            pandas.DataFrame: Balance sheet with financial metrics as rows
                             and reporting periods as columns
                             
        Example:
            >>> fetcher = FinancialFetcher()
            >>> balance_sheet = fetcher.fetch_balance_sheet("AAPL", "quarterly")
            >>> print(balance_sheet.head())
        """
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            period = self._validate_period(period)
            
            self.logger.info(f"Fetching {period} balance sheet for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch balance sheet data
            if period == "quarterly":
                balance_sheet = ticker.quarterly_balance_sheet
            else:  # annual
                balance_sheet = ticker.balance_sheet
            
            # Validate returned data
            if balance_sheet.empty:
                raise ValueError(f"No balance sheet data found for symbol {symbol}")
            
            # Clean and standardize the data
            balance_sheet = self._clean_financial_data(balance_sheet, symbol, "balance_sheet")
            
            self.logger.info(f"Successfully fetched {period} balance sheet for {symbol}: {balance_sheet.shape}")
            return balance_sheet
            
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet for {symbol}: {str(e)}")
            raise
    
    def fetch_cash_flow(
        self, 
        symbol: str, 
        period: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Fetch cash flow statement data for a stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            period: Data period - 'quarterly' or 'annual'. Default: 'quarterly'
            
        Returns:
            pandas.DataFrame: Cash flow statement with financial metrics as rows
                             and reporting periods as columns
                             
        Example:
            >>> fetcher = FinancialFetcher()
            >>> cash_flow = fetcher.fetch_cash_flow("AAPL", "quarterly")
            >>> print(cash_flow.head())
        """
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            period = self._validate_period(period)
            
            self.logger.info(f"Fetching {period} cash flow statement for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch cash flow data
            if period == "quarterly":
                cash_flow = ticker.quarterly_cashflow
            else:  # annual
                cash_flow = ticker.cashflow
            
            # Validate returned data
            if cash_flow.empty:
                raise ValueError(f"No cash flow data found for symbol {symbol}")
            
            # Clean and standardize the data
            cash_flow = self._clean_financial_data(cash_flow, symbol, "cash_flow")
            
            self.logger.info(f"Successfully fetched {period} cash flow for {symbol}: {cash_flow.shape}")
            return cash_flow
            
        except Exception as e:
            self.logger.error(f"Error fetching cash flow for {symbol}: {str(e)}")
            raise
    
    def fetch_all_statements(
        self, 
        symbol: str, 
        period: str = "quarterly"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all financial statements (income, balance sheet, cash flow) for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period - 'quarterly' or 'annual'. Default: 'quarterly'
            
        Returns:
            dict: Dictionary with keys 'income_statement', 'balance_sheet', 'cash_flow'
                  and DataFrame values
                  
        Example:
            >>> fetcher = FinancialFetcher()
            >>> statements = fetcher.fetch_all_statements("AAPL", "quarterly")
            >>> print(f"Income statement shape: {statements['income_statement'].shape}")
        """
        try:
            symbol = self._validate_symbol(symbol)
            
            self.logger.info(f"Fetching all {period} financial statements for {symbol}")
            
            statements = {}
            
            # Fetch each statement type
            statements['income_statement'] = self.fetch_income_statement(symbol, period)
            statements['balance_sheet'] = self.fetch_balance_sheet(symbol, period)
            statements['cash_flow'] = self.fetch_cash_flow(symbol, period)
            
            self.logger.info(f"Successfully fetched all financial statements for {symbol}")
            return statements
            
        except Exception as e:
            self.logger.error(f"Error fetching financial statements for {symbol}: {str(e)}")
            raise
    
    def get_key_metrics(
        self, 
        symbol: str, 
        period: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Extract key financial metrics from all statements.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period - 'quarterly' or 'annual'
            
        Returns:
            pandas.DataFrame: Key metrics with periods as columns
        """
        try:
            statements = self.fetch_all_statements(symbol, period)
            
            # Initialize metrics dictionary
            metrics = {}
            
            # Extract key metrics from income statement
            income = statements['income_statement']
            if not income.empty:
                metrics.update(self._extract_income_metrics(income))
            
            # Extract key metrics from balance sheet
            balance = statements['balance_sheet']
            if not balance.empty:
                metrics.update(self._extract_balance_metrics(balance))
            
            # Extract key metrics from cash flow
            cash_flow = statements['cash_flow']
            if not cash_flow.empty:
                metrics.update(self._extract_cashflow_metrics(cash_flow))
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Add symbol for identification
            metrics_df.loc['Symbol'] = symbol
            
            self.logger.info(f"Extracted {len(metrics_df)} key metrics for {symbol}")
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error extracting key metrics for {symbol}: {str(e)}")
            raise
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and clean stock symbol."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string")
        
        return symbol.strip().upper()
    
    def _validate_period(self, period: str) -> str:
        """Validate period parameter."""
        valid_periods = ["quarterly", "annual"]
        
        if period.lower() not in valid_periods:
            raise ValueError(f"Period must be one of {valid_periods}, got: {period}")
        
        return period.lower()
    
    def _clean_financial_data(self, data: pd.DataFrame, symbol: str, statement_type: str) -> pd.DataFrame:
        """Clean and standardize financial statement data."""
        if data.empty:
            return data
        
        # Sort columns by date (most recent first)
        data = data.sort_index(axis=1, ascending=False)
        
        # Convert to numeric data types where possible
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Add metadata
        data.attrs['symbol'] = symbol
        data.attrs['statement_type'] = statement_type
        data.attrs['fetch_date'] = datetime.now().isoformat()
        
        return data
    
    def _extract_income_metrics(self, income_stmt: pd.DataFrame) -> Dict:
        """Extract key metrics from income statement."""
        metrics = {}
        
        # Common income statement metrics
        metric_mappings = {
            'Total Revenue': ['Total Revenue', 'Revenue'],
            'Net Income': ['Net Income', 'Net Income Common Stockholders'],
            'Operating Income': ['Operating Income', 'Operating Revenue'],
            'Gross Profit': ['Gross Profit'],
            'EBITDA': ['EBITDA'],
            'EPS': ['Basic EPS', 'Diluted EPS']
        }
        
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in income_stmt.index:
                    metrics[metric_name] = income_stmt.loc[key]
                    break
        
        return metrics
    
    def _extract_balance_metrics(self, balance_sheet: pd.DataFrame) -> Dict:
        """Extract key metrics from balance sheet."""
        metrics = {}
        
        # Common balance sheet metrics
        metric_mappings = {
            'Total Assets': ['Total Assets'],
            'Total Equity': ['Total Equity Gross Minority Interest', 'Stockholders Equity'],
            'Total Debt': ['Total Debt'],
            'Cash And Equivalents': ['Cash And Cash Equivalents', 'Cash']
        }
        
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in balance_sheet.index:
                    metrics[metric_name] = balance_sheet.loc[key]
                    break
        
        return metrics
    
    def _extract_cashflow_metrics(self, cash_flow: pd.DataFrame) -> Dict:
        """Extract key metrics from cash flow statement."""
        metrics = {}
        
        # Common cash flow metrics
        metric_mappings = {
            'Operating Cash Flow': ['Operating Cash Flow'],
            'Free Cash Flow': ['Free Cash Flow'],
            'Capital Expenditure': ['Capital Expenditure']
        }
        
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in cash_flow.index:
                    metrics[metric_name] = cash_flow.loc[key]
                    break
        
        return metrics


def fetch_financial_statements(
    symbol: str, 
    period: str = "quarterly",
    statement_types: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch financial statements for a single stock.
    
    Args:
        symbol: Stock ticker symbol
        period: Data period - 'quarterly' or 'annual'
        statement_types: List of statement types to fetch. 
                        Options: ['income', 'balance', 'cash_flow']
                        If None, fetches all statements.
        
    Returns:
        dict: Dictionary with requested financial statements
        
    Example:
        >>> statements = fetch_financial_statements("AAPL", "quarterly", ["income", "balance"])
        >>> print(statements.keys())
    """
    fetcher = FinancialFetcher()
    
    if statement_types is None:
        return fetcher.fetch_all_statements(symbol, period)
    
    statements = {}
    
    if 'income' in statement_types:
        statements['income_statement'] = fetcher.fetch_income_statement(symbol, period)
    
    if 'balance' in statement_types:
        statements['balance_sheet'] = fetcher.fetch_balance_sheet(symbol, period)
    
    if 'cash_flow' in statement_types:
        statements['cash_flow'] = fetcher.fetch_cash_flow(symbol, period)
    
    return statements


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    fetcher = FinancialFetcher()
    
    try:
        # Fetch Apple quarterly income statement
        print("Fetching AAPL quarterly income statement...")
        income = fetcher.fetch_income_statement("AAPL", "quarterly")
        print(f"Income statement shape: {income.shape}")
        print("Latest quarter metrics:")
        print(income.iloc[:10, 0])  # First 10 rows, latest quarter
        
        # Fetch balance sheet
        print("\nFetching AAPL quarterly balance sheet...")
        balance = fetcher.fetch_balance_sheet("AAPL", "quarterly")
        print(f"Balance sheet shape: {balance.shape}")
        
        # Get key metrics
        print("\nFetching key metrics...")
        metrics = fetcher.get_key_metrics("AAPL", "quarterly")
        print(f"Key metrics shape: {metrics.shape}")
        print("Key metrics:")
        print(metrics.iloc[:, 0])  # Latest quarter metrics
        
    except Exception as e:
        print(f"Error in example: {e}")