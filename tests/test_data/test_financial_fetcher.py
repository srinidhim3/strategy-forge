"""
Test module for financial data fetching functionality.

This module contains comprehensive tests for the FinancialFetcher class,
including unit tests, integration tests, and error handling validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
import warnings

from src.data.financial_fetcher import FinancialFetcher


class TestFinancialFetcher:
    """Test suite for the FinancialFetcher class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.fetcher = FinancialFetcher()
        self.test_symbol = "AAPL"
        self.invalid_symbol = "INVALID_SYMBOL_123"
    
    def test_initialization(self):
        """Test FinancialFetcher initialization."""
        assert self.fetcher is not None
        assert hasattr(self.fetcher, 'fetch_income_statement')
        assert hasattr(self.fetcher, 'fetch_balance_sheet')
        assert hasattr(self.fetcher, 'fetch_cash_flow')
        assert hasattr(self.fetcher, 'fetch_all_statements')
        assert hasattr(self.fetcher, 'get_key_metrics')
    
    @pytest.mark.slow
    def test_fetch_income_statement_valid_symbol(self):
        """Test fetching income statement for a valid symbol."""
        result = self.fetcher.fetch_income_statement(self.test_symbol)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result.columns) > 0
        
        # Check that we have some expected financial metrics
        expected_metrics = [
            'Total Revenue', 'Net Income', 'Operating Income', 
            'Gross Profit', 'Operating Expense'
        ]
        available_metrics = [metric for metric in expected_metrics 
                           if metric in result.index]
        assert len(available_metrics) > 0
    
    @pytest.mark.slow
    def test_fetch_balance_sheet_valid_symbol(self):
        """Test fetching balance sheet for a valid symbol."""
        result = self.fetcher.fetch_balance_sheet(self.test_symbol)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result.columns) > 0
        
        # Check that we have some expected balance sheet items
        expected_items = [
            'Total Assets', 'Total Equity Gross Minority Interest',
            'Cash And Cash Equivalents', 'Total Debt'
        ]
        available_items = [item for item in expected_items 
                          if item in result.index]
        assert len(available_items) > 0
    
    @pytest.mark.slow
    def test_fetch_cash_flow_valid_symbol(self):
        """Test fetching cash flow statement for a valid symbol."""
        result = self.fetcher.fetch_cash_flow(self.test_symbol)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result.columns) > 0
        
        # Check that we have some expected cash flow items
        expected_items = [
            'Operating Cash Flow', 'Free Cash Flow',
            'Capital Expenditure', 'Net Income'
        ]
        available_items = [item for item in expected_items 
                          if item in result.index]
        assert len(available_items) > 0
    
    def test_fetch_income_statement_invalid_symbol(self):
        """Test fetching income statement for an invalid symbol."""
        with pytest.raises(ValueError, match="No income statement data found"):
            self.fetcher.fetch_income_statement(self.invalid_symbol)
    
    def test_fetch_balance_sheet_invalid_symbol(self):
        """Test fetching balance sheet for an invalid symbol."""
        with pytest.raises(ValueError, match="No balance sheet data found"):
            self.fetcher.fetch_balance_sheet(self.invalid_symbol)
    
    def test_fetch_cash_flow_invalid_symbol(self):
        """Test fetching cash flow for an invalid symbol."""
        with pytest.raises(ValueError, match="No cash flow data found"):
            self.fetcher.fetch_cash_flow(self.invalid_symbol)
    
    def test_fetch_income_statement_empty_symbol(self):
        """Test fetching income statement with empty symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            self.fetcher.fetch_income_statement("")
    
    def test_fetch_balance_sheet_none_symbol(self):
        """Test fetching balance sheet with None symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            self.fetcher.fetch_balance_sheet(None)
    
    def test_fetch_cash_flow_whitespace_symbol(self):
        """Test fetching cash flow with whitespace-only symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            self.fetcher.fetch_cash_flow("   ")
    
    @pytest.mark.slow
    def test_fetch_all_statements_valid_symbol(self):
        """Test fetching all financial statements for a valid symbol."""
        result = self.fetcher.fetch_all_statements(self.test_symbol)
        
        assert isinstance(result, dict)
        assert 'income_statement' in result
        assert 'balance_sheet' in result
        assert 'cash_flow' in result
        
        # All statements should be DataFrames or None
        for statement_name, statement_data in result.items():
            assert statement_data is None or isinstance(statement_data, pd.DataFrame)
    
    def test_fetch_all_statements_invalid_symbol(self):
        """Test fetching all statements for an invalid symbol."""
        with pytest.raises(ValueError, match="No income statement data found"):
            self.fetcher.fetch_all_statements(self.invalid_symbol)
    
    @pytest.mark.slow
    def test_get_key_metrics_valid_symbol(self):
        """Test extracting key metrics for a valid symbol."""
        result = self.fetcher.get_key_metrics(self.test_symbol)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        
        # Check for expected key metrics columns
        expected_metrics = [
            'Total Revenue', 'Net Income', 'Operating Income',
            'Total Assets', 'Total Equity', 'Operating Cash Flow'
        ]
        available_metrics = [metric for metric in expected_metrics 
                           if metric in result.columns]
        assert len(available_metrics) > 0
        
        # Check that index is datetime
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_get_key_metrics_invalid_symbol(self):
        """Test extracting key metrics for an invalid symbol."""
        with pytest.raises(ValueError, match="No income statement data found"):
            self.fetcher.get_key_metrics(self.invalid_symbol)
    
    @patch('src.data.financial_fetcher.yf.Ticker')
    def test_fetch_income_statement_with_mock(self, mock_ticker_class):
        """Test income statement fetching with mocked yfinance data."""
        # Create mock data
        mock_data = pd.DataFrame({
            '2024-01-01': [100000, 50000, 20000],
            '2023-01-01': [90000, 45000, 18000]
        }, index=['Total Revenue', 'Operating Income', 'Net Income'])
        
        # Configure mock
        mock_ticker = MagicMock()
        mock_ticker.quarterly_income_stmt = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        result = self.fetcher.fetch_income_statement("TEST")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_ticker_class.assert_called_once_with("TEST")
    
    @patch('src.data.financial_fetcher.yf.Ticker')
    def test_fetch_balance_sheet_with_mock(self, mock_ticker_class):
        """Test balance sheet fetching with mocked yfinance data."""
        # Create mock data
        mock_data = pd.DataFrame({
            '2024-01-01': [500000, 300000, 100000],
            '2023-01-01': [450000, 280000, 90000]
        }, index=['Total Assets', 'Total Equity Gross Minority Interest', 'Cash And Cash Equivalents'])
        
        # Configure mock
        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        result = self.fetcher.fetch_balance_sheet("TEST")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_ticker_class.assert_called_once_with("TEST")
    
    @patch('src.data.financial_fetcher.yf.Ticker')
    def test_fetch_cash_flow_with_mock(self, mock_ticker_class):
        """Test cash flow fetching with mocked yfinance data."""
        # Create mock data
        mock_data = pd.DataFrame({
            '2024-01-01': [80000, 70000, -10000],
            '2023-01-01': [75000, 65000, -9000]
        }, index=['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure'])
        
        # Configure mock
        mock_ticker = MagicMock()
        mock_ticker.quarterly_cashflow = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        result = self.fetcher.fetch_cash_flow("TEST")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_ticker_class.assert_called_once_with("TEST")
    
    @patch('src.data.financial_fetcher.yf.Ticker')
    def test_exception_handling(self, mock_ticker_class):
        """Test exception handling in financial data fetching."""
        # Configure mock to raise exception when accessing the property
        mock_ticker = MagicMock()
        type(mock_ticker).quarterly_income_stmt = PropertyMock(side_effect=Exception("API Error"))
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(Exception, match="API Error"):
            self.fetcher.fetch_income_statement("TEST")
    
    @patch('src.data.financial_fetcher.yf.Ticker')
    def test_empty_dataframe_handling(self, mock_ticker_class):
        """Test handling of empty DataFrames from yfinance."""
        # Configure mock to return empty DataFrame
        mock_ticker = MagicMock()
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError, match="No income statement data found"):
            self.fetcher.fetch_income_statement("TEST")
    
    def test_annual_vs_quarterly_frequency(self):
        """Test both annual and quarterly frequency options."""
        # Note: Current implementation doesn't support frequency parameter
        # This test is for future functionality
        pass
    
    def test_invalid_frequency(self):
        """Test invalid frequency parameter."""
        # Note: Current implementation doesn't support frequency parameter
        # This test is for future functionality
        pass


@pytest.mark.integration
class TestFinancialFetcherIntegration:
    """Integration tests for FinancialFetcher with real market data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = FinancialFetcher()
        # Use well-known symbols for integration tests
        self.us_symbol = "MSFT"  # Microsoft
        self.intl_symbol = "RELIANCE.NS"  # Reliance Industries (NSE)
    
    @pytest.mark.slow
    def test_us_stock_financial_data(self):
        """Test fetching financial data for US stock."""
        statements = self.fetcher.fetch_all_statements(self.us_symbol)
        
        assert isinstance(statements, dict)
        
        # At least one statement should be available
        available_statements = [stmt for stmt in statements.values() if stmt is not None]
        assert len(available_statements) > 0
        
        # Test key metrics extraction
        metrics = self.fetcher.get_key_metrics(self.us_symbol)
        if metrics is not None:
            assert isinstance(metrics, pd.DataFrame)
            assert len(metrics.columns) > 0
    
    @pytest.mark.slow
    def test_international_stock_financial_data(self):
        """Test fetching financial data for international stock."""
        statements = self.fetcher.fetch_all_statements(self.intl_symbol)
        
        assert isinstance(statements, dict)
        
        # International stocks might have limited data availability
        # So we just check that the method doesn't crash
        for statement_name, statement_data in statements.items():
            if statement_data is not None:
                assert isinstance(statement_data, pd.DataFrame)
    
    @pytest.mark.slow
    def test_data_consistency_across_methods(self):
        """Test consistency between individual and batch fetching methods."""
        symbol = self.us_symbol
        
        # Fetch individually
        income_individual = self.fetcher.fetch_income_statement(symbol)
        balance_individual = self.fetcher.fetch_balance_sheet(symbol)
        cash_flow_individual = self.fetcher.fetch_cash_flow(symbol)
        
        # Fetch as batch
        all_statements = self.fetcher.fetch_all_statements(symbol)
        
        # Compare results (should be identical)
        if income_individual is not None and all_statements['income_statement'] is not None:
            pd.testing.assert_frame_equal(income_individual, all_statements['income_statement'])
        
        if balance_individual is not None and all_statements['balance_sheet'] is not None:
            pd.testing.assert_frame_equal(balance_individual, all_statements['balance_sheet'])
        
        if cash_flow_individual is not None and all_statements['cash_flow'] is not None:
            pd.testing.assert_frame_equal(cash_flow_individual, all_statements['cash_flow'])


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])